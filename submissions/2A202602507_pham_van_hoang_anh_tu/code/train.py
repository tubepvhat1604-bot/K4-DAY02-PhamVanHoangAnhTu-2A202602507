"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Những gì mỗi lần chạy lưu trong run_dir(cfg) = <out_dir>/<exp_id>/seed<k>/ (đặt out_dir trên Drive, NGOÀI repo):
    config.json, history.csv, last.pt (checkpoint SAU MỖI EPOCH để resume), best.pt (macro-F1 val cao nhất),
    val_logits.npy + val_pred.csv, (Bước 4) test_logits.npy, done.json (tóm tắt; có file này thì run được bỏ qua)
Ảnh đường cong: <curves_dir>/<exp_id>[_seed<k>]_<desc>.png (nằm trong thư mục bài nộp).

Lựa chọn đã ghi lại:
  - Lịch LR cập nhật theo BƯỚC: warmup tuyến tính từ ~0 trong `warmup_epochs`, sau đó cosine về 0.
  - Chọn checkpoint theo macro-F1 val (eval.compute_metrics của repo gốc); hòa thì giữ epoch sớm hơn.
  - Có EMA thì đánh giá và chọn checkpoint bằng trọng số EMA; vẫn ghi macro-F1 val của trọng số thường.
  - Tái lập: seed random/numpy/torch, seed cho DataLoader theo từng epoch; cudnn.benchmark=True để nhanh
    (không bảo đảm giống từng bit giữa hai lần chạy trên GPU).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_eval():
    """Tìm eval.py của repo gốc (đi ngược lên từ thư mục chứa file này, rồi cwd)."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (d / "eval.py").exists() and (d / "tests").is_dir():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            break
    import eval as ev  # noqa: E402
    return ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves, ví dụ "resnet50", "cutmix"
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    pretrained: bool = True
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | dihedral
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    cache_images: bool = True         # nạp trước ảnh train/val vào RAM
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0      # loss="ls" mà để 0 thì dùng 0.1
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    grad_clip: float | None = None
    amp: bool = True
    num_workers: int = 2
    max_steps_per_epoch: int | None = None   # CHỈ để chạy thử nhanh; thí nghiệm thật để None
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    logs_dir: str | None = "logs"     # bản sao nhỏ (config/history/done) trong thư mục nộp để truy ngược exp_id
    expected_total: int | None = 17509  # chỉ đặt None khi chạy thử trên bộ ảnh mẫu
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    export_val_predictions: bool = False  # Bước 4: chép dự đoán val vào pred_dir (chấm I4b)


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """curves/<exp_id>_<desc>.png; nhiều seed (T00, F01) thì thêm _seed<k> cho seed > 0."""
    name = cfg.exp_id + (f"_seed{cfg.seed}" if cfg.seed else "") + (f"_{cfg.desc}" if cfg.desc else "")
    return Path(cfg.curves_dir) / f"{name}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). Seed DataLoader đặt riêng theo epoch trong run()."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (model.param_groups): weight decay không áp dụng cho norm/bias."""
    import torch
    from model import param_groups
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số nhân LR: warmup tuyến tính (bắt đầu 1/warmup_steps) rồi cosine về 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0, cập nhật theo bước (gọi scheduler.step() sau mỗi optimizer.step())."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W.

    Giữ một bản sao riêng (self.module) để đánh giá. Tham số dạng số thực được trung bình;
    buffer (running_mean/var của BatchNorm, num_batches_tracked) được CHÉP từ model đang train,
    nên thống kê BN luôn khớp với dữ liệu gần nhất.
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        import torch
        with torch.no_grad():
            for e, p in zip(self.module.parameters(), model.parameters()):
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            for e, b in zip(self.module.buffers(), model.buffers()):
                e.copy_(b)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd) -> None:
        self.module.load_state_dict(sd)


def _autocast(device, enabled: bool):
    import torch
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, rng: np.random.Generator | None = None,
                    lr_log: list | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss": ..., "lr": ..., "steps": ...}."""
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)  # backbone đóng băng thì BN/dropout của nó ở eval
    total, n, steps = 0.0, 0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast(device, cfg.amp):
            if cfg.mix:
                x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng=rng)
                loss = mixed_loss(criterion, model(x), targets)
            else:
                loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} ở bước {steps}; giảm LR hoặc kiểm tra dữ liệu")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if lr_log is not None:
            lr_log.append(optimizer.param_groups[0]["lr"])
        total += loss.item() * x.shape[0]
        n += x.shape[0]
        steps += 1
        if cfg.max_steps_per_epoch and steps >= cfg.max_steps_per_epoch:
            break
    return {"train_loss": total / max(n, 1), "lr": optimizer.param_groups[0]["lr"], "steps": steps}


def evaluate(model, loader, criterion, device, amp: bool = True):
    """Chạy model ở chế độ eval, KHÔNG gradient, giữ đúng thứ tự loader.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    """
    import torch
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    loss = float(criterion(logits, y_true)) if criterion is not None else float("nan")
    return names, y_true.numpy(), logits.numpy(), loss


def softmax_np(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list | None = None) -> None:
    """Ảnh đường cong: (1) loss train/val, (2) macro-F1 và top-1 val, (3) LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    ncol = 3 if lr_steps else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4), dpi=120)
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s--", label="val top-1")
    if "val_macro_f1_raw" in history[0]:
        ax[1].plot(ep, [h["val_macro_f1_raw"] for h in history], "^:", label="val macro-F1 (không EMA)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best ep {best['epoch']}: {best['val_macro_f1']:.4f}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metric"); ax[1].set_title("Val metric"); ax[1].legend(); ax[1].grid(alpha=.3)
    if lr_steps:
        ax[2].plot(np.arange(1, len(lr_steps) + 1), lr_steps)
        ax[2].set_xlabel("bước (iteration)"); ax[2].set_ylabel("LR backbone"); ax[2].set_title("Lịch LR"); ax[2].grid(alpha=.3)
    from matplotlib.ticker import MaxNLocator
    for a in ax[:2]:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _quick_latency_ms(model, img_size: int, device, amp: bool, warmup: int = 10, iters: int = 30) -> float:
    """Độ trễ SƠ BỘ batch 1 (p50) cho sheet Backbones; đo kỹ ở benchmark.py (Bước 3)."""
    import torch
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ts = []
    with torch.inference_mode(), _autocast(device, amp):
        for i in range(warmup + iters):
            sync(); t0 = time.perf_counter(); model(x); sync()
            if i >= warmup:
                ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _atomic_save(obj, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _versions() -> dict:
    import torch, torchvision, timm
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def _metrics(ev, y_true, logits) -> dict:
    probs = softmax_np(logits)
    m = ev.compute_metrics(y_true, probs.argmax(1), probs)
    return m


def export_logs(cfg: Config) -> None:
    """Chép config.json, history.csv, done.json (vài KB) sang <logs_dir>/<exp_id>_seed<k>/ trong thư mục nộp,
    để mọi số trong bảng truy ngược được tới log (RUBRIC P2). Checkpoint KHÔNG chép."""
    import shutil
    if not cfg.logs_dir:
        return
    dst = Path(cfg.logs_dir) / f"{cfg.exp_id}_seed{cfg.seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "history.csv", "done.json"):
        f = run_dir(cfg) / name
        if f.exists():
            shutil.copy(f, dst / name)


def collect_summaries(out_dir: str | Path, prefix: str = "") -> "pd.DataFrame":
    """Gom done.json của mọi run (exp_id bắt đầu bằng `prefix`) thành một bảng."""
    import pandas as pd
    rows = []
    for f in sorted(Path(out_dir).glob(f"{prefix}*/seed*/done.json")):
        d = json.loads(f.read_text())
        d.pop("val_f1_per_class", None)
        rows.append(d)
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    - Có done.json thì KHÔNG train lại (bỏ qua run đã xong); có last.pt thì train tiếp từ epoch đó (resume).
    - Test chỉ chạy khi cfg.save_test_predictions=True, đúng MỘT lần (đã có file test thì báo lỗi).
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import pandas as pd
    import torch
    from dataset import build_transforms, check_split, load_split, make_loader
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params, weight_tag

    ev = _import_eval()
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    done_file = rd / "done.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir, expected_total=cfg.expected_total, verbose=False)
    eval_tf = build_transforms(False, cfg.img_size)

    def new_model():
        m = build_model(cfg.backbone, cfg.pretrained, 9, cfg.drop_rate, cfg.init)
        return m.to(device)

    if done_file.exists():
        summary = json.loads(done_file.read_text())
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong trước đó (best epoch {summary['best_epoch']}, "
              f"macro-F1 val {summary['val_macro_f1']:.4f}): bỏ qua huấn luyện")
    else:
        summary = _train(cfg, rd, device, ev, train_df, val_df, eval_tf, new_model,
                         build_transforms, make_loader, build_criterion, class_weights,
                         count_params, count_gmacs, weight_tag)
    export_logs(cfg)

    # ---- xuất dự đoán (Bước 4): từ best.pt, không train lại ----
    if cfg.export_val_predictions:
        vp = pred_path(cfg, "val")
        logits = np.load(rd / "val_logits.npy")
        names = pd.read_csv(rd / "val_pred.csv")["Filename"].tolist()
        y = pd.read_csv(rd / "val_pred.csv")["y_true"].to_numpy()
        ev.save_predictions(vp, names, y, softmax_np(logits))
        print("Đã ghi", vp)
    if cfg.save_test_predictions:
        tp = pred_path(cfg, "test")
        if tp.exists() or (rd / "test_logits.npy").exists():
            raise RuntimeError(f"Test của {cfg.exp_id} seed{cfg.seed} ĐÃ chạy rồi ({tp}). "
                               "Quy tắc: test chạy đúng một lần mỗi seed; không chạy lại.")
        model = new_model()
        model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=True))
        test_loader = make_loader(test_df, cfg.images_dir, eval_tf, cfg.batch_size, train=False,
                                  num_workers=cfg.num_workers, seed=cfg.seed)
        names, y, logits, _ = evaluate(model, test_loader, None, device, cfg.amp)
        np.save(rd / "test_logits.npy", logits)
        pd.DataFrame({"Filename": names, "y_true": y}).to_csv(rd / "test_names.csv", index=False)
        ev.save_predictions(tp, names, y, softmax_np(logits))
        print("Đã ghi", tp, "(test chạy 1 lần; xem chỉ số bằng eval.py score)")
    return summary


def _train(cfg, rd, device, ev, train_df, val_df, eval_tf, new_model, build_transforms, make_loader,
           build_criterion, class_weights, count_params, count_gmacs, weight_tag) -> dict:
    import pandas as pd
    import torch

    set_seed(cfg.seed)
    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                               cfg.batch_size, train=True, sampler=cfg.sampler,
                               num_workers=cfg.num_workers, seed=cfg.seed, cache=cfg.cache_images)
    val_loader = make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size, train=False,
                             num_workers=cfg.num_workers, seed=cfg.seed, cache=cfg.cache_images)
    steps_per_epoch = len(train_loader)
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)

    model = new_model()
    counts = np.bincount(train_df["Label"].to_numpy(), minlength=9)  # CHỈ số liệu train
    kw = {"smoothing": cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        kw["weight"] = class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = build_criterion(cfg.loss, **kw)
    val_criterion = torch.nn.CrossEntropyLoss()  # val loss luôn là CE thường để so được giữa các run
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    info = {"config": dataclasses.asdict(cfg), "versions": _versions(), "weight_tag": weight_tag(model),
            "params_M": count_params(model), "gmacs": count_gmacs(model, cfg.img_size),
            "n_train": len(train_df), "n_val": len(val_df), "train_counts": counts.tolist(),
            "steps_per_epoch": steps_per_epoch, "device": str(device)}
    (rd / "config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    history, lr_steps, start_epoch = [], [], 1
    best = {"f1": -1.0, "epoch": 0}
    last = rd / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        history, lr_steps, best, start_epoch = ck["history"], ck["lr_steps"], ck["best"], ck["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] resume từ epoch {ck['epoch']} (best {best})")

    title = f"{cfg.exp_id} seed{cfg.seed} | {cfg.backbone} ({info['weight_tag'] or 'không pretrained'}) | {cfg.desc}"
    for epoch in range(start_epoch, cfg.epochs + 1):
        torch.manual_seed(cfg.seed * 10007 + epoch)          # augmentation trong main process
        train_loader.generator.manual_seed(cfg.seed * 10007 + epoch)  # thứ tự batch + seed worker
        rng = np.random.default_rng([cfg.seed, epoch])        # Mixup/CutMix
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device,
                             ema, rng, lr_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        eval_model = ema.module if ema is not None else model
        names, y, logits, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        m = _metrics(ev, y, logits)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "val_loss": vloss,
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"],
               "val_ece": m["ece"], "val_f1_chinee": m["f1"][0], "val_f1_snake": m["f1"][7],
               "lr_end": tr["lr"], "train_time_s": train_time, "eval_time_s": time.time() - t0 - train_time}
        if ema is not None:
            _, yr, lr_, _ = evaluate(model, val_loader, None, device, cfg.amp)
            row["val_macro_f1_raw"] = _metrics(ev, yr, lr_)["macro_f1"]
        history.append(row)
        if m["macro_f1"] > best["f1"]:  # hòa thì giữ epoch sớm hơn
            best = {"f1": m["macro_f1"], "epoch": epoch}
            _atomic_save(eval_model.state_dict(), rd / "best.pt")
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "ema": ema.state_dict() if ema is not None else None,
                      "history": history, "lr_steps": lr_steps, "best": best, "epoch": epoch}, last)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        plot_curves(history, curve_path(cfg), title, lr_steps)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch:2d}/{cfg.epochs} | train loss {tr['train_loss']:.4f} | "
              f"val loss {vloss:.4f} | val macro-F1 {m['macro_f1']:.4f} | top-1 {m['top1']:.4f} | "
              f"{train_time:.0f}s train" + (" | *best*" if best["epoch"] == epoch else ""), flush=True)

    # ---- nạp checkpoint tốt nhất, lưu logit + dự đoán val ----
    best_model = new_model()
    best_model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=True))
    names, y, logits, _ = evaluate(best_model, val_loader, None, device, cfg.amp)
    np.save(rd / "val_logits.npy", logits)
    ev.save_predictions(rd / "val_pred.csv", names, y, softmax_np(logits))
    m = _metrics(ev, y, logits)
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "desc": cfg.desc,
               "weight_tag": info["weight_tag"], "params_M": round(info["params_M"], 3),
               "gmacs": round(info["gmacs"], 3), "img_size": cfg.img_size, "epochs": cfg.epochs,
               "best_epoch": best["epoch"], "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"],
               "val_f1_per_class": [float(v) for v in m["f1"]],
               "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
               "latency_b1_ms_prelim": _quick_latency_ms(best_model, cfg.img_size, device, cfg.amp),
               "device": info["device"], "curve": str(curve_path(cfg))}
    (rd / "done.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    last.unlink(missing_ok=True)  # đã xong: bỏ checkpoint resume cho nhẹ Drive (best.pt vẫn giữ)
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, val = pair.split("=", 1)
        if key not in fields:
            raise KeyError(f"Config không có trường '{key}'. Các trường: {sorted(fields)}")
        typ = str(fields[key].type)
        low = val.strip().lower()
        if low in ("none", "null") and "None" in typ:
            out[key] = None
        elif typ.startswith("bool"):
            if low not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key} cần true/false, nhận '{val}'")
            out[key] = low in ("true", "1", "yes")
        elif typ.startswith("int"):
            out[key] = int(val)
        elif typ.startswith("float"):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def main() -> None:
    """`python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình Lab Day 2")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
