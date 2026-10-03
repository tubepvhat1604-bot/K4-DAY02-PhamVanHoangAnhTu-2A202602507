"""pipeline_checks.py - EDA và kiểm tra pipeline của Bước 0 (GUIDE.md mục 1.2 và 1.3).

Các hàm in số liệu và lưu ảnh để dán vào báo cáo:
    eda(train_df, val_df, test_df, images_dir, out_dir)   -> DataFrame số ảnh theo lớp
    initial_loss(backbone, loader, ...)                   -> loss CE của batch đầu (kỳ vọng ~ ln 9 = 2,197)
    overfit_one_batch(backbone, df, images_dir, ...)      -> list loss theo bước (kỳ vọng về gần 0)
    show_augmented(df, images_dir, aug, out_png, ...)     -> lưu ảnh sau augmentation + CutMix (đã giải chuẩn hoá)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, build_transforms, make_loader

# Table 1 của bài báo (Olsen et al. 2019), dùng để ĐỐI CHIẾU, không phải kết quả của mình.
PAPER_TABLE1 = {"Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
                "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
                "Negatives": 9106}


def eda(train_df, val_df, test_df, images_dir, out_dir, n_per_class: int = 3, n_stats: int = 300, seed: int = 0,
        labels_csv=None):
    """Biểu đồ phân bố lớp, đối chiếu Table 1, ảnh mẫu mỗi lớp, thống kê kích thước/kênh ảnh.
    Có `labels_csv` thì đối chiếu thêm nhãn trong 3 file fold với labels.csv (chỉ báo cáo, KHÔNG sửa CSV)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame({s: d["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                           for s, d in (("train", train_df), ("val", val_df), ("test", test_df))})
    counts.index = CLASS_NAMES
    counts["total"] = counts.sum(axis=1)
    counts["paper_table1"] = [PAPER_TABLE1[c] for c in CLASS_NAMES]
    counts["khớp_bài_báo"] = counts["total"] == counts["paper_table1"]
    print(counts.to_string())
    ratio = counts["total"].max() / counts["total"].min()
    print(f"\nLớp lớn nhất / lớp nhỏ nhất = {counts['total'].idxmax()} {counts['total'].max()} / "
          f"{counts['total'].idxmin()} {counts['total'].min()} = {ratio:.2f} lần")
    print(f"Tỉ lệ Negatives trên tổng: {100 * counts.loc['Negatives', 'total'] / counts['total'].sum():.1f}%")
    if labels_csv is not None:
        lab = pd.read_csv(labels_csv)[["Filename", "Label"]]
        allsplit = pd.concat([d.assign(split=s) for s, d in (("train", train_df), ("val", val_df), ("test", test_df))])
        m = allsplit.merge(lab, on="Filename", how="left", suffixes=("_fold", "_labelscsv"))
        diff = m[m["Label_fold"] != m["Label_labelscsv"]]
        print(f"\nĐối chiếu nhãn fold với labels.csv: {len(diff)} ảnh khác nhãn")
        if len(diff):
            print(diff[["Filename", "split", "Label_fold", "Label_labelscsv"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    x = np.arange(NUM_CLASSES)
    for i, s in enumerate(("train", "val", "test")):
        bars = ax.bar(x + (i - 1) * 0.27, counts[s], width=0.27, label=s)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, CLASS_NAMES, rotation=20)
    ax.set_ylabel("số ảnh"); ax.set_title("DeepWeeds fold 0: số ảnh theo lớp và theo tập")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "eda_class_distribution.png"); plt.show()

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(n_per_class, NUM_CLASSES, figsize=(2 * NUM_CLASSES, 2.1 * n_per_class), dpi=100)
    for c in range(NUM_CLASSES):
        files = train_df.loc[train_df["Label"] == c, "Filename"].to_numpy()
        for r, f in enumerate(rng.choice(files, n_per_class, replace=False)):
            a = axes[r, c]
            a.imshow(Image.open(os.path.join(images_dir, f)).convert("RGB"))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(CLASS_NAMES[c], fontsize=9)
    fig.suptitle("Ảnh mẫu tập train (mỗi cột một lớp)")
    fig.tight_layout(); fig.savefig(out_dir / "eda_samples.png"); plt.show()

    allf = pd.concat([train_df, val_df, test_df])["Filename"].to_numpy()
    sizes, modes = {}, {}
    for f in rng.choice(allf, min(n_stats, len(allf)), replace=False):
        with Image.open(os.path.join(images_dir, f)) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
    print(f"\nThống kê {sum(sizes.values())} ảnh ngẫu nhiên: kích thước {sizes}, chế độ màu {modes}")
    return counts


def _model(backbone, pretrained, device):
    from model import build_model
    return build_model(backbone, pretrained=pretrained, num_classes=NUM_CLASSES).to(device)


def initial_loss(backbone: str, loader, pretrained: bool = True, device=None) -> float:
    """CE của batch đầu với head mới (model.eval(), không gradient). Kỳ vọng ~ -ln(1/9) = 2,197."""
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = _model(backbone, pretrained, device).eval()
    x, y, _ = next(iter(loader))
    with torch.inference_mode():
        loss = torch.nn.functional.cross_entropy(model(x.to(device)).float(), y.to(device)).item()
    print(f"Loss ban đầu ({backbone}) = {loss:.4f}; kỳ vọng ln(9) = {math.log(9):.4f}; "
          f"lệch {abs(loss - math.log(9)):.4f}")
    return loss


def overfit_one_batch(backbone: str, df, images_dir, n: int = 16, steps: int = 60, lr: float = 1e-3,
                      img_size: int = 224, pretrained: bool = True, device=None) -> list[float]:
    """Huấn luyện lặp lại trên MỘT batch nhỏ cố định (không augmentation): loss phải về gần 0."""
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    sub = df.groupby("Label", group_keys=False).head(max(1, n // NUM_CLASSES + 1)).head(n)
    loader = make_loader(sub, images_dir, build_transforms(False, img_size), batch_size=n,
                         train=False, num_workers=0)
    x, y, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model = _model(backbone, pretrained, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    losses = []
    for s in range(steps):
        loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if s % 10 == 0 or s == steps - 1:
            print(f"  bước {s:3d}: loss {loss.item():.4f}")
    model.eval()
    with torch.inference_mode():
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"Overfit {len(y)} ảnh: loss {losses[0]:.3f} -> {losses[-1]:.4f}; accuracy trên batch (eval) {acc:.2f}")
    return losses


def denormalize(x):
    """Tensor (C,H,W) đã chuẩn hoá -> ảnh numpy (H,W,C) trong [0, 1]."""
    m = np.array(IMAGENET_MEAN)[:, None, None]
    s = np.array(IMAGENET_STD)[:, None, None]
    return np.clip(x.numpy() * s + m, 0, 1).transpose(1, 2, 0)


def show_augmented(df, images_dir, aug: str, out_png, n: int = 8, img_size: int = 224, seed: int = 0):
    """Hàng 1: ảnh gốc; hàng 2: sau augmentation `aug`; hàng 3: sau CutMix (in lam và hai nhãn)."""
    import matplotlib.pyplot as plt
    import torch
    from losses import mix_batch

    torch.manual_seed(seed)
    sub = df.sample(n, random_state=seed)
    plain = make_loader(sub, images_dir, build_transforms(False, img_size), n, train=False, num_workers=0)
    augl = make_loader(sub, images_dir, build_transforms(True, img_size, aug), n, train=False, num_workers=0)
    x0, y, names = next(iter(plain))
    x1, y1, names1 = next(iter(augl))
    assert list(names) == list(names1) and torch.equal(y, y1), "ảnh và nhãn lệch thứ tự"
    xm, (ya, yb, lam) = mix_batch(x1, y1, 1.0, "cutmix", rng=np.random.default_rng(seed))
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.6), dpi=100)
    for i in range(n):
        for r, (img, t) in enumerate(((x0[i], f"{CLASS_NAMES[y[i]]}"), (x1[i], f"aug: {aug}"),
                                      (xm[i], f"{CLASS_NAMES[ya[i]][:10]} + {CLASS_NAMES[yb[i]][:10]}"))):
            axes[r, i].imshow(denormalize(img)); axes[r, i].set_title(t, fontsize=8)
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    fig.suptitle(f"Hàng 1: gốc (val transform) | hàng 2: train aug '{aug}' | hàng 3: CutMix, lam = {lam:.3f} "
                 f"(phần ảnh gốc giữ lại)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.show()
    print(f"CutMix lam = {lam:.3f} (= 1 - diện tích hộp dán / diện tích ảnh)")


def time_backbones(backbones, train_df, images_dir, batch_size: int = 64, img_size: int = 224,
                   steps: int = 30, warmup: int = 5, num_workers: int = 2, n_val: int = 3501,
                   pretrained: bool = True):
    """Đo nhanh thời gian train mỗi bước (AMP, công thức nền) để ƯỚC LƯỢNG thời gian 1 epoch mỗi backbone.

    epoch ước lượng = s/bước x số bước một epoch (len(train)//batch) + thời gian đánh giá val.
    Đây là ước lượng để lập ngân sách; thời gian thật lấy từ history.csv của từng run.
    """
    import time
    import torch
    from model import build_model, param_groups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(train_df, images_dir, build_transforms(True, img_size), batch_size, train=True,
                         num_workers=num_workers, cache=True)
    steps_per_epoch = len(train_df) // batch_size
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    rows = []
    for name in backbones:
        model = build_model(name, pretrained=pretrained).to(device)
        opt = torch.optim.AdamW(param_groups(model, 1e-4, 1e-3, 0.05))
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        it = iter(loader)
        model.train()
        t_start = None
        for s in range(warmup + steps):
            if s == warmup:
                sync(); t_start = time.perf_counter()
            try:
                x, y, _ = next(it)
            except StopIteration:
                it = iter(loader); x, y, _ = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sync()
        s_per_step = (time.perf_counter() - t_start) / steps
        model.eval()
        xv = torch.randn(batch_size, 3, img_size, img_size, device=device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            model(xv); sync(); t0 = time.perf_counter()
            for _ in range(5):
                model(xv)
            sync()
        val_s = (time.perf_counter() - t0) / 5 * math.ceil(n_val / batch_size)
        epoch_min = (s_per_step * steps_per_epoch + val_s) / 60
        rows.append({"backbone": name, "s_per_step": round(s_per_step, 3), "img_per_s": round(batch_size / s_per_step),
                     "epoch_min_est": round(epoch_min, 2), "run_12ep_min_est": round(12 * epoch_min, 1)})
        print(rows[-1], flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
