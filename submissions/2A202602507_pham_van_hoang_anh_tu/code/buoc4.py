"""buoc4.py - Bước 4: chạy TEST đúng một lần mỗi seed cho chung kết F01 và mốc T00 + I00.

- F01: phương pháp suy luận đã chốt ở Bước 3 (logs/buoc3_choice.json) + temperature scaling,
  T khớp trên VAL của từng seed rồi áp dụng sang test. Ghi:
    predictions/F01_seed<k>_test.csv        (sau TS: dùng chấm I1-I4)
    predictions/F01_uncal_seed<k>_test.csv  (cùng cấu hình, chưa TS: chấm I4a)
    predictions/F01_seed<k>_val.csv         (dự đoán val của chung kết, sau TS: chấm I4b)
- T00: 1 view (I00), không TS: predictions/T00_seed<k>_test.csv
Đã có bất kỳ file *_test.csv nào thì DỪNG (quy tắc: test chạy một lần, không chạy lại).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import inference as inf
from buoc3 import LOADER_SIZE, _views


def run_test(runs_dir, images_dir, labels_dir, sub_dir, backbone="convnext_tiny", final_id="F01",
             base_id="T00", seeds=(0, 1, 2), device=None, batch_size=64, num_workers=2):
    import torch
    import train
    from dataset import build_transforms, load_split, make_loader
    from model import build_model

    ev = train._import_eval()
    sub = Path(sub_dir)
    pred = sub / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    done = sorted(pred.glob("*_test.csv"))
    if done:
        raise RuntimeError(f"Đã có {len(done)} file dự đoán test (ví dụ {done[0].name}): test chỉ chạy MỘT lần. "
                           "Không chạy lại ô này.")
    choice = json.loads((sub / "logs" / "buoc3_choice.json").read_text())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, val_df, test_df = load_split(labels_dir, 0)

    def loader(df, key):
        return make_loader(df, images_dir, build_transforms(False, LOADER_SIZE[key]), batch_size, train=False,
                           num_workers=num_workers)

    def load(exp_id, seed):
        m = build_model(backbone, pretrained=False, num_classes=9)
        m.load_state_dict(torch.load(Path(runs_dir) / exp_id / f"seed{seed}" / "best.pt", map_location="cpu",
                                     weights_only=True))
        return m.to(device).eval()

    key, vnames, space = choice["loader"], choice["views"], choice["aggregate"]
    vl, tl = loader(val_df, key), loader(test_df, key)
    info = {"choice": choice, "temperature": {}}
    for seed in seeds:
        model = load(final_id, seed)
        vn, vy, v_logits = inf.predict_views(model, vl, device, _views(vnames), amp=True)
        T = inf.fit_temperature_views(v_logits, vy, space)          # khớp trên VAL
        info["temperature"][f"seed{seed}"] = T
        ev.save_predictions(pred / f"{final_id}_seed{seed}_val.csv", vn, vy,
                            inf.aggregate_views([l / T for l in v_logits], space))
        tn, ty, t_logits = inf.predict_views(model, tl, device, _views(vnames), amp=True)   # TEST: một lần
        ev.save_predictions(pred / f"{final_id}_uncal_seed{seed}_test.csv", tn, ty, inf.aggregate_views(t_logits, space))
        ev.save_predictions(pred / f"{final_id}_seed{seed}_test.csv", tn, ty,
                            inf.aggregate_views([l / T for l in t_logits], space))
        print(f"{final_id} seed{seed}: T = {T:.3f}, đã ghi dự đoán val + test")
        del model

    tl1 = loader(test_df, "224")
    for seed in seeds:
        model = load(base_id, seed)
        tn, ty, logits = inf.predict_logits(model, tl1, device, amp=True)
        ev.save_predictions(pred / f"{base_id}_seed{seed}_test.csv", tn, ty, inf.apply_temperature(logits, 1.0))
        print(f"{base_id} seed{seed} (I00, không TS): đã ghi dự đoán test")
        del model
    (sub / "logs" / "buoc4_test_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def plot_confusion(pred_glob, labels_dir, path, title):
    """Ma trận nhầm lẫn test (cộng dồn các seed, chuẩn hoá theo hàng = recall), kèm số ảnh."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from dataset import CLASS_NAMES
    cm = np.zeros((9, 9), dtype=np.int64)
    files = sorted(glob.glob(str(pred_glob)))
    for f in files:
        d = pd.read_csv(f)
        np.add.at(cm, (d.y_true.to_numpy(), d.y_pred.to_numpy()), 1)
    rec = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    ax.imshow(rec, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if rec[i, j] > 0.5 else "black")
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(9), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(f"{title} ({len(files)} seed cộng dồn; màu = recall theo hàng)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return cm
