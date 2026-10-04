"""buoc3.py - Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại, không đụng test).

Mô hình: cấu hình cuối F01 (3 seed). Các phương pháp I00-I08 đo trên F01 seed 0; ensemble/soup dùng cả 3 seed.
Quy tắc chọn phương pháp suy luận cho chung kết (ghi sẵn, chỉ dùng val):
  trong các phương pháp MỘT mô hình (I00, I01, I02, I04), lấy macro-F1 val TRUNG BÌNH 3 seed cao nhất;
  hòa (chênh < 1e-4) thì lấy phương pháp ít view hơn. Temperature scaling (I07) luôn áp dụng thêm
  (T khớp trên val của từng seed), vì không đổi argmax mà chỉ sửa hiệu chuẩn.

Kết quả ghi vào <sub>/logs/: inference_val.csv, latency.csv, bn_fusion_check.json, buoc3_choice.json,
và biểu đồ <sub>/curves/buoc3_tradeoff.png. Có buoc3_choice.json thì bỏ qua (chạy lại không tính lại).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import inference as inf
from benchmark import latency_report, tta_latency

# phương pháp một mô hình: tên -> (loader, danh sách view, cách gộp, số view)
SINGLE = {
    "I00_1view":       ("224", ["id"], "logit", 1),
    "I01_hflip_prob":  ("224", ["id", "hflip"], "prob", 2),
    "I01_hflip_logit": ("224", ["id", "hflip"], "logit", 2),
    "I02_5crop_prob":  ("256", ["5crop"], "prob", 5),
    "I02_5crop_logit": ("256", ["5crop"], "logit", 5),
    "I04_res256":      ("256", ["id"], "logit", 1),
    "I04_res288":      ("288", ["id"], "logit", 1),
    "I04_res320":      ("320", ["id"], "logit", 1),
}
LOADER_SIZE = {"224": 224, "256": 256, "288": 288, "320": 320}   # kích thước tensor vào model (5crop: 224)


def _views(names):
    out = []
    for n in names:
        out.append({"id": inf.view_identity, "hflip": inf.view_hflip,
                    "5crop": lambda x: inf.views_multicrop(x, 224)}[n])
    return out


def _metrics(ev, y, probs):
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    return {"val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"],
            "val_ece": m["ece"], "val_nll": m["nll"], "f1_chinee": float(m["f1"][0]), "f1_snake": float(m["f1"][7])}


def _cross_fit_ece(ev, logits, y, seed=0):
    """ECE sau TS khi T khớp trên một nửa val và đo trên nửa còn lại (2-fold), để không lạc quan."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]
    probs = np.zeros_like(logits, dtype=np.float64)
    for a, b in (halves, halves[::-1]):
        probs[b] = inf.apply_temperature(logits[b], inf.fit_temperature(logits[a], y[a]))
    return ev.ece_score(probs, y)


def run_buoc3(runs_dir, images_dir, labels_dir, sub_dir, backbone="convnext_tiny", exp_id="F01",
              seeds=(0, 1, 2), device=None, latency_iters=100, bn_demo_backbone="resnet50",
              bn_demo_pretrained=True, batch_size=64, num_workers=2, force=False):
    import pandas as pd
    import torch
    import train
    from dataset import build_transforms, load_split, make_loader
    from model import build_model

    ev = train._import_eval()
    sub = Path(sub_dir)
    (sub / "logs").mkdir(parents=True, exist_ok=True)
    (sub / "curves").mkdir(parents=True, exist_ok=True)
    choice_file = sub / "logs" / "buoc3_choice.json"
    if choice_file.exists() and not force:
        print("Bước 3 đã chạy trước đó: đọc lại kết quả từ", sub / "logs")
        return (pd.read_csv(sub / "logs" / "inference_val.csv"), pd.read_csv(sub / "logs" / "latency.csv"),
                json.loads(choice_file.read_text()))

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, val_df, _ = load_split(labels_dir, 0)
    loaders = {k: make_loader(val_df, images_dir, build_transforms(False, s), batch_size, train=False,
                              num_workers=num_workers, cache=True) for k, s in LOADER_SIZE.items()}

    def load(seed, sd=None):
        m = build_model(backbone, pretrained=False, num_classes=9)
        m.load_state_dict(sd if sd is not None else
                          torch.load(Path(runs_dir) / exp_id / f"seed{seed}" / "best.pt", map_location="cpu",
                                     weights_only=True))
        return m.to(device).eval()

    # ---- 1) mọi phương pháp một mô hình, cho từng seed (logit mỗi view) ----
    per_seed = {}          # seed -> method -> probs
    raw = {}               # seed -> loader -> view -> logits (để gộp)
    y = names = None
    for seed in seeds:
        model = load(seed)
        raw[seed] = {}
        for key, ld in loaders.items():
            need = sorted({v for (lk, vs, _, _) in SINGLE.values() if lk == key for v in vs})
            n, yy, outs = inf.predict_views(model, ld, device, _views(need), amp=True)
            if y is None:
                y, names = yy, n
            assert list(n) == list(names), "thứ tự file val phải giống nhau giữa các loader"
            k = 0
            raw[seed][key] = {}
            for v in need:
                cnt = 5 if v == "5crop" else 1
                raw[seed][key][v] = outs[k:k + cnt]
                k += cnt
        per_seed[seed] = {}
        for meth, (lk, vs, space, _) in SINGLE.items():
            logit_list = [l for v in vs for l in raw[seed][lk][v]]
            per_seed[seed][meth] = inf.aggregate_views(logit_list, space)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"seed {seed}: xong các phương pháp một mô hình")

    s0 = seeds[0]
    rows = []
    for meth, (lk, vs, space, k) in SINGLE.items():
        m0 = _metrics(ev, y, per_seed[s0][meth])
        f1s = [_metrics(ev, y, per_seed[s][meth])["val_macro_f1"] for s in seeds]
        rows.append({"method": meth, "model": f"{exp_id} seed{s0}", "views": k, "aggregate": space,
                     "input_size": LOADER_SIZE[lk] if vs != ["5crop"] else 224, **m0,
                     "val_macro_f1_mean_3seed": float(np.mean(f1s)), "val_macro_f1_std_3seed": float(np.std(f1s, ddof=1))})

    # ---- 2) I05 ensemble 3 seed, I06 soup 3 seed (trên 1 view 224) ----
    p_ens = inf.ensemble_probs([per_seed[s]["I00_1view"] for s in seeds])
    rows.append({"method": "I05_ensemble3", "model": f"{exp_id} seed{','.join(map(str, seeds))}",
                 "views": len(seeds), "aggregate": "prob", "input_size": 224, **_metrics(ev, y, p_ens)})
    sds = [torch.load(Path(runs_dir) / exp_id / f"seed{s}" / "best.pt", map_location="cpu", weights_only=True)
           for s in seeds]
    soup = load(None, inf.uniform_soup(sds))
    _, _, l_soup = inf.predict_logits(soup, loaders["224"], device, amp=True)
    rows.append({"method": "I06_soup3", "model": f"{exp_id} soup seed{','.join(map(str, seeds))}", "views": 1,
                 "aggregate": "-", "input_size": 224, **_metrics(ev, y, inf.apply_temperature(l_soup, 1.0))})

    # ---- 3) I07 temperature scaling trên I00 (seed 0) ----
    l00 = raw[s0]["224"]["id"][0]
    T = inf.fit_temperature(l00, y)
    m_ts = _metrics(ev, y, inf.apply_temperature(l00, T))
    rows.append({"method": "I07_temp_scaling", "model": f"{exp_id} seed{s0}", "views": 1, "aggregate": "-",
                 "input_size": 224, **m_ts, "T": T, "val_ece_before": _metrics(ev, y, inf.apply_temperature(l00, 1.0))["val_ece"],
                 "val_ece_crossfit": _cross_fit_ece(ev, l00, y)})

    # ---- 4) I08 dtype: FP32 và FP16 (I00 là AMP) ----
    model = load(s0)
    _, _, l32 = inf.predict_logits(model, loaders["224"], device, amp=False)
    rows.append({"method": "I08_fp32", "model": f"{exp_id} seed{s0}", "views": 1, "aggregate": "-", "input_size": 224,
                 **_metrics(ev, y, inf.apply_temperature(l32, 1.0))})
    if device.type == "cuda":
        m16 = load(s0).half()
        outs = []
        with torch.inference_mode():
            for x, _, _ in loaders["224"]:
                outs.append(m16(x.to(device).half()).float().cpu())
        l16 = torch.cat(outs).numpy()
        rows.append({"method": "I08_fp16", "model": f"{exp_id} seed{s0}", "views": 1, "aggregate": "-",
                     "input_size": 224, **_metrics(ev, y, inf.apply_temperature(l16, 1.0)),
                     "max_abs_logit_diff_vs_fp32": float(np.abs(l16 - l32).max())})
        del m16
    res = pd.DataFrame(rows)
    i00 = res.loc[res.method == "I00_1view", "val_macro_f1"].item()
    res["delta_vs_I00"] = res["val_macro_f1"] - i00

    # ---- 5) độ trễ (batch 1, forward model, tensor đã ở GPU, không tính tiền xử lý) ----
    kw = dict(device=str(device), iters=latency_iters, warmup=10)
    lat = [latency_report(model, 1, 224, dtype="amp", label="I00_1view", **kw),
           latency_report(model, 1, 224, dtype="fp32", label="I08_fp32", **kw),
           latency_report(model, 32, 224, dtype="amp", label="I00_1view b32", **kw),
           latency_report(model, 32, 224, dtype="fp32", label="I08_fp32 b32", **kw),
           tta_latency(model, 2, 224, dtype="amp", label="I01_hflip", **kw),
           tta_latency(model, 5, 224, dtype="amp", label="I02_5crop", **kw)]
    if device.type == "cuda":
        lat += [latency_report(model, 1, 224, dtype="fp16", label="I08_fp16", **kw),
                latency_report(model, 32, 224, dtype="fp16", label="I08_fp16 b32", **kw)]
    for s in (256, 288, 320):
        lat.append(latency_report(model, 1, s, dtype="amp", label=f"I04_res{s}", **kw))
    lat.append(latency_report(soup, 1, 224, dtype="amp", label="I06_soup3", **kw))
    ens_models = [load(s) for s in seeds]
    xe = torch.randn(1, 3, 224, 224, device=device)
    import benchmark
    sync = torch.cuda.synchronize if device.type == "cuda" else None

    def ens_fn():
        with torch.inference_mode(), inf._autocast(device, True):
            for m in ens_models:
                m(xe)
    r = benchmark.bench(ens_fn, warmup=10, iters=latency_iters, sync=sync)
    lat.append({"label": "I05_ensemble3", "gpu": lat[0]["gpu"], "dtype": "amp", "batch": 1, "img_size": 224,
                "fused_bn": False, "preprocessing": "không tính", **r, "images_per_s": 1000.0 / r["p50"],
                "torch": torch.__version__})
    lat = pd.DataFrame(lat)
    del ens_models

    # ---- 6) gộp BN: ConvNeXt không có BatchNorm -> không áp dụng; kiểm tra công thức trên ResNet-50 ----
    n_cnx = inf.fuse_conv_bn(model).n_fused
    rn = build_model(bn_demo_backbone, pretrained=bn_demo_pretrained, num_classes=9).to(device).eval()
    rn_f = inf.fuse_conv_bn(rn)
    xb = torch.randn(4, 3, 224, 224, device=device)
    with torch.inference_mode():
        diff = float((rn(xb) - rn_f(xb)).abs().max())
    bn = {"convnext_tiny_n_fused": n_cnx, "demo_backbone": bn_demo_backbone, "demo_pretrained": bn_demo_pretrained,
          "demo_n_fused": rn_f.n_fused, "max_abs_diff_fp32": diff}
    l_rn = latency_report(rn, 1, 224, dtype="fp32", label=f"{bn_demo_backbone} chưa gộp BN", **kw)
    l_rnf = latency_report(rn_f, 1, 224, dtype="fp32", fused_bn=True, label=f"{bn_demo_backbone} đã gộp BN", **kw)
    lat = pd.concat([lat, pd.DataFrame([l_rn, l_rnf])], ignore_index=True)
    bn.update(p50_unfused_ms=l_rn["p50"], p50_fused_ms=l_rnf["p50"])
    print("Gộp BN:", bn)

    # gắn độ trễ batch 1 vào bảng phương pháp
    lat_map = {"I00_1view": "I00_1view", "I01_hflip_prob": "I01_hflip", "I01_hflip_logit": "I01_hflip",
               "I02_5crop_prob": "I02_5crop", "I02_5crop_logit": "I02_5crop", "I04_res256": "I04_res256",
               "I04_res288": "I04_res288", "I04_res320": "I04_res320", "I05_ensemble3": "I05_ensemble3",
               "I06_soup3": "I06_soup3", "I07_temp_scaling": "I00_1view", "I08_fp32": "I08_fp32",
               "I08_fp16": "I08_fp16"}
    by_label = lat.set_index("label")
    for col in ("p50", "p95", "p99"):
        res[f"lat_b1_{col}_ms"] = [by_label.loc[lat_map[m], col] if lat_map.get(m) in by_label.index else np.nan
                                   for m in res.method]
    res["cost_vs_I00"] = res["lat_b1_p50_ms"] / by_label.loc["I00_1view", "p50"]

    # ---- 7) chọn phương pháp suy luận cho chung kết (chỉ val, quy tắc ở đầu file) ----
    cand = res[res.method.isin(SINGLE)].copy()
    best = cand["val_macro_f1_mean_3seed"].max()
    cand = cand[cand["val_macro_f1_mean_3seed"] >= best - 1e-4].sort_values(["views", "method"])
    chosen = cand.iloc[0]
    choice = {"method": chosen.method, "loader": SINGLE[chosen.method][0], "views": SINGLE[chosen.method][1],
              "aggregate": SINGLE[chosen.method][2], "val_macro_f1_mean_3seed": float(chosen.val_macro_f1_mean_3seed),
              "plus_temperature_scaling": True, "realtime_p95_b1_ms": float(chosen.lat_b1_p95_ms),
              "rule": "max macro-F1 val trung bình 3 seed trong các phương pháp một mô hình; hòa thì ít view hơn"}

    res.to_csv(sub / "logs" / "inference_val.csv", index=False)
    lat.to_csv(sub / "logs" / "latency.csv", index=False)
    (sub / "logs" / "bn_fusion_check.json").write_text(json.dumps(bn, indent=2, ensure_ascii=False))
    choice_file.write_text(json.dumps(choice, indent=2, ensure_ascii=False))
    plot_tradeoff(res, sub / "curves" / "buoc3_tradeoff.png")
    print("Chọn cho chung kết:", choice)
    return res, lat, choice


def plot_tradeoff(res, path):
    """Scatter macro-F1 val theo độ trễ p50 batch 1 (thang log)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = res.dropna(subset=["lat_b1_p50_ms"])
    d = d[d.method != "I07_temp_scaling"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.scatter(d["lat_b1_p50_ms"], d["val_macro_f1"], s=36, color="#2a6fdb")
    for _, r in d.iterrows():
        ax.annotate(r.method, (r.lat_b1_p50_ms, r.val_macro_f1), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("độ trễ p50, batch 1 (ms, thang log)")
    ax.set_ylabel("macro-F1 val")
    ax.set_title("Bước 3: đánh đổi độ chính xác và độ trễ (F01 seed 0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
