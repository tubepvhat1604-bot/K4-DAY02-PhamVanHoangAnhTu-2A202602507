"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (kèm mean)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - Lựa chọn của bài này: KHÔNG tính tiền xử lý (đọc ảnh, resize, chuẩn hoá) và không tính chép ảnh
    CPU -> GPU; chỉ đo forward của model với tensor đã nằm sẵn trên GPU. TTA K view = forward 1 batch K ảnh.
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây): bỏ `warmup` lần đầu, đo `iters` lần, mỗi lần sync trước và sau."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
            "p99": float(np.percentile(ts, 99)), "mean": float(ts.mean()), "n": int(iters), "warmup": int(warmup)}


def _prepare(model, dtype: str, device):
    """Trả về (model đã đặt đúng dtype/thiết bị, dtype của input, có autocast không)."""
    import copy
    import torch
    m = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        return m.half(), torch.float16, False
    if dtype == "amp":
        return m, torch.float32, True
    if dtype == "fp32":
        return m, torch.float32, False
    raise ValueError(f"dtype không hợp lệ: {dtype}")


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, label: str = "") -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict ghi thẳng vào sheet `Latency` của results.xlsx.
    """
    import contextlib
    import torch
    dev = torch.device(device)
    m, in_dtype, use_amp = _prepare(model, dtype, dev)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=in_dtype)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    ctx = torch.autocast("cuda", dtype=torch.float16) if (use_amp and dev.type == "cuda") else contextlib.nullcontext()

    def fn():
        with torch.inference_mode(), ctx:
            m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    return {"label": label, "gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": fused_bn, "preprocessing": "không tính", **r,
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def tta_latency(model, k_views: int, img_size: int = 224, **kw) -> dict:
    """Độ trễ TTA K view cho MỘT ảnh: forward một batch K view (cách chạy thực tế trên robot).

    Trả về report có thêm `k_times_single_p50` (K lần p50 của 1 view) để so với số đo thật (slide trang 63).
    """
    single = latency_report(model, 1, img_size, **kw)
    r = latency_report(model, k_views, img_size, **kw)
    r.update(batch=1, k_views=k_views, images_per_s=1000.0 / r["p50"],
             k_times_single_p50=k_views * single["p50"])
    return r
