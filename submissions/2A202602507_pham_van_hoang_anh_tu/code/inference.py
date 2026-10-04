"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
Thêm:
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits mỗi view])
    uniform_soup(state_dicts)                        -> state_dict trung bình (model soup)
"""
from __future__ import annotations

import copy

import numpy as np


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _autocast(device, amp: bool):
    import contextlib
    import torch
    if amp and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def predict_views(model, loader, device, views, amp: bool = True):
    """Chạy model một lượt qua loader, mỗi batch áp K hàm `views` (mỗi hàm: batch -> batch hoặc list batch).

    Trả về (filenames, y_true[N], list K mảng logit [N, 9]) theo đúng thứ tự file. Không gradient.
    Đọc ảnh một lần cho cả K view (TTA rẻ hơn K lượt đọc riêng).
    """
    import torch
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            batches = []
            for v in views:
                r = v(x)
                batches.extend(r if isinstance(r, (list, tuple)) else [r])
            if outs is None:
                outs = [[] for _ in batches]
            with _autocast(device, amp):
                for k, b in enumerate(batches):
                    outs[k].append(model(b).float().cpu())
            ys.append(y)
            names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader và gom logit theo đúng thứ tự file. `view` là hàm biến đổi batch, hoặc None."""
    names, y, logits = predict_views(model, loader, device, [view or view_identity], amp=amp)
    return names, y, logits[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng (slide trang 75)."""
    import torch
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch (N, C, H, W); flip=True thêm bản lật (10 view)."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    t, l = (h - crop) // 2, (w - crop) // 2
    boxes = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (t, l)]
    out = [x[..., i:i + crop, j:j + crop] for i, j in boxes]
    if flip:
        out += [view_hflip(v) for v in out]
    return out


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias), trả về list các batch.

    ConvNeXt/ResNet có global pooling nên nhận được ảnh khác kích thước lúc train.
    ViT/DeiT/Swin cần nội suy vị trí hoặc cửa sổ: KHÔNG dùng hàm này cho các mạng đó.
    """
    import torch.nn.functional as F
    return [x if x.shape[-1] == s and x.shape[-2] == s else
            F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy TTA thành xác suất (N, 9) (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    arr = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])  # (K, N, 9)
    if space == "prob":
        p = np.mean([_softmax(l) for l in arr], axis=0)
    elif space == "logit":
        p = _softmax(arr.mean(0))
    else:
        raise ValueError(f"space không hợp lệ: {space}")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file). Chi phí = số mô hình."""
    p = np.mean([np.asarray(q, dtype=np.float64) for q in list_of_probs], axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T > 0 cực tiểu NLL trên VAL của softmax(logit / T) (slide trang 69).

    Tìm lưới thô trên log T trong [0,05; 20] rồi tinh bằng phương pháp lát cắt vàng (NLL lồi theo 1/T).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels, dtype=np.int64)

    def nll(log_t):
        s = z / np.exp(log_t)
        s = s - s.max(1, keepdims=True)
        return float(-(s[np.arange(len(y)), y] - np.log(np.exp(s).sum(1))).mean())

    grid = np.linspace(np.log(0.05), np.log(20.0), 121)
    i = int(np.argmin([nll(g) for g in grid]))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))


def fit_temperature_views(logits_per_view, val_labels, space: str = "logit") -> float:
    """Như fit_temperature nhưng cho TTA: cực tiểu NLL của aggregate_views([l / T cho mỗi view], space).
    Với 1 view, kết quả trùng fit_temperature."""
    views = [np.asarray(l, dtype=np.float64) for l in logits_per_view]
    y = np.asarray(val_labels, dtype=np.int64)

    def nll(log_t):
        p = aggregate_views([l / np.exp(log_t) for l in views], space)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

    grid = np.linspace(np.log(0.05), np.log(20.0), 121)
    i = int(np.argmin([nll(g) for g in grid]))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        if nll(c) < nll(d):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / float(T))


def uniform_soup(state_dicts):
    """Model soup đồng đều: trung bình từng tham số dạng số thực của nhiều checkpoint CÙNG kiến trúc
    (fine-tune từ cùng trọng số pretrained, khác seed). Buffer kiểu số nguyên lấy từ checkpoint đầu."""
    import torch
    out = copy.deepcopy(state_dicts[0])
    for k, v in out.items():
        if torch.is_floating_point(v):
            out[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(0).to(v.dtype)
    return out


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Duyệt cây module: trong mỗi module cha, cặp con liền kề (Conv2d rồi BatchNorm2d) được thay bằng
    conv mới có bias + Identity. Trả về BẢN SAO đã gộp (model gốc giữ nguyên) và gắn số cặp đã gộp vào
    thuộc tính `n_fused`. Kiến trúc không có BN (ConvNeXt, ViT, Swin dùng LayerNorm): n_fused = 0.
    """
    import torch
    from torch import nn

    fused = copy.deepcopy(model).eval()
    n = 0

    def fuse_pair(conv, bn):
        w = conv.weight.detach()
        b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0], device=w.device)
        scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                        conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(w.device)
        new.weight.data = w * scale.reshape(-1, 1, 1, 1)
        new.bias.data = bn.bias.detach() + (b - bn.running_mean) * scale
        return new

    def walk(parent):
        nonlocal n
        names = list(parent._modules.keys())
        for a, b in zip(names, names[1:]):
            m1, m2 = parent._modules[a], parent._modules[b]
            if isinstance(m1, nn.Conv2d) and type(m2) is nn.BatchNorm2d and m2.track_running_stats:
                parent._modules[a] = fuse_pair(m1, m2)
                parent._modules[b] = nn.Identity()
                n += 1
        for child in parent._modules.values():
            if child is not None:
                walk(child)

    walk(fused)
    fused.n_fused = n
    return fused
