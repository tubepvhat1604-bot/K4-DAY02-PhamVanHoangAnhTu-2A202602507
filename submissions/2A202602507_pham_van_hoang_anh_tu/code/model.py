"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Tag trọng số thật sự được tải ghi ở model.pretrained_cfg (xem weight_tag) và lưu vào config.json mỗi run.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp (timm tự thay head mới, khởi tạo ngẫu nhiên).

    `init` (trục A): "scratch" (không pretrained) | "frozen" (pretrained, chỉ train head) | "finetune".
    """
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(name, pretrained=(pretrained and init != "scratch"),
                              num_classes=num_classes, drop_rate=drop_rate)
    model.frozen_backbone = False
    model.weights_loaded = bool(pretrained and init != "scratch")
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tên đầy đủ của bộ trọng số, ví dụ 'resnet50.a1_in1k' (rỗng nếu không pretrained)."""
    if not getattr(model, "weights_loaded", True):
        return ""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", "")
    tag = cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else arch


def _head_param_ids(model) -> set[int]:
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head. BatchNorm của phần đóng băng được giữ ở eval
    trong train loop qua set_train_mode(model)."""
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì đưa mọi module ngoài head về eval
    (BatchNorm không cập nhật running_mean/var, dropout tắt)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m not in head_modules:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm tham số như slide Day 2, trang 52:
      - backbone ndim > 1  : lr_backbone, weight_decay
      - backbone ndim <= 1 (norm, bias): lr_backbone, weight_decay = 0
      - head mới           : lr_head, weight_decay (bias của head cũng không decay)
    Bỏ qua tham số requires_grad == False; bỏ nhóm rỗng.
    """
    head_ids = _head_param_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        kind = "decay" if p.ndim > 1 else "no_decay"
        groups[f"{part}_{kind}"].append(p)
    lr = {"backbone": lr_backbone, "head": lr_head}
    out = []
    for key, params in groups.items():
        if params:
            part, _, kind = key.partition("_")
            out.append({"name": key, "params": params, "lr": lr[part],
                        "weight_decay": weight_decay if kind == "decay" else 0.0})
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm FLOP của conv/matmul, 1 MAC = 2 FLOP),
    nên GMAC = FLOP / 2 / 1e9. Không đếm phép toán phần tử (activation, norm); có thể lệch vài %
    so với số trong bài báo/timm.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
