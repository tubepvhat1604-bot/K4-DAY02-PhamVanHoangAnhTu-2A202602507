"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (giữ nguyên để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn đã ghi lại:
  - Val/test ở 224: ảnh gốc 256x256 -> CenterCrop(224) (tương đương "resize 256 rồi center crop 224").
    Ở độ phân giải >= 256 (thí nghiệm I04): resize cả ảnh lên đúng kích thước, không crop.
  - Có thể nạp trước ảnh vào RAM (cache=True) để tránh nghẽn đọc đĩa trên Colab (2 nhân CPU).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi backbone timm dùng ở đây đều theo mean/std ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Trả về ba DataFrame đúng như file gốc (không sửa, không lọc, không chia lại).
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        df["Label"] = df["Label"].astype(int)
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, expected_total: int | None = TOTAL_IMAGES,
                verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và mỗi lớp trong từng tập (kèm tỉ lệ %)
    2. giao từng cặp tập theo Filename phải RỖNG
    3. hợp ba tập phải bằng đúng `expected_total` (17.509; chỉ đặt None khi chạy thử trên bộ mẫu)
    4. mọi Filename đều tồn tại trong `images_dir`
    Lỗi nào cũng raise AssertionError để dừng ngay.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert len(names[k]) == len(v), f"{k}: có Filename bị trùng trong cùng một tập"

    n = {k: len(v) for k, v in splits.items()}
    total = sum(n.values())
    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in splits.items()})
    per_class.index = [f"{i} {c}" for i, c in enumerate(CLASS_NAMES)]
    per_class["total"] = per_class.sum(axis=1)

    overlap = {
        "train∩val": len(names["train"] & names["val"]),
        "train∩test": len(names["train"] & names["test"]),
        "val∩test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])

    files = set(os.listdir(images_dir))
    missing = sorted((names["train"] | names["val"] | names["test"]) - files)

    if verbose:
        print("Số ảnh mỗi tập:", ", ".join(f"{k} {v} ({100 * v / total:.2f}%)" for k, v in n.items()))
        print("\nSố ảnh mỗi lớp trong từng tập:")
        print(per_class.to_string())
        print("\nGiao từng cặp (theo Filename):", overlap)
        print("Hợp ba tập:", union, "ảnh" + (f" (kỳ vọng {expected_total})" if expected_total else ""))
        print("File trong CSV nhưng không có trong thư mục ảnh:", len(missing))

    assert all(v == 0 for v in overlap.values()), f"Giao giữa các tập khác rỗng: {overlap}"
    if expected_total is not None:
        assert union == expected_total, f"Hợp ba tập = {union}, kỳ vọng {expected_total}"
    assert not missing, f"Thiếu {len(missing)} ảnh, ví dụ {missing[:5]}"
    if verbose:
        print("\nKIỂM TRA CHIA DỮ LIỆU: ĐẠT")
    return {"n": n, "pct": {k: round(100 * v / total, 2) for k, v in n.items()},
            "per_class": per_class, "overlap": overlap, "union": union, "missing": len(missing)}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` (trục B, GUIDE mục 3):

      "basic"    : RandomResizedCrop(img_size) + lật ngang (công thức nền T00)
      "color"    : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      "trivial"  : basic + TrivialAugmentWide
      "randaug"  : basic + RandAugment(2, 9)
      "dihedral" : basic + lật dọc + xoay bội 90 độ (ảnh chụp từ trên xuống nên hợp lệ về hình học)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    Đánh giá: không augmentation ngẫu nhiên (xem docstring đầu file).
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        if img_size < 256:
            return T.Compose([T.Resize(256), T.CenterCrop(img_size), *norm])
        return T.Compose([T.Resize((img_size, img_size)), *norm])

    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "dihedral":
        ops += [T.RandomVerticalFlip(), T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)])]
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*ops, *norm])


# Bộ nhớ đệm ảnh đã giải mã, dùng chung giữa các lần chạy trong cùng một phiên Python.
_IMAGE_CACHE: dict[str, np.ndarray] = globals().get("_IMAGE_CACHE", {})  # giữ cache khi importlib.reload


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def preload_images(filenames, images_dir: str | Path, verbose: bool = True) -> None:
    """Giải mã trước các ảnh vào RAM (khoảng 196 KB mỗi ảnh 256x256)."""
    todo = [f for f in filenames if os.path.join(str(images_dir), f) not in _IMAGE_CACHE]
    for i, f in enumerate(todo):
        p = os.path.join(str(images_dir), f)
        _IMAGE_CACHE[p] = _load_rgb(p)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  đã nạp {i + 1}/{len(todo)} ảnh vào RAM")
    if verbose:
        print(f"Cache RAM: {len(_IMAGE_CACHE)} ảnh")


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # để module vẫn import được khi chưa cài torch
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = str(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        if cache:
            preload_images(self.filenames, self.images_dir)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        from PIL import Image
        path = os.path.join(self.images_dir, self.filenames[i])
        arr = _IMAGE_CACHE.get(path)
        img = Image.fromarray(arr) if arr is not None else Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed cho từng worker của DataLoader (lấy từ seed gốc của torch)."""
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool = False):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc sampler="balanced": WeightedRandomSampler trọng số 1/n_lớp), drop_last=True
    - train=False: không shuffle, giữ đúng thứ tự df (để ghép logit với Filename)
    - generator + worker_init_fn có seed để tái lập
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[ds.labels]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
                      drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g)  # worker tạo lại mỗi epoch -> seed lại theo epoch
