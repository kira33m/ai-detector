"""
dataset.py — загрузка данных и аугментации.
Структура папок ожидается:
    data_mvp/
        train/real/  train/fake/
        val/real/    val/fake/
        test/real/   test/fake/
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import io

from src.config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATA_DIR


# ──────────────────────────────────────────────
# Трансформации
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class JPEGCompress:
    """
    Callable класс вместо lambda — нужен для совместимости с
    multiprocessing на Windows (lambda не сериализуется через pickle).
    """
    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        return jpeg_compress(img, self.quality)


def get_train_transforms():
    """Аугментации для обучения (по ТЗ раздел 5.3)."""
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
        JPEGCompress(quality=75),   # имитация JPEG при обучении
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Трансформации для val/test (без аугментаций)."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def jpeg_compress(img: Image.Image, quality: int = 75) -> Image.Image:
    """Имитация JPEG-сжатия: encode -> decode в памяти."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class AIDetectorDataset(Dataset):
    """
    Простой ImageFolder-совместимый датасет.
    label: 0 = real, 1 = fake
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, split: str, transform=None):
        """
        Args:
            split: 'train', 'val' или 'test'
            transform: torchvision transforms
        """
        self.split     = split
        self.transform = transform
        self.samples   = []   # (path, label)

        split_dir = Path(DATA_DIR) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {split_dir}\n"
                                    f"Сначала запустите scripts/prepare_data.py")

        for label_idx, class_name in enumerate(["real", "fake"]):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for f in class_dir.iterdir():
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"В папке {split_dir} не найдено изображений!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {
            "real": counts.get(0, 0),
            "fake": counts.get(1, 0),
            "total": len(self.samples),
        }


# ──────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────
def get_loaders():
    """Возвращает словарь DataLoader'ов для train/val/test."""
    transforms = {
        "train": get_train_transforms(),
        "val":   get_val_transforms(),
        "test":  get_val_transforms(),
    }
    loaders = {}
    for split, tfm in transforms.items():
        try:
            ds = AIDetectorDataset(split, transform=tfm)
            print(f"[{split}] {ds.class_distribution()}")
            loaders[split] = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=(split == "train"),
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
    return loaders
