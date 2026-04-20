"""
robustness.py — тесты устойчивости к JPEG-сжатию и resize.

Запуск:
    python -m src.robustness
"""

import io
import json
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from pathlib import Path

from src.config import (
    DEVICE, BEST_MODEL_PATH, REPORT_DIR,
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    JPEG_QUALITIES, RESIZE_FACTORS, USE_AMP
)
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, AIDetectorDataset
from src.model import load_model
import torchvision.transforms as T


# ──────────────────────────────────────────────
# Callable классы вместо lambda (Windows multiprocessing)
# ──────────────────────────────────────────────
class JPEGDistort:
    """Callable класс для JPEG-сжатия (lambda не работает с multiprocessing)."""
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class ResizeDistort:
    """Callable класс для масштабирования."""
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h  = img.size
        new_w = max(1, int(w * self.factor))
        new_h = max(1, int(h * self.factor))
        return img.resize((new_w, new_h), Image.LANCZOS)


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    return JPEGDistort(quality)(img)


def apply_resize(img: Image.Image, factor: float) -> Image.Image:
    return ResizeDistort(factor)(img)


# ──────────────────────────────────────────────
# Dataset с искажением
# ──────────────────────────────────────────────
class DistortedDataset(Dataset):
    """Обёртка над тестовым сплитом с применением искажения."""

    def __init__(self, base_dataset: AIDetectorDataset, distort_fn):
        self.samples    = base_dataset.samples
        self.distort_fn = distort_fn
        self.transform  = T.Compose([
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.distort_fn(img)
        img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────
# Расчёт F1 для одного DataLoader
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_f1(model, loader) -> float:
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        with autocast(DEVICE.type, enabled=(USE_AMP and DEVICE.type == "cuda")):
            logits = model(imgs)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
    return f1_score(all_labels, all_preds, zero_division=0)


# ──────────────────────────────────────────────
# Основной тест
# ──────────────────────────────────────────────
def run_robustness():
    model = load_model(BEST_MODEL_PATH, DEVICE)

    # Базовый тестовый датасет (без искажений)
    from src.dataset import AIDetectorDataset, get_val_transforms
    base_ds = AIDetectorDataset("test", transform=None)   # трансформации применяем в DistortedDataset

    results = []

    # ── Базовая F1 без искажений ──
    from src.dataset import get_val_transforms
    clean_ds = AIDetectorDataset("test", transform=get_val_transforms())
    clean_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE,
                              num_workers=0, shuffle=False)
    f1_clean = compute_f1(model, clean_loader)
    results.append({"type": "baseline", "param": "no distortion", "f1": f1_clean})
    print(f"Baseline F1 (no distortion): {f1_clean:.4f}")

    # ── JPEG сжатие ──
    print("\n🗜️  JPEG Robustness:")
    for quality in JPEG_QUALITIES:
        distort_fn = JPEGDistort(quality)          # класс вместо lambda
        ds = DistortedDataset(base_ds, distort_fn)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            num_workers=0, shuffle=False)
        f1 = compute_f1(model, loader)
        results.append({"type": "jpeg", "param": f"quality={quality}", "f1": f1})
        drop = f1_clean - f1
        print(f"  JPEG quality={quality:3d}: F1={f1:.4f}  (drop={drop:+.4f})")

    # ── Resize искажения ──
    print("\n🔍  Resize Robustness:")
    for factor in RESIZE_FACTORS:
        distort_fn = ResizeDistort(factor)         # класс вместо lambda
        ds = DistortedDataset(base_ds, distort_fn)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            num_workers=0, shuffle=False)
        f1 = compute_f1(model, loader)
        results.append({"type": "resize", "param": f"factor={factor}", "f1": f1})
        drop = f1_clean - f1
        print(f"  Resize factor={factor}: F1={f1:.4f}  (drop={drop:+.4f})")

    # ── Сохранение ──
    df = pd.DataFrame(results)
    csv_path = f"{REPORT_DIR}/robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Robustness результаты: {csv_path}")

    _plot_robustness(df, f1_clean)
    return df


def _plot_robustness(df: pd.DataFrame, baseline_f1: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # JPEG
    jpeg_df = df[df["type"] == "jpeg"]
    qualities = [int(p.split("=")[1]) for p in jpeg_df["param"]]
    ax1.plot(qualities, jpeg_df["f1"].values, "o-", color="steelblue", linewidth=2)
    ax1.axhline(baseline_f1, color="green", linestyle="--", label=f"Baseline F1={baseline_f1:.3f}")
    ax1.set_xlabel("JPEG Quality")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Robustness: JPEG Compression")
    ax1.legend()
    ax1.grid(True)
    ax1.invert_xaxis()

    # Resize
    resize_df = df[df["type"] == "resize"]
    factors = [float(p.split("=")[1]) for p in resize_df["param"]]
    ax2.plot(factors, resize_df["f1"].values, "s-", color="coral", linewidth=2)
    ax2.axhline(baseline_f1, color="green", linestyle="--", label=f"Baseline F1={baseline_f1:.3f}")
    ax2.set_xlabel("Resize Factor")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Robustness: Resize Distortion")
    ax2.legend()
    ax2.grid(True)

    path = f"{REPORT_DIR}/robustness_plot.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📈 График устойчивости: {path}")


if __name__ == "__main__":
    run_robustness()
