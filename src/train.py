"""
train.py — обучение модели с transfer learning.

Запуск:
    python -m src.train
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import (
    SEED, EPOCHS, FREEZE_EPOCHS,
    LR_HEAD, LR_FULL, WEIGHT_DECAY,
    USE_AMP, DEVICE, BEST_MODEL_PATH, REPORT_DIR
)
from src.dataset import get_loaders
from src.model import build_model, unfreeze_all, save_model


# ──────────────────────────────────────────────
# Воспроизводимость
# ──────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Одна эпоха обучения
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(DEVICE.type, enabled=(USE_AMP and DEVICE.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# Валидация
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, split="Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(DEVICE.type, enabled=(USE_AMP and DEVICE.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# Основной цикл
# ──────────────────────────────────────────────
def train():
    set_seed()
    print(f"🖥️  Устройство: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    loaders = get_loaders()
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("Нужны сплиты train и val. Запустите prepare_data.py")

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda", enabled=USE_AMP)

    # Начинаем с обучения только головы
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Разморозка после FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS:
            print(f"\n🔓 Epoch {epoch+1}: размораживаем всю сеть, lr={LR_FULL}")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - FREEZE_EPOCHS
            )

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, DEVICE, epoch
        )
        vl_loss, vl_acc = validate(model, loaders["val"], criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
            f"LR: {lr_now:.2e}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_model(model)

    # Сохраняем историю обучения
    history_path = f"{REPORT_DIR}/train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 История обучения сохранена: {history_path}")
    print(f"🏆 Лучший Val Acc: {best_val_acc:.4f}")

    _plot_history(history)


def _plot_history(history: dict):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    path = f"{REPORT_DIR}/training_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📈 График обучения: {path}")


if __name__ == "__main__":
    train()
