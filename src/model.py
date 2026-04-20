"""
model.py — создание модели EfficientNet-B0 через timm.
"""

import torch
import torch.nn as nn
import timm

from src.config import MODEL_NAME, NUM_CLASSES, BEST_MODEL_PATH, DEVICE


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Создаёт EfficientNet-B0 с заменённой головой для бинарной классификации.
    Начально все слои, кроме классификатора, заморожены.
    """
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)

    # Заморозить всё, кроме classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    return model


def unfreeze_all(model: nn.Module) -> None:
    """Разморозить все параметры (вызывается после FREEZE_EPOCHS)."""
    for param in model.parameters():
        param.requires_grad = True


def save_model(model: nn.Module, path: str = BEST_MODEL_PATH) -> None:
    torch.save(model.state_dict(), path)
    print(f"✅ Модель сохранена: {path}")


def load_model(path: str = BEST_MODEL_PATH, device=DEVICE) -> nn.Module:
    """Загружает сохранённые веса и переводит модель в eval режим."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Модель загружена: {path}")
    return model
