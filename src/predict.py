"""
predict.py — инференс одного изображения.
Используется Streamlit-приложением и может вызываться напрямую.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import DEVICE, BEST_MODEL_PATH, USE_AMP
from src.dataset import get_val_transforms
from src.model import load_model

# Кешируем модель чтобы не перезагружать при каждом вызове
_model_cache = None


def get_cached_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(BEST_MODEL_PATH, DEVICE)
    return _model_cache


def predict_image(image: Image.Image, model=None) -> dict:
    """
    Классифицирует одно изображение.

    Args:
        image: PIL.Image
        model: загруженная модель (если None — загрузит сама)

    Returns:
        dict с ключами:
            label   — "REAL" или "AI"
            prob_ai — вероятность AI (0..1)
            prob_real — вероятность REAL (0..1)
    """
    if model is None:
        model = get_cached_model()

    transform = get_val_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

    prob_real = float(probs[0])
    prob_ai   = float(probs[1])
    label     = "AI" if prob_ai >= 0.5 else "REAL"

    return {
        "label":     label,
        "prob_ai":   prob_ai,
        "prob_real": prob_real,
    }
