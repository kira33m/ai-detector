"""
gradcam.py — генерация тепловых карт Grad-CAM для объяснимости.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from src.config import DEVICE, BEST_MODEL_PATH, USE_AMP
from src.dataset import get_val_transforms
from src.model import load_model


class GradCAM:
    """Простая реализация Grad-CAM для EfficientNet."""

    def __init__(self, model, target_layer=None):
        self.model   = model
        self.gradients = None
        self.activations = None

        # Для EfficientNet-B0 через timm целевой слой — последний блок
        if target_layer is None:
            # timm EfficientNet: features[-1] — последний conv блок
            target_layer = model.conv_head

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: Image.Image) -> np.ndarray:
        """
        Генерирует тепловую карту для изображения.
        Возвращает numpy массив (H, W) с нормированными весами [0..1].
        """
        self.model.eval()
        transform = get_val_transforms()
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        logits = self.model(tensor)
        pred_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, pred_class].backward()

        # Веса = среднее градиентов по spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Накладывает тепловую карту на оригинальное изображение."""
    img_arr = np.array(image.resize((224, 224)).convert("RGB"))

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (img_arr * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def get_gradcam_overlay(image: Image.Image, model=None) -> Image.Image:
    """Полный pipeline: изображение → наложенная тепловая карта."""
    if model is None:
        from src.predict import get_cached_model
        model = get_cached_model()

    gc  = GradCAM(model)
    cam = gc.generate(image)
    return overlay_heatmap(image, cam)
