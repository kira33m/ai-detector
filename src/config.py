"""
config.py — централизованная конфигурация проекта.
Все гиперпараметры, пути и настройки берутся отсюда.
"""

import os
import torch

# ──────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data_mvp")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

# ──────────────────────────────────────────────
# Данные
# ──────────────────────────────────────────────
IMAGE_SIZE   = 224          # размер входа модели
BATCH_SIZE   = 32           # уменьши до 16 если нехватает VRAM
NUM_WORKERS  = 4            # потоки DataLoader (0 на Windows если ошибки)
PIN_MEMORY   = True         # ускорение передачи на GPU

# Метки классов
CLASS_NAMES  = ["real", "fake"]   # индекс 0 = real, 1 = fake
NUM_CLASSES  = 2

# ──────────────────────────────────────────────
# Модель
# ──────────────────────────────────────────────
MODEL_NAME   = "efficientnet_b0"  # из timm; можно попробовать efficientnet_b2

# ──────────────────────────────────────────────
# Обучение
# ──────────────────────────────────────────────
SEED           = 42
EPOCHS         = 10            # 8-12 по ТЗ
FREEZE_EPOCHS  = 2             # сколько эпох обучать только голову
LR_HEAD        = 1e-4          # lr для головы (первые FREEZE_EPOCHS)
LR_FULL        = 3e-5          # lr для всей сети после разморозки
WEIGHT_DECAY   = 1e-4
USE_AMP        = True          # Mixed Precision (экономия VRAM)

# ──────────────────────────────────────────────
# Устройство
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Robustness тесты
# ──────────────────────────────────────────────
JPEG_QUALITIES  = [95, 75, 50]       # уровни JPEG сжатия
RESIZE_FACTORS  = [0.5, 0.75, 1.5]  # факторы масштабирования (down/up)
