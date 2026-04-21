# 🔍 Детектор AI-сгенерированных изображений (Real vs AI)

Система для определения: является ли изображение **реальным** или **сгенерированным нейросетью**.  
Модель: **EfficientNet-B0** с transfer learning. Датасет: **GenImage** (nebula/GenImage-arrow).

---

## 📁 Структура проекта

```
ai_detector/
├── data_mvp/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
│   └── best.pt
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── robustness.csv
│   ├── robustness_plot.png
│   └── training_curves.png
├── src/
│   ├── config.py          # Гиперпараметры и пути
│   ├── dataset.py         # Датасет и трансформации
│   ├── model.py           # EfficientNet-B0
│   ├── train.py           # Обучение
│   ├── evaluate.py        # Оценка метрик
│   ├── robustness.py      # Тесты устойчивости
│   ├── predict.py         # Инференс одного изображения
│   └── gradcam.py         # Grad-CAM тепловые карты
├── scripts/
│   └── prepare_data.py    # Подготовка датасета
├── app/
│   └── app.py             # Streamlit веб-интерфейс
├── requirements.txt
└── README.md
```

---

## ⚙️ Шаг 1: Установка окружения

### Требования
- Python 3.10+
- NVIDIA GPU с CUDA (протестировано на RTX 4060 Laptop GPU)
- CUDA Toolkit 12.x

### Создание виртуального окружения

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Установка PyTorch с CUDA

> ⚠️ PyTorch с CUDA устанавливается отдельно, до остальных зависимостей!

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Проверить что CUDA работает:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Установка остальных зависимостей

```powershell
pip install -r requirements.txt
```

---

## 📥 Шаг 2: Подготовка датасета

Используется публичный датасет **nebula/GenImage-arrow** с HuggingFace.  
Токен не требуется. Скачивается в streaming-режиме — не нужно качать все 607 GB.

```powershell
# 2000 изображений на класс (рекомендуется)
python scripts/prepare_data.py --n 2000
```

---

## 🏋️ Шаг 3: Обучение модели

```powershell
python -m src.train
```

**Что происходит:**
- Эпохи 1-2: обучается только голова, backbone заморожен
- Эпохи 3-10: вся сеть разморожена, lr=3e-5
- Лучшая модель сохраняется в `models/best.pt`

**Время:** ~20-40 минут на RTX 4060

---

## 📊 Шаг 4: Оценка модели

```powershell
python -m src.evaluate
```

Сохраняет `reports/metrics.json` и `reports/confusion_matrix.png`

---

## 🛡️ Шаг 5: Тесты устойчивости

```powershell
python -m src.robustness
```

Тестирует при JPEG quality 95/75/50 и resize factor 0.5/0.75/1.5.  
Сохраняет `reports/robustness.csv` и `reports/robustness_plot.png`

---

## 🌐 Шаг 6: Запуск веб-демо

```powershell
streamlit run app/app.py
```

Откроется браузер на http://localhost:8501

**Функции:**
- Загрузка изображения (jpg/png/webp)
- Вердикт: REAL / AI + уровень уверенности
- Grad-CAM тепловая карта (в боковой панели)
- Метрики модели в боковой панели

---

## 🔧 Настройка параметров (`src/config.py`)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `BATCH_SIZE` | 32 | Уменьши до 16 при нехватке VRAM |
| `EPOCHS` | 10 | Количество эпох |
| `FREEZE_EPOCHS` | 2 | Эпох с замороженным backbone |
| `LR_HEAD` | 1e-4 | LR для головы |
| `LR_FULL` | 3e-5 | LR для всей сети |
| `NUM_WORKERS` | 0 | Обязательно 0 на Windows |
| `USE_AMP` | True | Mixed Precision (экономит VRAM) |

---

## 🚨 Типичные проблемы

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | Уменьши `BATCH_SIZE` до 16 в `config.py` |
| `FileNotFoundError: data_mvp` | Запусти `prepare_data.py` |
| `Can't get local object lambda` | Убедись что используешь актуальные файлы из репозитория |
| `CUDA is not available` | Переустанови PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| DataLoader ошибка на Windows | `NUM_WORKERS = 0` в `config.py` |

---

## 📈 Результаты модели (2000 изображений на класс)

### Метрики на тестовой выборке

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.728 |
| Precision | 0.687 |
| Recall | 0.835 |
| F1-score | **0.754** |
| ROC-AUC | **0.806** |

### Тесты устойчивости

| Тип | Параметр | F1 | Падение |
|-----|---------|-----|---------|
| Baseline | без искажений | 0.754 | — |
| JPEG | quality=95 | 0.749 | -0.005 |
| JPEG | quality=75 | 0.758 | +0.004 |
| JPEG | quality=50 | 0.765 | +0.011 |
| Resize | factor=0.5 | 0.747 | -0.007 |
| Resize | factor=0.75 | 0.749 | -0.005 |
| Resize | factor=1.5 | 0.751 | -0.003 |

> Максимальное падение F1 при искажениях: **0.007** — модель устойчива.

---

## ⚠️ Ограничения

- Модель обнаруживает статистические артефакты, а не "понимает" изображение
- Качество снижается на новых генераторах, не представленных в обучении
- Метаданные EXIF не используются

---

## 👤 Автор

Индивидуальный проект по направлению «Искусственный интеллект»  
Версия: 1.0 | 2026
