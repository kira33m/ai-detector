# 🔍 Детектор AI-сгенерированных изображений (Real vs AI)

Система для определения: является ли изображение **реальным** или **сгенерированным нейросетью**.  
Модель: **EfficientNet-B0** с transfer learning. Датасет: **GenImage**.

---

## 📁 Структура проекта

```
ai_detector/
├── data_mvp/              # Датасет (создаётся скриптом)
│   ├── train/
│   │   ├── real/          # Реальные фото (обучение)
│   │   └── fake/          # AI-изображения (обучение)
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
│   └── best.pt            # Веса лучшей модели (после обучения)
├── reports/
│   ├── metrics.json           # Метрики на тестовой выборке
│   ├── confusion_matrix.png   # Матрица ошибок
│   ├── robustness.csv         # Тесты устойчивости
│   ├── robustness_plot.png    # График устойчивости
│   └── training_curves.png    # Кривые обучения
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
│   └── prepare_data.py    # Подготовка датасета из GenImage
├── app/
│   └── app.py             # Streamlit веб-интерфейс
├── examples/              # Примеры изображений для демо
│   ├── real/
│   └── ai/
├── requirements.txt
└── README.md
```

---

## ⚙️ Шаг 1: Установка окружения

### Требования
- Python 3.10+
- NVIDIA GPU с CUDA (RTX 4060 8GB рекомендована)
- CUDA Toolkit 11.8 или 12.x

### Создание виртуального окружения

```bash
# Создать venv
python -m venv venv

# Активировать (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Активировать (Windows CMD)
venv\Scripts\activate.bat
```

### Установка PyTorch с CUDA

> ⚠️ Важно: PyTorch с CUDA устанавливается отдельно!

```bash
# Для CUDA 12.x (RTX 4060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Проверить что CUDA работает:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Установка остальных зависимостей

```bash
pip install -r requirements.txt
```

---

## 📥 Шаг 2: Подготовка датасета

Используется датасет **GenImage**: https://huggingface.co/datasets/GenImage

### Вариант А: Скачать с HuggingFace (автоматически)

1. Зарегистрируйся на https://huggingface.co
2. Получи токен: https://huggingface.co/settings/tokens
3. Запроси доступ к датасету: https://huggingface.co/datasets/GenImage
4. Установи токен:

```powershell
# PowerShell
$env:HF_TOKEN = "hf_ВашТокен"

# Запустить подготовку данных
python scripts/prepare_data.py --n 500
```

Параметр `--n 500` — количество изображений на генератор (500 × 4 генератора = 2000 fake + 2000 real).

### Вариант Б: Локальный датасет (если уже скачан)

```bash
python scripts/prepare_data.py --source "D:\GenImage" --n 500
```

Структура GenImage должна быть:
```
D:\GenImage\
    stable_diffusion_v_1_4\
        ai\        ← синтетика
        nature\    ← реальные
    midjourney\
        ai\
        nature\
```

### Добавить примеры для демо (опционально)

```bash
mkdir examples\real examples\ai
# Скопируй по 2-3 изображения из data_mvp/test/ для демо
```

---

## 🏋️ Шаг 3: Обучение модели

```bash
python -m src.train
```

**Что происходит:**
- Эпохи 1-2: обучается только голова (классификатор), основная сеть заморожена
- Эпохи 3-10: вся сеть разморожена и дообучается с меньшим lr
- Лучшая модель по val accuracy сохраняется в `models/best.pt`
- Кривые обучения сохраняются в `reports/training_curves.png`

**Ожидаемое время:** ~20-40 минут на RTX 4060 (зависит от размера датасета)

**Если кончается VRAM:** уменьши `BATCH_SIZE = 16` в `src/config.py`

---

## 📊 Шаг 4: Оценка модели

```bash
python -m src.evaluate
```

Выводит:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Полный Classification Report
- Сохраняет `reports/metrics.json` и `reports/confusion_matrix.png`

---

## 🛡️ Шаг 5: Тесты устойчивости

```bash
python -m src.robustness
```

Тестирует модель при:
- JPEG сжатии: quality 95 / 75 / 50
- Масштабировании: factor 0.5 / 0.75 / 1.5

Сохраняет `reports/robustness.csv` и `reports/robustness_plot.png`

---

## 🌐 Шаг 6: Запуск веб-демо

```bash
streamlit run app/app.py
```

Откроется браузер на http://localhost:8501

**Функции интерфейса:**
- Загрузка изображения (jpg/png/webp)
- Отображение вердикта: REAL / AI
- Уровень уверенности модели
- Grad-CAM тепловая карта (включается в боковой панели)
- Метрики модели в боковой панели (если есть `reports/metrics.json`)

---

## 🔧 Настройка параметров

Все гиперпараметры — в `src/config.py`:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `BATCH_SIZE` | 32 | Уменьши до 16 при нехватке VRAM |
| `EPOCHS` | 10 | Количество эпох |
| `FREEZE_EPOCHS` | 2 | Эпох с замороженной backbone |
| `LR_HEAD` | 1e-4 | LR для головы |
| `LR_FULL` | 3e-5 | LR для всей сети |
| `MODEL_NAME` | efficientnet_b0 | Попробуй b2 для лучшего качества |
| `USE_AMP` | True | Mixed Precision (экономит VRAM) |

---

## 🚨 Типичные проблемы

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | Уменьши `BATCH_SIZE` до 16 или 8 |
| `FileNotFoundError: data_mvp` | Сначала запусти `prepare_data.py` |
| `No module named 'timm'` | `pip install -r requirements.txt` |
| `RuntimeError: num_workers` | Поставь `NUM_WORKERS = 0` в config.py |
| DataLoader ошибка на Windows | `NUM_WORKERS = 0` в config.py |

---

## 📦 Версии библиотек

| Библиотека | Версия |
|-----------|--------|
| Python | 3.10+ |
| PyTorch | 2.1+ (CUDA 12.1) |
| timm | 0.9.12+ |
| Streamlit | 1.32+ |
| scikit-learn | 1.3+ |

Точные версии: `pip freeze > requirements_frozen.txt`

---

## 📈 Ожидаемые результаты (MVP)

| Метрика | Ожидаемое значение |
|---------|-------------------|
| Accuracy | 85-95% |
| F1-score | 0.85-0.95 |
| ROC-AUC | 0.90-0.98 |

Результаты зависят от размера датасета и выбранных генераторов.

---

## ⚠️ Ограничения

- Модель обнаруживает статистические артефакты синтетики, а не "понимает" изображение
- Качество снижается на новых генераторах (не представленных в обучении)
- JPEG-сжатие и изменение размера могут снижать точность
- Метаданные (EXIF) не используются

---

## 👤 Автор

Индивидуальный проект по направлению «Искусственный интеллект»  
Версия: MVP 1.0 | Март 2026
