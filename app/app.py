"""
app.py — Streamlit демо-интерфейс детектора AI-изображений.

Запуск:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import numpy as np
import os

# ──────────────────────────────────────────────
# Конфигурация страницы
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

# ──────────────────────────────────────────────
# CSS стили
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .result-real {
        background: linear-gradient(135deg, #1a7a4c, #27ae60);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }
    .result-ai {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Заголовок
# ──────────────────────────────────────────────
st.title("🔍 Детектор AI-сгенерированных изображений")
st.markdown("**Real vs AI** — загрузите изображение и узнайте, создано ли оно нейросетью.")
st.divider()


# ──────────────────────────────────────────────
# Боковая панель с настройками
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")
    show_gradcam = st.toggle("Показать Grad-CAM тепловую карту", value=False)
    st.info(
        "Grad-CAM показывает, какие области изображения "
        "повлияли на решение модели."
    )
    st.divider()
    st.markdown("**Модель:** EfficientNet-B0")
    st.markdown("**Обучена на:** GenImage subset")

    # Показываем метрики если есть
    metrics_path = Path(__file__).parent.parent / "reports" / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            m = json.load(f)
        st.divider()
        st.markdown("**📊 Метрики модели (test set):**")
        st.metric("Accuracy", f"{m['accuracy']:.3f}")
        st.metric("F1-score", f"{m['f1']:.3f}")
        st.metric("ROC-AUC",  f"{m['roc_auc']:.3f}")


# ──────────────────────────────────────────────
# Загрузка изображения
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png", "webp"],
    help="Поддерживаются форматы: JPEG, PNG, WebP",
)

# Примеры для быстрого теста
st.markdown("или протестируйте на примере:")
col_ex1, col_ex2, _ = st.columns([1, 1, 2])

example_image = None
examples_dir = Path(__file__).parent.parent / "examples"
if examples_dir.exists():
    real_ex = list((examples_dir / "real").glob("*.jpg"))
    ai_ex   = list((examples_dir / "ai").glob("*.jpg"))
    with col_ex1:
        if real_ex and st.button("📷 Реальное фото"):
            example_image = str(real_ex[0])
    with col_ex2:
        if ai_ex and st.button("🤖 AI-изображение"):
            example_image = str(ai_ex[0])


# ──────────────────────────────────────────────
# Обработка
# ──────────────────────────────────────────────
image = None

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif example_image:
    image = Image.open(example_image).convert("RGB")

if image is not None:
    # Превью
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Загруженное изображение", use_column_width=True)
    with col_info:
        w, h = image.size
        st.markdown(f"**Размер:** {w} × {h} px")
        st.markdown(f"**Режим:** {image.mode}")

    st.divider()

    # ── Классификация ──
    model_path = Path(__file__).parent.parent / "models" / "best.pt"
    if not model_path.exists():
        st.error(
            "❌ Модель не найдена!\n\n"
            "Сначала обучите модель:\n```\npython -m src.train\n```"
        )
    else:
        with st.spinner("🔄 Анализируем изображение..."):
            try:
                from src.predict import predict_image
                result = predict_image(image)

                label    = result["label"]
                prob_ai  = result["prob_ai"]
                prob_real = result["prob_real"]

            except Exception as e:
                st.error(f"Ошибка при классификации: {e}")
                st.stop()

        # ── Результат ──
        css_class = "result-ai" if label == "AI" else "result-real"
        emoji     = "🤖" if label == "AI" else "📷"
        verdict   = "AI-сгенерированное" if label == "AI" else "Реальное фото"

        st.markdown(
            f'<div class="{css_class}">{emoji} {verdict}</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence bars ──
        st.markdown("**Уверенность модели:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📷 REAL", f"{prob_real:.1%}")
            st.progress(prob_real)
        with col2:
            st.metric("🤖 AI", f"{prob_ai:.1%}")
            st.progress(prob_ai)

        # ── Grad-CAM ──
        if show_gradcam:
            st.divider()
            st.markdown("### 🌡️ Grad-CAM: зоны влияния на решение")
            with st.spinner("Генерируем тепловую карту..."):
                try:
                    from src.gradcam import get_gradcam_overlay
                    overlay = get_gradcam_overlay(image)
                    col_orig, col_cam = st.columns(2)
                    with col_orig:
                        st.image(image.resize((224, 224)), caption="Оригинал (224×224)")
                    with col_cam:
                        st.image(overlay, caption="Grad-CAM наложение")
                    st.info("🔴 Красные зоны = наибольшее влияние на решение модели")
                except Exception as e:
                    st.warning(f"Grad-CAM недоступен: {e}")

    st.divider()
    st.caption(
        "⚠️ Модель обнаруживает статистические признаки синтетики. "
        "Результат носит вероятностный характер и не является окончательным заключением."
    )
