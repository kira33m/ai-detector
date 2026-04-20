"""
evaluate.py — расчёт метрик на тестовой выборке.

Запуск:
    python -m src.evaluate
"""

import json
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

from src.config import DEVICE, BEST_MODEL_PATH, REPORT_DIR, USE_AMP
from src.dataset import get_loaders
from src.model import load_model


def evaluate(split: str = "test"):
    """Оцениваем модель на указанном сплите."""
    print(f"\n📊 Оценка модели на сплите: {split}")

    loaders = get_loaders()
    if split not in loaders:
        raise RuntimeError(f"Сплит '{split}' не найден!")

    model = load_model(BEST_MODEL_PATH, DEVICE)
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loaders[split], desc=f"Eval [{split}]"):
            imgs = imgs.to(DEVICE)
            with autocast(DEVICE.type, enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(imgs)
                probs  = torch.softmax(logits, dim=1)[:, 1]  # вероятность класса "fake"

            preds = logits.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)

    # ── Метрики ──
    metrics = {
        "split":     split,
        "accuracy":  float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall":    float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "f1":        float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels_arr, probs_arr)),
    }

    print("\n" + "="*50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("="*50)
    print("\n📋 Classification Report:")
    print(classification_report(labels_arr, preds_arr, target_names=["real", "fake"]))

    # ── Сохранение метрик ──
    metrics_path = f"{REPORT_DIR}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Метрики сохранены: {metrics_path}")

    # ── Confusion Matrix ──
    _plot_confusion_matrix(labels_arr, preds_arr)

    return metrics


def _plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted REAL", "Predicted AI"],
        yticklabels=["True REAL", "True AI"],
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    path = f"{REPORT_DIR}/confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Confusion Matrix сохранена: {path}")


if __name__ == "__main__":
    evaluate("test")
