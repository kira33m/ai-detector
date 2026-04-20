"""
prepare_data.py — подготовка subset из датасета nebula/GenImage-arrow.

Датасет: https://huggingface.co/datasets/nebula/GenImage-arrow
- Публичный, токен не нужен
- Streaming режим: скачивает по одному, не нужно 607 GB
- Метка берётся из image_path: /ai/ = fake, /nature/ = real

Запуск:
    python scripts/prepare_data.py

С параметрами:
    python scripts/prepare_data.py --n 600 --generators midjourney stable_diffusion_v_1_4
"""

import sys
import io
import random
import shutil
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, SEED

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────
IMAGES_PER_CLASS = 600

SELECTED_GENERATORS = [
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "midjourney",
    "wukong",
]

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────

def decode_image(image_field):
    """Декодирует поле image из датасета в PIL.Image."""
    try:
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        elif isinstance(image_field, bytes):
            return Image.open(io.BytesIO(image_field)).convert("RGB")
        elif isinstance(image_field, dict):
            raw = image_field.get("bytes") or image_field.get("path")
            if isinstance(raw, bytes):
                return Image.open(io.BytesIO(raw)).convert("RGB")
        elif isinstance(image_field, list):
            raw = bytes(image_field)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        return None
    except Exception:
        return None


def get_label_from_path(image_path: str):
    """Определяет метку из image_path: /ai/ -> fake, /nature/ -> real."""
    parts = image_path.lower().replace("\\", "/").split("/")
    if "ai" in parts:
        return "fake"
    if "nature" in parts:
        return "real"
    return None


def get_generator_from_path(image_path: str) -> str:
    """Извлекает имя генератора (первая часть пути)."""
    return image_path.split("/")[0].lower()


def save_image(img: Image.Image, dst_path: Path) -> bool:
    """Сохраняет изображение в JPEG."""
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"    Warning: {e}")
        return False


def distribute_to_splits(file_list, class_name: str):
    """Раскладывает файлы по train/val/test папкам."""
    random.seed(SEED)
    random.shuffle(file_list)

    n       = len(file_list)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])
    n_test  = n - n_train - n_val

    splits = (
        [(p, "train") for p in file_list[:n_train]]
        + [(p, "val")   for p in file_list[n_train:n_train + n_val]]
        + [(p, "test")  for p in file_list[n_train + n_val:]]
    )

    for src, split in splits:
        dst_dir = Path(DATA_DIR) / split / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst_dir / src.name)

    print(f"  [{class_name}] train={n_train}, val={n_val}, test={n_test}")


# ──────────────────────────────────────────────
# Основная функция
# ──────────────────────────────────────────────

def prepare(n_per_class: int, generators):
    print("=" * 55)
    print("  Датасет: nebula/GenImage-arrow")
    print("=" * 55)
    print(f"  Изображений на класс: {n_per_class}")
    print(f"  Генераторы (fake)   : {generators or 'любые'}")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Установи: pip install datasets")
        sys.exit(1)

    tmp_real = Path("/tmp/genimage_tmp/real")
    tmp_fake = Path("/tmp/genimage_tmp/fake")
    tmp_real.mkdir(parents=True, exist_ok=True)
    tmp_fake.mkdir(parents=True, exist_ok=True)

    print("Подключаемся к датасету (streaming, без лишних загрузок)...")
    ds = load_dataset(
        "nebula/GenImage-arrow",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED, buffer_size=5000)

    collected_real = []
    collected_fake = []
    hashes = set()
    total_seen = 0

    print(f"Собираем {n_per_class} real + {n_per_class} fake...\n")

    for sample in ds:
        if len(collected_real) >= n_per_class and len(collected_fake) >= n_per_class:
            break

        total_seen += 1
        if total_seen % 300 == 0:
            print(f"  просмотрено: {total_seen:5d} | "
                  f"real: {len(collected_real):4d}/{n_per_class} | "
                  f"fake: {len(collected_fake):4d}/{n_per_class}")

        image_path = sample.get("image_path", "")
        label      = get_label_from_path(image_path)
        generator  = get_generator_from_path(image_path)

        if label is None:
            continue

        # Фильтр по генераторам для fake
        if label == "fake" and generators:
            if not any(g in generator for g in generators):
                continue

        if label == "real"  and len(collected_real) >= n_per_class:
            continue
        if label == "fake"  and len(collected_fake) >= n_per_class:
            continue

        # Дедупликация
        md5_val = sample.get("md5", "")
        if md5_val:
            if md5_val in hashes:
                continue
            hashes.add(md5_val)

        img = decode_image(sample.get("image"))
        if img is None:
            continue

        idx   = len(collected_real) if label == "real" else len(collected_fake)
        fname = f"{generator}_{label}_{idx:05d}.jpg"

        if label == "real":
            dst = tmp_real / fname
            if save_image(img, dst):
                collected_real.append(dst)
        else:
            dst = tmp_fake / fname
            if save_image(img, dst):
                collected_fake.append(dst)

    print(f"\nСобрано: {len(collected_real)} real, {len(collected_fake)} fake")

    if not collected_real or not collected_fake:
        print("Ошибка: не удалось собрать изображения. Проверь интернет.")
        sys.exit(1)

    # Балансировка
    min_count      = min(len(collected_real), len(collected_fake))
    collected_real = collected_real[:min_count]
    collected_fake = collected_fake[:min_count]
    print(f"После балансировки: {min_count} real + {min_count} fake")

    # Очищаем старый датасет
    if Path(DATA_DIR).exists():
        print(f"\nОчищаем: {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    # Раскладываем по сплитам
    print(f"\nРаскладываем по train/val/test -> {DATA_DIR}")
    distribute_to_splits(collected_real, "real")
    distribute_to_splits(collected_fake, "fake")

    # Удаляем tmp
    shutil.rmtree("/tmp/genimage_tmp", ignore_errors=True)

    # Статистика
    print("\n" + "=" * 55)
    print(f"  Датасет готов: {DATA_DIR}")
    print("=" * 55)
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            p = Path(DATA_DIR) / split / cls
            if p.exists():
                n = len(list(p.glob("*.jpg")))
                total += n
                print(f"    {split}/{cls}: {n}")
    print(f"    ИТОГО: {total}")
    print("\nГотово! Теперь запускай: python -m src.train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=IMAGES_PER_CLASS,
                        help="Изображений на класс (default: 600)")
    parser.add_argument("--generators", nargs="+", default=SELECTED_GENERATORS,
                        help="Фильтр генераторов для fake-класса")
    args = parser.parse_args()
    prepare(args.n, args.generators)
