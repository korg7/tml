"""Утилиты: чтение изображений, resize, конвертация в LAB, хелперы."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Загрузка YAML-конфигурации."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_images(input_dir: str, extensions: List[str]) -> List[Path]:
    """Рекурсивный поиск изображений по расширениям."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")

    ext_set = {e.lower() for e in extensions}
    images = []
    for f in sorted(input_path.rglob("*")):
        if f.suffix.lower() in ext_set and f.is_file():
            images.append(f)
    return images


def read_and_preprocess(
    image_path: Path,
    max_side: int = 1280,
    blur_sigma: float = 1.0,
    return_original: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Читает изображение, уменьшает, размывает, конвертирует в LAB.

    Args:
        return_original: если True, возвращает копию BGR до размытия (для визуализации).
                         Если False, третий элемент — None (экономия памяти в воркерах).

    Returns:
        lab: изображение в LAB (uint8, OpenCV формат)
        L: яркостный канал, float32, нормализованный 0..1
        original_bgr: уменьшенное BGR-изображение (или None)
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Не удалось прочитать: {image_path}")

    # Resize по длинной стороне
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    original_bgr = bgr.copy() if return_original else None

    # Лёгкое размытие для подавления JPEG-артефактов
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1  # нечётный размер ядра
        ksize = max(ksize, 3)
        bgr = cv2.GaussianBlur(bgr, (ksize, ksize), blur_sigma)

    # Конвертация в LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # L-канал: float32, 0..1
    L = lab[:, :, 0].astype(np.float32) / 255.0

    return lab, L, original_bgr


def normalize_percentile(
    L: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    """Перцентильная нормализация яркости в [0, 1]."""
    lo = np.percentile(L, p_low)
    hi = np.percentile(L, p_high)
    eps = 1e-6
    L_norm = (L - lo) / (hi - lo + eps)
    return np.clip(L_norm, 0.0, 1.0)


def ensure_dir(path: str) -> Path:
    """Создать директорию если не существует."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
