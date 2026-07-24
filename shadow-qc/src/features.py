"""Извлечение признаков: глобальные, локальные, градиентные, теневые."""

from dataclasses import dataclass, asdict
from typing import Dict

import cv2
import numpy as np

from .shadow_mask import compute_shadow_mask, get_shadow_boundary


@dataclass
class FeatureVector:
    """Вектор признаков одного кадра."""

    # Глобальные
    rms_contrast: float = 0.0
    michelson_pct: float = 0.0
    dynamic_range: float = 0.0
    histogram_entropy: float = 0.0
    bimodality: float = 0.0
    clipped_highlight_ratio: float = 0.0

    # Локальные
    local_contrast_p95: float = 0.0
    local_contrast_p50: float = 0.0

    # Градиентные
    sobel_p95: float = 0.0
    edge_pixel_ratio: float = 0.0

    # Теневые
    shadow_area_ratio: float = 0.0
    boundary_gradient_mean: float = 0.0
    boundary_brightness_drop: float = 0.0
    boundary_length_norm: float = 0.0
    chroma_diff_at_boundary: float = 0.0

    # Итоговый
    shadow_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def extract_features(
    lab: np.ndarray,
    L_norm: np.ndarray,
    mask: np.ndarray,
    config: dict,
) -> FeatureVector:
    """
    Извлекает полный вектор признаков из подготовленного изображения.

    Args:
        lab: LAB изображение (uint8)
        L_norm: нормализованный L-канал (float32, 0..1)
        mask: теневая маска (uint8, 0/255)
        config: секция features из config.yaml
    """
    fv = FeatureVector()
    eps = 1e-6
    h, w = L_norm.shape

    # === Глобальные признаки ===
    fv.rms_contrast = float(np.std(L_norm))

    p1 = np.percentile(L_norm, 1)
    p5 = np.percentile(L_norm, 5)
    p95 = np.percentile(L_norm, 95)
    p99 = np.percentile(L_norm, 99)

    fv.michelson_pct = float((p99 - p1) / (p99 + p1 + eps))
    fv.dynamic_range = float(p95 - p5)

    # Энтропия гистограммы
    hist, _ = np.histogram(L_norm.ravel(), bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + eps)
    hist = hist[hist > 0]
    fv.histogram_entropy = float(-np.sum(hist * np.log2(hist)))

    # Бимодальность (межклассовая дисперсия Отсу)
    fv.bimodality = _otsu_separability(L_norm)

    # Доля пересвеченных пикселей
    fv.clipped_highlight_ratio = float(np.mean(L_norm > 0.98))

    # === Локальные признаки ===
    win = config.get("local_window", 48)
    local_std = _local_std(L_norm, win)
    fv.local_contrast_p95 = float(np.percentile(local_std, 95))
    fv.local_contrast_p50 = float(np.percentile(local_std, 50))

    # === Градиентные признаки ===
    L_uint8 = (L_norm * 255).astype(np.uint8)

    sobel_x = cv2.Sobel(L_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(L_uint8, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    fv.sobel_p95 = float(np.percentile(sobel_mag, 95))

    canny_low = config.get("canny_low", 50)
    canny_high = config.get("canny_high", 150)
    edges = cv2.Canny(L_uint8, canny_low, canny_high)
    fv.edge_pixel_ratio = float(np.mean(edges > 0))

    # === Теневые признаки ===
    total_pixels = h * w
    shadow_pixels = np.sum(mask > 0)
    fv.shadow_area_ratio = float(shadow_pixels / total_pixels)

    # Граница тени
    boundary = get_shadow_boundary(mask)
    boundary_pixels = boundary > 0
    boundary_count = np.sum(boundary_pixels)

    if boundary_count > 0:
        # Средний градиент на границе
        fv.boundary_gradient_mean = float(np.mean(sobel_mag[boundary_pixels]))

        # Перепад яркости поперёк границы
        fv.boundary_brightness_drop = _boundary_brightness_drop(
            L_norm, mask, boundary
        )

        # Длина границы (нормированная)
        diagonal = np.sqrt(h**2 + w**2)
        fv.boundary_length_norm = float(boundary_count / diagonal)

        # Цветовая разница на границе
        fv.chroma_diff_at_boundary = _chroma_diff_at_boundary(lab, boundary)
    else:
        fv.boundary_gradient_mean = 0.0
        fv.boundary_brightness_drop = 0.0
        fv.boundary_length_norm = 0.0
        fv.chroma_diff_at_boundary = 0.0

    # === Итоговый shadow_score ===
    fv.shadow_score = _compute_shadow_score(fv)

    return fv


def _local_std(L: np.ndarray, window: int) -> np.ndarray:
    """Локальное стандартное отклонение в скользящем окне."""
    ksize = window | 1  # нечётный
    local_mean = cv2.blur(L, (ksize, ksize))
    local_sq_mean = cv2.blur(L * L, (ksize, ksize))
    local_var = local_sq_mean - local_mean * local_mean
    local_var = np.clip(local_var, 0, None)
    return np.sqrt(local_var)


def _otsu_separability(L_norm: np.ndarray) -> float:
    """Межклассовая дисперсия Отсу как мера бимодальности."""
    L_uint8 = (L_norm * 255).astype(np.uint8)
    # cv2.threshold возвращает (thresh, dst); otsu_val = межклассовая дисперсия
    otsu_val, _ = cv2.threshold(
        L_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Нормируем на максимум (255^2 / 4 = теоретический максимум)
    return float(otsu_val / (255.0 * 255.0 / 4.0 + 1e-6))


def _boundary_brightness_drop(
    L_norm: np.ndarray, mask: np.ndarray, boundary: np.ndarray
) -> float:
    """
    Средний перепад яркости поперёк границы тени.
    Сравниваем среднюю яркость внутри тени и снаружи в окрестности границы.
    """
    # Расширяем маску и границу для захвата окрестностей
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Внутренняя сторона (тень у границы)
    inner = cv2.erode(mask, kernel, iterations=1)
    inner_boundary = cv2.bitwise_and(inner, cv2.dilate(boundary, kernel))

    # Внешняя сторона (свет у границы)
    inv_mask = cv2.bitwise_not(mask)
    outer = cv2.erode(inv_mask, kernel, iterations=1)
    outer_boundary = cv2.bitwise_and(outer, cv2.dilate(boundary, kernel))

    inner_px = inner_boundary > 0
    outer_px = outer_boundary > 0

    if np.sum(inner_px) == 0 or np.sum(outer_px) == 0:
        return 0.0

    mean_dark = np.mean(L_norm[inner_px])
    mean_light = np.mean(L_norm[outer_px])

    return float(max(0, mean_light - mean_dark))


def _chroma_diff_at_boundary(lab: np.ndarray, boundary: np.ndarray) -> float:
    """
    Средняя цветовая разница (|Δa| + |Δb|) на границе тени.
    Для тени цветность меняется мало; для объекта — много.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary_dilated = cv2.dilate(boundary, kernel, iterations=1)
    boundary_px = boundary_dilated > 0

    if np.sum(boundary_px) < 10:
        return 0.0

    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # Градиент цветности
    a_grad = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(
        a, cv2.CV_32F, 0, 1, ksize=3
    )
    b_grad = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(
        b, cv2.CV_32F, 0, 1, ksize=3
    )
    chroma_grad = np.abs(a_grad) + np.abs(b_grad)

    return float(np.mean(chroma_grad[boundary_px]))


def _compute_shadow_score(fv: FeatureVector) -> float:
    """
    Взвешенная комбинация теневых признаков → единый shadow_score [0..1].
    Веса начальные, калибруются на размеченном наборе.
    """
    # Нормируем каждый признак в [0, 1] эмпирическими максимумами
    brightness_drop_norm = min(fv.boundary_brightness_drop / 0.5, 1.0)
    gradient_norm = min(fv.boundary_gradient_mean / 80.0, 1.0)
    area_norm = min(fv.shadow_area_ratio / 0.5, 1.0)
    length_norm = min(fv.boundary_length_norm / 2.0, 1.0)
    # Инвертируем chroma_diff: мало цвета = больше похоже на тень
    chroma_norm = min(fv.chroma_diff_at_boundary / 50.0, 1.0)
    chroma_inv = 1.0 - chroma_norm

    score = (
        0.30 * brightness_drop_norm
        + 0.25 * gradient_norm
        + 0.20 * area_norm
        + 0.10 * length_norm
        + 0.15 * chroma_inv
    )
    return float(np.clip(score, 0.0, 1.0))
