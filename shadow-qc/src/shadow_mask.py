"""Построение теневой маски: комбинированный метод (LAB-порог + Silva CIELCh)."""

import cv2
import numpy as np


def compute_shadow_mask(
    lab: np.ndarray,
    L_norm: np.ndarray,
    method: str = "combined",
    dark_sigma_k: float = 0.7,
    morph_kernel: int = 5,
    min_area_px: int = 500,
    specthem_threshold: float = 1.0,
) -> np.ndarray:
    """
    Строит бинарную маску теней.

    Args:
        lab: изображение в LAB (uint8, OpenCV)
        L_norm: нормализованный L-канал (float32, 0..1)
        method: "combined" | "lab_threshold" | "silva"
        dark_sigma_k: коэффициент для адаптивного порога (L < mean - k*std)
        morph_kernel: размер ядра морфологии
        min_area_px: минимальная площадь связной области
        specthem_threshold: порог для Specthem ratio (Silva)

    Returns:
        mask: uint8, 0 или 255
    """
    h, w = L_norm.shape[:2]

    if method == "lab_threshold":
        mask = _mask_lab_threshold(L_norm, dark_sigma_k)
    elif method == "silva":
        mask = _mask_silva_specthem(lab, specthem_threshold)
    else:  # combined
        mask_lab = _mask_lab_threshold(L_norm, dark_sigma_k)
        mask_silva = _mask_silva_specthem(lab, specthem_threshold)
        # Объединение: пересечение (AND) для снижения ложных срабатываний
        mask = cv2.bitwise_and(mask_lab, mask_silva)

    # Морфологическая очистка
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Фильтрация мелких областей
    mask = _filter_small_regions(mask, min_area_px)

    return mask


def _mask_lab_threshold(L_norm: np.ndarray, k: float) -> np.ndarray:
    """
    Адаптивный порог: пиксель = тень, если L < local_mean - k * local_std.
    Работает по нормализованному L-каналу.
    """
    # Локальный фон (большое ядро размытия)
    ksize = 51  # достаточно большое для оценки фона
    local_mean = cv2.blur(L_norm, (ksize, ksize))

    # Локальное стандартное отклонение
    local_sq_mean = cv2.blur(L_norm * L_norm, (ksize, ksize))
    local_var = local_sq_mean - local_mean * local_mean
    local_var = np.clip(local_var, 0, None)
    local_std = np.sqrt(local_var)

    # Порог: тень там, где яркость ниже фона на k * std
    threshold = local_mean - k * local_std
    mask = (L_norm < threshold).astype(np.uint8) * 255

    return mask


def _mask_silva_specthem(lab: np.ndarray, threshold: float) -> np.ndarray:
    """
    Упрощённый метод Silva 2017: Specthem ratio в CIELCh.
    Тень имеет низкую яркость (L) и повышенную долю синего (hue сдвиг).

    В OpenCV LAB: L [0..255], a [0..255] (128=0), b [0..255] (128=0).
    """
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    # Перевод в LCh
    C = np.sqrt(a * a + b * b)  # Chroma
    h = np.arctan2(b, a)  # Hue в радианах

    # Specthem ratio: отношение яркости к цветности
    # В тени: L низкий, C относительно сохраняется → ratio низкий
    # На освещённом участке: L высокий → ratio высокий
    eps = 1e-6
    specthem = L / (C + eps)

    # Нормализация по перцентилям
    p5 = np.percentile(specthem, 5)
    p95 = np.percentile(specthem, 95)
    specthem_norm = (specthem - p5) / (p95 - p5 + eps)
    specthem_norm = np.clip(specthem_norm, 0, 1)

    # Тень: низкий specthem ratio + низкая яркость
    L_norm = L / 255.0
    L_median = np.median(L_norm)

    # Комбинированный критерий
    shadow_cond = (specthem_norm < threshold * 0.4) & (L_norm < L_median)
    mask = shadow_cond.astype(np.uint8) * 255

    return mask


def _filter_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Удаляет связные области меньше min_area пикселей."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    result = np.zeros_like(mask)
    for i in range(1, num_labels):  # пропускаем фон (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result


def get_shadow_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Извлекает границу теневой маски (внутренний контур).
    Returns: бинарное изображение границы (0/255).
    """
    # Dilate - original = boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    boundary = cv2.subtract(dilated, mask)
    return boundary
