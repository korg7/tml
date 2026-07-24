"""Debug-визуализация: overlay теневой маски и границ поверх фото."""

from pathlib import Path

import cv2
import numpy as np

from .features import extract_features
from .shadow_mask import compute_shadow_mask, get_shadow_boundary
from .utils import normalize_percentile, read_and_preprocess


def save_debug_for_image(image_path: Path, config: dict, debug_dir: str):
    """
    Сохраняет debug-изображения для одного файла:
    - masks/<name>_mask.png — бинарная теневая маска
    - overlays/<name>_overlay.png — фото с полупрозрачной маской и границами
    """
    img_cfg = config.get("image", {})
    feat_cfg = config.get("features", {})
    mask_cfg = config.get("shadow_mask", {})

    # Препроцессинг
    lab, L, original_bgr = read_and_preprocess(
        image_path,
        max_side=img_cfg.get("max_side", 1280),
        blur_sigma=img_cfg.get("blur_sigma", 1.0),
    )

    L_norm = normalize_percentile(
        L,
        p_low=feat_cfg.get("percentile_low", 1),
        p_high=feat_cfg.get("percentile_high", 99),
    )

    # Теневая маска
    mask = compute_shadow_mask(
        lab,
        L_norm,
        method=mask_cfg.get("method", "combined"),
        dark_sigma_k=feat_cfg.get("dark_sigma_k", 0.7),
        morph_kernel=mask_cfg.get("morph_kernel", 5),
        min_area_px=mask_cfg.get("min_area_px", 500),
        specthem_threshold=mask_cfg.get("specthem_threshold", 1.0),
    )

    boundary = get_shadow_boundary(mask)
    stem = image_path.stem

    # Сохраняем маску
    mask_path = Path(debug_dir) / "masks" / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # Создаём overlay
    overlay = _create_overlay(original_bgr, mask, boundary)
    overlay_path = Path(debug_dir) / "overlays" / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)


def _create_overlay(
    bgr: np.ndarray, mask: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    """
    Создаёт визуализацию: фото + полупрозрачная красная маска + жёлтые границы.
    """
    overlay = bgr.copy()

    # Полупрозрачная красная маска теней
    mask_color = np.zeros_like(bgr)
    mask_color[mask > 0] = [0, 0, 200]  # BGR: красный
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.35, 0)

    # Жёлтые границы теней
    overlay[boundary > 0] = [0, 255, 255]  # BGR: жёлтый

    return overlay


def create_comparison_grid(
    image_path: Path, config: dict, output_path: str
):
    """
    Создаёт сетку сравнения: оригинал | L-канал | маска | overlay.
    Полезно для быстрой визуальной проверки.
    """
    img_cfg = config.get("image", {})
    feat_cfg = config.get("features", {})
    mask_cfg = config.get("shadow_mask", {})

    lab, L, original_bgr = read_and_preprocess(
        image_path,
        max_side=img_cfg.get("max_side", 1280),
        blur_sigma=img_cfg.get("blur_sigma", 1.0),
    )

    L_norm = normalize_percentile(
        L,
        p_low=feat_cfg.get("percentile_low", 1),
        p_high=feat_cfg.get("percentile_high", 99),
    )

    mask = compute_shadow_mask(
        lab,
        L_norm,
        method=mask_cfg.get("method", "combined"),
        dark_sigma_k=feat_cfg.get("dark_sigma_k", 0.7),
        morph_kernel=mask_cfg.get("morph_kernel", 5),
        min_area_px=mask_cfg.get("min_area_px", 500),
        specthem_threshold=mask_cfg.get("specthem_threshold", 1.0),
    )

    boundary = get_shadow_boundary(mask)

    # L-канал как цветное изображение
    L_vis = (L_norm * 255).astype(np.uint8)
    L_vis = cv2.cvtColor(L_vis, cv2.COLOR_GRAY2BGR)

    # Маска как цветное
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Overlay
    overlay = _create_overlay(original_bgr, mask, boundary)

    # Сетка 2x2
    h, w = original_bgr.shape[:2]
    top = np.hstack([original_bgr, L_vis])
    bottom = np.hstack([mask_vis, overlay])
    grid = np.vstack([top, bottom])

    cv2.imwrite(output_path, grid)
