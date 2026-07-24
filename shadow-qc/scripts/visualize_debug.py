"""
Standalone-скрипт: визуальная отладка теневой маски для одного или нескольких изображений.

Использование:
  python scripts/visualize_debug.py --image photo.jpg
  python scripts/visualize_debug.py --dir data/input/ --count 10
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта для импортов
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, read_and_preprocess, normalize_percentile, ensure_dir
from src.shadow_mask import compute_shadow_mask, get_shadow_boundary
from src.visualize_debug import _create_overlay, create_comparison_grid


def visualize_single(image_path: Path, config: dict, output_dir: str):
    """Создаёт debug-визуализацию для одного изображения."""
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
    stem = image_path.stem

    ensure_dir(f"{output_dir}/masks")
    ensure_dir(f"{output_dir}/overlays")
    ensure_dir(f"{output_dir}/grids")

    import cv2

    # Маска
    cv2.imwrite(f"{output_dir}/masks/{stem}_mask.png", mask)

    # Overlay
    overlay = _create_overlay(original_bgr, mask, boundary)
    cv2.imwrite(f"{output_dir}/overlays/{stem}_overlay.png", overlay)

    # Сетка сравнения
    create_comparison_grid(image_path, config, f"{output_dir}/grids/{stem}_grid.png")

    # Статистика
    total_px = mask.shape[0] * mask.shape[1]
    shadow_px = (mask > 0).sum()
    print(f"  {image_path.name}: shadow_area={shadow_px/total_px:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Визуальная отладка теневой маски Shadow-QC."
    )
    parser.add_argument("--image", "-i", type=str, help="Одно изображение")
    parser.add_argument("--dir", "-d", type=str, help="Папка с изображениями")
    parser.add_argument("--count", "-n", type=int, default=10, help="Макс. число файлов из папки")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Конфиг")
    parser.add_argument("--output", "-o", type=str, default="debug", help="Папка вывода")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.image:
        images = [Path(args.image)]
    elif args.dir:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = sorted(
            [f for f in Path(args.dir).rglob("*") if f.suffix.lower() in exts]
        )[:args.count]
    else:
        print("Укажите --image или --dir")
        sys.exit(1)

    if not images:
        print("Изображения не найдены.")
        sys.exit(1)

    print(f"[Debug] Обработка {len(images)} изображений → {args.output}/")
    for img in images:
        try:
            visualize_single(img, config, args.output)
        except Exception as e:
            print(f"  [ОШИБКА] {img.name}: {e}")

    print(f"\n[Debug] Готово. Результаты в: {args.output}/")


if __name__ == "__main__":
    main()
