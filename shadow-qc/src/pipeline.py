"""Оркестрация пайплайна: обход файлов → анализ → классификация → запись."""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .classify import Decision, classify
from .features import FeatureVector, extract_features
from .shadow_mask import compute_shadow_mask
from .utils import (
    ensure_dir,
    find_images,
    normalize_percentile,
    read_and_preprocess,
)


def process_single_image(
    image_path: Path, config: dict
) -> Tuple[Path, FeatureVector, Decision]:
    """
    Обрабатывает одно изображение: препроцессинг → маска → признаки → класс.
    Вызывается в воркере.
    """
    img_cfg = config.get("image", {})
    feat_cfg = config.get("features", {})
    mask_cfg = config.get("shadow_mask", {})

    # Препроцессинг (original_bgr не нужен в воркере — экономим память)
    lab, L, _ = read_and_preprocess(
        image_path,
        max_side=img_cfg.get("max_side", 1280),
        blur_sigma=img_cfg.get("blur_sigma", 1.0),
        return_original=False,
    )

    # Нормализация
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

    # Извлечение признаков
    fv = extract_features(lab, L_norm, mask, feat_cfg)

    # Классификация
    decision = classify(fv, config)

    return image_path, fv, decision


def run_pipeline(
    config: dict,
    input_dir: Optional[str] = None,
    dry_run: Optional[bool] = None,
    workers: Optional[int] = None,
    save_debug: bool = False,
) -> pd.DataFrame:
    """
    Запуск полного пайплайна.

    Args:
        config: конфигурация
        input_dir: переопределение папки входа
        dry_run: переопределение режима (True = не писать метаданные)
        workers: число процессов (0 = auto)
        save_debug: сохранять debug-изображения

    Returns:
        DataFrame с результатами
    """
    # Параметры
    in_dir = input_dir or config.get("input_dir", "data/input")
    is_dry_run = dry_run if dry_run is not None else config.get("dry_run", True)
    img_cfg = config.get("image", {})
    extensions = img_cfg.get("extensions", [".jpg", ".jpeg", ".png", ".tif", ".tiff"])

    par_cfg = config.get("parallel", {})
    n_workers = workers if workers is not None else par_cfg.get("workers", 0)
    if n_workers == 0:
        import os
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Поиск изображений
    images = find_images(in_dir, extensions)
    if not images:
        print(f"[Shadow-QC] Изображения не найдены в: {in_dir}")
        return pd.DataFrame()

    print(f"[Shadow-QC] Найдено изображений: {len(images)}")
    print(f"[Shadow-QC] Режим: {'DRY-RUN (без записи)' if is_dry_run else 'ЗАПИСЬ метаданных'}")
    print(f"[Shadow-QC] Воркеров: {n_workers}")

    # Обработка
    results: List[Dict] = []
    decisions: Dict[Path, Decision] = {}
    start_time = time.time()

    if n_workers == 1:
        # Последовательная обработка (проще для отладки)
        for img_path in tqdm(images, desc="Анализ"):
            try:
                path, fv, decision = process_single_image(img_path, config)
                row = {"filename": str(path), "decision": decision.value}
                row.update(fv.to_dict())
                results.append(row)
                decisions[path] = decision
            except Exception as e:
                print(f"  [ОШИБКА] {img_path.name}: {e}")
                results.append({"filename": str(img_path), "decision": "error", "error": str(e)})
    else:
        # Параллельная обработка
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_single_image, img_path, config): img_path
                for img_path in images
            }
            with tqdm(total=len(images), desc="Анализ") as pbar:
                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        path, fv, decision = future.result()
                        row = {"filename": str(path), "decision": decision.value}
                        row.update(fv.to_dict())
                        results.append(row)
                        decisions[path] = decision
                    except Exception as e:
                        print(f"  [ОШИБКА] {img_path.name}: {e}")
                        results.append({"filename": str(img_path), "decision": "error", "error": str(e)})
                    pbar.update(1)

    elapsed = time.time() - start_time
    fps = len(images) / elapsed if elapsed > 0 else 0

    # DataFrame
    df = pd.DataFrame(results)

    # Статистика
    if not df.empty and "decision" in df.columns:
        counts = df["decision"].value_counts()
        print(f"\n[Shadow-QC] Результаты ({elapsed:.1f} сек, {fps:.1f} кадр/сек):")
        for dec in ["good", "review", "bad", "error"]:
            if dec in counts.index:
                print(f"  {dec}: {counts[dec]}")

    # Сохранение CSV
    output_csv = config.get("output_csv", "data/predictions.csv")
    ensure_dir(str(Path(output_csv).parent))
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[Shadow-QC] CSV сохранён: {output_csv}")

    # Запись метаданных
    if not is_dry_run and decisions:
        _write_metadata(config, decisions)

    # Debug-визуализация
    if save_debug:
        _save_debug_images(config, images)

    return df


def _write_metadata(config: dict, decisions: Dict[Path, Decision]):
    """Запись метаданных через ExifTool."""
    from .metadata import MetadataWriter

    meta_cfg = config.get("metadata", {})
    if not meta_cfg.get("enabled", True):
        print("[Shadow-QC] Запись метаданных отключена в конфиге.")
        return

    print(f"\n[Shadow-QC] Запись метаданных ({len(decisions)} файлов)...")
    try:
        writer = MetadataWriter(config)
        success, errors = writer.write_batch_grouped(
            decisions,
            progress_callback=lambda cur, tot: None,  # tqdm уже отработал
        )
        print(f"[Shadow-QC] Метаданные: успех={success}, ошибок={errors}")
        if errors > 0:
            print("[Shadow-QC] ВНИМАНИЕ: есть ошибки записи. Проверьте файлы.")
    except RuntimeError as e:
        print(f"[Shadow-QC] ОШИБКА: {e}")


def _save_debug_images(config: dict, images: List[Path]):
    """Сохраняет debug-изображения (маски поверх фото)."""
    from .visualize_debug import save_debug_for_image

    debug_dir = config.get("debug_dir", "debug")
    img_cfg = config.get("image", {})
    feat_cfg = config.get("features", {})
    mask_cfg = config.get("shadow_mask", {})

    ensure_dir(f"{debug_dir}/overlays")
    ensure_dir(f"{debug_dir}/masks")

    print(f"\n[Shadow-QC] Сохранение debug-изображений...")
    for img_path in tqdm(images[:50], desc="Debug"):  # максимум 50 для скорости
        try:
            save_debug_for_image(img_path, config, debug_dir)
        except Exception:
            pass
