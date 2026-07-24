"""CLI точка входа для Shadow-QC."""

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline
from .utils import load_config


def main():
    parser = argparse.ArgumentParser(
        prog="shadow-qc",
        description="Автоотбраковка кадров с жёсткими тенями для стройплощадки.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m src.cli --input photos/ --dry-run
  python -m src.cli --input photos/ --config my_config.yaml
  python -m src.cli --input photos/ --write --workers 4
  python -m src.cli --input photos/ --debug
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Папка с фотографиями (переопределяет config.yaml)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Путь к конфигурации (по умолчанию: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Только анализ и CSV, без записи метаданных",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Записать метаданные в файлы (отменяет dry_run из конфига)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Число параллельных процессов (0 = auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Сохранять debug-изображения (маски, overlay)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь к выходному CSV (переопределяет config.yaml)",
    )

    args = parser.parse_args()

    # Загрузка конфигурации
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Shadow-QC] Конфиг не найден: {config_path}")
        print("[Shadow-QC] Создайте config.yaml или укажите --config <path>")
        sys.exit(1)

    config = load_config(str(config_path))

    # Переопределения из CLI
    if args.output:
        config["output_csv"] = args.output

    # Определяем dry_run
    if args.write:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = config.get("dry_run", True)

    # Запуск
    print("=" * 60)
    print("  Shadow-QC: автоотбраковка кадров с жёсткими тенями")
    print("=" * 60)

    df = run_pipeline(
        config=config,
        input_dir=args.input,
        dry_run=dry_run,
        workers=args.workers,
        save_debug=args.debug,
    )

    if df.empty:
        print("\n[Shadow-QC] Нечего обрабатывать.")
        sys.exit(0)

    print(f"\n[Shadow-QC] Готово. Обработано: {len(df)} файлов.")
    if dry_run:
        print("[Shadow-QC] Режим DRY-RUN: метаданные НЕ записаны.")
        print("[Shadow-QC] Для записи запустите с флагом --write")


if __name__ == "__main__":
    main()
