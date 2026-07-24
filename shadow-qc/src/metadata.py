"""Запись метаданных через ExifTool: Rating, Label, Keywords для ACDSee."""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from .classify import Decision


class MetadataWriter:
    """Пакетная запись метаданных через ExifTool."""

    def __init__(self, config: dict):
        meta_cfg = config.get("metadata", {})
        self.exiftool_path = meta_cfg.get("exiftool_path", "exiftool")
        self.rating_map = {
            Decision.GOOD: meta_cfg.get("rating_good", 5),
            Decision.REVIEW: meta_cfg.get("rating_review", 3),
            Decision.BAD: meta_cfg.get("rating_bad", 1),
        }
        self.label_map = {
            Decision.GOOD: meta_cfg.get("label_good", "Green"),
            Decision.REVIEW: meta_cfg.get("label_review", "Yellow"),
            Decision.BAD: meta_cfg.get("label_bad", "Red"),
        }
        self.keyword_prefix = meta_cfg.get("keyword_prefix", "QC_")
        self._check_exiftool()

    def _check_exiftool(self):
        """Проверяет доступность ExifTool."""
        if shutil.which(self.exiftool_path) is None:
            raise RuntimeError(
                f"ExifTool не найден: '{self.exiftool_path}'. "
                f"Скачайте с https://exiftool.org/ и добавьте в PATH."
            )

    def _get_keywords(self, decision: Decision) -> List[str]:
        """Формирует список ключевых слов для решения."""
        prefix = self.keyword_prefix
        if decision == Decision.GOOD:
            return [f"{prefix}GOOD", "NO_HARD_SHADOW"]
        elif decision == Decision.REVIEW:
            return [f"{prefix}REVIEW", "POSSIBLE_SHADOW"]
        else:  # BAD
            return [f"{prefix}BAD", "HARD_SHADOW"]

    def write_single(self, filepath: Path, decision: Decision) -> bool:
        """Записывает метаданные в один файл. Возвращает True при успехе."""
        rating = self.rating_map[decision]
        label = self.label_map[decision]
        keywords = self._get_keywords(decision)

        cmd = [
            self.exiftool_path,
            "-overwrite_original",
            f"-XMP-xmp:Rating={rating}",
            f"-XMP-acdsee:Rating={rating}",
            f"-XMP-xmp:Label={label}",
        ]

        # Очищаем старые QC-ключевые слова перед добавлением новых
        cmd.append(f"-XMP-dc:Subject-={self.keyword_prefix}*")
        cmd.append(f"-IPTC:Keywords-={self.keyword_prefix}*")

        # Добавляем ключевые слова
        for kw in keywords:
            cmd.append(f"-XMP-dc:Subject+={kw}")
            cmd.append(f"-IPTC:Keywords+={kw}")

        cmd.append(str(filepath))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def write_batch(
        self, decisions: Dict[Path, Decision], progress_callback=None
    ) -> Tuple[int, int]:
        """
        Пакетная запись метаданных.

        Args:
            decisions: словарь {путь_к_файлу: решение}
            progress_callback: опциональный callback(current, total)

        Returns:
            (success_count, error_count)
        """
        total = len(decisions)
        success = 0
        errors = 0

        for i, (filepath, decision) in enumerate(decisions.items()):
            if self.write_single(filepath, decision):
                success += 1
            else:
                errors += 1

            if progress_callback:
                progress_callback(i + 1, total)

        return success, errors

    def write_batch_grouped(
        self, decisions: Dict[Path, Decision], progress_callback=None
    ) -> Tuple[int, int]:
        """
        Пакетная запись через ExifTool с группировкой по классам (минимум вызовов).
        """
        # Группируем файлы по решениям
        groups: Dict[Decision, List[Path]] = {
            Decision.GOOD: [],
            Decision.REVIEW: [],
            Decision.BAD: [],
        }
        for filepath, decision in decisions.items():
            groups[decision].append(filepath)

        total = len(decisions)
        success = 0
        errors = 0
        processed = 0

        for decision, files in groups.items():
            if not files:
                continue

            rating = self.rating_map[decision]
            label = self.label_map[decision]
            keywords = self._get_keywords(decision)

            # Формируем команду для группы файлов
            cmd = [
                self.exiftool_path,
                "-overwrite_original",
                f"-XMP-xmp:Rating={rating}",
                f"-XMP-acdsee:Rating={rating}",
                f"-XMP-xmp:Label={label}",
            ]
            # Очищаем старые QC-теги перед добавлением
            cmd.append(f"-XMP-dc:Subject-={self.keyword_prefix}*")
            cmd.append(f"-IPTC:Keywords-={self.keyword_prefix}*")
            for kw in keywords:
                cmd.append(f"-XMP-dc:Subject+={kw}")
                cmd.append(f"-IPTC:Keywords+={kw}")

            # Добавляем все файлы группы
            cmd.extend([str(f) for f in files])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 минут на группу
                )
                if result.returncode == 0:
                    success += len(files)
                else:
                    # ExifTool вернул ошибку — считаем по stderr
                    # Обычно часть файлов обработана
                    error_lines = [
                        l for l in result.stderr.splitlines() if "Error" in l
                    ]
                    err_count = len(error_lines) if error_lines else len(files)
                    errors += err_count
                    success += len(files) - err_count
            except (subprocess.TimeoutExpired, OSError):
                errors += len(files)

            processed += len(files)
            if progress_callback:
                progress_callback(processed, total)

        return success, errors
