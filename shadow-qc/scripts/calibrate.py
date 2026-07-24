"""Калибровка порогов на размеченном наборе данных."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

# Добавляем родительскую директорию для импортов
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classify import FEATURE_NAMES


def load_labeled_data(labeled_csv: str, features_csv: str) -> pd.DataFrame:
    """
    Объединяет ручную разметку с извлечёнными признаками.

    labeled_csv: filename, label (good/review/bad)
    features_csv: filename + все признаки (из predictions.csv)
    """
    labeled = pd.read_csv(labeled_csv)
    features = pd.read_csv(features_csv)

    # Нормализуем имена файлов для join
    labeled["filename"] = labeled["filename"].apply(lambda x: Path(x).name)
    features["filename"] = features["filename"].apply(lambda x: Path(x).name)

    merged = features.merge(labeled[["filename", "label"]], on="filename", how="inner")

    if merged.empty:
        raise ValueError(
            "Нет пересечений между labeled.csv и features.csv. "
            "Проверьте имена файлов."
        )

    print(f"[Калибровка] Загружено: {len(merged)} размеченных кадров")
    print(f"  good: {(merged['label'] == 'good').sum()}")
    print(f"  review: {(merged['label'] == 'review').sum()}")
    print(f"  bad: {(merged['label'] == 'bad').sum()}")

    return merged


def analyze_distributions(df: pd.DataFrame):
    """Выводит статистику признаков по классам."""
    print("\n" + "=" * 70)
    print("  Распределение признаков по классам")
    print("=" * 70)

    key_features = [
        "shadow_score",
        "local_contrast_p95",
        "shadow_area_ratio",
        "boundary_gradient_mean",
        "boundary_brightness_drop",
        "rms_contrast",
        "bimodality",
    ]

    for feat in key_features:
        if feat not in df.columns:
            continue
        print(f"\n  {feat}:")
        for label in ["good", "review", "bad"]:
            subset = df[df["label"] == label][feat]
            if len(subset) > 0:
                print(
                    f"    {label:8s}: mean={subset.mean():.4f}  "
                    f"std={subset.std():.4f}  "
                    f"[{subset.min():.4f} .. {subset.max():.4f}]"
                )


def suggest_thresholds(df: pd.DataFrame):
    """Предлагает пороги на основе пересечения распределений."""
    print("\n" + "=" * 70)
    print("  Рекомендуемые пороги")
    print("=" * 70)

    good = df[df["label"] == "good"]
    bad = df[df["label"] == "bad"]

    if good.empty or bad.empty:
        print("  Недостаточно данных для good или bad класса.")
        return

    # shadow_score: порог между good и bad
    good_ss = good["shadow_score"]
    bad_ss = bad["shadow_score"]
    threshold_ss = (good_ss.quantile(0.95) + bad_ss.quantile(0.05)) / 2
    print(f"\n  shadow_score:")
    print(f"    good P95 = {good_ss.quantile(0.95):.4f}")
    print(f"    bad  P05 = {bad_ss.quantile(0.05):.4f}")
    print(f"    → good_shadow_score_max = {threshold_ss:.4f}")
    print(f"    → bad_shadow_score_min  = {bad_ss.quantile(0.25):.4f}")

    # local_contrast_p95
    good_lc = good["local_contrast_p95"]
    bad_lc = bad["local_contrast_p95"]
    threshold_lc = (good_lc.quantile(0.95) + bad_lc.quantile(0.05)) / 2
    print(f"\n  local_contrast_p95:")
    print(f"    good P95 = {good_lc.quantile(0.95):.4f}")
    print(f"    bad  P05 = {bad_lc.quantile(0.05):.4f}")
    print(f"    → good_local_contrast_max = {threshold_lc:.4f}")
    print(f"    → bad_local_contrast_min  = {bad_lc.quantile(0.25):.4f}")


def train_ml_model(df: pd.DataFrame, output_path: str = "model.pkl"):
    """Обучает LogisticRegression на признаках."""
    print("\n" + "=" * 70)
    print("  Обучение ML-модели (LogisticRegression)")
    print("=" * 70)

    # Подготовка данных
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[available_features].values.astype(np.float32)
    y = df["label"].values

    # Замена NaN
    X = np.nan_to_num(X, nan=0.0)

    # Обучение
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )

    # Кросс-валидация для оценки
    if len(df) >= 10:
        y_pred = cross_val_predict(model, X, y, cv=min(5, len(df) // 3))
        print("\n  Cross-validation (5-fold):")
        print(classification_report(y, y_pred, zero_division=0))
        print("  Confusion matrix:")
        print(confusion_matrix(y, y_pred, labels=["good", "review", "bad"]))

    # Финальное обучение на всех данных
    model.fit(X, y)

    # Сохранение
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Модель сохранена: {output_path}")
    print(f"  Классы: {list(model.classes_)}")
    print(f"  Признаки ({len(available_features)}): {available_features}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Калибровка порогов Shadow-QC на размеченном наборе."
    )
    parser.add_argument(
        "--labeled", "-l",
        type=str,
        default="data/labeled.csv",
        help="CSV с ручной разметкой (filename, label)",
    )
    parser.add_argument(
        "--features", "-f",
        type=str,
        default="data/predictions.csv",
        help="CSV с извлечёнными признаками (из pipeline)",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Обучить ML-модель и сохранить",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="model.pkl",
        help="Путь для сохранения модели",
    )

    args = parser.parse_args()

    # Загрузка данных
    df = load_labeled_data(args.labeled, args.features)

    # Анализ распределений
    analyze_distributions(df)

    # Рекомендации по порогам
    suggest_thresholds(df)

    # ML-модель
    if args.train_ml:
        train_ml_model(df, args.model_output)

    print("\n[Калибровка] Готово.")


if __name__ == "__main__":
    main()
