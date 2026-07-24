"""Классификация кадров: правила по порогам или ML-модель."""

import logging
import pickle
from enum import Enum
from pathlib import Path

import numpy as np

from .features import FeatureVector

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    GOOD = "good"
    REVIEW = "review"
    BAD = "bad"


def classify_by_rules(fv: FeatureVector, config: dict) -> Decision:
    """
    Классификация по пороговым правилам.

    Логика:
      - GOOD: shadow_score < good_max AND local_contrast_p95 < good_lc_max
      - BAD:  shadow_score > bad_min OR local_contrast_p95 > bad_lc_min
      - REVIEW: всё остальное
    """
    cls_cfg = config.get("classification", {})

    good_shadow_max = cls_cfg.get("good_shadow_score_max", 0.25)
    good_lc_max = cls_cfg.get("good_local_contrast_max", 0.22)
    bad_shadow_min = cls_cfg.get("bad_shadow_score_min", 0.55)
    bad_lc_min = cls_cfg.get("bad_local_contrast_min", 0.38)

    # Дополнительная проверка: пересвет
    if fv.clipped_highlight_ratio > 0.20:
        return Decision.REVIEW

    # GOOD: низкий shadow_score И низкий локальный контраст
    if fv.shadow_score < good_shadow_max and fv.local_contrast_p95 < good_lc_max:
        return Decision.GOOD

    # BAD: высокий shadow_score ИЛИ высокий локальный контраст
    if fv.shadow_score > bad_shadow_min or fv.local_contrast_p95 > bad_lc_min:
        return Decision.BAD

    return Decision.REVIEW


def classify_by_ml(
    fv: FeatureVector, model_path: str, config: dict,
    prob_bad: float = 0.70, prob_good: float = 0.70,
) -> Decision:
    """
    Классификация через обученную модель (LogisticRegression / RandomForest).
    Модель должна быть обучена на векторе признаков и возвращать вероятности.
    """
    model = _load_model(model_path)
    if model is None:
        # Fallback на правила с полным конфигом пользователя
        logger.warning(
            "ML-модель не найдена (%s), fallback на правила.", model_path
        )
        return classify_by_rules(fv, config)

    # Формируем вектор признаков в том же порядке, что при обучении
    feature_vec = _feature_vector_to_array(fv)
    probs = model.predict_proba(feature_vec.reshape(1, -1))[0]

    # Предполагаем порядок классов: [bad, good, review] или [0, 1, 2]
    classes = model.classes_
    prob_dict = {str(c): p for c, p in zip(classes, probs)}

    p_bad = prob_dict.get("bad", prob_dict.get("2", 0.0))
    p_good = prob_dict.get("good", prob_dict.get("0", 0.0))

    if p_bad > prob_bad:
        return Decision.BAD
    elif p_good > prob_good:
        return Decision.GOOD
    else:
        return Decision.REVIEW


def classify(fv: FeatureVector, config: dict) -> Decision:
    """Единая точка входа классификации."""
    cls_cfg = config.get("classification", {})
    mode = cls_cfg.get("mode", "rules")

    if mode == "ml":
        model_path = cls_cfg.get("ml_model_path", "")
        prob_bad = cls_cfg.get("prob_bad_threshold", 0.70)
        prob_good = cls_cfg.get("prob_good_threshold", 0.70)
        return classify_by_ml(fv, model_path, config, prob_bad, prob_good)
    else:
        return classify_by_rules(fv, config)


def _feature_vector_to_array(fv: FeatureVector) -> np.ndarray:
    """Преобразует FeatureVector в numpy-массив для ML-модели."""
    return np.array(
        [getattr(fv, name) for name in FEATURE_NAMES], dtype=np.float32
    )


FEATURE_NAMES = [
    "rms_contrast",
    "michelson_pct",
    "dynamic_range",
    "histogram_entropy",
    "bimodality",
    "clipped_highlight_ratio",
    "local_contrast_p95",
    "local_contrast_p50",
    "sobel_p95",
    "edge_pixel_ratio",
    "shadow_area_ratio",
    "boundary_gradient_mean",
    "boundary_brightness_drop",
    "boundary_length_norm",
    "chroma_diff_at_boundary",
    "shadow_score",
]


def _load_model(path: str):
    """Загрузка обученной модели из pickle."""
    if not path or not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Не удалось загрузить модель '%s': %s", path, e)
        return None
