"""
Model training module for car valuation.

This module handles training AutoGluon models for car price prediction:
- Main point-estimate models
- Quantile regression models for confidence intervals
- Tail-specific Ridge regression models for extreme value handling
"""

import os
import pickle
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import Ridge

from model_cache import clear_cache


def train_autogluon(
    data: pd.DataFrame, model_path: str, training_config: dict
) -> tuple:
    """Train an AutoGluon predictor on the given data.

    Args:
        data: DataFrame with features and a "price" target column.
        model_path: Directory path where the model will be saved.
        training_config: Dictionary with training parameters:
            - preset: AutoGluon preset (default: "best_quality")
            - time_limit: Training time limit in seconds (default: 600)
            - eval_metric: Evaluation metric (default: "mean_absolute_error")
            - num_bag_folds: Number of bagging folds (default: 5)

    Returns:
        Tuple of (predictor, n_samples) where predictor is the trained
        TabularPredictor and n_samples is the number of training samples.
    """
    preset = training_config.get("preset", "best_quality")
    time_limit = training_config.get("time_limit", 600)
    eval_metric = training_config.get("eval_metric", "mean_absolute_error")
    num_bag_folds = training_config.get("num_bag_folds", 5)

    predictor = TabularPredictor(
        label="price",
        path=model_path,
        eval_metric=eval_metric,
    )
    predictor.fit(
        train_data=data,
        presets=preset,
        time_limit=time_limit,
        num_bag_folds=num_bag_folds,
    )
    clear_cache()  # Invalidate any cached models at this path
    return predictor, len(data)


def train_quantile_model(
    data: pd.DataFrame, model_path: str, training_config: dict
) -> TabularPredictor:
    """Train a quantile regression model for confidence intervals and tail detection.

    Trains an AutoGluon model with quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
    to provide prediction intervals.

    Args:
        data: DataFrame with features and a "price" target column.
        model_path: Directory path where the model will be saved.
        training_config: Dictionary with training parameters:
            - time_limit: Training time limit in seconds (default: 600)

    Returns:
        Trained TabularPredictor configured for quantile regression.
    """
    time_limit = training_config.get("time_limit", 600)

    predictor = TabularPredictor(
        label="price",
        path=model_path,
        problem_type="quantile",
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
    predictor.fit(
        train_data=data,
        presets="best_quality",
        time_limit=time_limit,
    )
    clear_cache()  # Invalidate any cached models at this path
    return predictor


def train_tail_models(data: pd.DataFrame, training_config: dict) -> dict:
    """Fit Ridge regression models on the lower and upper tails of the price distribution.

    These tail models are blended with the main AutoGluon predictions when
    the predicted value falls in extreme regions (below Q25 or above Q75).

    Args:
        data: DataFrame with features and a "price" target column.
        training_config: Training configuration (currently unused but kept
            for future extension).

    Returns:
        Dictionary containing:
            - q25: 25th percentile price boundary
            - q75: 75th percentile price boundary
            - feature_cols: List of numeric feature column names
            - low_model: Ridge model for low tail (or None if insufficient data)
            - low_n: Number of samples in low tail
            - high_model: Ridge model for high tail (or None if insufficient data)
            - high_n: Number of samples in high tail
    """
    q25 = data["price"].quantile(0.25)
    q75 = data["price"].quantile(0.75)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "price"]

    tail_info = {"q25": q25, "q75": q75, "feature_cols": feature_cols}

    # Low tail
    low_tail = data[data["price"] < q25]
    if len(low_tail) >= 10:
        X_low = low_tail[feature_cols].fillna(0)
        y_low = low_tail["price"]
        ridge_low = Ridge(alpha=1.0)
        ridge_low.fit(X_low, y_low)
        tail_info["low_model"] = ridge_low
        tail_info["low_n"] = len(low_tail)
    else:
        tail_info["low_model"] = None
        tail_info["low_n"] = len(low_tail)

    # High tail
    high_tail = data[data["price"] > q75]
    if len(high_tail) >= 10:
        X_high = high_tail[feature_cols].fillna(0)
        y_high = high_tail["price"]
        ridge_high = Ridge(alpha=1.0)
        ridge_high.fit(X_high, y_high)
        tail_info["high_model"] = ridge_high
        tail_info["high_n"] = len(high_tail)
    else:
        tail_info["high_model"] = None
        tail_info["high_n"] = len(high_tail)

    return tail_info


def save_tail_models(tail_info: dict, model_path: str) -> None:
    """Save tail model info to disk.

    Args:
        tail_info: Dictionary from train_tail_models().
        model_path: Path to the main model directory.
    """
    tail_path = os.path.join(model_path, "tail_info.pkl")
    with open(tail_path, "wb") as f:
        pickle.dump(tail_info, f)


def load_tail_models(model_path: str) -> dict:
    """Load tail model info from disk.

    Args:
        model_path: Path to the main model directory.

    Returns:
        Dictionary containing tail model info, or empty dict if not found.
    """
    tail_path = os.path.join(model_path, "tail_info.pkl")
    if os.path.exists(tail_path):
        with open(tail_path, "rb") as f:
            return pickle.load(f)
    return {}
