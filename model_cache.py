"""Centralized model caching for AutoGluon predictors.

This module provides an LRU cache for AutoGluon TabularPredictor models
to avoid repeated disk loads during valuation sessions. Models are cached
in memory with a configurable maximum size.
"""

import os
from collections import OrderedDict

from autogluon.tabular import TabularPredictor

_model_cache: OrderedDict[str, TabularPredictor] = OrderedDict()
_max_cache_size: int = 5  # ~500MB-1.5GB memory depending on model complexity


def get_predictor(path: str) -> TabularPredictor:
    """Load a TabularPredictor, using cache if available.

    Args:
        path: Path to the AutoGluon model directory.

    Returns:
        Loaded TabularPredictor instance.
    """
    path = os.path.normpath(path)

    if path in _model_cache:
        _model_cache.move_to_end(path)
        return _model_cache[path]

    predictor = TabularPredictor.load(path)
    _model_cache[path] = predictor
    _model_cache.move_to_end(path)

    while len(_model_cache) > _max_cache_size:
        _model_cache.popitem(last=False)

    return predictor


def clear_cache() -> None:
    """Clear all cached models.

    Call this after training to invalidate stale models.
    """
    _model_cache.clear()
