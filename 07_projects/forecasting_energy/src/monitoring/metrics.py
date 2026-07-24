"""Quantitative forecast error metrics for energy forecasting monitoring.

This module provides small, reusable metric functions for comparing observed
values with predicted values. Metrics operate directly on the numeric values
provided by the caller: no scaling, inverse scaling, feature construction, data
loading, plotting, or model-specific logic is performed here.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

ArrayLike: TypeAlias = Any

__all__ = ["mean_absolute_error", "root_mean_squared_error"]


def _as_numeric_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert array-like metric input to a numeric NumPy array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")

    return array


def _validate_metric_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return validated metric inputs as numeric arrays with identical shape."""
    true_values = _as_numeric_array(y_true, "y_true")
    predicted_values = _as_numeric_array(y_pred, "y_pred")

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape: "
            f"received {true_values.shape} and {predicted_values.shape}."
        )

    return true_values, predicted_values


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean absolute error between observed and predicted values.

    The metric is computed as the arithmetic mean of ``abs(y_true - y_pred)``.
    Inputs may be one-dimensional or multidimensional numeric array-like
    objects, provided both inputs have exactly the same shape. Values are used as
    received; callers are responsible for passing values in the desired scale.

    Args:
        y_true: Observed numeric values.
        y_pred: Predicted numeric values with the same shape as ``y_true``.

    Returns:
        The mean absolute error as a Python ``float``.

    Raises:
        TypeError: If either input cannot be interpreted as numeric values.
        ValueError: If either input is empty, contains non-finite values, or the
            two inputs do not have identical shapes.
    """
    true_values, predicted_values = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean(np.abs(true_values - predicted_values)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the root mean squared error between observed and predicted values.

    The metric is computed as the square root of the arithmetic mean of
    ``(y_true - y_pred) ** 2``. Inputs may be one-dimensional or
    multidimensional numeric array-like objects, provided both inputs have
    exactly the same shape. Values are used as received; callers are responsible
    for passing values in the desired scale.

    Args:
        y_true: Observed numeric values.
        y_pred: Predicted numeric values with the same shape as ``y_true``.

    Returns:
        The root mean squared error as a Python ``float``.

    Raises:
        TypeError: If either input cannot be interpreted as numeric values.
        ValueError: If either input is empty, contains non-finite values, or the
            two inputs do not have identical shapes.
    """
    true_values, predicted_values = _validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true_values - predicted_values))))
