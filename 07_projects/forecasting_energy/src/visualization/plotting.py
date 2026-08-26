"""Visualization layer for the energy forecasting project.

This module contains only plotting helpers. All inputs are expected to be
pre-computed by upstream layers and already expressed in their physical scale.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_training_history",
    "plot_prediction_window",
    "plot_error_distribution",
    "plot_evaluation_metrics",
    "plot_future_forecast",
    "plot_grid_exchange",
    "plot_cumulative_grid_exchange",
]

_DEFAULT_PAIR_COLORS = ("tab:blue", "tab:orange")


def _validate_label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _numeric_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return arr


def _pair_colors(colors: Sequence[str] | None) -> tuple[str, str]:
    if colors is None:
        return _DEFAULT_PAIR_COLORS
    if isinstance(colors, str) or len(colors) != 2:
        raise ValueError("colors must contain exactly two colors")
    return colors[0], colors[1]


def _validated_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _validate_matrix(values: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    arr = values.astype(float, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_windows, horizon) or (n_windows, horizon, 1)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return arr


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas.Series")
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    values = pd.to_numeric(series, errors="raise")
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"{name} must contain only finite numeric values")
    return values


def _validate_aligned_series(left: pd.Series, right: pd.Series, left_name: str, right_name: str) -> tuple[pd.Series, pd.Series]:
    left = _validate_series(left, left_name)
    right = _validate_series(right, right_name)
    if len(left) != len(right):
        raise ValueError("series must have the same length")
    if not left.index.equals(right.index):
        raise ValueError("series must have the same aligned DatetimeIndex")
    return left, right


def _validate_ordered_unique_index(series: pd.Series, name: str) -> None:
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be ordered")
    if series.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")


def plot_training_history(history: Mapping[str, Any], *, figsize: tuple[float, float] = (8, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot training and validation loss by epoch."""
    if not isinstance(history, Mapping):
        raise TypeError("history must be a mapping")
    missing = {"loss", "val_loss"} - set(history)
    if missing:
        raise ValueError(f"history is missing required keys: {sorted(missing)}")
    loss = _numeric_1d(history["loss"], "loss")
    val_loss = _numeric_1d(history["val_loss"], "val_loss")
    if len(loss) != len(val_loss):
        raise ValueError("loss and val_loss must have the same length")
    train_color, val_color = _pair_colors(colors)
    epochs = np.arange(1, len(loss) + 1)
    fig, ax = _validated_figure(figsize)
    ax.plot(epochs, loss, marker="o", label="Training loss", color=train_color)
    ax.plot(epochs, val_loss, marker="o", label="Validation loss", color=val_color)
    ax.set_title("Training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_prediction_window(y_true: np.ndarray, y_pred: np.ndarray, *, target_name: str, window_index: int = 0, timestamps: Sequence[Any] | None = None, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (10, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot real and predicted values for one prediction window."""
    target_name = _validate_label(target_name, "target_name")
    unit = _validate_label(unit, "unit")
    actual = _validate_matrix(y_true, "y_true")
    predicted = _validate_matrix(y_pred, "y_pred")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not isinstance(window_index, int) or window_index < 0 or window_index >= actual.shape[0]:
        raise IndexError("window_index is out of range")
    x_values = np.arange(actual.shape[1]) if timestamps is None else list(timestamps)
    if len(x_values) != actual.shape[1]:
        raise ValueError("timestamps length must match the prediction horizon")
    true_color, pred_color = _pair_colors(colors)
    fig, ax = _validated_figure(figsize)
    ax.plot(x_values, actual[window_index], marker="o", label="Actual", color=true_color)
    ax.plot(x_values, predicted[window_index], marker="o", label="Predicted", color=pred_color)
    ax.set_title(title or f"{target_name} prediction window")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{target_name} ({unit})")
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_error_distribution(errors: Any, *, target_name: str, bins: int = 50, kde: bool = True, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (8, 5), color: str | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot the distribution of pre-computed residuals/errors."""
    target_name = _validate_label(target_name, "target_name")
    unit = _validate_label(unit, "unit")
    arr = _numeric_1d(errors, "errors")
    fig, ax = _validated_figure(figsize)
    sns.histplot(arr, bins=bins, kde=kde, color=color or "tab:blue", ax=ax)
    ax.set_title(title or f"{target_name} error distribution")
    ax.set_xlabel(f"Error ({unit})")
    ax.set_ylabel("Count")
    ax.grid(show_grid)
    return fig, ax


def plot_evaluation_metrics(metrics: Mapping[str, Any], *, target_name: str, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (6, 4), colors: Sequence[str] | None = None, show_grid: bool = False) -> tuple[Figure, Axes]:
    """Plot pre-computed MAE and RMSE values."""
    target_name = _validate_label(target_name, "target_name")
    unit = _validate_label(unit, "unit")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    missing = {"mae", "rmse"} - set(metrics)
    if missing:
        raise ValueError(f"metrics is missing required keys: {sorted(missing)}")
    values = _numeric_1d([metrics["mae"], metrics["rmse"]], "metrics")
    if np.any(values < 0):
        raise ValueError("metric values must be non-negative")
    bar_colors = _pair_colors(colors)
    fig, ax = _validated_figure(figsize)
    ax.bar(["MAE", "RMSE"], values, color=bar_colors)
    ax.set_title(title or f"{target_name} evaluation metrics")
    ax.set_ylabel(unit)
    ax.grid(show_grid, axis="y")
    return fig, ax


def plot_future_forecast(consumption: pd.Series, production: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot future consumption and production forecasts."""
    unit = _validate_label(unit, "unit")
    consumption, production = _validate_aligned_series(consumption, production, "consumption", "production")
    c_color, p_color = _pair_colors(colors)
    fig, ax = _validated_figure(figsize)
    ax.plot(consumption.index, consumption.values, label="Consumption", color=c_color)
    ax.plot(production.index, production.values, label="Production", color=p_color)
    ax.set_title(title or "Future forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_grid_exchange(grid_import: pd.Series, grid_export: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot pre-computed grid import and grid export series."""
    unit = _validate_label(unit, "unit")
    grid_import, grid_export = _validate_aligned_series(grid_import, grid_export, "grid_import", "grid_export")
    i_color, e_color = _pair_colors(colors)
    fig, ax = _validated_figure(figsize)
    ax.plot(grid_import.index, grid_import.values, label="Grid import", color=i_color)
    ax.plot(grid_export.index, grid_export.values, label="Grid export", color=e_color)
    ax.set_title(title or "Grid exchange")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_cumulative_grid_exchange(cumulative_grid_import: pd.Series, cumulative_grid_export: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot pre-computed cumulative grid import and export series."""
    unit = _validate_label(unit, "unit")
    cumulative_grid_import, cumulative_grid_export = _validate_aligned_series(cumulative_grid_import, cumulative_grid_export, "cumulative_grid_import", "cumulative_grid_export")
    _validate_ordered_unique_index(cumulative_grid_import, "cumulative_grid_import")
    _validate_ordered_unique_index(cumulative_grid_export, "cumulative_grid_export")
    i_color, e_color = _pair_colors(colors)
    fig, ax = _validated_figure(figsize)
    ax.plot(cumulative_grid_import.index, cumulative_grid_import.values, label="Cumulative grid import", color=i_color)
    ax.plot(cumulative_grid_export.index, cumulative_grid_export.values, label="Cumulative grid export", color=e_color)
    ax.set_title(title or "Cumulative grid exchange")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax
