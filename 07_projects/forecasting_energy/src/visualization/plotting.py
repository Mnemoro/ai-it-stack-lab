"""Plotting helpers for already-computed energy forecasting results."""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like as _is_color_like
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


def _validate_label(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_title(title: Any) -> str | None:
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    return title


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validate_figsize(figsize: Any) -> tuple[float, float]:
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise TypeError("figsize must contain exactly two numeric dimensions")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in figsize):
        raise TypeError("figsize dimensions must be numeric")
    size = tuple(float(v) for v in figsize)
    if not all(np.isfinite(size)) or not all(v > 0 for v in size):
        raise ValueError("figsize dimensions must be finite and positive")
    return size


def _figure(figsize: Any) -> tuple[Figure, Axes]:
    return plt.subplots(figsize=_validate_figsize(figsize))


def _pair_colors(colors: Any, count: int = 2) -> tuple[str, ...]:
    if colors is None:
        return _DEFAULT_PAIR_COLORS[:count]
    if isinstance(colors, (str, bytes)) or not isinstance(colors, Sequence):
        raise TypeError("colors must be a sequence of Matplotlib colors")
    if len(colors) != count:
        raise ValueError(f"colors must contain exactly {count} colors")
    for color in colors:
        if not _is_color_like(color):
            raise ValueError(f"invalid Matplotlib color: {color!r}")
    return tuple(colors)


def _numeric_1d(values: Any, name: str, *, flatten: bool = False, invalid_type: type[Exception] = TypeError) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise invalid_type(f"{name} must contain numeric values")
    if not isinstance(values, (Sequence, np.ndarray)):
        raise TypeError(f"{name} must be a numeric sequence or array")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise invalid_type(f"{name} must contain numeric values") from exc
    if flatten:
        raw = raw.reshape(-1)
    elif raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind not in "iuf":
        raise invalid_type(f"{name} must contain numeric values")
    arr = raw.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_matrix(values: Any, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain numeric values")
    arr = values.astype(float, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_windows, horizon) or (n_windows, horizon, 1)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_series(series: Any, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas.Series")
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if series.index.hasnans:
        raise ValueError(f"{name} index must not contain NaT")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be ordered increasingly")
    if series.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")
    try:
        values = series.to_numpy()
        if values.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain numeric values")
        numeric = values.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TypeError) and str(exc).endswith("must contain numeric values"):
            raise
        raise TypeError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return series


def _aligned_series(left: Any, right: Any, left_name: str, right_name: str) -> tuple[pd.Series, pd.Series]:
    left = _validate_series(left, left_name)
    right = _validate_series(right, right_name)
    if not left.index.equals(right.index):
        raise ValueError("series must have identical DatetimeIndex values")
    return left, right


def _common_config(title: Any, figsize: Any, show_grid: Any) -> tuple[str | None, tuple[float, float], bool]:
    return _validate_title(title), _validate_figsize(figsize), _validate_bool(show_grid, "show_grid")


def plot_training_history(history: Mapping[str, Any], *, title: str | None = None, figsize: tuple[float, float] = (8, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot pre-computed training and validation loss by epoch."""
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    if not isinstance(history, Mapping):
        raise TypeError("history must be a mapping")
    missing = {"loss", "val_loss"} - set(history)
    if missing:
        raise ValueError(f"history is missing required keys: {sorted(missing)}")
    loss = _numeric_1d(history["loss"], "loss", invalid_type=ValueError)
    val_loss = _numeric_1d(history["val_loss"], "val_loss", invalid_type=ValueError)
    if len(loss) != len(val_loss):
        raise ValueError("loss and val_loss must have the same length")
    train_color, val_color = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    epochs = np.arange(1, len(loss) + 1)
    ax.plot(epochs, loss, marker="o", label="Training loss", color=train_color)
    ax.plot(epochs, val_loss, marker="o", label="Validation loss", color=val_color)
    ax.set_title(title if title is not None else "Training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_prediction_window(y_true: np.ndarray, y_pred: np.ndarray, *, target_name: str, window_index: int = 0, timestamps: pd.DatetimeIndex | None = None, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (10, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot actual and predicted values for one pre-computed window."""
    target_name, unit = _validate_label(target_name, "target_name"), _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    if isinstance(window_index, bool) or not isinstance(window_index, (int, np.integer)):
        raise TypeError("window_index must be an integer")
    actual, predicted = _validate_matrix(y_true, "y_true"), _validate_matrix(y_pred, "y_pred")
    if y_true.shape != y_pred.shape or actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if window_index < 0 or window_index >= actual.shape[0]:
        raise ValueError("window_index is out of range")
    if timestamps is None:
        x_values = np.arange(actual.shape[1])
    else:
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError("timestamps must be a pandas.DatetimeIndex")
        if len(timestamps) != actual.shape[1]:
            raise ValueError("timestamps length must match the prediction horizon")
        x_values = timestamps
    true_color, pred_color = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, actual[window_index], marker="o", label=f"Actual {target_name}", color=true_color)
    ax.plot(x_values, predicted[window_index], marker="o", label=f"Predicted {target_name}", color=pred_color)
    ax.set_title(title if title is not None else f"{target_name} prediction window")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{target_name} ({unit})")
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_error_distribution(errors: Any, *, target_name: str, bins: int = 50, kde: bool = True, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (8, 5), color: str | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot a pre-computed error distribution without deriving errors."""
    target_name, unit = _validate_label(target_name, "target_name"), _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    kde = _validate_bool(kde, "kde")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise TypeError("bins must be an integer")
    if bins <= 0:
        raise ValueError("bins must be positive")
    arr = _numeric_1d(errors, "errors", flatten=isinstance(errors, np.ndarray))
    if color is not None:
        if not _is_color_like(color):
            raise ValueError("color must be a valid Matplotlib color")
    color = color if color is not None else "tab:blue"
    fig, ax = plt.subplots(figsize=figsize)
    use_kde = kde and np.ptp(arr) > 0
    sns.histplot(arr, bins=int(bins), kde=use_kde, color=color, ax=ax)
    ax.axvline(0, color="black", linestyle="--", label="Zero error")
    ax.set_title(title if title is not None else f"{target_name} error distribution")
    ax.set_xlabel(f"Error ({unit})")
    ax.set_ylabel("Count")
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_evaluation_metrics(metrics: Mapping[str, Any], *, target_name: str, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (6, 4), colors: Sequence[str] | None = None, show_grid: bool = False) -> tuple[Figure, Axes]:
    """Plot pre-computed MAE and RMSE values."""
    target_name, unit = _validate_label(target_name, "target_name"), _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    missing = {"mae", "rmse"} - set(metrics)
    if missing:
        raise ValueError(f"metrics is missing required keys: {sorted(missing)}")
    values = _numeric_1d([metrics["mae"], metrics["rmse"]], "metrics", invalid_type=ValueError)
    if np.any(values < 0):
        raise ValueError("metric values must be non-negative")
    bar_colors = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["MAE", "RMSE"], values, color=bar_colors)
    ax.set_title(title if title is not None else f"{target_name} evaluation metrics")
    ax.set_ylabel(unit)
    ax.grid(show_grid, axis="y")
    return fig, ax


def plot_future_forecast(consumption: pd.Series, production: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot aligned, pre-computed future consumption and production."""
    unit = _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    consumption, production = _aligned_series(consumption, production, "consumption", "production")
    c_color, p_color = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(consumption.index, consumption.to_numpy(), label="Consumption", color=c_color)
    ax.plot(production.index, production.to_numpy(), label="Production", color=p_color)
    ax.set_title(title if title is not None else "Future forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_grid_exchange(grid_import: pd.Series, grid_export: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot pre-computed, aligned grid import and export series."""
    unit = _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    grid_import, grid_export = _aligned_series(grid_import, grid_export, "grid_import", "grid_export")
    i_color, e_color = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid_import.index, grid_import.to_numpy(), label="Grid import", color=i_color)
    ax.plot(grid_export.index, grid_export.to_numpy(), label="Grid export", color=e_color)
    ax.set_title(title if title is not None else "Grid exchange")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax


def plot_cumulative_grid_exchange(cumulative_grid_import: pd.Series, cumulative_grid_export: pd.Series, *, unit: str = "kWh", title: str | None = None, figsize: tuple[float, float] = (14, 5), colors: Sequence[str] | None = None, show_grid: bool = True) -> tuple[Figure, Axes]:
    """Plot pre-computed cumulative import and export without summing them."""
    unit = _validate_label(unit, "unit")
    title, figsize, show_grid = _common_config(title, figsize, show_grid)
    cumulative_grid_import, cumulative_grid_export = _aligned_series(
        cumulative_grid_import, cumulative_grid_export, "cumulative_grid_import", "cumulative_grid_export"
    )
    i_color, e_color = _pair_colors(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumulative_grid_import.index, cumulative_grid_import.to_numpy(), label="Cumulative grid import", color=i_color)
    ax.plot(cumulative_grid_export.index, cumulative_grid_export.to_numpy(), label="Cumulative grid export", color=e_color)
    ax.set_title(title if title is not None else "Cumulative grid exchange")
    ax.set_xlabel("Time")
    ax.set_ylabel(unit)
    ax.grid(show_grid)
    ax.legend()
    return fig, ax
