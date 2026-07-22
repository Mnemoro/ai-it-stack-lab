"""Prediction workflow for Seq2Seq LSTM energy forecasting models.

This module keeps inference orchestration intentionally thin. Callers are
responsible for preparing and scaling historical encoder features and future
known decoder features before invoking this layer. The predictor only builds the
single forecast sequence, calls a trained Keras-like model, restores predictions
to the target's original scale, and attaches the requested future timestamps.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd

from features.scaling import inverse_transform_target
from features.sequence_builder import build_prediction_sequence

ArrayLike: TypeAlias = Any
Scaler: TypeAlias = Any

__all__ = ["predict_single_target"]


class PredictiveSeq2SeqModel(Protocol):
    """Minimal Keras-like protocol required for inference."""

    def predict(self, x: Any, **kwargs: Any) -> ArrayLike:
        """Return scaled forecast values for encoder/decoder inputs."""
        ...


def _validate_positive_int(value: int, name: str) -> None:
    """Validate an integer inference-window parameter."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_prediction_inputs(
    *,
    model: PredictiveSeq2SeqModel,
    features_past: pd.DataFrame,
    features_future: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    lookback: int,
    future_window: int,
) -> None:
    """Validate the minimal prepared-input contract for prediction."""
    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError("model must expose a callable predict method.")
    if not isinstance(features_past, pd.DataFrame):
        raise TypeError("features_past must be a pandas DataFrame.")
    if not isinstance(features_future, pd.DataFrame):
        raise TypeError("features_future must be a pandas DataFrame.")
    if not isinstance(future_index, pd.DatetimeIndex):
        raise TypeError("future_index must be a pandas DatetimeIndex.")

    _validate_positive_int(lookback, "lookback")
    _validate_positive_int(future_window, "future_window")

    if len(features_past) < lookback:
        raise ValueError("features_past must contain at least lookback rows.")
    if len(features_future) < future_window:
        raise ValueError("features_future must contain at least future_window rows.")
    if len(future_index) < future_window:
        raise ValueError("future_index must contain at least future_window timestamps.")
    if features_past.shape[1] == 0:
        raise ValueError("features_past must contain at least one feature column.")
    if features_future.shape[1] == 0:
        raise ValueError("features_future must contain at least one feature column.")


def _normalize_prediction_output(predictions: ArrayLike, future_window: int) -> np.ndarray:
    """Return model predictions as a two-dimensional ``(horizon, 1)`` array."""
    array = np.asarray(predictions)

    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("prediction output must have a single batch item.")
        if array.shape[2] != 1:
            raise ValueError("prediction output must contain exactly one target per timestep.")
        array = array[0]
    elif array.ndim == 2:
        if array.shape == (1, future_window):
            array = array.reshape(future_window, 1)
        elif array.shape[1] != 1:
            raise ValueError("2D prediction output must have shape (future_window, 1).")
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    else:
        raise ValueError(f"prediction output must be 1D, 2D, or 3D, received ndim={array.ndim}.")

    if array.shape != (future_window, 1):
        raise ValueError(
            "prediction output horizon does not match future_window: "
            f"received shape {array.shape}, expected ({future_window}, 1)."
        )
    return array


def _prediction_series(
    values: np.ndarray,
    future_index: pd.DatetimeIndex,
    future_window: int,
    name: str | None,
) -> pd.Series:
    """Associate restored predictions with the requested future timestamps."""
    return pd.Series(
        values.reshape(future_window),
        index=future_index[:future_window],
        name=name,
    )


def predict_single_target(
    model: PredictiveSeq2SeqModel,
    *,
    features_past: pd.DataFrame,
    features_future: pd.DataFrame,
    target_scaler: Scaler,
    future_index: pd.DatetimeIndex,
    lookback: int,
    future_window: int,
    prediction_name: str | None = "prediction",
    predict_kwargs: dict[str, Any] | None = None,
) -> pd.Series:
    """Predict one energy target over a future horizon.

    The same generic workflow supports both consumption and photovoltaic
    production models. Inputs must already be feature-engineered and scaled by
    upstream layers. This function constructs the single Seq2Seq inference
    window via :func:`features.sequence_builder.build_prediction_sequence`, calls
    ``model.predict([encoder_input, decoder_input])``, inverse-transforms the
    scaled target forecast via :func:`features.scaling.inverse_transform_target`,
    and returns a timestamp-aligned pandas series.

    Args:
        model: Trained Keras-compatible Seq2Seq model with a ``predict`` method.
        features_past: Prepared and scaled historical encoder features.
        features_future: Prepared and scaled future-known decoder features.
        target_scaler: Fitted target scaler used to restore the target scale.
        future_index: Timestamps corresponding to the forecast horizon.
        lookback: Number of historical timesteps in the encoder input.
        future_window: Number of future timesteps to forecast.
        prediction_name: Optional name for the returned series.
        predict_kwargs: Optional keyword arguments forwarded to ``model.predict``.

    Returns:
        A pandas Series containing original-scale predictions indexed by the
        first ``future_window`` values of ``future_index``.
    """
    _validate_prediction_inputs(
        model=model,
        features_past=features_past,
        features_future=features_future,
        future_index=future_index,
        lookback=lookback,
        future_window=future_window,
    )

    encoder_input, decoder_input = build_prediction_sequence(
        features_past=features_past,
        features_future=features_future,
        lookback=lookback,
        future_window=future_window,
    )
    scaled_predictions = model.predict(
        [encoder_input, decoder_input],
        **(predict_kwargs or {}),
    )
    scaled_horizon = _normalize_prediction_output(scaled_predictions, future_window)
    restored_horizon = inverse_transform_target(
        scaled_horizon,
        target_scaler,
        expected_ndim=2,
    )

    return _prediction_series(
        values=np.asarray(restored_horizon),
        future_index=future_index,
        future_window=future_window,
        name=prediction_name,
    )
