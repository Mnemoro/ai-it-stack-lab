"""LSTM Seq2Seq model factory for the Forecasting Layer.

This module contains only the construction and compilation of the neural
forecasting architecture extracted from the original notebook. It intentionally
keeps data preparation, scaling, sequence construction, training, prediction,
evaluation, persistence, and visualization outside of this layer.
"""

from __future__ import annotations

from typing import Any

from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed  # type: ignore[import-untyped]
from tensorflow.keras.models import Model  # type: ignore[import-untyped]

__all__ = ["build_seq2seq_lstm_model"]


def _validate_positive_int(value: int, name: str) -> None:
    """Validate an integer architecture parameter that must be strictly positive."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_dropout_rate(dropout_rate: float) -> None:
    """Validate a Keras dropout rate."""
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate must be a float between 0 and 1.")
    if not 0 <= float(dropout_rate) < 1:
        raise ValueError("dropout_rate must be greater than or equal to 0 and less than 1.")


def build_seq2seq_lstm_model(
    n_features_enc: int,
    n_features_dec: int,
    *,
    latent_dim: int = 64,
    future_window: int | None = None,
    dropout_rate: float = 0.3,
    output_dim: int = 1,
    optimizer: str | Any = "adam",
    loss: str | Any = "mse",
    metrics: list[str | Any] | tuple[str | Any, ...] | None = None,
) -> Model:
    """Build and compile the notebook-compatible Seq2Seq LSTM model.

    The returned model preserves the original encoder/decoder architecture used
    for both consumption and photovoltaic-production forecasting in the source
    notebook:

    * encoder input with shape ``(None, n_features_enc)``;
    * encoder ``LSTM(latent_dim, return_state=True)``;
    * dropout applied independently to the encoder hidden and cell states;
    * decoder input with shape ``(None, n_features_dec)`` by default, or
      ``(future_window, n_features_dec)`` only when callers explicitly request
      a fixed decoder horizon;
    * decoder ``LSTM(latent_dim, return_sequences=True, return_state=True)``
      initialized from the regularized encoder states;
    * dropout applied to the decoder sequence output;
    * time-distributed dense projection to ``output_dim`` target values per
      forecast step;
    * compilation with Adam and MSE by default.

    Parameters
    ----------
    n_features_enc:
        Number of features in each encoder timestep.
    n_features_dec:
        Number of future-known decoder features in each decoder timestep.
    latent_dim:
        Number of LSTM units in both encoder and decoder layers. Defaults to
        ``64``, matching the notebook.
    future_window:
        Optional forecast horizon used to fix the decoder time dimension. The
        default ``None`` preserves the original notebook's variable-length
        Keras ``Input(shape=(None, n_features_dec))``. Pass a positive integer
        only when a caller needs an explicit decoder horizon.
    dropout_rate:
        Dropout rate applied to encoder states and decoder outputs. Defaults to
        ``0.3``, matching the notebook.
    output_dim:
        Number of target values emitted per timestep. Defaults to ``1`` for the
        single-target consumption/production models used by the notebook.
    optimizer:
        Optimizer passed to ``Model.compile``. Defaults to ``"adam"``.
    loss:
        Loss passed to ``Model.compile``. Defaults to ``"mse"``.
    metrics:
        Optional metrics passed to ``Model.compile``. The notebook did not
        compile metrics, so the default is ``None``.

    Returns
    -------
    tensorflow.keras.models.Model
        A compiled Keras functional model with two inputs
        ``[encoder_inputs, decoder_inputs]`` and one sequence output.
    """
    _validate_positive_int(n_features_enc, "n_features_enc")
    _validate_positive_int(n_features_dec, "n_features_dec")
    _validate_positive_int(latent_dim, "latent_dim")
    _validate_positive_int(output_dim, "output_dim")
    _validate_dropout_rate(dropout_rate)
    if future_window is not None:
        _validate_positive_int(future_window, "future_window")

    encoder_inputs = Input(shape=(None, n_features_enc), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)

    state_h = Dropout(float(dropout_rate), name="dropout_state_h")(state_h)
    state_c = Dropout(float(dropout_rate), name="dropout_state_c")(state_c)
    encoder_states = [state_h, state_c]

    decoder_timesteps = future_window if future_window is not None else None
    decoder_inputs = Input(
        shape=(decoder_timesteps, n_features_dec),
        name="decoder_inputs",
    )
    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = Dropout(float(dropout_rate), name="dropout_decoder_output")(
        decoder_outputs
    )
    decoder_dense = TimeDistributed(Dense(output_dim), name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    compile_kwargs: dict[str, Any] = {"optimizer": optimizer, "loss": loss}
    if metrics is not None:
        compile_kwargs["metrics"] = list(metrics)
    model.compile(**compile_kwargs)
    return model
