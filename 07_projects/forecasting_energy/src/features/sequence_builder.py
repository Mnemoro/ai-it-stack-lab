"""Utilities for building Seq2Seq time-series windows.

This module contains only the sequence construction logic extracted from the
original energy forecasting notebook. It does not load data, create features,
scale values, validate input dataframes, or train models.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

Array3D: TypeAlias = np.ndarray
SeriesOrFrame: TypeAlias = pd.Series | pd.DataFrame

__all__ = [
    "build_sequences",
    "build_training_sequences",
    "build_prediction_sequence",
]


def _to_numpy(values: SeriesOrFrame) -> np.ndarray:
    """Return the underlying NumPy values preserving pandas row order."""
    return values.values


def build_sequences(
    df_features_past: pd.DataFrame,
    df_targets_past: SeriesOrFrame,
    df_features_future: pd.DataFrame,
    df_targets_future: SeriesOrFrame,
    lookback: int,
    future_window: int,
) -> tuple[Array3D, Array3D, Array3D]:
    """Build sliding windows for a Seq2Seq forecasting model.

    The implementation mirrors the notebook function ``crea_sequence_dataset``:
    for every chronological position, it creates an encoder window from past
    features, a decoder window from future-known features, and a target window
    from future targets.

    Parameters
    ----------
    df_features_past:
        Chronologically ordered feature dataframe used by the encoder.
    df_targets_past:
        Past targets kept for API parity with the original notebook. The
        notebook did not use this argument because past targets were already
        expected to be included in ``df_features_past`` when needed.
    df_features_future:
        Chronologically ordered dataframe of future-known features used by the
        decoder.
    df_targets_future:
        Chronologically ordered target series or dataframe to forecast.
    lookback:
        Number of past time steps included in each encoder window. This is the
        notebook's ``past_window`` parameter.
    future_window:
        Number of future time steps included in each decoder and target window.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(X_enc, X_dec, y)`` where ``X_enc`` contains encoder windows,
        ``X_dec`` contains decoder windows, and ``y`` contains target windows.
        The chronological order of generated windows is preserved.
    """
    del df_targets_past  # Preserved for behavioral/API parity with notebook.

    x_enc_list: list[np.ndarray] = []
    x_dec_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    total_len = len(df_features_past)
    for i in range(total_len - lookback - future_window + 1):
        enc_slice = _to_numpy(df_features_past.iloc[i : i + lookback])
        dec_features = _to_numpy(
            df_features_future.iloc[i + lookback : i + lookback + future_window]
        )
        y_slice = _to_numpy(
            df_targets_future.iloc[i + lookback : i + lookback + future_window]
        )

        x_enc_list.append(enc_slice)
        x_dec_list.append(dec_features)
        y_list.append(y_slice)

    x_enc = np.array(x_enc_list)
    x_dec = np.array(x_dec_list)
    y = np.array(y_list)

    return x_enc, x_dec, y


def build_training_sequences(
    features_past: pd.DataFrame,
    targets_past: SeriesOrFrame,
    features_future: pd.DataFrame,
    targets_future: SeriesOrFrame,
    lookback: int,
    future_window: int,
) -> tuple[Array3D, Array3D, Array3D]:
    """Prepare training, validation, or test sequences.

    This is a semantic wrapper around :func:`build_sequences` for fold-specific
    datasets already prepared by the caller. Scaling, feature engineering, and
    chronological train/validation/test splitting remain external to this
    module.

    Parameters
    ----------
    features_past:
        Fold-specific encoder features in chronological order.
    targets_past:
        Fold-specific historical targets, retained for parity with the original
        notebook sequence builder.
    features_future:
        Fold-specific decoder features in chronological order.
    targets_future:
        Fold-specific future targets in chronological order.
    lookback:
        Number of past time steps in each encoder sequence.
    future_window:
        Number of future time steps in each decoder and target sequence.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Encoder inputs, decoder inputs, and target sequences.
    """
    return build_sequences(
        df_features_past=features_past,
        df_targets_past=targets_past,
        df_features_future=features_future,
        df_targets_future=targets_future,
        lookback=lookback,
        future_window=future_window,
    )


def build_prediction_sequence(
    features_past: pd.DataFrame,
    features_future: pd.DataFrame,
    lookback: int,
    future_window: int,
) -> tuple[Array3D, Array3D]:
    """Build the single encoder/decoder sequence used for prediction.

    The logic mirrors the notebook forecast step: take the last ``lookback``
    rows from the already prepared past features for the encoder, take the
    first ``future_window`` rows from the already prepared future-known decoder
    features, and reshape both arrays to include a batch dimension of one.

    Parameters
    ----------
    features_past:
        Chronologically ordered, already prepared encoder features.
    features_future:
        Chronologically ordered, already prepared decoder features for the
        forecast horizon.
    lookback:
        Number of past time steps used by the encoder.
    future_window:
        Number of future time steps used by the decoder.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(X_enc_future, X_dec_future)`` with shapes
        ``(1, lookback, n_encoder_features)`` and
        ``(1, future_window, n_decoder_features)``.
    """
    n_features_enc = features_past.shape[1]
    n_features_dec = features_future.shape[1]

    x_enc_future = features_past.iloc[-lookback:].values.reshape(
        1,
        lookback,
        n_features_enc,
    )
    x_dec_future = features_future.iloc[:future_window].values.reshape(
        1,
        future_window,
        n_features_dec,
    )

    return x_enc_future, x_dec_future
