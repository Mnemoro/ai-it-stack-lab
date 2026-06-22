"""Scaling utilities for the energy forecasting feature layer.

This module contains only the normalization logic extracted from the original
notebook. Scalers are always fitted on the training fold to avoid data leakage,
then reused to transform validation, test, historical, or future datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = [
    "fit_feature_scaler",
    "fit_target_scaler",
    "transform_features",
    "transform_target",
    "inverse_transform_target",
]


def _to_2d_array(data: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """Return input values as a two-dimensional NumPy array."""
    values = data.to_numpy() if isinstance(data, (pd.DataFrame, pd.Series)) else np.asarray(data)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"Sono attesi dati 1D o 2D, ricevuto array con ndim={values.ndim}.")
    return values


def _validate_no_missing(values: np.ndarray, data_name: str) -> None:
    """Raise an error when values contain missing numeric entries."""
    if np.isnan(values).any():
        raise ValueError(f"{data_name} contiene NaN: impossibile applicare lo scaling.")


def _as_scaled_series(values: np.ndarray, template: pd.Series) -> pd.Series:
    """Build a scaled series preserving the input index and name."""
    return pd.Series(values.reshape(-1), index=template.index, name=template.name)


def _as_scaled_dataframe(values: np.ndarray, template: pd.DataFrame) -> pd.DataFrame:
    """Build a scaled dataframe preserving the input index and columns."""
    return pd.DataFrame(values, index=template.index, columns=template.columns)


def fit_feature_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit the feature scaler on training features only.

    The original notebook used ``StandardScaler`` for encoder features and
    fitted it exclusively on ``df_features_past_*.loc[df_train.index]``. This
    function preserves that behavior and returns the fitted scaler for later
    transformations.

    Args:
        train_features: Training-fold feature dataframe. Columns must match the
            dataframes later passed to :func:`transform_features`.

    Returns:
        A fitted ``StandardScaler`` instance.

    Raises:
        TypeError: If ``train_features`` is not a dataframe.
        ValueError: If the dataframe is empty or contains missing values.
    """
    if not isinstance(train_features, pd.DataFrame):
        raise TypeError("train_features deve essere un pandas DataFrame.")
    if train_features.empty:
        raise ValueError("train_features non può essere vuoto.")

    values = _to_2d_array(train_features)
    _validate_no_missing(values, "train_features")
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def fit_target_scaler(train_target: pd.Series | pd.DataFrame) -> StandardScaler:
    """Fit the target scaler on the original-scale training target only.

    The notebook used a separate ``StandardScaler`` for each target
    (consumption and production), fitted only on the corresponding training
    target reshaped to one column.

    Args:
        train_target: Training-fold target series or one-column dataframe.

    Returns:
        A fitted ``StandardScaler`` instance.

    Raises:
        TypeError: If the input is not a pandas series or dataframe.
        ValueError: If the target is empty, has more than one column, or
            contains missing values.
    """
    if not isinstance(train_target, (pd.Series, pd.DataFrame)):
        raise TypeError("train_target deve essere una Series o un DataFrame pandas.")
    if train_target.empty:
        raise ValueError("train_target non può essere vuoto.")
    if isinstance(train_target, pd.DataFrame) and train_target.shape[1] != 1:
        raise ValueError("train_target DataFrame deve contenere una sola colonna.")

    values = _to_2d_array(train_target)
    _validate_no_missing(values, "train_target")
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def transform_features(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Transform feature data with a previously fitted feature scaler.

    Args:
        features: Feature dataframe to scale. Its columns must have the same
            order and meaning used when fitting ``scaler``.
        scaler: Fitted ``StandardScaler`` returned by
            :func:`fit_feature_scaler`.

    Returns:
        A scaled dataframe with the same index and columns as ``features``.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features deve essere un pandas DataFrame.")

    values = _to_2d_array(features)
    _validate_no_missing(values, "features")
    return _as_scaled_dataframe(scaler.transform(values), features)


def transform_target(
    target: pd.Series | pd.DataFrame,
    scaler: StandardScaler,
) -> pd.Series | pd.DataFrame:
    """Transform target values preserving pandas metadata.

    Args:
        target: Target series or one-column dataframe to scale.
        scaler: Fitted target ``StandardScaler`` returned by
            :func:`fit_target_scaler`.

    Returns:
        A scaled target with the same pandas type, index, and name or columns as
        ``target``.
    """
    if not isinstance(target, (pd.Series, pd.DataFrame)):
        raise TypeError("target deve essere una Series o un DataFrame pandas.")
    if isinstance(target, pd.DataFrame) and target.shape[1] != 1:
        raise ValueError("target DataFrame deve contenere una sola colonna.")

    values = _to_2d_array(target)
    _validate_no_missing(values, "target")
    scaled = scaler.transform(values)
    if isinstance(target, pd.Series):
        return _as_scaled_series(scaled, target)
    return _as_scaled_dataframe(scaled, target)


def inverse_transform_target(
    values: pd.Series | pd.DataFrame | np.ndarray,
    scaler: StandardScaler,
    expected_ndim: int | None = None,
) -> pd.Series | pd.DataFrame | np.ndarray:
    """Restore scaled target values to the original physical scale.

    This is the reusable equivalent of the notebook's ``inverse_transform_array``:
    it reshapes any 1D, 2D, or 3D prediction array to one column for
    ``StandardScaler.inverse_transform`` and then restores the original shape.
    Pandas series and one-column dataframes keep their metadata.

    Args:
        values: Scaled target values or predictions.
        scaler: Fitted target ``StandardScaler``.
        expected_ndim: Optional dimensionality check used by callers that want
            to validate prediction shapes before inverse transformation.

    Returns:
        Values converted back to the original target scale.

    Raises:
        ValueError: If ``expected_ndim`` is provided and does not match the
            actual number of dimensions.
    """
    array = values.to_numpy() if isinstance(values, (pd.Series, pd.DataFrame)) else np.asarray(values)
    original_shape = array.shape
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"Array con ndim={array.ndim}, atteso ndim={expected_ndim}.")

    restored = scaler.inverse_transform(array.reshape(-1, 1)).reshape(original_shape)
    if isinstance(values, pd.Series):
        return _as_scaled_series(restored, values)
    if isinstance(values, pd.DataFrame):
        return _as_scaled_dataframe(restored, values)
    return restored
