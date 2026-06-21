"""Preprocessing utilities for the energy forecasting feature layer.

This module extracts the notebook preprocessing steps that prepare historical
weather, consumption, and production data before scaling and sequence creation.
It intentionally stops at dataframe preparation, target extraction, temporal
feature exposure, and chronological train/validation/test splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from data.validator import clean_time_series_dataframe, validate_datetime_index

TEST_RATIO = 0.15
VAL_RATIO = 0.15
TRAIN_RATIO = 1 - TEST_RATIO - VAL_RATIO
PAST_WINDOW = 168
WEATHER_COLUMNS = ["temperature", "irradiance", "wind_speed"]
CONSUMPTION_COLUMN = "consumption"
PRODUCTION_COLUMN = "production"
TARGET_COLUMNS = [CONSUMPTION_COLUMN, PRODUCTION_COLUMN]


__all__ = [
    "PreparedTrainingDataset",
    "build_historical_dataframe",
    "prepare_dataframe",
    "build_time_features",
    "build_training_dataset",
    "split_train_validation_test",
]


@dataclass(frozen=True)
class PreparedTrainingDataset:
    """Container for the dataframe objects needed by later training modules.

    Attributes:
        historical: Complete historical dataframe with weather, consumption,
            and production columns.
        train: Chronological training fold.
        validation: Chronological validation fold, including the notebook's
            past-window buffer before the validation boundary.
        test: Chronological test fold, including the notebook's past-window
            buffer before the test boundary.
        time_features: Timestamp-derived feature dataframe for the historical
            index.
        consumption_features: Encoder dataframe for the consumption model,
            made of weather columns plus the historical consumption target.
        production_features: Encoder dataframe for the production model,
            made of weather columns plus the historical production target.
        consumption_target: Historical consumption target series.
        production_target: Historical production target series.
    """

    historical: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    time_features: pd.DataFrame
    consumption_features: pd.DataFrame
    production_features: pd.DataFrame
    consumption_target: pd.Series
    production_target: pd.Series


def _as_dataframe(series: pd.Series, column_name: str) -> pd.DataFrame:
    """Return ``series`` as a one-column dataframe with ``column_name``."""
    dataframe = series.to_frame(name=column_name)
    validate_datetime_index(dataframe, dataframe_name=f"{column_name} dataframe")
    return dataframe


def _clean_notebook_style(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a dataframe using the notebook's time interpolation behavior.

    The original notebook inserted random missing values before interpolation
    for synthetic experimentation. Data cleaning itself was temporal
    interpolation followed by forward-fill and backward-fill; that deterministic
    cleaning behavior is delegated to the Data Layer validator.
    """
    return clean_time_series_dataframe(
        df,
        numeric_columns=list(df.columns),
        dataframe_name="preprocessing dataframe",
        sort_index=True,
        impute=True,
    )


def build_historical_dataframe(
    weather_df: pd.DataFrame,
    consumption: pd.Series,
    production: pd.Series,
) -> pd.DataFrame:
    """Build the historical dataframe used in the notebook.

    Args:
        weather_df: Weather dataframe indexed by timestamp and containing
            ``temperature``, ``irradiance``, and ``wind_speed``.
        consumption: Historical consumption series indexed by timestamp.
        production: Historical production series indexed by timestamp.

    Returns:
        A cleaned dataframe equivalent to the notebook's ``df_hist``:
        weather columns, ``consumption``, and ``production`` concatenated on the
        same ``DatetimeIndex``.

    Raises:
        TypeError: If any input is not indexed by a ``DatetimeIndex``.
        ValueError: If required columns are missing, indexes are not aligned, or
            numeric values cannot be imputed.
    """
    weather = clean_time_series_dataframe(
        weather_df,
        required_columns=WEATHER_COLUMNS,
        numeric_columns=WEATHER_COLUMNS,
        dataframe_name="weather dataframe",
        sort_index=True,
        impute=True,
    )
    consumption_df = _clean_notebook_style(_as_dataframe(consumption, CONSUMPTION_COLUMN))
    production_df = _clean_notebook_style(_as_dataframe(production, PRODUCTION_COLUMN))

    historical = pd.concat([weather[WEATHER_COLUMNS], consumption_df, production_df], axis=1)
    return clean_time_series_dataframe(
        historical,
        required_columns=[*WEATHER_COLUMNS, *TARGET_COLUMNS],
        numeric_columns=[*WEATHER_COLUMNS, *TARGET_COLUMNS],
        dataframe_name="historical dataframe",
        sort_index=True,
        impute=True,
    )


def prepare_dataframe(
    weather_df: pd.DataFrame,
    consumption: pd.Series,
    production: pd.Series,
) -> pd.DataFrame:
    """Prepare the historical dataframe by validating and concatenating inputs.

    This compatibility wrapper keeps the public API clear while delegating the
    historical dataframe construction to ``build_historical_dataframe``.

    Args:
        weather_df: Weather dataframe indexed by timestamp and containing
            ``temperature``, ``irradiance``, and ``wind_speed``.
        consumption: Historical consumption series indexed by timestamp.
        production: Historical production series indexed by timestamp.

    Returns:
        A cleaned dataframe equivalent to the notebook's ``df_hist``.
    """
    return build_historical_dataframe(weather_df, consumption, production)


def build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Expose timestamp-derived fields without changing notebook formulas.

    Args:
        index: Timestamp index used by historical or future hourly datasets.

    Returns:
        A dataframe indexed by ``index`` with the temporal fields used by the
        notebook formulas: ``hour``, ``dayofyear``, and ``month``.

    Raises:
        ValueError: If ``index`` is not a pandas ``DatetimeIndex``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("L'indice deve essere un DatetimeIndex.")

    return pd.DataFrame(
        {
            "hour": index.hour,
            "dayofyear": index.dayofyear,
            "month": index.month,
        },
        index=index,
    )


def split_train_validation_test(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    past_window: int = PAST_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series dataframe exactly as in the notebook.

    Validation and test folds include a backward buffer of up to ``past_window``
    rows so later sequence-building code has enough encoder history.
    """
    validate_datetime_index(df, dataframe_name="split dataframe")
    n_rows = len(df)
    idx_train_end = int(n_rows * train_ratio)
    idx_val_end = int(n_rows * (train_ratio + val_ratio))
    train_buffer = min(past_window, idx_train_end)

    df_train = df.iloc[:idx_train_end]
    val_start_with_buffer = max(0, idx_train_end - train_buffer)
    df_val = df.iloc[val_start_with_buffer:idx_val_end]
    test_start_with_buffer = max(0, idx_val_end - train_buffer)
    df_test = df.iloc[test_start_with_buffer:]
    return df_train, df_val, df_test


def build_training_dataset(
    weather_df: pd.DataFrame,
    consumption: pd.Series,
    production: pd.Series,
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    past_window: int = PAST_WINDOW,
) -> PreparedTrainingDataset:
    """Prepare historical dataframes and targets for later training modules.

    The function only orchestrates preprocessing for already available inputs:
    it builds the historical dataframe, constructs timestamp-derived features,
    performs the chronological split, and returns original-scale feature and
    target dataframes for subsequent modules.

    Args:
        weather_df: Weather dataframe indexed by timestamp.
        consumption: Historical consumption series indexed by timestamp.
        production: Historical production series indexed by timestamp.
        train_ratio: Chronological training ratio from the notebook.
        val_ratio: Chronological validation ratio from the notebook.
        past_window: Historical buffer size used by validation and test folds.

    Returns:
        A ``PreparedTrainingDataset`` containing historical folds, feature
        dataframes, and original-scale target series.
    """
    historical = prepare_dataframe(weather_df, consumption, production)
    time_features = build_time_features(historical.index)
    train, validation, test = split_train_validation_test(
        historical,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        past_window=past_window,
    )
    consumption_features = pd.concat(
        [historical[WEATHER_COLUMNS], historical[[CONSUMPTION_COLUMN]]],
        axis=1,
    )
    production_features = pd.concat(
        [historical[WEATHER_COLUMNS], historical[[PRODUCTION_COLUMN]]],
        axis=1,
    )

    return PreparedTrainingDataset(
        historical=historical,
        train=train,
        validation=validation,
        test=test,
        time_features=time_features,
        consumption_features=consumption_features,
        production_features=production_features,
        consumption_target=historical[CONSUMPTION_COLUMN],
        production_target=historical[PRODUCTION_COLUMN],
    )
