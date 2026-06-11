"""Validation and cleaning helpers for energy forecasting time-series data.

The functions in this module centralize the dataframe checks that were
previously embedded in the exploratory notebook.  They intentionally depend
only on pandas so that the data layer remains independent from modeling and
visualization libraries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

ColumnName: TypeAlias = str


__all__ = [
    "validate_datetime_index",
    "validate_required_columns",
    "validate_no_empty_dataframe",
    "validate_aligned_index",
    "validate_numeric_columns",
    "impute_missing_values",
    "clean_time_series_dataframe",
]


def _format_column_list(columns: Iterable[object]) -> str:
    """Return a stable, readable representation of column names."""
    return ", ".join(str(column) for column in columns)


def validate_no_empty_dataframe(
    df: pd.DataFrame,
    *,
    dataframe_name: str = "dataframe",
) -> pd.DataFrame:
    """Validate that a dataframe exists and contains at least one row and column.

    Args:
        df: Dataframe to validate.
        dataframe_name: Human-readable name used in error messages.

    Returns:
        The original dataframe, enabling fluent validation chains.

    Raises:
        TypeError: If ``df`` is not a pandas ``DataFrame``.
        ValueError: If the dataframe has no rows or no columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataframe_name} must be a pandas DataFrame.")

    if df.empty or len(df.columns) == 0:
        raise ValueError(f"{dataframe_name} must not be empty.")

    return df


def validate_datetime_index(
    df: pd.DataFrame,
    *,
    dataframe_name: str = "dataframe",
    require_monotonic: bool = False,
    require_unique: bool = True,
) -> pd.DataFrame:
    """Validate that a dataframe uses a ``DatetimeIndex`` suitable for time series.

    Args:
        df: Dataframe to validate.
        dataframe_name: Human-readable name used in error messages.
        require_monotonic: When ``True``, require timestamps to be sorted in
            increasing order.
        require_unique: When ``True``, reject duplicated timestamps.

    Returns:
        The original dataframe, enabling fluent validation chains.

    Raises:
        TypeError: If ``df`` is not a dataframe or its index is not a
            ``DatetimeIndex``.
        ValueError: If monotonicity or uniqueness constraints are violated.
    """
    validate_no_empty_dataframe(df, dataframe_name=dataframe_name)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{dataframe_name} must be indexed by a pandas DatetimeIndex.")

    if require_monotonic and not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{dataframe_name} DatetimeIndex must be sorted in increasing order."
        )

    if require_unique and not df.index.is_unique:
        duplicated = df.index[df.index.duplicated()].unique()
        preview = _format_column_list(duplicated[:5])
        raise ValueError(
            f"{dataframe_name} DatetimeIndex contains duplicated timestamps"
            f"{': ' + preview if preview else ''}."
        )

    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[ColumnName],
    *,
    dataframe_name: str = "dataframe",
) -> pd.DataFrame:
    """Validate that a dataframe contains all required columns.

    Args:
        df: Dataframe to validate.
        required_columns: Column names that must be present.
        dataframe_name: Human-readable name used in error messages.

    Returns:
        The original dataframe, enabling fluent validation chains.

    Raises:
        TypeError: If ``required_columns`` is not a sequence of strings.
        ValueError: If one or more required columns are missing.
    """
    validate_no_empty_dataframe(df, dataframe_name=dataframe_name)

    if isinstance(required_columns, (str, bytes)):
        raise TypeError("required_columns must be a sequence of column names, not a string.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required column(s): {_format_column_list(missing)}."
        )

    return df


def validate_aligned_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str = "left dataframe",
    right_name: str = "right dataframe",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that two time-series dataframes share the same datetime index.

    Args:
        left: First dataframe to compare.
        right: Second dataframe to compare.
        left_name: Human-readable name for the first dataframe.
        right_name: Human-readable name for the second dataframe.

    Returns:
        The original pair ``(left, right)`` when indices are aligned.

    Raises:
        TypeError: If either dataframe does not use a ``DatetimeIndex``.
        ValueError: If indexes differ by length, order, timestamp values, or timezone.
    """
    validate_datetime_index(left, dataframe_name=left_name)
    validate_datetime_index(right, dataframe_name=right_name)

    if not left.index.equals(right.index):
        raise ValueError(
            f"{left_name} and {right_name} must have aligned DatetimeIndex values "
            "with identical length, order, timestamps, and timezone."
        )

    return left, right


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[ColumnName] | None = None,
    *,
    dataframe_name: str = "dataframe",
    allow_bool: bool = False,
) -> pd.DataFrame:
    """Validate that selected dataframe columns contain numeric data.

    Args:
        df: Dataframe to validate.
        columns: Columns to check. When omitted, all dataframe columns are
            validated.
        dataframe_name: Human-readable name used in error messages.
        allow_bool: Whether boolean columns should be accepted as numeric.

    Returns:
        The original dataframe, enabling fluent validation chains.

    Raises:
        ValueError: If a requested column is missing or has a non-numeric dtype.
    """
    validate_no_empty_dataframe(df, dataframe_name=dataframe_name)

    columns_to_check = list(df.columns if columns is None else columns)
    validate_required_columns(df, columns_to_check, dataframe_name=dataframe_name)

    non_numeric = []
    for column in columns_to_check:
        dtype = df[column].dtype
        if not is_numeric_dtype(dtype) or (is_bool_dtype(dtype) and not allow_bool):
            non_numeric.append(column)

    if non_numeric:
        raise ValueError(
            f"{dataframe_name} contains non-numeric column(s): "
            f"{_format_column_list(non_numeric)}."
        )

    return df


def impute_missing_values(
    df: pd.DataFrame,
    columns: Sequence[ColumnName] | None = None,
    *,
    dataframe_name: str = "dataframe",
    interpolation_method: str = "time",
    fill_edges: bool = True,
) -> pd.DataFrame:
    """Impute missing values in numeric time-series columns.

    The default strategy mirrors the notebook workflow: temporal interpolation
    followed by forward-fill and backward-fill to handle missing values at the
    edges of the series.

    Args:
        df: Dataframe to clean. The input object is not mutated.
        columns: Numeric columns to impute. When omitted, all numeric columns are
            selected.
        dataframe_name: Human-readable name used in error messages.
        interpolation_method: Interpolation method passed to
            ``DataFrame.interpolate``. The default ``"time"`` requires a
            ``DatetimeIndex`` and is appropriate for regularly sampled energy
            time series.
        fill_edges: When ``True``, apply forward-fill and backward-fill after
            interpolation.

    Returns:
        A copy of ``df`` with missing values imputed in the selected columns.

    Raises:
        TypeError: If the dataframe does not use a ``DatetimeIndex``.
        ValueError: If selected columns are missing, non-numeric, or still
            contain missing values after imputation.
    """
    validate_datetime_index(df, dataframe_name=dataframe_name)

    cleaned = df.copy()
    selected_columns = (
        cleaned.select_dtypes(include="number").columns
        if columns is None
        else columns
    )
    columns_to_impute = list(selected_columns)

    if not columns_to_impute:
        return cleaned

    validate_numeric_columns(
        cleaned,
        columns_to_impute,
        dataframe_name=dataframe_name,
    )

    cleaned.loc[:, columns_to_impute] = cleaned.loc[:, columns_to_impute].interpolate(
        method=interpolation_method
    )

    if fill_edges:
        cleaned.loc[:, columns_to_impute] = cleaned.loc[:, columns_to_impute].ffill().bfill()

    columns_with_missing = [column for column in columns_to_impute if cleaned[column].isna().any()]
    if columns_with_missing:
        raise ValueError(
            f"{dataframe_name} still contains missing values after imputation in column(s): "
            f"{_format_column_list(columns_with_missing)}."
        )

    return cleaned


def clean_time_series_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[ColumnName] | None = None,
    numeric_columns: Sequence[ColumnName] | None = None,
    aligned_to: pd.DataFrame | None = None,
    dataframe_name: str = "dataframe",
    aligned_to_name: str = "reference dataframe",
    sort_index: bool = True,
    impute: bool = True,
) -> pd.DataFrame:
    """Validate and clean a dataframe for the forecasting data layer.

    The function performs the standard data-layer sequence for time-series
    inputs: non-empty dataframe validation, datetime-index validation, optional
    index sorting, required-column checks, numeric-column checks, optional
    alignment against another dataframe, and optional missing-value imputation.

    Args:
        df: Input dataframe. The input object is not mutated.
        required_columns: Columns that must be present before cleaning.
        numeric_columns: Columns expected to be numeric and, when ``impute`` is
            true, the subset considered for imputation. When omitted, imputation
            is applied to all numeric columns.
        aligned_to: Optional reference dataframe whose index must match ``df``.
        dataframe_name: Human-readable name used in error messages.
        aligned_to_name: Human-readable name for the reference dataframe.
        sort_index: When ``True``, return data ordered by timestamp.
        impute: When ``True``, impute missing numeric values.

    Returns:
        A cleaned copy of ``df`` ready for downstream feature generation or
        forecasting steps.

    Raises:
        TypeError: If input objects have invalid types or datetime indexes.
        ValueError: If column, numeric, alignment, or missing-value validation
            fails.
    """
    validate_datetime_index(df, dataframe_name=dataframe_name)

    cleaned = df.copy()
    if sort_index:
        cleaned = cleaned.sort_index()

    if required_columns is not None:
        validate_required_columns(
            cleaned,
            required_columns,
            dataframe_name=dataframe_name,
        )

    if numeric_columns is not None:
        validate_numeric_columns(
            cleaned,
            numeric_columns,
            dataframe_name=dataframe_name,
        )

    if aligned_to is not None:
        validate_aligned_index(
            cleaned,
            aligned_to,
            left_name=dataframe_name,
            right_name=aligned_to_name,
        )

    if impute:
        cleaned = impute_missing_values(
            cleaned,
            numeric_columns,
            dataframe_name=dataframe_name,
        )

    return cleaned
