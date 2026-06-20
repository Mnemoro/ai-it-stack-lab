"""CSV loading helpers for historical energy forecasting datasets.

This module contains the data-loading responsibilities that were previously
embedded in the exploratory notebook.  It intentionally handles only CSV-based
historical datasets and stays independent from model-training and visualization
libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ParseDates = bool | list[str] | list[int] | list[list[str]] | list[list[int]] | dict[str, Any]

DATA_DIRECTORY = Path.cwd() / "data"
CONSUMPTION_HISTORY_FILENAME = "consumo_storico.csv"
PRODUCTION_HISTORY_FILENAME = "produzione_storico.csv"

__all__ = [
    "ensure_data_directory",
    "load_dataframe",
    "load_consumption_history",
    "load_production_history",
]


def ensure_data_directory() -> Path:
    """Create and return the notebook-compatible data directory.

    The original notebook stores generated and externally supplied historical
    CSV files in a ``data`` directory rooted at the current working directory.
    This helper preserves that convention while making directory creation
    explicit and reusable by loading code.

    Returns:
        The absolute path to the ``data`` directory.

    Raises:
        OSError: If the directory cannot be created or accessed.
    """
    try:
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Unable to create data directory at {DATA_DIRECTORY}.") from exc

    if not DATA_DIRECTORY.is_dir():
        raise OSError(f"Data path exists but is not a directory: {DATA_DIRECTORY}.")

    return DATA_DIRECTORY


def load_dataframe(
    csv_path: str | Path,
    index_column: str | int,
    parse_dates: ParseDates,
) -> pd.DataFrame:
    """Load a CSV time-series dataset into a dataframe with a datetime index.

    Args:
        csv_path: Path to the CSV file to load.
        index_column: Column name or position to use as the dataframe index.
        parse_dates: Date parsing configuration forwarded to ``pandas.read_csv``.
            Use ``True`` to parse the index column, matching the notebook's CSV
            loading behavior.

    Returns:
        A pandas ``DataFrame`` indexed by a ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV file is empty or the parsed index is not a
            ``DatetimeIndex``.
        OSError: If ``csv_path`` exists but is not a regular file.
        pandas.errors.ParserError: If pandas cannot parse the CSV content.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}.")

    if not path.is_file():
        raise OSError(f"CSV path exists but is not a file: {path}.")

    try:
        dataframe = pd.read_csv(path, index_col=index_column, parse_dates=parse_dates)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}.") from exc

    if dataframe.empty:
        raise ValueError(f"CSV file contains no data rows: {path}.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError(
            f"CSV file must parse column {index_column!r} as a pandas DatetimeIndex: {path}."
        )

    return dataframe


def _load_single_series(csv_path: Path, *, dataset_name: str) -> pd.Series:
    """Load a one-column historical CSV dataset as a time-indexed series.

    Args:
        csv_path: Path to the historical CSV file.
        dataset_name: Human-readable dataset name used in exception messages.

    Returns:
        A pandas ``Series`` indexed by a ``DatetimeIndex``.

    Raises:
        ValueError: If the CSV does not contain exactly one data column.
        FileNotFoundError: If the CSV file does not exist.
        OSError: If the CSV path is invalid.
    """
    dataframe = load_dataframe(csv_path, index_column=0, parse_dates=True)

    if len(dataframe.columns) != 1:
        raise ValueError(
            f"{dataset_name} CSV must contain exactly one data column; "
            f"found {len(dataframe.columns)} in {csv_path}."
        )

    series = dataframe.iloc[:, 0]
    series.index = dataframe.index
    return series


def load_consumption_history() -> pd.Series:
    """Load the historical consumption dataset from ``data/consumo_storico.csv``.

    Returns:
        A pandas ``Series`` containing historical consumption values indexed by
        a ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If ``data/consumo_storico.csv`` does not exist.
        ValueError: If the file is empty, has an invalid datetime index, or does
            not contain exactly one data column.
        OSError: If the data directory or CSV path cannot be accessed.
    """
    data_directory = ensure_data_directory()
    return _load_single_series(
        data_directory / CONSUMPTION_HISTORY_FILENAME,
        dataset_name="Consumption history",
    )


def load_production_history() -> pd.Series:
    """Load the historical production dataset from ``data/produzione_storico.csv``.

    Returns:
        A pandas ``Series`` containing historical production values indexed by a
        ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If ``data/produzione_storico.csv`` does not exist.
        ValueError: If the file is empty, has an invalid datetime index, or does
            not contain exactly one data column.
        OSError: If the data directory or CSV path cannot be accessed.
    """
    data_directory = ensure_data_directory()
    return _load_single_series(
        data_directory / PRODUCTION_HISTORY_FILENAME,
        dataset_name="Production history",
    )
