from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import loader, synthetic_data, validator


def test_validation_and_cleaning_interpolates_without_mutating_input():
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    original = pd.DataFrame({"load": [1.0, np.nan, 3.0, 4.0]}, index=idx)
    before = original.copy(deep=True)
    cleaned = validator.clean_time_series_dataframe(original, required_columns=["load"])
    assert cleaned["load"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert original.equals(before)
    assert cleaned is not original


@pytest.mark.parametrize("bad", [pd.DataFrame(), pd.DataFrame({"x": [1]}, index=[0])])
def test_data_validation_rejects_empty_and_non_datetime_inputs(bad):
    with pytest.raises((TypeError, ValueError)):
        validator.validate_datetime_index(bad)


def test_data_validation_rejects_duplicate_index_missing_columns_and_non_numeric():
    idx = pd.DatetimeIndex(["2026-01-01", "2026-01-01"])
    with pytest.raises(ValueError, match="duplicated"):
        validator.validate_datetime_index(pd.DataFrame({"x": [1, 2]}, index=idx))
    valid = pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2026-01-01", periods=2, freq="h"))
    with pytest.raises(ValueError, match="missing"):
        validator.validate_required_columns(valid, ["absent"])
    with pytest.raises(ValueError, match="non-numeric"):
        validator.validate_numeric_columns(valid.assign(x=["a", "b"]))


def test_time_series_cleaning_sorts_rows():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    unordered = pd.DataFrame({"load": [3.0, 1.0, 2.0]}, index=idx[[2, 0, 1]])
    cleaned = validator.clean_time_series_dataframe(unordered, sort_index=True, impute=False)
    assert cleaned.index.equals(idx)
    assert cleaned["load"].tolist() == [1.0, 2.0, 3.0]


def test_synthetic_generators_are_seed_reproducible_and_keep_contract():
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="Europe/Rome")
    random_state = np.random.get_state()
    try:
        np.random.seed(17)
        weather1 = synthetic_data.generate_weather_data(idx)
        consumption1 = synthetic_data.generate_consumption_data(idx, weather1)
        production1 = synthetic_data.generate_production_data(idx, weather1)
        np.random.seed(17)
        weather2 = synthetic_data.generate_weather_data(idx)
        consumption2 = synthetic_data.generate_consumption_data(idx, weather2)
        production2 = synthetic_data.generate_production_data(idx, weather2)
    finally:
        np.random.set_state(random_state)
    assert weather1.equals(weather2)
    assert consumption1.equals(consumption2) and production1.equals(production2)
    assert list(weather1.columns) == ["temperature", "irradiance", "wind_speed"]
    assert consumption1.name == "consumption" and production1.name == "production"
    assert weather1.index.equals(idx) and (production1 >= 0).all() and (consumption1 >= 0).all()


def test_loader_reads_synthetic_csv_and_reports_invalid_files(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("timestamp,load\n2026-01-01T00:00:00,1.5\n2026-01-01T01:00:00,2.5\n", encoding="utf-8")
    result = loader.load_dataframe(path, index_column="timestamp", parse_dates=True)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result["load"].tolist() == [1.5, 2.5]
    with pytest.raises(FileNotFoundError):
        loader.load_dataframe(tmp_path / "missing.csv", index_column=0, parse_dates=True)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        loader.load_dataframe(empty, index_column=0, parse_dates=True)


def test_specific_history_loaders_read_their_configured_single_series(tmp_path, monkeypatch):
    idx = pd.date_range("2026-02-01", periods=3, freq="h")
    monkeypatch.setattr(loader, "DATA_DIRECTORY", tmp_path)
    (tmp_path / loader.CONSUMPTION_HISTORY_FILENAME).write_text(
        "timestamp,consumption\n" + "\n".join(f"{stamp.isoformat()},{value}" for stamp, value in zip(idx, [1.0, 2.0, 3.0])),
        encoding="utf-8",
    )
    (tmp_path / loader.PRODUCTION_HISTORY_FILENAME).write_text(
        "timestamp,production\n" + "\n".join(f"{stamp.isoformat()},{value}" for stamp, value in zip(idx, [0.0, 0.5, 1.0])),
        encoding="utf-8",
    )
    consumption = loader.load_consumption_history()
    production = loader.load_production_history()
    assert consumption.name == "consumption" and consumption.index.equals(idx)
    assert consumption.tolist() == [1.0, 2.0, 3.0]
    assert production.name == "production" and production.index.equals(idx)
    assert production.tolist() == [0.0, 0.5, 1.0]

    (tmp_path / loader.CONSUMPTION_HISTORY_FILENAME).write_text(
        "timestamp,consumption,extra\n" + "\n".join(f"{stamp.isoformat()},1,2" for stamp in idx),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        loader.load_consumption_history()
