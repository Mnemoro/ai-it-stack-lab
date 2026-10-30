"""Deferred Issue #6 contract tests retained for future defect issues.

These seven pytest cases are deliberately stored outside ``tests/`` in a file
whose name does not match pytest's default ``test_*.py`` discovery pattern.
They are not part of the ordinary suite until their corresponding production
behavior is corrected and the test is copied back into ``tests/``.

Recovery map:
* Data Layer Issue: regular-frequency validation and weather/index alignment.
* Feature Engineering Issue: historical input alignment and sequence alignment.
* Forecasting Issue: NaN forecast rejection; +/-Inf remain subject to an
  architectural decision about requiring finite outputs.

Architecture references (v15): Data Layer frequency, section 15.4 p. 87 and
section 5.3 p. 26; data alignment, section 17.3 p. 101; feature/target
alignment, sections 15.5 p. 88, 16.5 p. 95, 18.2-18.3 pp. 105-106; forecasting
output, section 15.6 p. 88.

To reintegrate, move the applicable test function(s) into the corresponding
``tests/test_*.py`` file. First split the combined frequency/sorting test so
the sorting contract remains in the ordinary Data Layer suite. Preserve the
NaN case as required; reintegrate +/-Inf as required only if finiteness is
formally adopted as part of the Forecasting contract.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import synthetic_data, validator
from features.preprocessing import (
    build_historical_dataframe,
    build_training_dataset,
    build_time_features,
)
from features.sequence_builder import build_training_sequences
from features.scaling import fit_target_scaler
from forecasting.predictor import predict_single_target


class FakeModel:
    def __init__(self, predictions=None):
        self.predictions = predictions
        self.fit_call = None
        self.predict_call = None

    def predict(self, inputs, **kwargs):
        self.predict_call = (inputs, kwargs)
        return self.predictions

    def fit(self, inputs, targets, **kwargs):
        self.fit_call = (inputs, targets, kwargs)
        return None


def _inputs(n=12):
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    weather = pd.DataFrame({"temperature": np.arange(n, dtype=float), "irradiance": np.linspace(0, 1, n), "wind_speed": np.ones(n)}, index=idx)
    return idx, weather, pd.Series(np.arange(n, dtype=float) + 10, index=idx, name="consumption"), pd.Series(np.arange(n, dtype=float) + 2, index=idx, name="production")


def test_time_series_validation_requires_regular_frequency_and_cleaning_sorts_rows():
    irregular = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 03:00"]),
    )
    with pytest.raises(ValueError, match="frequency|regular|interval"):
        validator.validate_datetime_index(irregular)

    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    unordered = pd.DataFrame({"load": [3.0, 1.0, 2.0]}, index=idx[[2, 0, 1]])
    cleaned = validator.clean_time_series_dataframe(unordered, sort_index=True, impute=False)
    assert cleaned.index.equals(idx)
    assert cleaned["load"].tolist() == [1.0, 2.0, 3.0]


def test_synthetic_consumption_rejects_weather_with_misaligned_timestamps():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    weather = pd.DataFrame(
        {"temperature": [18.0, 19.0, 20.0], "irradiance": [0.0, 0.5, 1.0]},
        index=idx + pd.Timedelta(hours=1),
    )
    with pytest.raises(ValueError, match="align|index|timestamp"):
        synthetic_data.generate_consumption_data(idx, weather)


def test_historical_preparation_rejects_series_with_misaligned_indexes():
    idx, weather, consumption, production = _inputs(5)
    shifted = consumption.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="align|index|timestamp"):
        build_historical_dataframe(weather, shifted, production)


def test_sequence_builder_rejects_future_feature_target_timestamp_mismatch():
    idx = pd.date_range("2026-01-01", periods=6, freq="h")
    features = pd.DataFrame({"x": np.arange(6, dtype=float)}, index=idx)
    shifted_features = pd.DataFrame(
        {"weather": np.arange(6, dtype=float)},
        index=idx + pd.Timedelta(hours=1),
    )
    targets = pd.Series(np.arange(6, dtype=float), index=idx)
    with pytest.raises(ValueError, match="align|index|timestamp"):
        build_training_sequences(features, targets, shifted_features, targets, 2, 2)


@pytest.mark.parametrize("bad_prediction", [np.nan, np.inf, -np.inf])
def test_predictor_rejects_non_finite_forecasts(bad_prediction):
    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    scaler = fit_target_scaler(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite|non-finite|NaN|inf"):
        predict_single_target(
            FakeModel(np.array([[[bad_prediction]]])),
            features_past=pd.DataFrame({"x": [1.0, 2.0]}, index=idx),
            features_future=pd.DataFrame({"x": [3.0, 4.0]}, index=idx),
            target_scaler=scaler,
            future_index=idx,
            lookback=1,
            future_window=1,
        )
