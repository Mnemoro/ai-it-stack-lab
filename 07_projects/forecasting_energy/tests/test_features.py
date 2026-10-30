from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.preprocessing import build_historical_dataframe, build_training_dataset, build_time_features, split_train_validation_test
from features.scaling import fit_feature_scaler, fit_target_scaler, inverse_transform_target, transform_features, transform_target
from features.sequence_builder import build_prediction_sequence, build_training_sequences


def _inputs(n=12):
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    weather = pd.DataFrame({"temperature": np.arange(n, dtype=float), "irradiance": np.linspace(0, 1, n), "wind_speed": np.ones(n)}, index=idx)
    return idx, weather, pd.Series(np.arange(n, dtype=float) + 10, index=idx, name="consumption"), pd.Series(np.arange(n, dtype=float) + 2, index=idx, name="production")


def test_preparation_features_split_and_missing_value_handling():
    idx, weather, consumption, production = _inputs()
    weather.loc[idx[3], "temperature"] = np.nan
    before = weather.copy(deep=True)
    data = build_training_dataset(weather, consumption, production, train_ratio=0.5, val_ratio=0.25, past_window=2)
    assert data.historical.shape == (12, 5) and data.historical.isna().sum().sum() == 0
    assert data.train.index[-1] < data.test.index[-1]
    assert data.time_features.loc[idx[0]].to_dict() == {"hour": 0, "dayofyear": 1, "month": 1}
    assert data.consumption_target.name == "consumption"
    assert weather.equals(before)
    assert data.train.index.equals(idx[:6])
    assert data.validation.index.equals(idx[4:9])
    assert data.test.index.equals(idx[7:])
    # The fold buffers overlap historical encoder rows, while forecast targets
    # begin at the chronological validation/test boundaries.
    assert data.validation.index[2] == idx[6]
    assert data.test.index[2] == idx[9]
    with pytest.raises(ValueError):
        build_time_features(pd.Index([1, 2]))


def test_split_short_input_keeps_chronology_and_required_buffer():
    idx, weather, consumption, production = _inputs(3)
    frame = pd.DataFrame({"x": range(3)}, index=idx)
    train, val, test = split_train_validation_test(frame, train_ratio=0.5, val_ratio=0.25, past_window=10)
    assert len(train) == 1 and len(val) == 2 and len(test) == 2
    assert train.index.max() < val.index.max() < test.index.max()


def test_scaling_preserves_pandas_metadata_roundtrips_and_rejects_missing_values():
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [4., 3., 2., 1.]}, index=idx)
    original = frame.copy(deep=True)
    train = frame.iloc[:2]
    validation = frame.iloc[2:]
    scaler = fit_feature_scaler(train)
    scaled = transform_features(frame, scaler)
    assert scaled.index.equals(idx) and list(scaled.columns) == ["a", "b"]
    np.testing.assert_allclose(transform_features(train, scaler).mean().to_numpy(), 0, atol=1e-12)
    # Validation uses training statistics: these values intentionally do not
    # have a zero mean after transformation.
    np.testing.assert_allclose(transform_features(validation, scaler)["a"].to_numpy(), [3.0, 5.0])
    np.testing.assert_allclose(transform_features(validation, scaler)["b"].to_numpy(), [-3.0, -5.0])
    assert frame.equals(original)
    target = pd.Series([10., 20., 30., 40.], index=idx, name="load")
    target_scaler = fit_target_scaler(target)
    scaled_target = transform_target(target, target_scaler)
    assert isinstance(scaled_target, pd.Series) and scaled_target.name == "load" and scaled_target.index.equals(idx)
    pd.testing.assert_series_equal(inverse_transform_target(scaled_target, target_scaler), target)
    with pytest.raises(ValueError):
        fit_feature_scaler(frame.assign(a=[0., np.nan, 2., 3.]))
    with pytest.raises(ValueError):
        transform_target(pd.DataFrame({"a": [1], "b": [2]}), target_scaler)


def test_sequence_windows_have_expected_values_shapes_and_edge_case():
    idx = pd.date_range("2026-01-01", periods=6, freq="h")
    features = pd.DataFrame({"x": np.arange(6)}, index=idx)
    targets = pd.Series(np.arange(6) * 10, index=idx)
    original = features.copy(deep=True)
    enc, dec, y = build_training_sequences(features, targets, features, targets, lookback=2, future_window=2)
    assert enc.shape == (3, 2, 1) and dec.shape == (3, 2, 1) and y.shape == (3, 2)
    np.testing.assert_array_equal(enc[0, :, 0], [0, 1])
    np.testing.assert_array_equal(dec[0, :, 0], [2, 3])
    np.testing.assert_array_equal(y[0], [20, 30])
    assert features.equals(original)
    enc_p, dec_p = build_prediction_sequence(features, features, lookback=2, future_window=2)
    assert enc_p.shape == (1, 2, 1) and dec_p.shape == (1, 2, 1)
    empty = build_training_sequences(features.iloc[:3], targets.iloc[:3], features.iloc[:3], targets.iloc[:3], 2, 2)
    assert all(x.shape[0] == 0 for x in empty)
