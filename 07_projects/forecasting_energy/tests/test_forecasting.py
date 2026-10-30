from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.scaling import fit_target_scaler, transform_target
from forecasting.predictor import predict_single_target
from forecasting.trainer import TrainingConfig, train_seq2seq_model
from forecasting.lstm_model import build_seq2seq_lstm_model


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
        return SimpleNamespace(history={"loss": [1.0, 0.5]})


def test_predictor_builds_windows_restores_scale_and_keeps_index():
    past_idx = pd.date_range("2026-01-01", periods=4, freq="h")
    future_idx = pd.date_range("2026-01-01 04:00", periods=3, freq="h")
    past = pd.DataFrame({"x": [0., 1., 2., 3.]}, index=past_idx)
    future = pd.DataFrame({"x": [4., 5., 6.]}, index=future_idx)
    train_target = pd.Series([10., 20., 30.], name="load")
    scaler = fit_target_scaler(train_target)
    expected_scaled = transform_target(pd.Series([20., 30.], name="load"), scaler).to_numpy().reshape(1, 2, 1)
    model = FakeModel(expected_scaled)
    past_before, future_before = past.copy(deep=True), future.copy(deep=True)
    result = predict_single_target(model, features_past=past, features_future=future, target_scaler=scaler,
                                   future_index=future_idx, lookback=2, future_window=2,
                                   prediction_name="load", predict_kwargs={"verbose": 0})
    assert result.name == "load" and result.index.equals(future_idx[:2])
    np.testing.assert_allclose(result.to_numpy(), [20., 30.])
    assert model.predict_call[0][0].shape == (1, 2, 1)
    assert model.predict_call[0][1].shape == (1, 2, 1)
    assert model.predict_call[1] == {"verbose": 0}
    assert past.equals(past_before) and future.equals(future_before)


@pytest.mark.parametrize("bad", [np.ones((2, 2, 1)), np.ones((1, 2, 2))])
def test_predictor_rejects_invalid_prediction_shapes(bad):
    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    scaler = fit_target_scaler(pd.Series([1., 2.]))
    with pytest.raises(ValueError):
        predict_single_target(FakeModel(bad), features_past=pd.DataFrame({"x": [1., 2.]}, index=idx),
                              features_future=pd.DataFrame({"x": [3., 4.]}, index=idx), target_scaler=scaler,
                              future_index=idx, lookback=1, future_window=1)


def test_predictor_rejects_short_inputs_and_bad_model():
    idx = pd.date_range("2026-01-01", periods=1, freq="h")
    args = dict(features_past=pd.DataFrame({"x": [1.]}, index=idx), features_future=pd.DataFrame({"x": [1.]}, index=idx),
                target_scaler=fit_target_scaler(pd.Series([1., 2.])), future_index=idx, lookback=2, future_window=1)
    with pytest.raises(ValueError):
        predict_single_target(FakeModel([1.]), **args)
    args["lookback"] = 1
    with pytest.raises(TypeError):
        predict_single_target(object(), **args)


def test_trainer_passes_batched_inputs_config_callbacks_and_optional_artifacts(tmp_path):
    model = FakeModel()
    train_enc, train_dec, train_y = np.ones((2, 3, 1)), np.ones((2, 2, 1)), np.ones((2, 2, 1))
    val_enc, val_dec, val_y = np.zeros((1, 3, 1)), np.zeros((1, 2, 1)), np.zeros((1, 2, 1))
    config = TrainingConfig(epochs=4, batch_size=2, shuffle=False)
    history = train_seq2seq_model(model, train_encoder_sequences=train_enc, train_decoder_sequences=train_dec,
                                  train_targets=train_y, validation_encoder_sequences=val_enc,
                                  validation_decoder_sequences=val_dec, validation_targets=val_y,
                                  config=config, callbacks=[], artifacts=None)
    assert history.history == {"loss": [1.0, 0.5]}
    inputs, targets, kwargs = model.fit_call
    assert inputs == [train_enc, train_dec] and targets is train_y
    assert kwargs["epochs"] == 4 and kwargs["batch_size"] == 2 and kwargs["shuffle"] is False
    assert kwargs["validation_data"] == ([val_enc, val_dec], val_y) and kwargs["callbacks"] == []

    class SavingModel(FakeModel):
        def save(self, path):
            Path(path).write_text("model", encoding="utf-8")

    persisted = SavingModel()
    train_seq2seq_model(persisted, train_encoder_sequences=train_enc, train_decoder_sequences=train_dec,
                        train_targets=train_y, validation_encoder_sequences=val_enc,
                        validation_decoder_sequences=val_dec, validation_targets=val_y,
                        config=config, callbacks=[],
                        artifacts=__import__("forecasting.trainer", fromlist=["TrainingArtifacts"]).TrainingArtifacts(
                            output_dir=tmp_path, timestamp="fixed"))
    assert (tmp_path / "model_fixed.h5").is_file()
    assert (tmp_path / "history_fixed.json").read_text(encoding="utf-8").find('"loss"') >= 0


def test_trainer_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        train_seq2seq_model(FakeModel(), train_encoder_sequences=[], train_decoder_sequences=[], train_targets=[],
                            validation_encoder_sequences=[], validation_decoder_sequences=[], validation_targets=[],
                            config=TrainingConfig(epochs=0), callbacks=[])


def test_seq2seq_factory_builds_two_input_single_target_model():
    model = build_seq2seq_lstm_model(
        n_features_enc=3, n_features_dec=2, latent_dim=2, dropout_rate=0.0,
        future_window=4, output_dim=1,
    )
    assert [tuple(shape[1:]) for shape in model.input_shape] == [(None, 3), (4, 2)]
    assert tuple(model.output_shape[1:]) == (4, 1)
    assert model.loss == "mse"
    with pytest.raises(ValueError):
        build_seq2seq_lstm_model(n_features_enc=0, n_features_dec=2)
