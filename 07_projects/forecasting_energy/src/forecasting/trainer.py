"""Training orchestration for Seq2Seq LSTM energy forecasting models.

This module preserves the training behavior used by the original notebook for
both consumption and photovoltaic-production targets while keeping model
construction, feature engineering, scaling, sequence construction, prediction,
evaluation, and visualization outside the trainer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, TypeAlias

ArrayLike: TypeAlias = Any
Callback: TypeAlias = Any
History: TypeAlias = Any

__all__ = [
    "TrainingArtifacts",
    "TrainingConfig",
    "build_and_train_seq2seq_model",
    "train_seq2seq_model",
]


class TrainableSeq2SeqModel(Protocol):
    """Minimal Keras-like protocol required by the trainer."""

    def fit(self, x: Any, y: ArrayLike, **kwargs: Any) -> History:
        """Fit the model and return a Keras-compatible history object."""
        ...

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Persist the model to ``filepath``."""
        ...


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for notebook-compatible model training.

    Defaults mirror the source notebook: ``epochs=30``, ``batch_size=32``, one
    ``EarlyStopping`` callback monitoring ``val_loss`` with ``patience=5`` and
    ``restore_best_weights=True``, and Keras' default shuffle behavior because
    the notebook did not explicitly pass a ``shuffle`` argument to ``fit``.

    Attributes:
        epochs: Maximum number of training epochs passed to ``model.fit``.
        batch_size: Batch size passed to ``model.fit``.
        early_stopping_monitor: Quantity monitored by the default early-stopping
            callback.
        early_stopping_patience: Number of epochs without improvement tolerated
            by the default early-stopping callback.
        restore_best_weights: Whether early stopping restores the best weights.
        shuffle: Optional explicit ``model.fit`` shuffle value. Leave as
            ``None`` to preserve the notebook behavior of not passing the
            argument and therefore using Keras' default.
        extra_fit_kwargs: Additional fit keyword arguments for callers that need
            Keras-native training controls without changing this module's API.
    """

    epochs: int = 30
    batch_size: int = 32
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 5
    restore_best_weights: bool = True
    shuffle: bool | None = None
    extra_fit_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TrainingArtifacts:
    """Optional model-training artifact configuration.

    The notebook saved each trained model and its ``History.history`` dictionary
    under a timestamped filename in a ``models`` directory. This configuration
    keeps that behavior opt-in so training orchestration remains usable without
    side effects by default.

    Attributes:
        output_dir: Directory where artifacts are written when configured.
        model_prefix: Prefix used for the saved model filename.
        history_prefix: Prefix used for the saved history JSON filename.
        timestamp: Optional timestamp suffix. If omitted, the notebook format
            ``YYYYmmdd_HHMMSS`` is generated at save time.
        model_extension: File extension used for saved Keras models. Defaults
            to ``.h5`` to match the notebook.
    """

    output_dir: str | Path = "models"
    model_prefix: str = "model"
    history_prefix: str = "history"
    timestamp: str | None = None
    model_extension: str = ".h5"


def _validate_training_config(config: TrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be greater than or equal to zero.")
    if not config.early_stopping_monitor:
        raise ValueError("early_stopping_monitor must not be empty.")


def _default_callbacks(config: TrainingConfig) -> list[Callback]:
    """Create the notebook's default EarlyStopping callback lazily."""
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-untyped]

    return [
        EarlyStopping(
            monitor=config.early_stopping_monitor,
            patience=config.early_stopping_patience,
            restore_best_weights=config.restore_best_weights,
        )
    ]


def _fit_inputs(encoder_sequences: ArrayLike, decoder_sequences: ArrayLike) -> list[ArrayLike]:
    """Return Keras input ordering used by the original notebook."""
    return [encoder_sequences, decoder_sequences]


def _history_payload(history: History) -> dict[str, list[float]]:
    raw_history = getattr(history, "history", history)
    if not isinstance(raw_history, Mapping):
        raise TypeError("history must expose a mapping-like 'history' attribute.")
    return {str(key): [float(value) for value in values] for key, values in raw_history.items()}


def _save_artifacts(model: TrainableSeq2SeqModel, history: History, artifacts: TrainingArtifacts) -> None:
    from datetime import datetime

    timestamp = artifacts.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(artifacts.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = artifacts.model_extension if artifacts.model_extension.startswith(".") else f".{artifacts.model_extension}"
    model_path = output_dir / f"{artifacts.model_prefix}_{timestamp}{extension}"
    history_path = output_dir / f"{artifacts.history_prefix}_{timestamp}.json"

    model.save(str(model_path))
    history_path.write_text(json.dumps(_history_payload(history), indent=2), encoding="utf-8")


def train_seq2seq_model(
    model: TrainableSeq2SeqModel,
    *,
    train_encoder_sequences: ArrayLike,
    train_decoder_sequences: ArrayLike,
    train_targets: ArrayLike,
    validation_encoder_sequences: ArrayLike,
    validation_decoder_sequences: ArrayLike,
    validation_targets: ArrayLike,
    config: TrainingConfig | None = None,
    callbacks: Sequence[Callback] | None = None,
    artifacts: TrainingArtifacts | None = None,
) -> History:
    """Train a prepared Seq2Seq model on prepared sequence arrays.

    The same orchestration applies to consumption and photovoltaic-production
    models in the notebook: two model inputs ``[X_enc_train, X_dec_train]``, one
    target tensor, validation data in the same two-input form, 30 epochs, batch
    size 32, and early stopping on validation loss. No prediction or data
    preparation is performed here.

    Args:
        model: Compiled Seq2Seq model created by
            :func:`forecasting.lstm_model.build_seq2seq_lstm_model` or an
            equivalent Keras-compatible model instance.
        train_encoder_sequences: Prepared encoder training sequences.
        train_decoder_sequences: Prepared decoder training sequences.
        train_targets: Prepared training target sequences.
        validation_encoder_sequences: Prepared encoder validation sequences.
        validation_decoder_sequences: Prepared decoder validation sequences.
        validation_targets: Prepared validation target sequences.
        config: Optional training configuration. Defaults preserve notebook
            behavior.
        callbacks: Optional explicit callbacks. If omitted, the notebook's
            ``EarlyStopping(monitor='val_loss', patience=5,
            restore_best_weights=True)`` callback is created.
        artifacts: Optional artifact persistence configuration. If provided,
            saves the trained model and JSON history after fitting, matching the
            isolated training-specific persistence behavior in the notebook.

    Returns:
        The Keras ``History`` object returned by ``model.fit``.
    """
    effective_config = config or TrainingConfig()
    _validate_training_config(effective_config)

    fit_kwargs: dict[str, Any] = {
        "epochs": effective_config.epochs,
        "batch_size": effective_config.batch_size,
        "validation_data": (
            _fit_inputs(validation_encoder_sequences, validation_decoder_sequences),
            validation_targets,
        ),
        "callbacks": list(callbacks) if callbacks is not None else _default_callbacks(effective_config),
    }
    if effective_config.shuffle is not None:
        fit_kwargs["shuffle"] = effective_config.shuffle
    if effective_config.extra_fit_kwargs:
        fit_kwargs.update(dict(effective_config.extra_fit_kwargs))

    history = model.fit(
        _fit_inputs(train_encoder_sequences, train_decoder_sequences),
        train_targets,
        **fit_kwargs,
    )

    if artifacts is not None:
        _save_artifacts(model, history, artifacts)

    return history


def build_and_train_seq2seq_model(
    *,
    n_features_enc: int,
    n_features_dec: int,
    train_encoder_sequences: ArrayLike,
    train_decoder_sequences: ArrayLike,
    train_targets: ArrayLike,
    validation_encoder_sequences: ArrayLike,
    validation_decoder_sequences: ArrayLike,
    validation_targets: ArrayLike,
    model_kwargs: Mapping[str, Any] | None = None,
    config: TrainingConfig | None = None,
    callbacks: Sequence[Callback] | None = None,
    artifacts: TrainingArtifacts | None = None,
) -> tuple[TrainableSeq2SeqModel, History]:
    """Build a Seq2Seq LSTM model with the model factory and train it.

    This convenience API keeps model architecture ownership in
    :mod:`forecasting.lstm_model` while allowing callers to perform the notebook
    training flow in a single step when they have not already constructed a
    model.

    Returns:
        A tuple containing the compiled model and the history returned by
        :func:`train_seq2seq_model`.
    """
    from forecasting.lstm_model import build_seq2seq_lstm_model

    model = build_seq2seq_lstm_model(
        n_features_enc=n_features_enc,
        n_features_dec=n_features_dec,
        **dict(model_kwargs or {}),
    )
    history = train_seq2seq_model(
        model,
        train_encoder_sequences=train_encoder_sequences,
        train_decoder_sequences=train_decoder_sequences,
        train_targets=train_targets,
        validation_encoder_sequences=validation_encoder_sequences,
        validation_decoder_sequences=validation_decoder_sequences,
        validation_targets=validation_targets,
        config=config,
        callbacks=callbacks,
        artifacts=artifacts,
    )
    return model, history
