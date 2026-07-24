"""Forecast evaluation orchestration for the energy monitoring layer.

The notebook evaluation workflow compares observed target arrays with model
predictions and reports MAE and RMSE for both consumption and photovoltaic
production. This module captures that reusable orchestration without taking over
responsibilities that belong to upstream or downstream layers: callers must pass
values that are already aligned and already expressed in the desired physical
scale, while visualization, persistence, training, prediction, feature
engineering, scaling, and inverse scaling remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from monitoring.metrics import mean_absolute_error, root_mean_squared_error

ArrayLike: TypeAlias = Any

__all__ = ["EvaluationResult", "evaluate_forecast"]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured quantitative evaluation for one forecast target and split.

    The same result shape is used for every target, including energy consumption
    and photovoltaic production. Optional labels identify the evaluated target
    and dataset split without changing the metric computation.

    Attributes:
        mae: Mean absolute error computed by ``monitoring.metrics``.
        rmse: Root mean squared error computed by ``monitoring.metrics``.
        target_name: Optional logical target label, for example ``"consumption"``
            or ``"production"``.
        dataset_name: Optional dataset/split label, for example ``"TRAIN"``,
            ``"VAL"``, or ``"TEST"``.
    """

    mae: float
    rmse: float
    target_name: str | None = None
    dataset_name: str | None = None

    @property
    def metrics(self) -> dict[str, float]:
        """Return metric values in a notebook- and dashboard-friendly mapping."""
        return {"mae": self.mae, "rmse": self.rmse}

    def as_dict(self) -> dict[str, float | str | None]:
        """Return the full evaluation result as a serializable dictionary."""
        return {
            "target_name": self.target_name,
            "dataset_name": self.dataset_name,
            "mae": self.mae,
            "rmse": self.rmse,
        }


def evaluate_forecast(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    target_name: str | None = None,
    dataset_name: str | None = None,
) -> EvaluationResult:
    """Evaluate forecast predictions with the project monitoring metrics.

    This function coordinates quantitative evaluation only. It compares the
    observed values supplied by the caller with the corresponding predicted
    values supplied by the caller, using MAE and RMSE from
    :mod:`monitoring.metrics`. Inputs are not scaled, inverse-scaled, loaded,
    reshaped into sequences, predicted, persisted, or visualized here.

    Args:
        y_true: Observed numeric values in the desired evaluation scale.
        y_pred: Predicted numeric values with the same shape as ``y_true``.
        target_name: Optional target label reused by callers for consumption,
            photovoltaic production, or future single-target workflows.
        dataset_name: Optional dataset/split label reused by notebooks,
            monitoring workflows, or dashboards.

    Returns:
        An immutable :class:`EvaluationResult` containing MAE and RMSE plus the
        optional labels.

    Raises:
        TypeError: Propagated from the metric functions when values cannot be
            interpreted as numeric arrays.
        ValueError: Propagated from the metric functions when inputs are empty,
            non-finite, or have incompatible shapes.
    """
    return EvaluationResult(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        target_name=target_name,
        dataset_name=dataset_name,
    )
