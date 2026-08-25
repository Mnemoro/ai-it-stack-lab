"""Deterministic energy-balance calculations for forecasted energy flows.

The Energy Balance Layer receives consumption and production values that are
already aligned, numeric, finite, and expressed in the same physical scale. It
computes only the canonical operational balance fields and deliberately avoids
forecasting, scaling, plotting, persistence, costs, tariffs, batteries, or model
specific concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, TypeAlias

ArrayLike: TypeAlias = Any

__all__ = ["EnergyBalanceResult", "calculate_energy_balance"]


@dataclass(frozen=True, slots=True)
class EnergyBalanceResult:
    """Per-timestep canonical energy-balance result.

    Attributes:
        consumption: Consumption values used as received, converted only to
            floats for numeric computation.
        production: Production values used as received, converted only to
            floats for numeric computation.
        net_balance: Canonical ``production - consumption`` values.
        grid_import: Positive grid purchases derived from negative balances.
        grid_export: Positive grid injections derived from positive balances.
        index: Optional pandas-compatible index preserved from aligned indexed
            inputs. Array-like inputs without indexes remain positionally
            aligned and leave this field as ``None``.
    """

    consumption: tuple[float, ...]
    production: tuple[float, ...]
    net_balance: tuple[float, ...]
    grid_import: tuple[float, ...]
    grid_export: tuple[float, ...]
    index: Any | None = None

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        """Return canonical balance columns in a serializable mapping."""
        return {
            "consumption": self.consumption,
            "production": self.production,
            "net_balance": self.net_balance,
            "grid_import": self.grid_import,
            "grid_export": self.grid_export,
        }


def _extract_index(values: ArrayLike) -> Any | None:
    """Return a pandas-like index when present, otherwise ``None``."""
    index = getattr(values, "index", None)
    if callable(index):
        return None
    return index


def _indexes_equal(left: Any, right: Any) -> bool:
    """Compare pandas-like indexes without reindexing or implicit joins."""
    equals = getattr(left, "equals", None)
    if callable(equals):
        return bool(equals(right))
    return left == right


def _as_numeric_tuple(values: ArrayLike, name: str) -> tuple[float, ...]:
    """Convert a one-dimensional input to finite numeric float values."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain numeric values.")

    try:
        raw_values = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numeric values.") from exc

    if not raw_values:
        raise ValueError(f"{name} must not be empty.")

    numeric_values: list[float] = []
    for value in raw_values:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must contain numeric values.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values.") from exc
        if not isfinite(numeric_value):
            raise ValueError(f"{name} must contain only finite numeric values.")
        numeric_values.append(numeric_value)

    return tuple(numeric_values)


def _validate_alignment(consumption: ArrayLike, production: ArrayLike) -> Any | None:
    """Validate positional length or explicit pandas-index alignment."""
    consumption_index = _extract_index(consumption)
    production_index = _extract_index(production)

    if consumption_index is not None and production_index is not None:
        if not _indexes_equal(consumption_index, production_index):
            raise ValueError("consumption and production pandas indexes must be aligned.")
        return consumption_index

    try:
        consumption_length = len(consumption)
        production_length = len(production)
    except TypeError as exc:
        raise TypeError("consumption and production must be sized iterables.") from exc

    if consumption_length != production_length:
        raise ValueError(
            "consumption and production must have the same length: "
            f"received {consumption_length} and {production_length}."
        )

    return consumption_index if consumption_index is not None else production_index


def calculate_energy_balance(consumption: ArrayLike, production: ArrayLike) -> EnergyBalanceResult:
    """Compute canonical per-timestep energy-balance fields.

    The canonical formulas are ``net_balance = production - consumption``,
    ``grid_export = max(net_balance, 0)``, and
    ``grid_import = max(-net_balance, 0)``. Negative input values are not
    clipped or otherwise modified; callers are responsible for passing values
    that satisfy the domain contract.

    Args:
        consumption: Numeric consumption values in the original physical scale.
        production: Numeric production values in the same unit and time order as
            ``consumption``.

    Returns:
        An immutable :class:`EnergyBalanceResult` containing ``consumption``,
        ``production``, ``net_balance``, ``grid_import``, and ``grid_export``.

    Raises:
        TypeError: If either input cannot be interpreted as numeric values.
        ValueError: If inputs are empty, non-finite, have incompatible lengths,
            or have incompatible pandas-style indexes.
    """
    result_index = _validate_alignment(consumption, production)
    consumption_values = _as_numeric_tuple(consumption, "consumption")
    production_values = _as_numeric_tuple(production, "production")

    if len(consumption_values) != len(production_values):
        raise ValueError(
            "consumption and production must have the same length: "
            f"received {len(consumption_values)} and {len(production_values)}."
        )

    net_balance = tuple(
        production - consumption
        for consumption, production in zip(consumption_values, production_values)
    )
    grid_export = tuple(max(balance, 0.0) for balance in net_balance)
    grid_import = tuple(max(0.0, -balance) for balance in net_balance)

    return EnergyBalanceResult(
        consumption=consumption_values,
        production=production_values,
        net_balance=net_balance,
        grid_import=grid_import,
        grid_export=grid_export,
        index=result_index,
    )
