"""KPI aggregations for canonical Energy Balance Layer outputs.

This module consumes results produced by ``energy_balance.balance_calculator``
and derives totals and cumulative grid-flow series. It does not recompute the
underlying balance formulas and does not perform forecasting, monitoring
metrics, plotting, persistence, costs, tariffs, or unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, TypeAlias

from energy_balance.balance_calculator import EnergyBalanceResult

BalanceResult: TypeAlias = Any

__all__ = [
    "EnergyBalanceKPI",
    "CumulativeGridFlows",
    "calculate_energy_balance_kpis",
    "calculate_cumulative_grid_flows",
]

_REQUIRED_BALANCE_FIELDS = ("net_balance", "grid_import", "grid_export")


@dataclass(frozen=True, slots=True)
class EnergyBalanceKPI:
    """Aggregate quantitative indicators derived from balance results."""

    total_grid_export: float
    total_grid_import: float
    total_net_balance: float

    def as_dict(self) -> dict[str, float]:
        """Return KPI values in a serializable mapping."""
        return {
            "total_grid_export": self.total_grid_export,
            "total_grid_import": self.total_grid_import,
            "total_net_balance": self.total_net_balance,
        }


@dataclass(frozen=True, slots=True)
class CumulativeGridFlows:
    """Cumulative import/export series derived from balance results."""

    cumulative_grid_export: tuple[float, ...]
    cumulative_grid_import: tuple[float, ...]
    index: Any | None = None

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        """Return cumulative series in a serializable mapping."""
        return {
            "cumulative_grid_export": self.cumulative_grid_export,
            "cumulative_grid_import": self.cumulative_grid_import,
        }


def _field_values(balance_result: BalanceResult, field: str) -> Any:
    """Extract a canonical field from supported balance result containers."""
    if isinstance(balance_result, EnergyBalanceResult):
        return getattr(balance_result, field)
    if isinstance(balance_result, dict):
        if field not in balance_result:
            raise ValueError(f"balance_result is missing required field: {field}.")
        return balance_result[field]
    if hasattr(balance_result, field):
        return getattr(balance_result, field)
    if hasattr(balance_result, "__getitem__"):
        try:
            return balance_result[field]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"balance_result is missing required field: {field}.") from exc
    raise TypeError("balance_result must be an EnergyBalanceResult or a mapping-like balance output.")


def _index_value(balance_result: BalanceResult) -> Any | None:
    """Return an optional index from supported balance result containers."""
    return getattr(balance_result, "index", None)


def _as_finite_tuple(values: Any, field: str) -> tuple[float, ...]:
    """Validate one balance-result field as finite numeric values."""
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise TypeError(f"balance_result['{field}'] must be iterable.") from exc

    if not raw_values:
        raise ValueError("balance_result must not be empty.")

    numeric_values: list[float] = []
    for value in raw_values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"balance_result['{field}'] must contain numeric values.") from exc
        if not isfinite(numeric_value):
            raise ValueError(f"balance_result['{field}'] must contain only finite numeric values.")
        numeric_values.append(numeric_value)
    return tuple(numeric_values)


def _validate_balance_result(
    balance_result: BalanceResult,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return validated canonical balance-result fields."""
    net_balance = _as_finite_tuple(_field_values(balance_result, "net_balance"), "net_balance")
    grid_import = _as_finite_tuple(_field_values(balance_result, "grid_import"), "grid_import")
    grid_export = _as_finite_tuple(_field_values(balance_result, "grid_export"), "grid_export")

    lengths = {len(net_balance), len(grid_import), len(grid_export)}
    if len(lengths) != 1:
        raise ValueError("balance_result canonical fields must have the same length.")

    return net_balance, grid_import, grid_export


def calculate_energy_balance_kpis(balance_result: BalanceResult) -> EnergyBalanceKPI:
    """Compute total import/export KPIs from canonical balance output.

    Args:
        balance_result: Result produced by
            ``balance_calculator.calculate_energy_balance`` or an equivalent
            mapping-like object containing the canonical fields.

    Returns:
        An immutable :class:`EnergyBalanceKPI` with total grid export, total grid
        import, and total net balance.
    """
    net_balance, grid_import, grid_export = _validate_balance_result(balance_result)
    return EnergyBalanceKPI(
        total_grid_export=sum(grid_export),
        total_grid_import=sum(grid_import),
        total_net_balance=sum(net_balance),
    )


def _cumulative_sum(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return cumulative sums for a numeric sequence."""
    total = 0.0
    cumulative_values: list[float] = []
    for value in values:
        total += value
        cumulative_values.append(total)
    return tuple(cumulative_values)


def calculate_cumulative_grid_flows(balance_result: BalanceResult) -> CumulativeGridFlows:
    """Compute cumulative import/export series from canonical balance output.

    Args:
        balance_result: Result produced by
            ``balance_calculator.calculate_energy_balance`` or an equivalent
            mapping-like object containing the canonical fields.

    Returns:
        A :class:`CumulativeGridFlows` result containing
        ``cumulative_grid_export`` and ``cumulative_grid_import``.
    """
    _, grid_import, grid_export = _validate_balance_result(balance_result)
    return CumulativeGridFlows(
        cumulative_grid_export=_cumulative_sum(grid_export),
        cumulative_grid_import=_cumulative_sum(grid_import),
        index=_index_value(balance_result),
    )
