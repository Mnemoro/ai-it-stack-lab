from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_balance.balance_calculator import EnergyBalanceResult, calculate_energy_balance
from energy_balance.kpi import calculate_cumulative_grid_flows, calculate_energy_balance_kpis


def test_balance_formulas_types_index_and_input_immutability():
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    consumption = pd.Series([5., 3., 2., 4.], index=idx, name="consumption")
    production = pd.Series([2., 3., 6., 1.], index=idx, name="production")
    c_before, p_before = consumption.copy(), production.copy()
    result = calculate_energy_balance(consumption, production)
    assert isinstance(result, EnergyBalanceResult) and result.index.equals(idx)
    assert result.net_balance == (-3., 0., 4., -3.)
    assert result.grid_import == (3., 0., 0., 3.)
    assert result.grid_export == (0., 0., 4., 0.)
    assert result.as_dict()["consumption"] == (5., 3., 2., 4.)
    assert consumption.equals(c_before) and production.equals(p_before)


@pytest.mark.parametrize("consumption,production,error", [([], [], ValueError), ([1.], [1., 2.], ValueError), ([1., np.nan], [0., 0.], ValueError), ([1.], [np.inf], ValueError), (["bad"], [1.], TypeError)])
def test_balance_rejects_empty_mismatched_or_invalid_values(consumption, production, error):
    with pytest.raises(error):
        calculate_energy_balance(consumption, production)


def test_balance_rejects_different_series_indexes():
    left = pd.Series([1.], index=pd.date_range("2026-01-01", periods=1, freq="h"))
    right = pd.Series([2.], index=pd.date_range("2026-01-02", periods=1, freq="h"))
    with pytest.raises(ValueError, match="aligned"):
        calculate_energy_balance(left, right)


def test_kpis_and_cumulative_flows_match_balance_and_validate_fields():
    result = calculate_energy_balance([5., 3., 2., 4.], [2., 3., 6., 1.])
    kpis = calculate_energy_balance_kpis(result)
    assert (kpis.total_grid_export, kpis.total_grid_import, kpis.total_net_balance) == (4., 6., -2.)
    cumulative = calculate_cumulative_grid_flows(result)
    assert cumulative.cumulative_grid_export == (0., 0., 4., 4.)
    assert cumulative.cumulative_grid_import == (3., 3., 3., 6.)
    with pytest.raises(ValueError):
        calculate_energy_balance_kpis({"net_balance": [0], "grid_import": [0], "grid_export": [0, 1]})
    with pytest.raises(ValueError):
        calculate_cumulative_grid_flows({"net_balance": [0], "grid_import": [np.nan], "grid_export": [0]})


def test_kpis_accept_mapping_and_preserve_optional_index():
    idx = pd.date_range("2026-01-01", periods=1, freq="h")
    flows = calculate_cumulative_grid_flows({"net_balance": (1,), "grid_import": (0,), "grid_export": (1,), "index": idx})
    assert flows.index is None  # Mapping outputs carry no object-level index attribute.
