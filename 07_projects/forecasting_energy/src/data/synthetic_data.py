"""Synthetic historical data generators for the energy forecasting project.

This module extracts the synthetic data generation logic from the exploratory
notebook without changing the mathematical formulas or random-number usage.
It intentionally depends only on pandas and numpy, and it performs no file I/O,
interactive prompting, or console output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORICAL_MONTHS = 6
FREQ = "h"
PANEL_CAPACITY = 5.0
EFFICIENCY = 0.18
BASE_CONSUMPTION = 1.5
TEMP_EFFECT_FACTOR = 0.05

TIMEZONE = "Europe/Rome"
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
DAILY_TEMPERATURE_AMPLITUDE = 10
SEASONAL_TEMPERATURE_AMPLITUDE = 15
BASE_TEMPERATURE = 20
TEMPERATURE_NOISE_STD = 2
IRRADIANCE_NOISE_STD = 0.05
BASE_WIND_SPEED = 3
WIND_NOISE_STD = 1
CONSUMPTION_NOISE_STD = 0.2
PRODUCTION_NOISE_STD = 0.1
COMFORT_TEMPERATURE = 20
TEMPERATURE_DEADBAND = 2
DAILY_PATTERN_OFFSET_HOUR = 8

__all__ = [
    "generate_date_range",
    "generate_weather_data",
    "generate_consumption_data",
    "generate_production_data",
]


def _validate_datetime_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``index`` after enforcing the notebook's DatetimeIndex check.

    Args:
        index: Candidate datetime index used by the synthetic data generators.

    Returns:
        The original ``DatetimeIndex`` when validation succeeds.

    Raises:
        ValueError: If ``index`` is not a pandas ``DatetimeIndex``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("L'indice deve essere un DatetimeIndex.")

    return index


def generate_date_range(
    months: int = HISTORICAL_MONTHS,
    freq: str = FREQ,
    tz: str | None = None,
) -> pd.DatetimeIndex:
    """Generate the notebook-compatible historical hourly datetime index.

    This is the English-named equivalent of ``genera_date_range_storico``.  The
    function preserves the notebook behavior exactly: it ignores the optional
    ``tz`` argument, takes the current time in the ``Europe/Rome`` timezone,
    floors it to the current hour, subtracts ``months`` calendar months, and
    returns a pandas ``DatetimeIndex`` between the two endpoints using ``freq``.

    Args:
        months: Number of calendar months to subtract from the current hourly
            timestamp. Defaults to the notebook value of ``6``.
        freq: Frequency string passed to ``pandas.date_range``. Defaults to the
            notebook value of hourly frequency, ``"h"``.
        tz: Unused compatibility parameter retained from the notebook function.

    Returns:
        A pandas ``DatetimeIndex`` covering the historical period.
    """
    end = pd.Timestamp.now(tz=TIMEZONE).floor("h")
    start = end - pd.DateOffset(months=months)
    return pd.date_range(start=start, end=end, freq=freq)


def generate_weather_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate synthetic hourly weather data from the notebook equations.

    This is the English-named equivalent of ``genera_meteo_sintetico``.  It
    returns the same three columns as the notebook: ``temperature``,
    ``irradiance``, and ``wind_speed``.  Temperature is modeled as a daily
    sinusoid plus a yearly sinusoid and Gaussian noise; irradiance is modeled as
    a clipped daily sinusoid with Gaussian noise; wind speed is modeled as a
    clipped Gaussian perturbation around the base wind speed.

    Args:
        index: Datetime index used as the time axis for generated weather rows.

    Returns:
        A dataframe indexed by ``index`` with synthetic weather columns.

    Raises:
        ValueError: If ``index`` is not a pandas ``DatetimeIndex``.
    """
    rng = _validate_datetime_index(index)
    n = len(rng)

    hours = np.arange(n)

    daily = DAILY_TEMPERATURE_AMPLITUDE * np.sin(2 * np.pi * (rng.hour / HOURS_PER_DAY))
    day_of_year = rng.dayofyear.values
    seasonal = SEASONAL_TEMPERATURE_AMPLITUDE * np.sin(
        2 * np.pi * (day_of_year / DAYS_PER_YEAR)
    )
    temp = BASE_TEMPERATURE + daily + seasonal + np.random.normal(0, TEMPERATURE_NOISE_STD, n)
    hour_angle = (rng.hour - 6) / 12
    irradiance = np.clip(np.sin(np.pi * (rng.hour / HOURS_PER_DAY)), 0, None)
    irradiance = irradiance + np.random.normal(0, IRRADIANCE_NOISE_STD, n)
    irradiance = np.clip(irradiance, 0, 1)
    wind = BASE_WIND_SPEED + np.random.normal(0, WIND_NOISE_STD, n)
    wind = np.clip(wind, 0, None)
    df = pd.DataFrame(
        {"temperature": temp, "irradiance": irradiance, "wind_speed": wind},
        index=rng,
    )
    return df


def generate_consumption_data(index: pd.DatetimeIndex, weather_df: pd.DataFrame) -> pd.Series:
    """Generate synthetic hourly electrical consumption from notebook equations.

    This is the English-named equivalent of ``genera_consumo_sintetico``.
    Consumption is built from the notebook's daily load profile, temperature
    effect, Gaussian noise, and final clipping at zero.

    Args:
        index: Datetime index used as the time axis for generated consumption.
        weather_df: Weather dataframe containing a ``temperature`` column.

    Returns:
        A series named ``consumption`` indexed by ``index``.
    """
    rng = index
    n = len(rng)
    hour = rng.hour.values
    daily_pattern = 0.5 + 0.5 * (
        np.sin(2 * np.pi * (hour - DAILY_PATTERN_OFFSET_HOUR) / HOURS_PER_DAY) * 0.5 + 0.5
    )
    consumo = BASE_CONSUMPTION * daily_pattern
    temp = weather_df["temperature"].values
    temp_effect = TEMP_EFFECT_FACTOR * np.maximum(
        0,
        np.abs(temp - COMFORT_TEMPERATURE) - TEMPERATURE_DEADBAND,
    )
    consumo = consumo + temp_effect
    consumo = consumo + np.random.normal(0, CONSUMPTION_NOISE_STD, n)
    consumo = np.clip(consumo, 0, None)
    return pd.Series(consumo, index=rng, name="consumption")


def generate_production_data(index: pd.DatetimeIndex, weather_df: pd.DataFrame) -> pd.Series:
    """Generate synthetic photovoltaic production from notebook equations.

    This is the English-named equivalent of ``genera_produzione_sintetica``.
    Production is computed from irradiance, panel capacity, efficiency, Gaussian
    noise, and final clipping at zero.

    Args:
        index: Datetime index used as the time axis for generated production.
        weather_df: Weather dataframe containing an ``irradiance`` column.

    Returns:
        A series named ``production`` indexed by ``index``.
    """
    irr = weather_df["irradiance"].values
    prod = PANEL_CAPACITY * EFFICIENCY * irr
    prod = prod + np.random.normal(0, PRODUCTION_NOISE_STD, len(irr))
    prod = np.clip(prod, 0, None)
    return pd.Series(prod, index=index, name="production")
