"""Data-quality checks for 1-minute PV monitoring data.

Every check returns a boolean Series/DataFrame where True means "bad", so the
flags can be combined, counted and reported per check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

SENTINEL = -7999.0  # PVDAQ "no data" value

# Physically plausible bounds for each channel (generous on purpose: they catch
# broken readings, not unusual weather).
BOUNDS = {
    "ac_power": (-50, 1300),   # W, 1 kW array behind an 1800 W inverter
    "dc_power": (-50, 1300),   # W
    "poa": (-20, 1500),        # W/m2
    "t_amb": (-35, 45),        # degC, Golden CO
    "t_mod1": (-40, 85),
    "t_mod2": (-40, 85),
    "t_mod3": (-40, 85),
}


def site_location(meta: dict) -> pvlib.location.Location:
    s = meta["Site"]
    # PVDAQ timestamps for NREL systems are local standard time (no DST).
    return pvlib.location.Location(float(s["latitude"]), float(s["longitude"]),
                                   "Etc/GMT+7", float(s["elevation"]))


def solar_context(index: pd.DatetimeIndex, loc, tilt: float, azimuth: float) -> pd.DataFrame:
    """Solar elevation and modelled clear-sky plane-of-array irradiance."""
    idx = index.tz_localize(loc.tz)
    sp = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx, solar_position=sp)
    poa_cs = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, sp.apparent_zenith, sp.azimuth, cs.dni, cs.ghi, cs.dhi)["poa_global"]
    out = pd.DataFrame({"elevation": sp.apparent_elevation.values,
                        "poa_clearsky": poa_cs.fillna(0).values}, index=index)
    return out


# --- individual checks -----------------------------------------------------

def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def flag_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    return df.eq(SENTINEL)


def flag_range(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, (lo, hi) in BOUNDS.items():
        if col in df:
            v = df[col]
            out[col] = v.notna() & ~v.eq(SENTINEL) & ((v < lo) | (v > hi))
    return out


def flag_stuck(s: pd.Series, window: int = 30, daylight: pd.Series | None = None,
               tol: float = 1e-6) -> pd.Series:
    """Readings that do not change at all over `window` consecutive minutes.

    Only evaluated in daylight when a daylight mask is given, because flat
    readings at night (e.g. zero power) are expected.
    """
    rng = s.rolling(window, min_periods=window).max() - s.rolling(window, min_periods=window).min()
    stuck = rng.lt(tol)
    # mark the whole window, not just its last sample
    stuck = stuck[::-1].rolling(window, min_periods=1).max()[::-1].astype(bool)
    if daylight is not None:
        stuck &= daylight
    return stuck & s.notna()


def flag_module_temp_disagreement(df: pd.DataFrame, cols=("t_mod1", "t_mod2", "t_mod3"),
                                  max_dev: float = 5.0) -> pd.DataFrame:
    """A back-of-module sensor that sits > max_dev degC away from the median of the three."""
    t = df[list(cols)].where(~df[list(cols)].eq(SENTINEL))
    med = t.median(axis=1)
    return t.sub(med, axis=0).abs().gt(max_dev)


def flag_irradiance_vs_clearsky(poa: pd.Series, poa_cs: pd.Series, elevation: pd.Series,
                                factor: float = 1.35, margin: float = 100.0) -> pd.Series:
    """POA well above what a clear sky can deliver (cloud-edge enhancement allowed for)."""
    return (elevation > 5) & (poa > factor * poa_cs + margin)


def flag_power_without_sun(ac: pd.Series, elevation: pd.Series, threshold: float = 20.0) -> pd.Series:
    """Meaningful AC output while the sun is below the horizon."""
    return (elevation < -2) & (ac > threshold)


def flag_no_output_in_sun(ac: pd.Series, poa: pd.Series, poa_min: float = 200.0,
                          ac_max: float = 20.0) -> pd.Series:
    """Inverter producing (almost) nothing while the irradiance sensor sees real sun.

    This is an availability problem (outage, tripped breaker, disconnected
    string), not a performance problem, so these minutes are excluded from
    performance-ratio and degradation analysis and reported separately.
    """
    return (poa > poa_min) & (ac < ac_max)


# --- diagnostics -------------------------------------------------------------

def night_offsets(df: pd.DataFrame, elevation: pd.Series) -> pd.DataFrame:
    """Median night-time reading of each power/irradiance channel, per year.

    A perfect system reads 0 at night. A constant non-zero value means a sensor
    or meter offset that should be removed before computing energy.
    """
    night = elevation < -10
    cols = [c for c in ("ac_power", "dc_power", "poa") if c in df]
    clean = df[cols].where(~df[cols].eq(SENTINEL))
    return clean[night].groupby(clean[night].index.year).median()


def daily_clock_offset(signal: pd.Series, reference: pd.Series, max_shift: int = 90) -> pd.Series:
    """Per-day time shift (minutes) that best aligns `signal` with `reference`.

    Uses cross-correlation over +/- max_shift minutes on a complete 1-minute day.
    Positive = the data are recorded later than the sun says they should be.
    """
    shifts = np.arange(-max_shift, max_shift + 1)
    out = {}
    for day, g in signal.groupby(signal.index.date):
        r = reference.loc[g.index]
        if len(g) < 1380 or g.isna().mean() > 0.02 or r.max() <= 0:
            continue
        x = g.fillna(0).to_numpy(float)
        y = r.to_numpy(float)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        corr = [np.dot(np.roll(y, k), x) for k in shifts]
        out[pd.Timestamp(day)] = shifts[int(np.argmax(corr))]
    return pd.Series(out, name="clock_offset_min", dtype=float)


def clear_days(poa: pd.Series, poa_cs: pd.Series, min_corr: float = 0.99,
               ratio=(0.9, 1.15)) -> pd.Index:
    """Days whose measured POA profile closely follows the clear-sky model."""
    d = pd.DataFrame({"m": poa.clip(lower=0), "c": poa_cs})
    days = []
    for day, g in d.groupby(d.index.date):
        g = g.dropna()
        if len(g) < 1380 or g.c.sum() <= 0:
            continue
        r = g.m.sum() / g.c.sum()
        if ratio[0] <= r <= ratio[1] and g.m.corr(g.c) >= min_corr:
            days.append(pd.Timestamp(day))
    return pd.Index(days)


# --- pipeline ----------------------------------------------------------------

def run_qc(df: pd.DataFrame, solar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply every check. Returns (clean data with bad values set to NaN, per-check flag counts)."""
    daylight = solar.elevation > 5
    flags = {
        "sentinel (-7999)": flag_sentinel(df),
        "out of range": flag_range(df),
    }
    stuck = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        stuck[c] = flag_stuck(df[c].where(~df[c].eq(SENTINEL)), daylight=daylight)
    flags["stuck / flat-lined"] = stuck

    tdis = pd.DataFrame(False, index=df.index, columns=df.columns)
    tdis[["t_mod1", "t_mod2", "t_mod3"]] = flag_module_temp_disagreement(df)
    flags["module temp disagreement"] = tdis

    irr = pd.DataFrame(False, index=df.index, columns=df.columns)
    irr["poa"] = flag_irradiance_vs_clearsky(df.poa, solar.poa_clearsky, solar.elevation)
    flags["POA above clear-sky limit"] = irr

    pw = pd.DataFrame(False, index=df.index, columns=df.columns)
    pw["ac_power"] = flag_power_without_sun(df.ac_power, solar.elevation)
    flags["AC power at night"] = pw

    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    poa_ok = df.poa.where(~(flags["sentinel (-7999)"].poa | flags["out of range"].poa))
    out["ac_power"] = flag_no_output_in_sun(df.ac_power, poa_ok)
    out["dc_power"] = out["ac_power"]
    flags["no output in sunlight (outage)"] = out

    bad = pd.DataFrame(False, index=df.index, columns=df.columns)
    for f in flags.values():
        bad |= f
    clean = df.mask(bad)
    temps = clean[["t_mod1", "t_mod2", "t_mod3"]]
    clean["t_mod"] = temps.median(axis=1)  # robust module temperature from 3 sensors
    counts = pd.DataFrame({k: v.sum() for k, v in flags.items()}).T
    counts.loc["missing (no row value)"] = df.isna().sum()
    return clean, counts


def daily_completeness(clean: pd.DataFrame, daylight: pd.Series, cols=("ac_power", "poa", "t_mod")) -> pd.DataFrame:
    """Share of daylight minutes with a usable value, per day and channel."""
    present = pd.DataFrame({c: clean[c].notna() for c in cols if c in clean})
    present = present[daylight]
    return present.groupby(present.index.date).mean()
