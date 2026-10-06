"""Performance ratio and degradation-rate estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_power(poa: pd.Series, t_mod: pd.Series, p_stc: float, gamma: float) -> pd.Series:
    """DC-rated power scaled by irradiance and corrected to 25 degC module temperature.

    p_stc: nameplate power (W) at 1000 W/m2 and 25 degC
    gamma: power temperature coefficient (1/degC), e.g. -0.0030 for -0.30 %/degC
    """
    return p_stc * (poa / 1000.0) * (1.0 + gamma * (t_mod - 25.0))


def daily_pr(clean: pd.DataFrame, elevation: pd.Series, p_stc: float, gamma: float,
             poa_min: float = 200.0, min_minutes: int = 120) -> pd.DataFrame:
    """Temperature-corrected daily performance ratio from 1-minute data.

    Only minutes with good AC power, POA and module temperature, the sun above
    10 degrees and POA >= poa_min are used, which removes low-light and
    high-incidence-angle effects that are not degradation.
    """
    ok = (clean[["ac_power", "poa", "t_mod"]].notna().all(axis=1)
          & (elevation > 10) & (clean.poa >= poa_min))
    d = clean.loc[ok, ["ac_power", "poa", "t_mod"]].copy()
    d["p_exp"] = expected_power(d.poa, d.t_mod, p_stc, gamma)
    g = d.groupby(d.index.normalize())
    out = pd.DataFrame({
        "pr": g.ac_power.sum() / g.p_exp.sum(),
        "pr_uncorrected": g.ac_power.sum() / (p_stc * g.poa.sum() / 1000.0),
        "minutes": g.size(),
        "poa_kwh_m2": g.poa.sum() / 60 / 1000,
        "t_mod_mean": g.t_mod.mean(),
    })
    out = out[out.minutes >= min_minutes]
    return out


def remove_outliers(pr: pd.Series, window: int = 61, k: float = 3.5) -> pd.Series:
    """Drop days far from a rolling median (snow, partial shading, short outages).

    Uses a robust z-score based on the rolling median absolute deviation.
    """
    med = pr.rolling(window, center=True, min_periods=15).median()
    mad = (pr - med).abs().rolling(window, center=True, min_periods=15).median()
    z = (pr - med) / (1.4826 * mad)
    return pr[z.abs() <= k]


def yoy_degradation(pr: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Year-on-year degradation rate (%/year), the method used by NREL's RdTools.

    Every day is compared with the same calendar day one year later; the
    median of all those one-year changes is the degradation rate. Seasonal
    effects cancel because each pair shares the same season. The confidence
    interval comes from bootstrapping the yearly changes.
    """
    pr = pr.dropna()
    later = pr.copy()
    later.index = later.index - pd.Timedelta(days=365)
    pairs = pd.concat([pr.rename("y0"), later.rename("y1")], axis=1, join="inner")
    rates = (pairs.y1 / pairs.y0 - 1.0) * 100.0
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(rates.values, (n_boot, len(rates)), replace=True), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rate": float(np.median(rates)), "ci_low": float(lo), "ci_high": float(hi),
            "n_pairs": int(len(rates)), "rates": rates}


def linear_trend(pr: pd.Series) -> dict:
    """Ordinary least squares on monthly medians, as a simple cross-check (%/year)."""
    m = pr.resample("MS").median().dropna()
    t = (m.index - m.index[0]).days / 365.25
    slope, intercept = np.polyfit(t, m.values, 1)
    return {"rate": float(slope / intercept * 100), "monthly": m}
