"""Gaussian-process correction of the DST upper bound from lab measurements. No physics changes.

Residual r = measured DST − dst_high (model, Monte Carlo mean), fitted against
X = [log10 total surfactant (% w/v), log10 surface age (ms)] with a fixed kernel
ConstantKernel(σ_model²) × RBF(1.0) + WhiteKernel(0.25) (mN/m², no hyperparameter fitting).
Calibrated DST = dst_high + GP mean; σ² = σ_MC² + GP std². A calibrated value is an estimate,
not a pessimistic bound.
Ref: Rasmussen & Williams, Gaussian Processes for Machine Learning (2006), ch. 2.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from wetopt.physics import DSTStats, SurfactantArrays, monte_carlo_dst

COLUMNS = ["surfactant_1", "conc_1_wv", "surfactant_2", "conc_2_wv", "surface_age_ms", "dst_mN_m"]
WV_TO_KG_M3 = 10.0


def read_measurements(text: str) -> pd.DataFrame:
    """CSV text → DataFrame. Whole lines starting with '#' are notes; '#' inside a name (such as
    'C14H28O6 #1') is kept, so pandas' comment option is not used."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return pd.read_csv(io.StringIO(body), dtype={"surfactant_1": str, "surfactant_2": str})


def features(total_wv: np.ndarray, age_ms: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log10(total_wv), np.log10(age_ms)])


def clean(df: pd.DataFrame, lib: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Valid rows (known surfactants, positive numbers) and how many rows were skipped."""
    df = df.reindex(columns=COLUMNS)
    names = set(lib["name"])
    numeric = ["conc_1_wv", "conc_2_wv", "surface_age_ms", "dst_mN_m"]
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in numeric}

    def text(col: str) -> pd.Series:
        return df[col].map(lambda v: "" if pd.isna(v) else str(v).strip())

    s1, s2 = text("surfactant_1"), text("surfactant_2")
    c2 = num["conc_2_wv"].fillna(0.0)
    ok = (
        s1.isin(names)
        & (num["conc_1_wv"] > 0)
        & (num["surface_age_ms"] > 0)
        & num["dst_mN_m"].between(15, 75)
        & ((s2 == "") & (c2 == 0) | s2.isin(names) & (c2 > 0))
    )
    out = pd.DataFrame(
        {
            "surfactant_1": s1,
            "conc_1_wv": num["conc_1_wv"],
            "surfactant_2": s2,
            "conc_2_wv": c2,
            "surface_age_ms": num["surface_age_ms"],
            "dst_mN_m": num["dst_mN_m"],
        }
    )[ok.to_numpy()]
    return out.reset_index(drop=True), int((~ok).sum())


def model_dst_high(rows: pd.DataFrame, lib: pd.DataFrame, sigma_model: float) -> np.ndarray:
    """Monte Carlo mean dst_high (mN/m) for each measurement row (same draws as the optimiser)."""
    props = SurfactantArrays.from_library(lib.reset_index(drop=True))
    index = {n: i for i, n in enumerate(lib["name"])}
    out = np.empty(len(rows))
    for j, r in enumerate(rows.itertuples()):
        i1 = index[r.surfactant_1]
        i2 = index.get(r.surfactant_2, i1)
        idx = np.array([[i1, i2]])
        wv = np.array([[r.conc_1_wv, r.conc_2_wv if r.surfactant_2 else 0.0]])
        conc = WV_TO_KG_M3 * wv / lib["mw"].to_numpy()[idx]
        s = monte_carlo_dst([r.surface_age_ms / 1e3], idx, conc, props, sigma_model=sigma_model)
        out[j] = s.high_mean[0, 0] * 1e3
    return out


@dataclass
class Calibration:
    gp: GaussianProcessRegressor
    n: int
    skipped: int
    rmse_before: float  # mN/m, measured vs dst_high
    rmse_after: float  # mN/m, measured vs calibrated (on the same data)
    sigma_model: float  # N/m

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP mean and std of the residual (mN/m) at X = features(total_wv, age_ms)."""
        return self.gp.predict(X, return_std=True)

    def apply(self, ages_s: np.ndarray, wv: np.ndarray, stats: DSTStats) -> DSTStats:
        """Replace both DST bounds at finite ages by the calibrated estimate dst_high + GP mean,
        with σ² = σ_MC² + GP std² (σ_model is replaced by the GP's own uncertainty)."""
        cols = np.flatnonzero(np.isfinite(ages_s))
        if not len(cols):
            return stats
        total = wv.sum(1)
        X = features(np.repeat(total, len(cols)), np.tile(ages_s[cols] * 1e3, len(total)))
        mean, std = self.predict(X)
        mean, std = (
            mean.reshape(len(total), len(cols)) / 1e3,
            std.reshape(len(total), len(cols)) / 1e3,
        )
        hm, hs = stats.high_mean.copy(), stats.high_std.copy()
        var_mc = np.maximum(hs[:, cols] ** 2 - self.sigma_model**2, 0.0)
        hm[:, cols] = hm[:, cols] + mean
        hs[:, cols] = np.sqrt(var_mc + std**2)
        lm, ls = stats.low_mean.copy(), stats.low_std.copy()
        lm[:, cols], ls[:, cols] = hm[:, cols], hs[:, cols]
        return DSTStats(lm, ls, hm, hs)


def fit(df: pd.DataFrame, lib: pd.DataFrame, sigma_model: float) -> Calibration | None:
    """Fit the fixed-kernel GP to measured − dst_high; None if no valid rows."""
    rows, skipped = clean(df, lib)
    if rows.empty:
        return None
    resid = rows["dst_mN_m"].to_numpy() - model_dst_high(rows, lib, sigma_model)
    X = features(
        (rows["conc_1_wv"] + rows["conc_2_wv"]).to_numpy(), rows["surface_age_ms"].to_numpy()
    )
    s2 = (sigma_model * 1e3) ** 2
    kernel = ConstantKernel(s2, "fixed") * RBF(1.0, "fixed") + WhiteKernel(0.25, "fixed")
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, resid)
    after = resid - gp.predict(X)
    return Calibration(
        gp,
        len(rows),
        skipped,
        float(np.sqrt(np.mean(resid**2))),
        float(np.sqrt(np.mean(after**2))),
        sigma_model,
    )
