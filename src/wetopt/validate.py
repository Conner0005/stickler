"""Model checks against SurfPro values that the model was not fitted to."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from wetopt import config
from wetopt.physics import T0, equilibrium_surface_tension, gamma_water, szyszkowski_K

MEASURED = ["cmc_measured", "gamma_cmc_measured", "gamma_max_measured", "pc20_measured"]


class PC20Check(NamedTuple):
    n: int
    rmse: float  # N/m
    bias: float  # N/m
    table: pd.DataFrame


def pc20_check(lib: pd.DataFrame, T: float = T0) -> PC20Check:
    """Out-of-sample check of the single-surfactant isotherm at C20.

    K comes from (CMC, γ_CMC, Γ_max) only; C20 = 10^(−pC20) mol/L is, by definition, where
    γ = γ0 − 20 mN/m. Residual r = γ_model(C20) − (γ0 − 20 mN/m), with
    γ_model = γ0 − n·R·T·Γ∞·ln(1 + K·C20) (capped at γ_CMC above the CMC). Uses candidates
    whose four properties are all measured. Caveat: some SurfPro pC20 values are 'calculated'
    from the authors' isotherm fits, which flatters the check.
    Ref: Rosen & Kunjappu, Surfactants and Interfacial Phenomena, 4th ed. (2012), pC20;
    Szyszkowski, Z. Phys. Chem. 64, 385 (1908).
    """
    sub = lib[lib[MEASURED].all(axis=1)]
    if sub.empty:
        return PC20Check(0, float("nan"), float("nan"), pd.DataFrame())
    resid = pc20_errors(sub, T)
    c20_over_cmc = 10.0 ** (3.0 - sub["pc20"].to_numpy()) / sub["cmc"].to_numpy()
    table = pd.DataFrame({"name": sub["name"], "c20_over_cmc": c20_over_cmc, "residual": resid})
    return PC20Check(len(sub), float(np.sqrt(np.mean(resid**2))), float(resid.mean()), table)


def pc20_errors(sub: pd.DataFrame, T: float = T0) -> np.ndarray:
    """pC20-check error per row, N/m: e = γ_model(C20) − (γ0 − 20 mN/m) (see pc20_check)."""
    g0 = gamma_water(T)
    n = sub["ionic_class"].map(config.N_GIBBS).to_numpy(float)
    cmc, gmax = sub["cmc"].to_numpy(), sub["gamma_max"].to_numpy()
    K = szyszkowski_K(cmc, sub["gamma_cmc"].to_numpy(), gmax, n, g0, T)
    c20 = 10.0 ** (3.0 - sub["pc20"].to_numpy())  # mol/L -> mol/m³
    col = (slice(None), None)
    gamma = equilibrium_surface_tension(c20[col], cmc[col], K[col], gmax[col], n[col], g0, T=T)
    return gamma - (g0 - 0.020)


INCONSISTENT = "inconsistent source data"


def sigma_model(lib: pd.DataFrame) -> float:
    """σ_model (N/m) = max(config.SIGMA_MODEL, pC20-check RMSE), or the config value if none."""
    check = pc20_check(lib)
    return max(config.SIGMA_MODEL, check.rmse) if check.n else config.SIGMA_MODEL


def data_quality(lib: pd.DataFrame, limit: float | None = None, T: float = T0) -> pd.DataFrame:
    """Per-candidate QC status: 'consistent', INCONSISTENT or 'unchecked', with a reason.

    For candidates with a measured pC20, e_i = pc20_errors (the measured pC20 against the
    candidate's own CMC, γ_CMC and Γ_max under the isotherm). |e_i| > limit → INCONSISTENT,
    where limit defaults to config.QC_SIGMA_MULT · σ_model. Candidates without a measured pC20
    are 'unchecked' and are not excluded.
    Ref: see pc20_check.
    """
    if limit is None:
        limit = config.QC_SIGMA_MULT * sigma_model(lib)
    out = pd.DataFrame(
        {
            "name": lib["name"],
            "status": "unchecked",
            "pc20_error": np.nan,
            "reason": "no measured pC20",
        }
    )
    checked = lib["pc20_measured"].to_numpy(bool)
    if not checked.any():
        return out
    e = pc20_errors(lib[checked], T)
    bad = np.abs(e) > limit
    lim = f"{config.QC_SIGMA_MULT:g}·σ_model = {limit * 1e3:.1f} mN/m"
    out.loc[checked, "pc20_error"] = e
    out.loc[checked, "status"] = np.where(bad, INCONSISTENT, "consistent")
    out.loc[checked, "reason"] = [
        f"pC20-check error {x * 1e3:+.1f} mN/m, |error| {'>' if b else '≤'} {lim}"
        for x, b in zip(e, bad, strict=True)
    ]
    return out
