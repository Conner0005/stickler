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
    g0 = gamma_water(T)
    n = sub["ionic_class"].map(config.N_GIBBS).to_numpy(float)
    cmc, gmax = sub["cmc"].to_numpy(), sub["gamma_max"].to_numpy()
    K = szyszkowski_K(cmc, sub["gamma_cmc"].to_numpy(), gmax, n, g0, T)
    c20 = 10.0 ** (3.0 - sub["pc20"].to_numpy())  # mol/L -> mol/m³
    col = (slice(None), None)
    gamma = equilibrium_surface_tension(c20[col], cmc[col], K[col], gmax[col], n[col], g0, T=T)
    resid = gamma - (g0 - 0.020)
    table = pd.DataFrame({"name": sub["name"], "c20_over_cmc": c20 / cmc, "residual": resid})
    return PC20Check(len(sub), float(np.sqrt(np.mean(resid**2))), float(resid.mean()), table)
