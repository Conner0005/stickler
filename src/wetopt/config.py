"""Paths and modelling assumptions. Assumptions here are not data; data lives in data/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SURFPRO_DIR = ROOT / "data" / "raw" / "surfpro" / "data"
LIBRARY_PATH = ROOT / "data" / "library.parquet"

T_RANGE_C = (20.0, 30.0)  # keep SurfPro entries measured in this range, °C
N_CANDIDATES = 40  # demo candidates to select (30–50)

# Assumed 1σ for measured values (inter-lab reproducibility), in SurfPro units:
# pCMC and pC20 in log10 units, AW_ST_CMC in mN/m, Gamma_max in µmol/m².
MEASURED_SIGMA = {"pCMC": 0.05, "AW_ST_CMC": 1.0, "Gamma_max": 0.3, "pC20": 0.1}

# σ for imputed values: test-set RMSE of the AttentiveFP_all64d ensemble that produced
# surfpro_imputed.csv (Hödl et al., Digital Discovery 2025, 4, 1176, Table 3; values supplied
# by the user from the paper), same units as MEASURED_SIGMA. None would fall back to the
# per-entry ensemble std in surfpro_imputed.csv.
SURFPRO_TEST_RMSE: dict[str, float] | None = {
    "pCMC": 0.358,
    "AW_ST_CMC": 3.52,
    "Gamma_max": 0.84,
    "pC20": 0.39,
}

# ASSUMPTION: model-form error of the DST/γ_eq predictions, added in quadrature to the
# Monte Carlo std (N/m).
SIGMA_MODEL = 2e-3

# Gibbs prefactor n in γ = γ0 − nRTΓ∞·ln(1 + Kc), by schema ionic class:
# 1 for neutral species, 2 for 1:1 ionic surfactants without added salt.
N_GIBBS = {"nonionic": 1, "zwitterionic": 1, "anionic": 2, "cationic": 2}

# ASSUMPTION: monomer diffusion coefficient (m²/s) used when Le Bas has no increment for an
# atom (e.g. Si), so Wilke–Chang cannot be applied. Order of magnitude of Wilke–Chang values
# for surfactants of molar mass ~0.5–2 kg/mol; not measured data.
D_CLASS_DEFAULT = {"siloxane": 3e-10}
D_DEFAULT = 4e-10
