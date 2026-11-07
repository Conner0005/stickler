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

# Test-set RMSE of the SurfPro ensemble that produced surfpro_imputed.csv
# (Hödl et al., Digital Discovery 2025, 4, 1176), same units as MEASURED_SIGMA.
# PENDING: the paper could not be fetched (publisher blocks automated access).
# While None, imputed σ falls back to the per-entry ensemble std in surfpro_imputed.csv.
SURFPRO_TEST_RMSE: dict[str, float] | None = None
