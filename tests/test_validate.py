"""pC20 check: exact on a surfactant whose C20 is placed by the model; runs on the library."""

import numpy as np
import pandas as pd

from wetopt import config
from wetopt.physics import T0, R, gamma_water, szyszkowski_K
from wetopt.validate import INCONSISTENT, data_quality, pc20_check


def test_pc20_check_exact_and_library():
    cmc, gcmc, gmax, n = 8.0, 0.039, 2.0e-6, 2  # synthetic test inputs
    K = szyszkowski_K(cmc, gcmc, gmax, n, gamma_water())
    c20 = np.expm1(0.020 / (n * R * T0 * gmax)) / K  # γ0 − 20 mN/m on the model isotherm
    row = {
        "name": "toy",
        "ionic_class": "anionic",
        "cmc": cmc,
        "gamma_cmc": gcmc,
        "gamma_max": gmax,
        "pc20": 3.0 - np.log10(c20),
        **{f"{c}_measured": True for c in ["cmc", "gamma_cmc", "gamma_max", "pc20"]},
    }
    check = pc20_check(pd.DataFrame([row]))
    assert check.n == 1 and check.rmse < 1e-9
    lib_check = pc20_check(pd.read_parquet(config.LIBRARY_PATH))
    assert lib_check.n >= 10 and 0 < lib_check.rmse < 0.02


def test_data_quality_flags_outliers_not_unchecked():
    lib = pd.read_parquet(config.LIBRARY_PATH)
    qc = data_quality(lib)
    status = dict(zip(qc["name"], qc["status"], strict=True))
    assert status["C14H29NaO5S #1"] == INCONSISTENT  # pC20-check error +16 mN/m
    unchecked = ~lib["pc20_measured"]
    assert (qc.loc[unchecked.to_numpy(), "status"] == "unchecked").all()
    assert set(qc["status"]) <= {"consistent", INCONSISTENT, "unchecked"}
