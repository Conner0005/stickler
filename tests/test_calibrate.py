"""GP calibration: no data leaves results unchanged; the demo recovers its −3 mN/m offset."""

import numpy as np
import pandas as pd
import yaml

from wetopt import config
from wetopt.calibrate import COLUMNS, features, fit, read_measurements
from wetopt.scenario import optimise_scenario
from wetopt.schema import Scenario
from wetopt.validate import sigma_model

LIB = pd.read_parquet(config.LIBRARY_PATH)
DEMO = config.ROOT / "examples" / "demo_measurements.csv"


def test_no_data_leaves_results_unchanged():
    scn = Scenario.model_validate(
        yaml.safe_load((config.ROOT / "examples" / "cereal.yaml").read_text())
    )
    cal = fit(pd.DataFrame(columns=COLUMNS), LIB, sigma_model(LIB))
    assert cal is None
    base, same = optimise_scenario(LIB, scn), optimise_scenario(LIB, scn, calibration=cal)
    assert same.result.model_dump() == base.result.model_dump()
    assert same.feasible.iloc[0]["total_wv"] == base.feasible.iloc[0]["total_wv"]


def test_demo_recovers_offset():
    cal = fit(read_measurements(DEMO.read_text()), LIB, sigma_model(LIB))
    assert cal.n == 8 and cal.skipped == 0 and cal.rmse_after < cal.rmse_before
    mean, std = cal.predict(features(np.array([0.16]), np.array([200.0])))
    assert abs(mean[0] - (-3.0)) < 1.0 and std[0] > 0
