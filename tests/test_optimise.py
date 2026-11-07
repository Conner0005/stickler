"""Optimiser tests on a 4-surfactant toy library (synthetic test inputs, not data)."""

import time

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from wetopt import config
from wetopt.cli import app
from wetopt.optimise import log_grid, make_problem, mc_eval, optimise, subsets
from wetopt.schema import Constraints, Spec


def toy_library() -> pd.DataFrame:
    rows = [
        (
            "A",
            "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
            "anionic",
            "alkyl sulfate",
            0.288,
            8.0,
            0.039,
            2.0e-6,
        ),
        (
            "B",
            "CCCCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
            "anionic",
            "alkyl sulfate",
            0.344,
            0.6,
            0.036,
            2.6e-6,
        ),
        (
            "C",
            "CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCO",
            "nonionic",
            "alcohol ethoxylate",
            0.539,
            0.08,
            0.034,
            2.5e-6,
        ),
        (
            "D",
            "CCCCCCCCOS(=O)(=O)[O-].[Na+]",
            "anionic",
            "alkyl sulfate",
            0.232,
            130.0,
            0.040,
            3.0e-6,
        ),
    ]
    cols = ["name", "smiles", "ionic_class", "family", "mw", "cmc", "gamma_cmc", "gamma_max"]
    df = pd.DataFrame(rows, columns=cols).assign(inchikey=lambda d: d["name"], pc20=np.nan)
    df["cmc_sigma_log10"], df["gamma_cmc_sigma"], df["gamma_max_sigma"] = 0.05, 1e-3, 0.3e-6
    for c in ["cmc", "gamma_cmc", "gamma_max", "pc20"]:
        df[f"{c}_measured"] = c != "pc20"
    return df


def spec(*bands) -> Spec:
    return Spec.model_validate({"bands": list(bands)})


DST100 = {"metric": "dst", "surface_age_ms": 100, "lo": 0.0, "hi": 0.060}
EQ40 = {"metric": "gamma_eq", "lo": 0.0, "hi": 0.040}


def brute_force(lib, sp, cons, levels=40):
    """Minimal k and minimal feasible loading on a fine log grid, Monte Carlo classified."""
    p, _ = make_problem(lib, sp, cons, config.SIGMA_MODEL)
    for k in range(1, cons.max_k + 1):
        S, G = subsets(p, k), log_grid(k, cons.max_total_wv, levels)
        idx, wv = np.repeat(S, len(G), 0), np.tile(G, (len(S), 1))
        _, pess, _ = mc_eval(p, idx, wv)
        ok = pess.min(1) >= 0
        if ok.any():
            return k, wv[ok].sum(1).min()
    return None, None


def test_matches_brute_force():
    lib, sp, cons = toy_library(), spec(DST100, EQ40), Constraints(max_k=2, max_total_wv=0.1)
    k_bf, load_bf = brute_force(lib, sp, cons)
    o = optimise(lib, sp, cons)
    assert k_bf == 2 and o.result is not None and o.result.k == k_bf
    load = o.feasible.iloc[0]["total_wv"]
    assert 0.75 * load_bf <= load <= 1.05 * load_bf
    assert min(o.result.margins) >= 0


def test_certificate_when_k1_infeasible():
    # Every single fails even on its optimistic side, so k = 1 is certified infeasible.
    tough = {"metric": "dst", "surface_age_ms": 20, "lo": 0.0, "hi": 0.040}
    o = optimise(toy_library(), spec(tough), Constraints(max_k=2, max_total_wv=0.01))
    assert o.result is None and o.certified
    assert "No single surfactant meets DST(20 ms) ≤ 40 mN/m" in o.certificate
    # A k = 1 winner is certified against water alone.
    easy = {"metric": "dst", "surface_age_ms": 200, "lo": 0.0, "hi": 0.065}
    o = optimise(toy_library(), spec(easy), Constraints(max_k=2, max_total_wv=0.1))
    assert o.result.k == 1 and o.certified and "water alone fails" in o.certificate


def test_cli_optimise(tmp_path):
    lib_path, spec_path = tmp_path / "toy.parquet", tmp_path / "toy.yaml"
    toy_library().to_parquet(lib_path)
    doc = {"spec": {"bands": [DST100, EQ40]}, "constraints": {"max_k": 2, "max_total_wv": 0.1}}
    spec_path.write_text(yaml.safe_dump(doc))
    args = ["optimise", str(spec_path), "--library", str(lib_path), "--out-dir", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Winner (k = 2)" in result.output and (tmp_path / "toy.json").exists()


@pytest.mark.slow
def test_runtime_40_candidates_k3():
    lib = pd.read_parquet(config.LIBRARY_PATH)
    # Cereal band at the original 0.1 % w/v cap: infeasible, so every k is searched.
    dst200 = {"metric": "dst", "surface_age_ms": 200, "lo": 0.0, "hi": 0.050}
    sp, cons = spec(dst200), Constraints(max_k=3, max_total_wv=0.1)
    t0 = time.perf_counter()
    o = optimise(lib, sp, cons)
    runtime = time.perf_counter() - t0
    print(f"\n40 candidates, k ≤ 3, full search: {runtime:.1f} s; levels {o.levels}")
    assert [lv["k"] for lv in o.levels] == [1, 2, 3]
    assert runtime < 60
