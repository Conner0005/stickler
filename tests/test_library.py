"""Smoke tests: library build from SurfPro raw data, selection ranges, and the CLI."""

import pytest
from typer.testing import CliRunner

from wetopt import config
from wetopt.cli import app
from wetopt.library import build_library, load_surfpro, select_candidates

pytestmark = pytest.mark.skipif(
    not (config.SURFPRO_DIR / "surfpro_literature.csv").exists(),
    reason="SurfPro raw data not in data/raw/surfpro (see docs/data.md)",
)


@pytest.fixture(scope="module")
def sel():
    return select_candidates(build_library(*load_surfpro()))


def test_cmc_in_range(sel):
    cmc_molar = sel["cmc"] * 1e-3  # mol/m³ -> mol/L
    assert cmc_molar.between(1e-7, 1).all()


def test_gamma_cmc_in_range(sel):
    assert sel["gamma_cmc"].between(15e-3, 72e-3).all()


def test_no_nan_in_selection(sel):
    assert 30 <= len(sel) <= 50
    assert not sel.isna().any().any()


def test_cli_library(tmp_path):
    out = tmp_path / "library.parquet"
    result = CliRunner().invoke(app, ["library", "--path", str(out), "--rebuild"])
    assert result.exit_code == 0, result.output
    assert out.exists() and "candidate library" in result.output
