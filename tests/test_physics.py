"""Physics checks. Surfactant property values below are synthetic test inputs, not data."""

import numpy as np
import pandas as pd
import pytest

from wetopt import config
from wetopt.physics import (
    SurfactantArrays,
    diffusion_coefficient,
    dynamic_surface_tension,
    equilibrium_surface_tension,
    free_monomer,
    gamma_water,
    mixed_cmc,
    monte_carlo_gamma,
    szyszkowski_K,
)

G0 = gamma_water()
# Two synthetic surfactants: an ionic one (n = 2) and a more surface-active nonionic (n = 1).
CMC = np.array([8.0, 0.1])  # mol/m³
GCMC = np.array([0.039, 0.032])  # N/m
GMAX = np.array([3.0e-6, 2.5e-6])  # mol/m²
N = np.array([2.0, 1.0])
D = np.array([5e-10, 3e-10])  # m²/s
K = szyszkowski_K(CMC, GCMC, GMAX, N, G0)
T_AGES = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, np.inf])  # s
TOL = 1e-12


def _gammas(conc, cmc, k_, gmax, n, d):
    dyn = dynamic_surface_tension(T_AGES, conc, cmc, k_, gmax, n, d, G0)
    eq = equilibrium_surface_tension(conc, cmc, k_, gmax, n, G0)
    return dyn, eq


@pytest.mark.parametrize("i", [0, 1])
def test_single_monotone_and_bounded(i):
    s = slice(i, i + 1)
    conc = CMC[i] * np.logspace(-3, 2, 40)[:, None]
    dyn, eq = _gammas(conc, CMC[s], K[s], GMAX[s], N[s], D[s])
    assert (np.diff(dyn, axis=-1) <= TOL).all()  # non-increasing in t
    assert (np.diff(dyn, axis=0) <= TOL).all()  # non-increasing in concentration
    assert ((dyn >= eq[:, None] - TOL) & (dyn <= G0 + TOL)).all()
    assert np.allclose(dyn[:, 0], G0) and np.allclose(dyn[:, -1], eq)


def test_binary_monotone_and_bounded():
    alpha = np.array([0.7, 0.3])
    cmc_mix = mixed_cmc(alpha, CMC)
    conc = cmc_mix * np.logspace(-3, 2, 40)[:, None] * alpha
    dyn, eq = _gammas(conc, CMC, K, GMAX, N, D)
    assert (np.diff(dyn, axis=-1) <= TOL).all()
    assert ((dyn >= eq[:, None] - TOL) & (dyn <= G0 + TOL)).all()
    # Monotone in concentration up to CMC_mix; above it, micelles can deplete the more
    # surface-active monomer (documented in docs/physics.md).
    below = conc.sum(-1) <= cmc_mix
    assert (np.diff(dyn[below], axis=0) <= TOL).all()


def test_single_at_cmc_gives_gamma_cmc():
    for i in range(2):
        s = slice(i, i + 1)
        conc = np.array([[CMC[i]], [3 * CMC[i]]])
        eq = equilibrium_surface_tension(conc, CMC[s], K[s], GMAX[s], N[s], G0)
        assert np.allclose(eq, GCMC[i], rtol=1e-9)


def test_beta_zero_reproduces_clint():
    rng = np.random.default_rng(1)
    conc, cmc = rng.uniform(0.1, 5, (50, 3)), rng.uniform(0.01, 10, (50, 3))
    alpha = conc / conc.sum(-1, keepdims=True)
    clint = 1 / (alpha / cmc).sum(-1)
    zero = np.zeros((3, 3))
    assert np.allclose(mixed_cmc(conc, cmc), clint)
    assert np.allclose(mixed_cmc(conc, cmc, zero), clint)
    assert np.allclose(free_monomer(conc, cmc, zero), free_monomer(conc, cmc))
    # Rubingh, symmetric binary: X = ½, f = exp(β/4) ⇒ CMC_mix = CMC·exp(β/4).
    beta = np.array([[0.0, -4.0], [-4.0, 0.0]])
    assert np.isclose(mixed_cmc([1.0, 1.0], [2.0, 2.0], beta), 2.0 * np.exp(-1.0))


def _props(scale):
    return SurfactantArrays(
        cmc=CMC,
        cmc_sigma_log10=scale * np.array([0.05, 0.3]),
        gamma_cmc=GCMC,
        gamma_cmc_sigma=scale * np.array([1e-3, 3e-3]),
        gamma_max=GMAX,
        gamma_max_sigma=scale * np.array([0.3e-6, 0.4e-6]),
        n=N,
        D=D,
        D_assumed=np.array([False, False]),
    )


def test_monte_carlo_std_vanishes_with_sigma():
    idx = np.array([[0, 1], [0, 0], [1, 1]])
    conc = np.array([[2.0, 0.05], [4.0, 0.0], [0.2, 0.0]])
    t = np.array([0.01, 0.1, np.inf])
    stds = [monte_carlo_gamma(t, idx, conc, _props(s))[1] for s in (1.0, 0.1, 0.01, 0.0)]
    assert stds[0].shape == (3, 3)
    assert (stds[0] > stds[1]).all() and (stds[1] > stds[2]).all()
    assert (stds[3] < 1e-12).all()


def test_diffusion_and_library_monte_carlo():
    D_sds, assumed = diffusion_coefficient("CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]", "alkyl sulfate")
    assert 1e-10 < D_sds < 1e-9 and not assumed
    lib = pd.read_parquet(config.LIBRARY_PATH)
    props = SurfactantArrays.from_library(lib)
    assert (props.D_assumed == (lib["family"] == "siloxane")).all()
    idx = np.array([[0, 1, 2], [3, 4, 5]])
    conc = np.full((2, 3), 0.5)
    mean, std = monte_carlo_gamma([0.01, 0.1, np.inf], idx, conc, props)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert ((mean > 0.015) & (mean < G0)).all()
