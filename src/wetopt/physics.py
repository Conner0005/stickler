"""Surface tension of surfactant blends: equilibrium and diffusion-limited dynamics.

Conventions: SI throughout (mol/m³, N/m, mol/m², m²/s, s, K). Per-surfactant arrays carry the
component axis LAST (shape ``(..., k)``); leading axes (blends, Monte Carlo draws) broadcast.
Time-dependent outputs put the time axis last (``(..., T)``). All functions are pure.
See docs/physics.md for the full model and its assumptions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from rdkit import Chem

from wetopt import config

R = 8.314462618  # J/(mol K), CODATA 2018 (exact)
T0 = 298.15  # K
ETA_WATER = 0.8900e-3  # Pa s at 298.15 K, IAPWS 2008 (Huber et al., JPCRD 38, 101, 2009)
M_WATER = 18.015e-3  # kg/mol
GAMMA_MAX_FLOOR = 1e-7  # mol/m²; floor for Monte Carlo draws (SurfPro minimum is 1.28e-7)


def gamma_water(T: ArrayLike = T0) -> np.ndarray:
    """Surface tension of pure water, N/m.

    γ0 = B·τ^μ·(1 + b·τ), τ = 1 − T/Tc, B = 235.8 mN/m, b = −0.625, μ = 1.256, Tc = 647.096 K.
    Gives 71.97 mN/m at 298.15 K.
    Ref: IAPWS R1-76(2014), Revised Release on Surface Tension of Ordinary Water Substance.
    """
    tau = 1.0 - np.asarray(T, float) / 647.096
    return 235.8e-3 * tau**1.256 * (1.0 - 0.625 * tau)


def szyszkowski_K(
    cmc: ArrayLike,
    gamma_cmc: ArrayLike,
    gamma_max: ArrayLike,
    n: ArrayLike,
    gamma0: ArrayLike,
    T: float = T0,
) -> np.ndarray:
    """Langmuir adsorption constant K (m³/mol) found by fitting γ_CMC at the CMC.

    γ_CMC = γ0 − n·R·T·Γ∞·ln(1 + K·CMC)  ⇒  K = expm1((γ0 − γ_CMC)/(n·R·T·Γ∞)) / CMC,
    with Γ∞ = Γ_max; K = 0 if γ_CMC ≥ γ0 (no surface activity).
    Ref: Szyszkowski, Z. Phys. Chem. 64, 385 (1908); Chang & Franses, Colloids Surf. A 100,
    1 (1995), eqs. for the Langmuir–Szyszkowski isotherm/EOS.
    """
    gamma_max = np.asarray(gamma_max, float)
    x = (np.asarray(gamma0) - np.asarray(gamma_cmc)) / (np.asarray(n) * R * T * gamma_max)
    return np.expm1(np.maximum(x, 0.0)) / np.asarray(cmc, float)


def ln_activity_coefficients(X: ArrayLike, beta: ArrayLike) -> np.ndarray:
    """ln f_i of surfactant i in a mixed micelle (regular solution, pairwise β_ij).

    gᴱ/RT = Σ_{j<k} β_jk X_j X_k  ⇒  ln f_i = Σ_j β_ij X_j − ½ Σ_jk β_jk X_j X_k
    (β symmetric, zero diagonal). For a binary this is Rubingh's ln f_1 = β·X_2².
    Ref: Rubingh, in Solution Chemistry of Surfactants vol. 1, 337 (1979);
    Holland & Rubingh, J. Phys. Chem. 87, 1984 (1983).
    """
    X = np.asarray(X, float)
    bx = np.einsum("...ij,...j->...i", np.asarray(beta, float), X)
    return bx - 0.5 * np.sum(X * bx, axis=-1, keepdims=True)


def mixed_cmc(conc: ArrayLike, cmc: ArrayLike, beta: ArrayLike | None = None) -> np.ndarray:
    """Mixed CMC (mol/m³) for bulk composition α_i = C_i/ΣC.

    Ideal (Clint): 1/CMC_mix = Σ α_i/CMC_i.
    Non-ideal (Rubingh): 1/CMC_mix = Σ α_i/(f_i·CMC_i), with micelle composition
    X_i = α_i·CMC_mix/(f_i·CMC_i) solved by damped fixed-point iteration.
    Ref: Clint, J. Chem. Soc. Faraday Trans. 1 71, 1327 (1975); Rubingh (1979).
    """
    conc, cmc = np.asarray(conc, float), np.asarray(cmc, float)
    alpha = conc / conc.sum(-1, keepdims=True)
    lnf = np.zeros(np.broadcast_shapes(alpha.shape, cmc.shape))
    for _ in range(1 if beta is None else 200):
        w = alpha / (cmc * np.exp(lnf))
        cmc_mix = 1.0 / w.sum(-1)
        if beta is None:
            break
        lnf = 0.5 * lnf + 0.5 * ln_activity_coefficients(w * cmc_mix[..., None], beta)
    return cmc_mix


def free_monomer(conc: ArrayLike, cmc: ArrayLike, beta: ArrayLike | None = None) -> np.ndarray:
    """Free monomer concentration c_i (mol/m³) in a mixed-micelle solution.

    Pseudo-phase separation: c_i = X_i·f_i·CMC_i, Σ X_i = 1, C_i = c_i + X_i·M, where M is
    the total micellised amount. Eliminating X_i: Σ_i C_i/(M + f_i·CMC_i) = 1, solved for M ≥ 0
    by bisection; c_i = C_i·f_i·CMC_i/(M + f_i·CMC_i). Below CMC_mix there is no root, M = 0
    and c_i = C_i. f_i = 1 when β is None; otherwise f_i(X) from ln_activity_coefficients.
    Ref: Clint (1975); Holland & Rubingh (1983); Rosen & Kunjappu, Surfactants and
    Interfacial Phenomena, 4th ed. (2012).
    """
    conc, cmc = np.asarray(conc, float), np.asarray(cmc, float)
    shape = np.broadcast_shapes(conc.shape, cmc.shape)
    conc = np.broadcast_to(conc, shape)
    total = conc.sum(-1, keepdims=True)
    lnf = np.zeros(shape)
    for _ in range(1 if beta is None else 200):
        fc = cmc * np.exp(lnf)
        lo, hi = np.zeros_like(total), total.copy()
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            above = (conc / (mid + fc)).sum(-1, keepdims=True) > 1.0
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        below_cmc = (conc / fc).sum(-1, keepdims=True) <= 1.0
        M = np.where(below_cmc, 0.0, 0.5 * (lo + hi))
        if beta is None:
            break
        X = conc / (M + fc)
        X = X / np.where(X.sum(-1, keepdims=True) > 0, X.sum(-1, keepdims=True), 1.0)
        lnf = 0.5 * lnf + 0.5 * ln_activity_coefficients(X, beta)
    return conc * fc / (M + fc)


def langmuir_coverage(c: ArrayLike, K: ArrayLike) -> np.ndarray:
    """Competitive Langmuir fractional coverage θ_i = K_i·c_i / (1 + Σ_j K_j·c_j).

    Ref: Chang & Franses, Colloids Surf. A 100, 1 (1995) (multicomponent Langmuir).
    """
    x = np.asarray(K, float) * np.asarray(c, float)
    return x / (1.0 + x.sum(-1, keepdims=True))


def surface_tension_eos(
    phi: ArrayLike,
    n: ArrayLike,
    gamma_max: ArrayLike,
    gamma0: ArrayLike,
    T: float = T0,
    ln_vacancy: ArrayLike | None = None,
) -> np.ndarray:
    """Langmuir–Szyszkowski equation of state for a mixed monolayer, N/m.

    γ = γ0 + (nΓ∞)_eff·R·T·ln(1 − Σφ_i), (nΓ∞)_eff = Σ φ_i·n_i·Γ∞,i / Σ φ_i,
    with φ_i = Γ_i/Γ∞,i. APPROXIMATION: the exact multicomponent Langmuir EOS needs equal Γ∞;
    here the product n·Γ∞ is coverage-weighted. Weighting n and Γ∞ separately would let a
    low-nΓ∞ species raise γ; the product weighting keeps γ non-increasing in every φ_i and is
    exact for one surfactant. Pass ``ln_vacancy`` = ln(1 − Σφ) when it is known in closed
    form (avoids cancellation as Σφ → 1).
    Ref: Szyszkowski (1908); Chang & Franses (1995); Fainerman, Miller & Aksenenko,
    Adv. Colloid Interface Sci. 96, 339 (2002) for mixed-layer EOS.
    """
    phi = np.asarray(phi, float)
    s = phi.sum(-1)
    lnv = np.log1p(-s) if ln_vacancy is None else np.asarray(ln_vacancy, float)
    w = (phi * np.asarray(n) * np.asarray(gamma_max)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_gamma_eff = np.where(s > 0, w / s, 0.0)
    return np.asarray(gamma0) + R * T * n_gamma_eff * lnv


def adsorbed_amount(
    t: ArrayLike, c: ArrayLike, D: ArrayLike, theta: ArrayLike, gamma_max: ArrayLike
) -> np.ndarray:
    """Diffusion-limited surface excess Γ_i(t), mol/m², shape (..., T, k).

    Γ_i(t) = min(2·c_i·sqrt(D_i·t/π), θ_i·Γ∞,i): short-time Ward–Tordai (no back-diffusion),
    capped at the equilibrium coverage. Only free monomer adsorbs (micelles ignored).
    Ref: Ward & Tordai, J. Chem. Phys. 14, 453 (1946); Eastoe & Dalton, Adv. Colloid
    Interface Sci. 85, 103 (2000).
    """
    t = np.asarray(t, float)[:, None]
    c, D = np.asarray(c, float)[..., None, :], np.asarray(D, float)[..., None, :]
    sat = (np.asarray(theta) * np.asarray(gamma_max))[..., None, :]
    with np.errstate(invalid="ignore"):
        diffusive = np.where(c > 0, 2.0 * c * np.sqrt(D * t / np.pi), 0.0)
    return np.minimum(diffusive, sat)


def equilibrium_surface_tension(
    conc: ArrayLike,
    cmc: ArrayLike,
    K: ArrayLike,
    gamma_max: ArrayLike,
    n: ArrayLike,
    gamma0: ArrayLike,
    beta: ArrayLike | None = None,
    T: float = T0,
) -> np.ndarray:
    """Equilibrium γ_eq (N/m) of a blend with total concentrations ``conc``.

    c = free_monomer(C); θ_i = K_i·c_i/(1 + ΣK_j·c_j); γ_eq = surface_tension_eos(θ) with
    ln(1 − Σθ) = −ln(1 + ΣK_j·c_j). One surfactant at C ≥ CMC returns its γ_CMC.
    Ref: Szyszkowski (1908); Clint (1975); Chang & Franses (1995).
    """
    c = free_monomer(conc, cmc, beta)
    x = np.asarray(K) * c
    theta = langmuir_coverage(c, K)
    return surface_tension_eos(theta, n, gamma_max, gamma0, T, -np.log1p(x.sum(-1)))


def dynamic_surface_tension(
    t: ArrayLike,
    conc: ArrayLike,
    cmc: ArrayLike,
    K: ArrayLike,
    gamma_max: ArrayLike,
    n: ArrayLike,
    D: ArrayLike,
    gamma0: ArrayLike,
    beta: ArrayLike | None = None,
    T: float = T0,
) -> np.ndarray:
    """Dynamic surface tension γ(t) (N/m) at surface ages ``t`` (s), shape (..., T).

    Γ_i(t) from adsorbed_amount; φ_i = Γ_i(t)/Γ∞,i = θ_i·r_i fed into surface_tension_eos,
    with ln(1 − Σφ) = ln(1 + ΣK_i·c_i·(1 − r_i)) − ln(1 + ΣK_j·c_j). t = ∞ gives γ_eq, t = 0
    gives γ0. Micelle breakdown is ignored: above CMC_mix only free monomer diffuses, so γ(t)
    is over-predicted at short times (a conservative upper estimate of DST). The short-time
    Ward–Tordai limit ignores back-diffusion, which under-predicts γ(t) at intermediate times;
    see docs/physics.md.
    Ref: Ward & Tordai (1946); Eastoe & Dalton (2000); Fainerman et al. (2002).
    """
    t = np.atleast_1d(np.asarray(t, float))
    c = free_monomer(conc, cmc, beta)
    x = np.asarray(K) * c
    theta = langmuir_coverage(c, K)
    gamma_max = np.asarray(gamma_max, float)
    G = adsorbed_amount(t, c, D, theta, gamma_max)
    sat = (theta * gamma_max)[..., None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(sat > 0, G / sat, 1.0)
    lnv = np.log1p((x[..., None, :] * (1.0 - r)).sum(-1)) - np.log1p(x.sum(-1, keepdims=True))
    n = np.asarray(n)[..., None, :]
    return surface_tension_eos(theta[..., None, :] * r, n, gamma_max[..., None, :], gamma0, T, lnv)


# Le Bas increments, cm³/mol (Poling, Prausnitz & O'Connell 2001, ch. 4).
_LE_BAS = {"C": 14.8, "H": 3.7, "O": 7.4, "S": 25.6, "F": 8.7, "Cl": 24.6, "Br": 27.0, "I": 37.0}
_LE_BAS_RING = {3: -6.0, 4: -8.5, 5: -11.5, 6: -15.0}


def le_bas_volume(smiles: str) -> float:
    """Le Bas molar volume at the normal boiling point, m³/mol (NaN if an atom has no increment).

    V_b = Σ atom increments + ring corrections, for the largest fragment (surfactant ion;
    counterions dropped). C 14.8, H 3.7, O 7.4, S 25.6, F 8.7, Cl 24.6, Br 27.0, I 37.0;
    N 15.6 (double-bonded), 10.5 (one heavy neighbour), else 12.0; rings 3: −6.0, 4: −8.5,
    5: −11.5, 6: −15.0 cm³/mol. Atom-count version: the ether/ester/acid O variants are ignored.
    Ref: Le Bas, The Molecular Volumes of Liquid Chemical Compounds (1915); Poling,
    Prausnitz & O'Connell, The Properties of Gases and Liquids, 5th ed. (2001), ch. 4.
    """
    mol = Chem.MolFromSmiles(smiles)
    ion = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    v = 0.0
    for atom in ion.GetAtoms():
        sym = atom.GetSymbol()
        v += 3.7 * atom.GetTotalNumHs()
        if sym == "N":
            double = any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())
            v += 15.6 if double else 10.5 if atom.GetDegree() == 1 else 12.0
        elif sym in _LE_BAS:
            v += _LE_BAS[sym]
        else:
            return float("nan")
    v += sum(_LE_BAS_RING.get(len(ring), 0.0) for ring in ion.GetRingInfo().AtomRings())
    return v * 1e-6


def wilke_chang_D(V_b: ArrayLike, T: float = T0, eta: float = ETA_WATER) -> np.ndarray:
    """Infinite-dilution diffusion coefficient of a solute in water, m²/s.

    D [cm²/s] = 7.4e-8·(φ·M_w)^½·T / (η·V_b^0.6), φ = 2.6 (water), M_w in g/mol, η in cP,
    V_b in cm³/mol (Le Bas). Typical error ~10–20 %.
    Ref: Wilke & Chang, AIChE J. 1, 264 (1955); Poling et al. (2001), ch. 11.
    """
    V_cm3 = np.asarray(V_b, float) * 1e6
    D_cm2 = 7.4e-8 * np.sqrt(2.6 * M_WATER * 1e3) * T / (eta * 1e3 * V_cm3**0.6)
    return D_cm2 * 1e-4


def diffusion_coefficient(smiles: str, family: str, T: float = T0) -> tuple[float, bool]:
    """Monomer D (m²/s) and whether it is an assumption.

    D = wilke_chang_D(le_bas_volume(smiles)); if Le Bas has no increment for an atom (e.g. Si),
    D = config.D_CLASS_DEFAULT[family] (or config.D_DEFAULT), flagged as assumed.
    Ref: Wilke & Chang (1955); Le Bas (1915).
    """
    V = le_bas_volume(smiles)
    if np.isfinite(V):
        return float(wilke_chang_D(V, T)), False
    return config.D_CLASS_DEFAULT.get(family, config.D_DEFAULT), True


@dataclass(frozen=True)
class SurfactantArrays:
    """Per-surfactant property arrays, shape (S,), SI; σ as in data/library.parquet."""

    cmc: np.ndarray
    cmc_sigma_log10: np.ndarray
    gamma_cmc: np.ndarray
    gamma_cmc_sigma: np.ndarray
    gamma_max: np.ndarray
    gamma_max_sigma: np.ndarray
    n: np.ndarray
    D: np.ndarray
    D_assumed: np.ndarray

    @classmethod
    def from_library(cls, lib: pd.DataFrame, T: float = T0) -> "SurfactantArrays":
        """Arrays from library rows; n from config.N_GIBBS, D from diffusion_coefficient."""
        D, assumed = zip(
            *(
                diffusion_coefficient(s, f, T)
                for s, f in zip(lib["smiles"], lib["family"], strict=True)
            ),
            strict=True,
        )
        cols = ["cmc", "cmc_sigma_log10", "gamma_cmc", "gamma_cmc_sigma", "gamma_max"]
        return cls(
            **{c: lib[c].to_numpy(float) for c in [*cols, "gamma_max_sigma"]},
            n=lib["ionic_class"].map(config.N_GIBBS).to_numpy(float),
            D=np.array(D),
            D_assumed=np.array(assumed),
        )


def monte_carlo_gamma(
    t: ArrayLike,
    idx: ArrayLike,
    conc: ArrayLike,
    props: SurfactantArrays,
    beta: ArrayLike | None = None,
    n_draws: int = 200,
    seed: int = 0,
    T: float = T0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of γ(t) (N/m), each of shape (B, T), over property uncertainty.

    Blends: ``idx`` (B, k) surfactant indices into ``props`` and ``conc`` (B, k) mol/m³
    (zero-padded allowed); ``beta`` is an optional (S, S) pairwise matrix; t = ∞ gives γ_eq.
    Per draw: log10 CMC ~ N(log10 CMC, σ_log10²), γ_CMC ~ N(γ_CMC, σ²) clipped to [0, γ0],
    Γ_max ~ N(Γ_max, σ²) floored at GAMMA_MAX_FLOOR; K is re-derived per draw. The same draws
    are shared by all blends (common random numbers).
    Ref: Szyszkowski (1908); Ward & Tordai (1946); see dynamic_surface_tension.
    """
    idx, conc = np.asarray(idx), np.asarray(conc, float)
    z = np.random.default_rng(seed).standard_normal((3, n_draws, props.cmc.size))
    g0 = gamma_water(T)
    cmc = props.cmc * 10.0 ** (props.cmc_sigma_log10 * z[0])
    gamma_cmc = np.clip(props.gamma_cmc + props.gamma_cmc_sigma * z[1], 0.0, g0)
    gamma_max = np.maximum(props.gamma_max + props.gamma_max_sigma * z[2], GAMMA_MAX_FLOOR)
    K = szyszkowski_K(cmc, gamma_cmc, gamma_max, props.n, g0, T)
    beta_b = None if beta is None else np.asarray(beta)[idx[..., :, None], idx[..., None, :]]
    g = dynamic_surface_tension(
        t,
        conc,
        cmc[:, idx],
        K[:, idx],
        gamma_max[:, idx],
        props.n[idx],
        props.D[idx],
        g0,
        beta_b,
        T,
    )
    return g.mean(0), g.std(0)
