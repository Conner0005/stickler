# Physics model (`src/wetopt/physics.py`)

SI throughout; T = 298.15 K; R = 8.314462618 J/(mol K). Component axis last, blends and Monte
Carlo draws broadcast over leading axes. Inputs per surfactant i: CMC_i, γ_CMC,i, Γ_max,i (from
`data/library.parquet`), Gibbs prefactor n_i, diffusion coefficient D_i.

## 1. Water
γ0 = B·τ^μ·(1 + bτ), τ = 1 − T/Tc (B = 235.8 mN/m, b = −0.625, μ = 1.256, Tc = 647.096 K)
→ 71.97 mN/m at 298.15 K [IAPWS R1-76(2014)].

## 2. Single surfactant (Langmuir–Szyszkowski)
γ = γ0 − n·R·T·Γ∞·ln(1 + K·c), with Γ∞ = Γ_max. K is found by requiring the curve to pass
through (CMC, γ_CMC): K = expm1((γ0 − γ_CMC)/(nRTΓ∞)) / CMC [Szyszkowski 1908; Chang & Franses 1995].
n = 1 for nonionic and zwitterionic surfactants, 2 for 1:1 ionic surfactants without salt (`config.N_GIBBS`).

## 3. Mixed CMC
Ideal: 1/CMC_mix = Σ α_i/CMC_i, α_i = C_i/ΣC [Clint 1975].
Non-ideal (optional pairwise β_ij, default 0): 1/CMC_mix = Σ α_i/(f_i·CMC_i), with
ln f_i = Σ_j β_ij X_j − ½ Σ_jk β_jk X_j X_k, which reduces to ln f_1 = β·X_2² for a binary
[Rubingh 1979; Holland & Rubingh 1983]. Solved by damped fixed-point iteration.

## 4. Free monomer
Pseudo-phase separation: c_i = X_i·f_i·CMC_i, Σ X_i = 1, C_i = c_i + X_i·M (M = total micellised).
Eliminating X_i gives Σ C_i/(M + f_i·CMC_i) = 1, solved for M ≥ 0 by bisection, and then
c_i = C_i·f_i·CMC_i/(M + f_i·CMC_i). Below CMC_mix, M = 0 and c_i = C_i [Clint 1975; Holland & Rubingh 1983].

## 5. Equilibrium surface tension
Competitive Langmuir: θ_i = K_i c_i/(1 + Σ K_j c_j).
EOS: γ_eq = γ0 + (nΓ∞)_eff·R·T·ln(1 − Σθ_i), where (nΓ∞)_eff = Σ θ_i n_i Γ∞,i / Σ θ_i and
ln(1 − Σθ) = −ln(1 + ΣK_j c_j).
**Approximation.** The exact multicomponent Langmuir EOS requires equal Γ∞ for all species
[Fainerman et al. 2002]. Here the *product* n·Γ∞ is coverage-weighted. Weighting n and Γ∞
separately can make γ rise when a species with small n·Γ∞ adsorbs: to leading order in
coverage the change is ∝ Γ_eff·(1 − n_eff), which is negative for ionic-rich layers. The
product form keeps γ non-increasing in every coverage and is exact for a single surfactant.

## 6. Dynamics (diffusion-limited)
Γ_i(t) = min(2·c_i·√(D_i t/π), θ_i·Γ∞,i): short-time Ward–Tordai, capped at equilibrium
coverage [Ward & Tordai 1946; Eastoe & Dalton 2000]. With φ_i = Γ_i/Γ∞,i = θ_i r_i, the same
EOS gives γ(t), where ln(1 − Σφ) = ln(1 + Σ K_i c_i (1 − r_i)) − ln(1 + Σ K_j c_j) for numerical
stability. γ(0) = γ0, γ(∞) = γ_eq, and γ is non-increasing in t.

## 7. Diffusion coefficients
Wilke–Chang: D = 7.4e-8·(2.6·18.015)^½·T/(η·V_b^0.6) cm²/s, with η = 0.8900 cP
[Wilke & Chang 1955; IAPWS 2008 viscosity, Huber et al. 2009]. V_b is the Le Bas volume of
the surfactant ion (largest fragment) from atom counts plus ring corrections [Le Bas 1915;
Poling et al. 2001]. If an atom has no Le Bas increment (Si), the code uses the **assumed**
class default `config.D_CLASS_DEFAULT` (siloxane 3e-10 m²/s, else 4e-10 m²/s) and sets
`D_assumed`. All 7 siloxanes in the current library use this default; everything else uses
Wilke–Chang (2.5–5.6e-10 m²/s).

## 8. Uncertainty
Monte Carlo, 200 draws (seeded, same draws shared by all blends):
log10 CMC ~ N(·, σ_log10²); γ_CMC ~ N(·, σ²) clipped to [0, γ0]; Γ_max ~ N(·, σ²) floored at
1e-7 mol/m². K is re-derived per draw. Returns the mean and std of γ at the requested surface
ages (t = ∞ gives γ_eq), shape (blends, ages). D, n and β are not sampled. Model-form error
is not included, so σ is a lower bound on the true error.

## Assumptions and known behaviour
1. Langmuir adsorption with Γ∞ = Γ_max (SurfPro). No interaction parameter in the monolayer and no salt or counterion-binding effects beyond n.
2. Mixed-layer EOS weights n·Γ∞ by coverage (§5). This is an approximation.
3. β = 0 unless supplied. With β = 0, an anionic/nonionic test binary gives a γ_eq that **rises by ~5 mN/m above CMC_mix**, because micelles deplete the more surface-active monomer. Real anionic/nonionic pairs usually have β < 0, which weakens the effect. Do not assume more surfactant always lowers γ.
4. Dynamics: micelles do not release monomer. Above CMC_mix this **over-predicts γ(t)** at short times, so DST is a conservative upper estimate for "DST ≤ hi" bands.
5. Dynamics: the short-time Ward–Tordai limit has no back-diffusion and no adsorption barrier. This **under-predicts γ(t)** at intermediate times, partly offsetting (4). It is not conservative for bands of the form "DST ≥ lo".
6. D comes from Wilke–Chang/Le Bas (~10–20 % error; the atom-count Le Bas ignores ether-O variants), or from the class default for Si.
7. T = 298.15 K; SurfPro values measured at 20–30 °C are used as if at 25 °C.

## References
- IAPWS R1-76(2014), Revised Release on Surface Tension of Ordinary Water Substance.
- Szyszkowski, Z. Phys. Chem. 64, 385 (1908).
- Chang & Franses, Colloids Surf. A 100, 1 (1995).
- Clint, J. Chem. Soc. Faraday Trans. 1 71, 1327 (1975).
- Rubingh, in *Solution Chemistry of Surfactants*, vol. 1, 337 (1979).
- Holland & Rubingh, J. Phys. Chem. 87, 1984 (1983).
- Fainerman, Miller & Aksenenko, Adv. Colloid Interface Sci. 96, 339 (2002).
- Ward & Tordai, J. Chem. Phys. 14, 453 (1946).
- Eastoe & Dalton, Adv. Colloid Interface Sci. 85, 103 (2000).
- Wilke & Chang, AIChE J. 1, 264 (1955).
- Huber et al., J. Phys. Chem. Ref. Data 38, 101 (2009).
- Le Bas, *The Molecular Volumes of Liquid Chemical Compounds* (1915).
- Poling, Prausnitz & O'Connell, *The Properties of Gases and Liquids*, 5th ed. (2001).
- Rosen & Kunjappu, *Surfactants and Interfacial Phenomena*, 4th ed. (2012).
