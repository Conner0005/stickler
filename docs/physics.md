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
Competitive Langmuir: θ_i = K_i c_i/(1 + Σ K_j c_j); Γ_i,eq = θ_i·Γ∞,i.
EOS: γ = γ0 + (nΓ∞)_eff·R·T·ln(1 − Σφ_i), φ_i = Γ_i/Γ∞,i, (nΓ∞)_eff = Σ φ_i n_i Γ∞,i / Σ φ_i.
At equilibrium φ = θ and ln(1 − Σθ) = −ln(1 + ΣK_j c_j).
**Approximation.** The exact multicomponent Langmuir EOS requires equal Γ∞ for all species
[Fainerman et al. 2002]. Here the *product* n·Γ∞ is coverage-weighted. Weighting n and Γ∞
separately can make γ rise when a species with small n·Γ∞ adsorbs: to leading order in
coverage the change is ∝ Γ_eff·(1 − n_eff), which is negative for ionic-rich layers. The
product form keeps γ non-increasing in every coverage and is exact for a single surfactant.

## 6. Dynamics: a DST bracket
Both bounds use the same Γ_i,eq (from the equilibrium free monomer) and the same EOS (§5),
with φ_i = Γ_i(t)/Γ∞,i and the stable form ln(1 − Σφ) = ln(1 + Σ x_i(1 − r_i)) − ln(1 + Σ x_j),
where x_i = K_i c_i and r_i = Γ_i(t)/Γ_i,eq.

**dst_low (lower bound): adsorbs too fast.**
Γ_i(t) = min(2·C_i·√(D_i t/π), Γ_i,eq), with C_i the **total** concentration of i: micelles act
as instantly available monomer. This is the short-time Ward–Tordai limit, which has no
back-diffusion and is cut off at equilibrium [Ward & Tordai 1946; Eastoe & Dalton 2000].

**dst_high (upper bound): adsorbs too slowly.**
Sutherland's exact solution for diffusion-controlled adsorption with a linear isotherm,
using the secant Henry constant K_H,i = Γ_i,eq/c_i [Sutherland 1952]:
Γ_i(t) = Γ_i,eq·(1 − erfcx(√(t/τ_i))), τ_i = (Γ_i,eq/c_i)²/D_i, erfcx(x) = exp(x²)·erfc(x),
with c_i = **free monomer** (micelles contribute nothing). The Langmuir isotherm is concave,
so the secant overstates back-diffusion and this curve adsorbs no faster than the Langmuir
case. Limits: 2c_i√(D_i t/π) as t → 0, Γ_i,eq as t → ∞.

**Ordering.** 1 − erfcx(x) ≤ 2x/√π and c_i ≤ C_i, so Γ_i,high(t) ≤ Γ_i,low(t) for every
component. γ is non-increasing in every φ_i (§5), so dst_low ≤ dst_high. Both are
non-increasing in t, equal γ0 at t = 0 and γ_eq at t = ∞.

**Which bound to use.** For a "DST ≤ hi" band, test dst_high; for a "DST ≥ lo" band, test dst_low.
The bracket is wide above the CMC: at 5× CMC and 10 ms, some candidates span 30–68 mN/m.

## 7. Diffusion coefficients
Wilke–Chang: D = 7.4e-8·(2.6·18.015)^½·T/(η·V_b^0.6) cm²/s, with η = 0.8900 cP
[Wilke & Chang 1955; IAPWS 2008 viscosity, Huber et al. 2009]. V_b is the Le Bas volume of
the surfactant ion (largest fragment) from atom counts plus ring corrections [Le Bas 1915;
Poling et al. 2001]. If an atom has no Le Bas increment (Si), the code uses the **assumed**
class default `config.D_CLASS_DEFAULT` (siloxane 3e-10 m²/s, else 4e-10 m²/s) and sets
`D_assumed`. All 7 siloxanes in the current library use this default; everything else uses
Wilke–Chang (2.5–5.6e-10 m²/s).

## 8. Uncertainty
Monte Carlo on both bounds, 200 draws (seeded; the same draws are shared by all blends):
- log10 CMC ~ N(log10 CMC, σ_log10²), i.e. log-normal CMC.
- γ_CMC ~ N(γ_CMC, σ²). A draw ≥ γ0 gives K = 0.
- Γ_max ~ N(Γ_max, σ²) truncated at 1e-7 mol/m², drawn by inversion [Devroye 1986].

K is re-derived per draw. The reported std is √(std_MC² + σ_model²), where σ_model =
`config.SIGMA_MODEL` = 2 mN/m is an **assumption** for model-form error. D, n and β are not sampled.

## Assumptions and known behaviour
1. Langmuir adsorption with Γ∞ = Γ_max (SurfPro). No interaction parameter in the monolayer and no salt or counterion-binding effects beyond n.
2. The mixed-layer EOS weights n·Γ∞ by coverage (§5). This is an approximation.
3. β = 0 unless supplied. With β = 0, an anionic/nonionic test binary gives a γ_eq that **rises by ~5 mN/m above CMC_mix**, because micelles deplete the more surface-active monomer. Real anionic/nonionic pairs usually have β < 0, which weakens the effect. Do not assume more surfactant always lowers γ.
4. The **short-time capped form** (dst_low) ignores back-diffusion, so it **over-predicts adsorption and under-predicts DST**. It is optimistic for "DST ≤ hi" bands and conservative for "DST ≥ lo" bands. Treating micelles as instant monomer pushes it further in the same direction.
5. **Ignoring micelle breakdown** (dst_high, free monomer only) errs the other way: it under-predicts adsorption above CMC_mix and over-predicts DST. It is conservative for "DST ≤ hi" bands and optimistic for "DST ≥ lo" bands. The secant linear isotherm adds to this.
6. Each component diffuses independently to a cut-off set by the competitive equilibrium. Competition during the transient is not modelled, and there is no adsorption barrier.
7. D comes from Wilke–Chang/Le Bas (~10–20 % error; the atom-count Le Bas ignores ether-O variants), or from the class default for Si.
8. T = 298.15 K; SurfPro values measured at 20–30 °C are used as if at 25 °C.

## References
- IAPWS R1-76(2014), Revised Release on Surface Tension of Ordinary Water Substance.
- Szyszkowski, Z. Phys. Chem. 64, 385 (1908).
- Chang & Franses, Colloids Surf. A 100, 1 (1995).
- Clint, J. Chem. Soc. Faraday Trans. 1 71, 1327 (1975).
- Rubingh, in *Solution Chemistry of Surfactants*, vol. 1, 337 (1979).
- Holland & Rubingh, J. Phys. Chem. 87, 1984 (1983).
- Fainerman, Miller & Aksenenko, Adv. Colloid Interface Sci. 96, 339 (2002).
- Ward & Tordai, J. Chem. Phys. 14, 453 (1946).
- Sutherland, Aust. J. Sci. Res. A 5, 683 (1952).
- Eastoe & Dalton, Adv. Colloid Interface Sci. 85, 103 (2000).
- Wilke & Chang, AIChE J. 1, 264 (1955).
- Huber et al., J. Phys. Chem. Ref. Data 38, 101 (2009).
- Le Bas, *The Molecular Volumes of Liquid Chemical Compounds* (1915).
- Poling, Prausnitz & O'Connell, *The Properties of Gases and Liquids*, 5th ed. (2001).
- Devroye, *Non-Uniform Random Variate Generation* (1986).
- Rosen & Kunjappu, *Surfactants and Interfacial Phenomena*, 4th ed. (2012).
