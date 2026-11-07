# Data: SurfPro

**Source.** Hödl, Hermans, Dankloff, Piruska, Huck, Robinson, *SurfPro – a curated database and
predictive model of experimental properties of surfactants*, Digital Discovery 2025, 4, 1176–1187,
doi:10.1039/D4DD00393D. Repo: <https://github.com/BigChemistry-RobotLab/SurfPro>, commit
`35b8906` (2026-05-20, "SurfProV2": 1504 structures, 1360 pCMC, 932 γ_CMC, 700 Γ_max).

**Fetch** (not committed; `data/raw/surfpro/` is git-ignored):

    git clone https://github.com/BigChemistry-RobotLab/SurfPro data/raw/surfpro
    git -C data/raw/surfpro checkout 35b8906

**Licence: CC BY-NC-SA 4.0** (`LICENSE.md` in the repo). You must cite the paper (attribution),
use the data for **non-commercial purposes only**, and release anything derived from it under the
same licence (share-alike). `data/library.parquet` is derived from SurfPro, so it is CC BY-NC-SA 4.0 too.
The paper's own licence is unconfirmed: the publisher and the Radboud repository both refuse automated fetches.

## Files used

| file | content |
|---|---|
| `data/surfpro_literature.csv` | measured values with per-property DOIs (`Ref_*`), `Temp_Celsius`, `InChIKey`, `Molecular_Weight` |
| `data/surfpro_imputed.csv` | every property for train+test structures (1476 rows, no `mixture`); missing values filled with predictions from the 10-fold AttentiveFP_all64d ensemble (paper, Table 3; the repo's `params.yaml` shows the last run config, AttentiveFP-96d); `<prop>_std` = ensemble std |

## Units (confirmed from the repo; the paper could not be fetched)

| property | SurfPro column | unit in literature.csv | unit in imputed.csv | evidence | wetopt column (SI) |
|---|---|---|---|---|---|
| CMC | `pCMC` | −log10(CMC / M) | same | README; `pCMC = −log10(CMC)` holds to 2e-4 | `cmc` mol/m³ = 10^(3−pCMC); `cmc_sigma_log10` |
| γ_CMC | `AW_ST_CMC` | mN/m | mN/m | README table; range 17.8–61.9 | `gamma_cmc` N/m |
| Γ_max | `Gamma_max` | **mol/m²** | **µmol/m²** (×1e6) | README table; `params.yaml` unit `mol/m^2 * 1e6`; `src/surfpro/dataloader.py:128`; imputed = lit × 1e6 on 698/699 measured | `gamma_max` mol/m² |
| pC20 | `pC20` | −log10(C20 / M) | same | README | `pc20` (kept in M, Rosen convention) |
| T | `Temp_Celsius` | °C | absent | column name; 1232 rows at 25 °C, 158 blank | `temp_c` °C |
| M | `Molecular_Weight` | g/mol | absent | values (SDS 288.4) | `mw` kg/mol |

Caveats found while checking:
- The `C20` column disagrees with `10^−pC20` in 253 of 636 rows, and the gap is not a constant unit factor. We use `pC20`, the column SurfPro's models train on, and ignore `C20`. Some `pC20` values have `Ref_pC20 = calculated`.
- Measured values sometimes lack a per-property DOI (γ_CMC 45, Γ_max 61, pC20 114 rows). Their source reads `SurfPro lit. (no ref)`.
- In `surfpro_imputed.csv`, `std == 0` marks a measured value, with one exception per property (SurfProV2 edits). We take the measured flag from `surfpro_literature.csv`.

## Library build (`src/wetopt/library.py`)

1. Keep types `anionic`, `non-ionic`, `sugar-based non-ionic`; `--all-types` adds cationic, zwitterionic and gemini. `mixture` (organic counterions) is always dropped because SurfPro has no imputed values for it.
2. Keep 20 ≤ `Temp_Celsius` ≤ 30; rows with a blank temperature are dropped.
3. Deduplicate by InChIKey (median). SurfProV2 has no duplicate InChIKeys, so this step changes nothing today.
4. Each property uses the measured value if one exists, otherwise the imputed value. Every value records `_measured` and `_source` (DOI, or the imputed-DB citation).
5. σ: measured values use `config.MEASURED_SIGMA`, an *assumption*: pCMC 0.05, γ 1.0 mN/m, Γ 0.3 µmol/m², pC20 0.1. Imputed values use the test-set RMSE of the AttentiveFP_all64d ensemble (paper, Table 3; values supplied by the user, since the paper could not be fetched here): pCMC 0.358, γ_CMC 3.52 mN/m, Γ_max 0.84 µmol/m², pC20 0.39 (`config.SURFPRO_TEST_RMSE`).
   **Caveat:** SurfPro's test set is a random split, stratified only by surfactant type. Structures similar to the test set are well represented in training, so this RMSE probably **understates** the error for agro-like surfactants that are scarce in SurfPro (sulfosuccinates, alkylbenzene sulfonates, trisiloxanes). The candidates in those families have almost every property except CMC imputed.
6. Family is assigned from RDKit SMARTS, first match wins: siloxane (any Si), sulfosuccinate, alkylbenzene sulfonate, alkyl sulfate (includes ether sulfates), sugar-based (ring or open-chain polyol), alcohol ethoxylate (C/H/O only, non-aromatic), other.
7. Selection: the six named families count as agro-like unless fluorinated. They are taken round-robin, and within a family entries with more measured properties rank first. 40 candidates are saved to `data/library.parquet`, each with a `reason`.

## Data-quality filter (`validate.data_quality`, default on)
For each candidate with a **measured** pC20, the pC20-check error is e = γ_model(C20) − (γ0 − 20 mN/m),
where the model uses the candidate's own CMC, γ_CMC and Γ_max. If |e| > 3·σ_model
(`config.QC_SIGMA_MULT`; 10.5 mN/m with σ_model = 3.5 mN/m), the candidate is flagged
**inconsistent source data** and excluded from optimisation, with the reason shown. Turn this off
with `config.EXCLUDE_INCONSISTENT` or `wetopt optimise --include-inconsistent`. Candidates without a
measured pC20 are **unchecked** and are not excluded.
The cut uses σ_model, not the spread of e: many SurfPro pC20 values come from the same isotherm
fits as CMC and γ_CMC, so their spread (1.4826·MAD = 0.6 mN/m) is not an independent error scale.
The filter catches gross data errors, not normal model error. Current library: 28 checked, 1 flagged
(C14H29NaO5S #1, +16.0 mN/m).

## Regulatory flags (`data/regulations.yaml`, `regulatory.py`)
This is a hardcoded snapshot (as_of 2026-11-07) and **not legal advice**; the disclaimer in the
YAML is shown wherever flags appear. Each ingredient is classified by listed CAS and by RDKit
structure, never by name:
- nonylphenol/octylphenol family (not_permitted/restricted): a benzene ring with an acyclic alkyl substituent of exactly 9/8 C and either a free phenol OH or a pure (CH2CH2O)n chain ending in a free –OH. These are the forms Annex III entries 56–139 and REACH entry 46 name.
- alkylphenol_derivative (**check**, EU and GB): any other structure carrying that nonyl/octylphenol unit (PO or mixed EO/PO chains, capped, sulfated or phosphated ends, esters).
- tallow amine ethoxylate: a pattern match only, to be confirmed by CAS.
- PFAS (OECD 2021), PFOS/PFHxS (C8/C6 sulfonate), PFOA (C7F15COO) and long-chain PFCAs (CnF2n+1COO, 8 ≤ n ≤ 20, as defined in Stockholm decision SC-12/12).

`Constraints.markets` (EU, GB, US, Global) and `policy`:
- exclude (default): removes anything not_permitted, restricted or eliminated in a selected market before the search.
- strict: also removes 'check' flags.
- warn: flags only.

Rule hits over the 40 library candidates, all markets: only `eu_alkylphenol_check` and
`gb_alkylphenol_check`, 3 each. These are the three nonylphenol propoxylate sulfates
(C24H41NaO7S, C33H59NaO10S, C42H77NaO13S), which only `strict` removes. Over the 528 SurfPro
surfactants (default types, 20–30 °C): EU Annex III 15, REACH Annex XVII entry 46 3, GB 15,
alkylphenol check EU 7 and GB 7, PFOS 1, PFOA 4, PFHxS 0, LC-PFCA 1, US PFAS check 39.
**Resolved gap:** the propoxylate sulfates were previously unflagged. The legal rules are not
stretched to cover them; they get the 'check' tier with the note "Not named in Annex III or REACH
entry 46, but contains the nonyl/octylphenol unit those rules target; confirm before use."

Names: `CnEm` for linear alcohol ethoxylates, otherwise the molecular formula, with ` #k` added when isomers share a formula. SurfPro has no common names.
