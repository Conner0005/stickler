# wetopt: fewest-surfactant spray adjuvant blends

**Goal:** given a wetting spec (bands on dynamic surface tension at a surface age and/or equilibrium γ), find the adjuvant blend with the fewest surfactants that meets every band at mean ± z·σ.
**Proof:** show that no smaller blend meets the spec under the model, and emit that as certificate text.
**Demo:** load a spec → minimal-k blend, predicted DST/γ_eq ± σ against the bands, margins, certificate.

## Units
- SI internally: mol/m³, m²/s, s, N/m (also mol/m² for Γ, kg/mol for M). T = 298.15 K.
- mM, % w/v, mN/m and ms (and µmol/m² for display) appear only at I/O (CLI, app, raw-data ingest). 1 mM = 1 mol/m³.
- Exceptions: pc20 = −log10(C20/M); CMC σ is `cmc_sigma_log10` (1σ of log10 CMC).
- Schema fields are SI unless the name carries a unit suffix: `surface_age_ms`, `max_total_wv`.

## Rules
- Physics functions are pure; each docstring gives the equation and a reference.
- Never invent property values: each value carries a `source` or `imputed: true` with σ (enforced by `schema.Quantity`).
- Speed over polish, but every module gets a smoke test.
- Run `uv run pytest -q` and `uv run ruff check` before saying a step is done.
- Commit after each H-step with NO Co-Authored-By trailer.

## Layout
- `src/wetopt/schema.py`: pydantic v2 models (Quantity, Surfactant, Band, Spec, Constraints, Blend, Prediction, Result).
- `src/wetopt/library.py`: SurfPro → `data/library.parquet` (40 candidates). `config.py`: paths, σ/n/D assumptions. `cli.py`: `wetopt`.
- `src/wetopt/physics.py`: Szyszkowski + Clint/Rubingh γ_eq; DST bracket (dst_low capped Ward–Tordai, dst_high Sutherland); Monte Carlo σ ⊕ σ_model. See `docs/physics.md`.
- `tests/` smoke tests · `data/raw/` source data as obtained · `examples/` spec and constraint YAMLs · `docs/data.md` units, licence, provenance.
- SurfPro is CC BY-NC-SA 4.0 (non-commercial, share-alike); fetch it per `docs/data.md`.

## Commands
`uv sync` · `uv run pytest -q` · `uv run ruff check` · `uv run wetopt library [--rebuild] [--all-types]`
