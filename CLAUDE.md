# wetopt: fewest-surfactant spray adjuvant blends

**Goal:** given a wetting spec (bands on dynamic surface tension at a surface age and/or equilibrium γ), find the adjuvant blend with the fewest surfactants that meets every band at mean ± z·σ.
**Proof:** show that no smaller blend meets the spec under the model, and emit that as certificate text.
**Demo:** load a spec → minimal-k blend, predicted DST/γ_eq ± σ against the bands, margins, certificate.

## Units
- SI internally: mol/m³, m²/s, s, N/m (also mol/m² for Γ, kg/mol for M). T = 298.15 K.
- mM, % w/v, mN/m and ms appear only at I/O (CLI, app, raw-data ingest). 1 mM = 1 mol/m³.
- Schema fields are SI unless the name carries a unit suffix: `surface_age_ms`, `max_total_wv`.

## Rules
- Physics functions are pure; each docstring gives the equation and a reference.
- Never invent property values: each value carries a `source` or `imputed: true` with σ (enforced by `schema.Quantity`).
- Speed over polish, but every module gets a smoke test.
- Run `uv run pytest -q` and `uv run ruff check` before saying a step is done.
- Commit after each H-step with NO Co-Authored-By trailer.

## Layout
- `src/wetopt/schema.py`: pydantic v2 models (Quantity, Surfactant, Band, Spec, Constraints, Blend, Prediction, Result).
- `tests/` smoke tests · `data/raw/` source data as obtained · `examples/` spec and constraint YAMLs · `docs/`.

## Commands
`uv sync` · `uv run pytest -q` · `uv run ruff check`
