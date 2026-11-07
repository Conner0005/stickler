"""Multipage website (Streamlit Community Cloud): Home, Optimiser, Regulations, Method.

Entry point: streamlit_app.py at the repo root calls run(). Pages are plain functions so each can
be rendered on its own (tests use streamlit.testing.v1.AppTest.from_function).
"""

import pandas as pd
import streamlit as st

from wetopt import config, regulatory
from wetopt.app import TIME_LIMIT_S, optimiser_page

PAGES: dict[str, st.Page] = {}
SURFPRO = "https://doi.org/10.1039/D4DD00393D"
MIT_2016 = "https://doi.org/10.1038/ncomms12560"


def home_page() -> None:
    st.title("wetopt")
    st.subheader(
        "Find the spray adjuvant blend with the fewest surfactants that wets the crop, "
        "and show that nothing smaller can."
    )
    st.markdown(
        f"""
- **Most spray misses.** Only about 2 % of sprayed pesticide ends up sticking to the plant;
  the rest bounces or runs off ([MIT, 2016]({MIT_2016})).
- **Every extra co-formulant is extra regulatory risk.** Each one has to clear REACH and product
  authorisation in the EU and GB, and EPA's approved inert list in the US.
- **wetopt finds the fewest-surfactant blend and proves nothing smaller works.** It searches a
  curated surfactant library against a dynamic-surface-tension spec, ranks the blends that pass
  with margin, and certifies, under the model, that no blend with fewer surfactants can.
"""
    )
    if (
        st.button("Try it", type="primary", icon=":material/arrow_forward:")
        and "optimiser" in PAGES
    ):
        st.switch_page(PAGES["optimiser"])
    st.divider()
    st.caption(
        f"Surfactant data: SurfPro (Hödl et al., *Digital Discovery* 2025, 4, 1176; [{SURFPRO}]"
        f"({SURFPRO}); github.com/BigChemistry-RobotLab/SurfPro), licensed CC BY-NC-SA 4.0. The "
        "candidate library used here is derived from it and shared under the same licence, for "
        "non-commercial use. A hackathon prototype: predictions are model estimates, not "
        "measurements."
    )


def regulations_page() -> None:
    regs = regulatory.load()
    st.title("Regulations")
    st.warning(regs.disclaimer)
    st.caption(f"Hardcoded snapshot as of {regs.as_of} (data/regulations.yaml).")
    rows = [
        {
            "region": r.region_label,
            "status": r.status.replace("_", " "),
            "rule": r.rule,
            "applies to": ", ".join(regs.class_labels[c] for c in sorted(r.classes))
            or ", ".join(sorted(r.cas)),
            "source": r.source,
        }
        for r in regs.rules
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={"source": st.column_config.LinkColumn("source", display_text="open")},
    )
    st.markdown(
        """
**How the Optimiser uses these rules.** Each ingredient is classified by listed CAS number and by
chemical structure (RDKit substructure tests, never by name), then matched against the rules of
the target markets:
- *Exclude* (default) drops anything **not permitted**, **restricted** or **eliminated** in a
  target market before the search.
- *Strict* also drops **check** flags.
- *Warn* keeps everything and shows badges.

Tallow amine ethoxylates are a structural pattern match, to be confirmed by CAS. Nonyl- and
octylphenol forms that the rules do not name (propoxylates, sulfates, phosphates, capped chains)
are flagged **check**, not banned.
"""
    )


def method_page() -> None:
    st.title("Method")
    st.markdown(
        f"""
**Data.** 40 candidates from SurfPro (CMC, γ_CMC, Γ_max, pC20). Measured values are used where
they exist; otherwise SurfPro's GNN-imputed values with σ = the paper's test-set RMSE. Candidates
whose measured pC20 contradicts their own CMC, γ_CMC and Γ_max by more than 3·σ_model are
flagged *inconsistent source data* and excluded by default.

**Equilibrium.** Langmuir–Szyszkowski isotherm per surfactant (K fitted through the CMC and
γ_CMC), ideal mixed micelles (Clint), and competitive Langmuir adsorption from the free monomer.
The mixed-layer equation of state weights n·Γ∞ by coverage.

**Dynamic surface tension as a bracket.** The real DST(t) lies between two bounds:
- *dst_low* (adsorbs too fast): short-time diffusion 2C√(Dt/π) with micelles counted as instant
  monomer and no back-diffusion, capped at equilibrium coverage.
- *dst_high* (adsorbs too slowly): Sutherland's solution for a linear isotherm with the secant
  Henry constant, free monomer only.

A **DST ≤** band must hold on *dst_high*; a **DST ≥** band on *dst_low*; γ_eq bands use the
equilibrium value.

**Robustness and certificate.** Each prediction carries σ from 200 Monte Carlo draws over CMC
(log-normal), γ_CMC and Γ_max (truncated normal), combined with σ_model, the larger of 2 mN/m
and the pC20-check RMSE. A blend is *feasible* when every band holds at mean ± z·σ on its
pessimistic side. It has *failed* when some band fails even on its optimistic side. Anything
in between is *undetermined*: worth testing in the lab, and it blocks the certificate. The search
stops at the smallest k with a feasible blend; the certificate states that every (k − 1) blend
failed. On this site a run is capped at {TIME_LIMIT_S} s, and by default Monte Carlo is run only
on the top candidates (fast mode). Tick *exhaustive certificate check* to classify every smaller
blend.

**What is not validated.**
- No prediction here has been checked against measured dynamic surface tension for these blends.
- The bracket can be wide above the CMC (tens of mN/m at short surface ages), so many blends end
  up *undetermined*.
- Ideal mixing (β = 0) and no transient competition: under this model a blend rarely beats the
  best single surfactant, and certificates beyond "water alone fails" are rarely reachable.
- Diffusion coefficients come from Wilke–Chang/Le Bas, with an assumed value for siloxanes.
- SurfPro's imputed values carry a random-split RMSE that probably understates error for
  agro-like surfactants.
- The pC20 check is weak, because many SurfPro pC20 values come from the same isotherm fits.
- Regulatory flags are a hardcoded snapshot, not legal advice.

**Next steps (not implemented):** surface synergy β^σ, transient competition, and a full
Ward–Tordai/Langmuir lower bound.
"""
    )
    doc = config.ROOT / "docs" / "physics.md"
    if doc.exists():
        with st.expander("Full model notes (docs/physics.md)"):
            st.markdown(doc.read_text())


def run() -> None:
    """Build the navigation and run the selected page."""
    st.set_page_config(page_title="wetopt", page_icon=":material/water_drop:", layout="wide")
    PAGES.update(
        home=st.Page(
            home_page, title="Home", icon=":material/home:", url_path="home", default=True
        ),
        optimiser=st.Page(
            optimiser_page, title="Optimiser", icon=":material/science:", url_path="optimiser"
        ),
        regulations=st.Page(
            regulations_page, title="Regulations", icon=":material/gavel:", url_path="regulations"
        ),
        method=st.Page(method_page, title="Method", icon=":material/functions:", url_path="method"),
    )
    st.navigation(list(PAGES.values()), position="top").run()
