"""Stickler website (Streamlit Community Cloud): Home, Find a recipe, Regulations, Method.

Entry point: streamlit_app.py at the repo root calls run(). Pages are plain functions so each can
be rendered on its own (tests use streamlit.testing.v1.AppTest.from_function).
"""

import numpy as np
import pandas as pd
import streamlit as st

from wetopt import config, regulatory, ui
from wetopt.app import AGES_MS, TIME_LIMIT_S, dst_curves, load_library, optimiser_page
from wetopt.questions import Answers, to_scenario
from wetopt.scenario import optimise_scenario

PAGES: dict[str, st.Page] = {}
SURFPRO = "https://doi.org/10.1039/D4DD00393D"
MIT_2016 = "https://doi.org/10.1038/ncomms12560"


@st.cache_data
def hero_curve() -> tuple[str, float, np.ndarray] | None:
    """The default recipe (wheat + azoxystrobin + EU) and its upper-bound DST curve, mN/m."""
    lib = load_library(str(config.LIBRARY_PATH))
    o = optimise_scenario(lib, to_scenario(Answers()), exhaustive=False)
    if o.result is None:
        return None
    top = o.feasible.iloc[0]
    curve = dst_curves(lib, tuple(top["names"]), tuple(float(w) for w in top["wv"]), o.sigma_model)
    return top["blend"], float(top["total_wv"]), curve["high_mean"]


def _go(page: str) -> None:
    if page in PAGES:
        st.switch_page(PAGES[page])


def home_page() -> None:
    ui.inject()
    left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with left:
        ui.render(
            '<div class="sk"><h1 class="sk-hero-title">The fewest surfactants that make your '
            'spray stick.</h1><p class="sk-lead">Tell Stickler what you are spraying and where it '
            "will be sold. It finds the adjuvant recipe with the fewest ingredients that wets the "
            "leaf in time, and shows that nothing smaller would.</p></div>"
        )
        st.write("")
        c1, c2, _ = st.columns([1, 1, 0.6])
        if c1.button("Find a recipe", type="primary", use_container_width=True):
            _go("optimiser")
        if c2.button("How it works", use_container_width=True):
            _go("method")
    with right:
        hero = hero_curve()
        if hero is not None:
            blend, wv, curve = hero
            svg = ui.dst_hero_svg(AGES_MS, curve, 200.0, 50.0, f"{blend}, {wv:.2g} % w/v")
            ui.render(
                f'<figure class="sk sk-figure" style="margin:0">{svg}<figcaption class="sk-figcaption">'
                f"Wheat sprayed with azoxystrobin: one surfactant, {ui.e(blend)} at {wv:.2g} % w/v, "
                "brings the droplet below the target in time. Upper-bound prediction from the "
                "Stickler model.</figcaption></figure>"
            )
    ui.render(
        f"""<div class="sk sk-points">
<div class="sk-point"><h3>Most spray misses</h3><p>Only about 2 % of sprayed pesticide stays on
the plant; the rest bounces or runs off (<a href="{MIT_2016}">MIT, 2016</a>).</p></div>
<div class="sk-point"><h3>Every co-formulant adds regulatory risk</h3><p>Each one needs REACH and product
authorisation in the EU and GB, and EPA approval as an inert in the US.</p></div>
<div class="sk-point"><h3>Fewer, with proof</h3><p>Stickler certifies, under its model, that no
smaller blend would meet your wetting target.</p></div>
</div>
<h2 class="sk sk-h2">How it works</h2>
<ol class="sk sk-steps">
<li><span><b>Answer five questions.</b> What you spray onto, how strongly it must stick, what is
in the tank, where it will be sold, and your limits.</span></li>
<li><span><b>Stickler checks every recipe.</b> One-, two- and three-surfactant blends from 40
curated surfactants, each tested against a dynamic surface tension bracket with uncertainty and
screened for regulations.</span></li>
<li><span><b>Get the smallest recipe that passes.</b> With the dose, a proof that nothing smaller
works, or the one change that would make your target reachable.</span></li>
</ol>
<p class="sk sk-credit">Surfactant data: SurfPro (Hödl et al., <i>Digital Discovery</i> 2025, 4,
1176; <a href="{SURFPRO}">doi:10.1039/D4DD00393D</a>), CC BY-NC-SA 4.0. The candidate library is
derived from it and shared under the same licence, for non-commercial use. Stickler is a
prototype: its predictions are model estimates, not measurements.</p>"""
    )


def regulations_page() -> None:
    ui.inject()
    regs = regulatory.load()
    ui.render(
        ui.page_header(
            "Regulations",
            f"The co-formulant rules Stickler screens for, as of {regs.as_of}. Pick your markets "
            "on the Find a recipe page.",
        )
    )
    ui.render(ui.note(regs.disclaimer))
    st.write("")
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
    ui.render(
        """<div class="sk sk-prose">
<h3>How the rules are applied</h3>
<p>Each ingredient is classified by its CAS number when listed and by its chemical structure
(RDKit substructure tests, never by name), then matched against the rules of your markets.</p>
<ul>
<li><b>Standard mode</b> leaves out anything not permitted, restricted or eliminated in a market
you sell in.</li>
<li><b>Precautionary mode</b> also leaves out anything flagged 'check'.</li>
</ul>
<p>Tallow amine ethoxylates are a structural pattern match, to be confirmed by CAS. Nonyl- and
octylphenol forms that the rules do not name (propoxylates, sulfates, phosphates, capped chains)
are flagged 'check', not banned.</p>
</div>"""
    )


def method_page() -> None:
    ui.inject()
    ui.render(
        ui.page_header(
            "Method",
            "How Stickler turns your answers into a recipe and a proof, and what it does not yet "
            "know.",
        )
    )
    ui.render(
        f"""<div class="sk sk-prose">
<h3>Data</h3>
<p>40 candidate surfactants from SurfPro, with CMC, surface tension at the CMC, maximum surface
excess and pC20. Measured values are used where they exist. Otherwise Stickler uses SurfPro's
GNN-imputed values, with σ set to the paper's test-set error. A candidate whose measured pC20
contradicts its own CMC, γ_CMC and Γ_max by more than 3·σ_model is excluded as inconsistent
source data.</p>
<h3>Equilibrium</h3>
<p>Each surfactant follows a Langmuir–Szyszkowski isotherm fitted through its CMC and γ_CMC.
Mixed micelles are ideal (Clint), and the surfactants compete for the surface by Langmuir
adsorption from the free monomer.</p>
<h3>Dynamic surface tension as a bracket</h3>
<p>The real DST(t) lies between two bounds. The lower bound, dst_low, adsorbs too fast: it uses
short-time diffusion with micelles counted as instant monomer and no back-diffusion. The upper
bound, dst_high, adsorbs too slowly: it uses Sutherland's linear-isotherm solution with free
monomer only. A 'DST ≤' target must hold on the upper bound.</p>
<h3>Robustness and proof</h3>
<p>Every prediction carries σ from 200 Monte Carlo draws over the inputs, combined with σ_model.
A recipe passes when every target holds at mean ± z·σ on its pessimistic side. It fails when a
target fails even on its optimistic side. Anything in between is undetermined: worth testing in
the lab, and it blocks the certificate. Stickler stops at the smallest number of ingredients
with a recipe that passes. The proof states that every smaller blend failed. Runs on this site
stop after {TIME_LIMIT_S} s.</p>
<h3>Not yet validated</h3>
<ul>
<li>No prediction here has been checked against measured DST for these blends.</li>
<li>The bracket can be wide above the CMC, so many blends end up undetermined.</li>
<li>Ideal mixing and no transient competition mean a blend rarely beats the best single
surfactant, and proofs beyond 'water alone fails' are rarely reachable.</li>
<li>Diffusion coefficients come from Wilke–Chang/Le Bas, with an assumed value for siloxanes.</li>
<li>The active ingredient's own surface activity, water temperature, hardness and droplet size
are recorded, not modelled.</li>
<li>Regulatory flags are a hardcoded snapshot, not legal advice.</li>
</ul>
<h3>Next steps</h3>
<p>Surface synergy (β^σ), transient competition, and a full Ward–Tordai/Langmuir lower bound.</p>
</div>"""
    )
    doc = config.ROOT / "docs" / "physics.md"
    if doc.exists():
        with st.expander("Full model notes"):
            st.markdown(doc.read_text())


def run() -> None:
    """Build the navigation and run the selected page."""
    st.set_page_config(
        page_title="Stickler: adjuvant recipes that stick",
        page_icon=str(config.ROOT / "assets" / "stickler-icon.svg"),
        layout="wide",
    )
    st.logo(
        str(config.ROOT / "assets" / "stickler-logo.svg"),
        size="large",
    )
    PAGES.update(
        home=st.Page(home_page, title="Home", icon=":material/home:", default=True),
        optimiser=st.Page(
            optimiser_page, title="Find a recipe", icon=":material/water_drop:", url_path="recipe"
        ),
        regulations=st.Page(
            regulations_page, title="Regulations", icon=":material/gavel:", url_path="regulations"
        ),
        method=st.Page(method_page, title="Method", icon=":material/functions:", url_path="method"),
    )
    st.navigation(list(PAGES.values()), position="top").run()
