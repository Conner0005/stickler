"""Optimiser page: set a wetting spec, get the fewest-surfactant blend and its certificate.

Part of the multipage site (streamlit_app.py → wetopt.site); also runs alone with
`uv run streamlit run src/wetopt/app.py`. Built on library.py (data/library.parquet),
physics.py, optimise.py and regulatory.py; no physics of its own.
Display units (mN/m, ms, % w/v, mM) appear only here.
"""

import hashlib
import json
from typing import get_args

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

from wetopt import config, regulatory
from wetopt import optimise as opt
from wetopt.physics import SurfactantArrays, gamma_water, monte_carlo_dst
from wetopt.questions import (
    CONFIDENCE,
    OTHER_AI,
    OTHER_CROP,
    STICK,
    STICK_ORDER,
    STICK_SOURCE,
    TIPS,
    Answers,
    suggest_relaxation,
    to_scenario,
    what_this_means,
)
from wetopt.scenario import (
    AI_NOTE,
    AI_PRESETS,
    DEFAULT_AGE_MS,
    DIFFICULT_LEAVES,
    NOT_MODELLED,
    coformulant_tank_cap_wv,
    compatibility_exclusions,
    hard_water_flag,
    input_summary,
    optimise_scenario,
    temperature_warning,
    time_to_impact_ms,
)
from wetopt.schema import ActiveIngredient, AICharge, IonicClass, Market, Scenario, Spec
from wetopt.validate import INCONSISTENT, data_quality, sigma_model

PROPS = {"cmc": "CMC", "gamma_cmc": "γ_CMC", "gamma_max": "Γ_max", "pc20": "pC20"}
# Series colours: dataviz reference palette slots 1–2 (validated, CVD ΔE 24.7); windows neutral.
WINNER, RUNNER = "#2a78d6", "#eb6834"
WINDOW_FILL = "rgba(137, 135, 129, 0.22)"
EQ_FILL = "rgba(137, 135, 129, 0.12)"
AGES_MS = np.geomspace(10, 1000, 41)
TIME_LIMIT_S = 60  # cap per optimiser run on the website (Community Cloud free tier)
RELAX_TIME_S = 15  # cap per trial when looking for a single relaxation
STATUS_ICON = {"not_permitted": "⛔", "eliminated": "⛔", "restricted": "🚫", "check": "⚠️"}


@st.cache_data
def load_library(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).reset_index(drop=True)


@st.cache_data
def surfactant_arrays(lib: pd.DataFrame) -> SurfactantArrays:
    return SurfactantArrays.from_library(lib)


@st.cache_data
def dst_curves(lib: pd.DataFrame, names: tuple, wv: tuple, sigma_model: float) -> dict:
    """Monte Carlo DST bracket (mN/m) for one blend over AGES_MS (same draws as the optimiser)."""
    idx = np.array([[int(np.flatnonzero(lib["name"] == n)[0]) for n in names]])
    conc = opt.WV_TO_KG_M3 * np.array([wv]) / lib["mw"].to_numpy()[idx]
    s = monte_carlo_dst(AGES_MS / 1e3, idx, conc, surfactant_arrays(lib), sigma_model=sigma_model)
    return {k: getattr(s, k)[0] * 1e3 for k in ("low_mean", "low_std", "high_mean", "high_std")}


@st.cache_data
def regulatory_flags(lib: pd.DataFrame, markets: tuple[str, ...]) -> dict[str, list]:
    """Name → regulatory flags in the target markets (regulatory.screen)."""
    sc = regulatory.screen(lib, list(markets))
    return dict(zip(sc["name"], sc["flags"], strict=True))


@st.cache_data
def quality_table(lib: pd.DataFrame) -> pd.DataFrame:
    """Data-quality status and reason per candidate (validate.data_quality at 3·σ_model)."""
    return data_quality(lib, config.QC_SIGMA_MULT * sigma_model(lib))


def badge(flags: list) -> str:
    """Compact regulatory badge for one ingredient, e.g. '⚠️ EU check · ⚠️ GB check'."""
    if not flags:
        return "no flags"
    return " · ".join(
        sorted({f"{STATUS_ICON[f.status]} {f.market} {f.status.replace('_', ' ')}" for f in flags})
    )


def badges(names: list[str], flags: dict[str, list]) -> str:
    return "; ".join(f"{n}: {badge(flags.get(n, []))}" for n in names)


def _w(widget, label: str, key: str, default, **kwargs):
    """Widget whose initial value lives in session state, so buttons can set it."""
    if key not in st.session_state:
        st.session_state[key] = default
    return widget(label, key=key, **kwargs)


def questions() -> Answers:
    """The five plain-language questions (sidebar)."""
    sb, d = st.sidebar, Answers()
    sb.header("1. What are you spraying onto?")
    crops = [c.capitalize() for c in DIFFICULT_LEAVES] + [OTHER_CROP]
    crop = _w(
        sb.selectbox,
        "Crop (all listed crops are hard to wet)",
        "q:crop",
        "Wheat",
        options=crops,
        help=TIPS["crop"],
    )
    crop = crop if crop == OTHER_CROP else crop.lower()
    beads = "not sure"
    if crop == OTHER_CROP:
        beads = _w(
            sb.radio,
            "Do water droplets bead up and roll off the leaf?",
            "q:beads",
            "not sure",
            options=["yes", "not sure", "no"],
            horizontal=True,
            help=TIPS["beads"],
        )
    hairy = _w(sb.toggle, "Hairy leaves?", "q:hairy", False, help=TIPS["hairy"])
    if hairy:
        sb.caption("Hairy leaves shatter droplets: consider the Strong stick level.")

    sb.header("2. How strongly should the spray stick?")
    hard = crop in DIFFICULT_LEAVES or beads != "no"
    stick = _w(
        sb.radio,
        "Stick level",
        f"q:stick:{'hard' if hard else 'easy'}",
        "standard" if hard else "gentle",
        options=STICK_ORDER,
        horizontal=True,
        format_func=str.capitalize,
        help=TIPS["stick"],
    )
    sb.caption(
        f"{stick.capitalize()} = DST(200 ms) ≤ {STICK[stick] * 1e3:g} mN/m ({STICK_SOURCE[stick]})."
    )

    sb.header("3. What's in the tank?")
    ai = _w(
        sb.selectbox,
        "Active ingredient",
        "q:ai",
        d.ai,
        options=[*AI_PRESETS, OTHER_AI],
        help=TIPS["ai"],
    )
    ai_name, ai_charge = "", "nonionic"
    if ai == OTHER_AI:
        ai_name = _w(sb.text_input, "Name", "q:ai_name", "")
        ai_charge = _w(
            sb.radio,
            "Charge",
            "q:ai_charge",
            "nonionic",
            options=list(get_args(AICharge)),
            horizontal=True,
        )
    c1, c2 = sb.columns([2, 1])
    rate = _w(
        c1.number_input,
        "Product rate (from your product label)",
        "q:rate",
        None,
        min_value=0.0,
        placeholder="—",
        help=TIPS["rate"],
    )
    rate_unit = _w(c2.selectbox, "unit", "q:rate_unit", "L/ha", options=["L/ha", "kg/ha"])
    volume = _w(
        sb.number_input,
        "Spray volume, L/ha (from your product label)",
        "q:volume",
        None,
        min_value=0.0,
        placeholder="—",
        help=TIPS["volume"],
    )
    c1, c2 = sb.columns(2)
    hardness = _w(
        c1.selectbox,
        "Water hardness",
        "q:hardness",
        "medium",
        options=["soft", "medium", "hard"],
        help=TIPS["hardness"],
    )
    drops = _w(
        c2.selectbox,
        "Nozzle droplet size",
        "q:drops",
        "medium",
        options=["fine", "medium", "coarse"],
        help=TIPS["droplets"],
    )

    sb.header("4. Where will it be sold?")
    cols = sb.columns(4)
    markets = [
        m
        for m, c in zip(get_args(Market), cols, strict=True)
        if _w(c.checkbox, m, f"q:mk:{m}", m in d.markets, help=TIPS["markets"])
    ]
    precautionary = _w(
        sb.toggle, "Precautionary mode", "q:precaution", False, help=TIPS["precaution"]
    )

    sb.header("5. Preferences")
    max_k = _w(
        sb.radio,
        "Most ingredients you'd accept",
        "q:max_k",
        3,
        options=[1, 2, 3],
        horizontal=True,
        help=TIPS["max_k"],
    )
    cap = _w(
        sb.number_input,
        "Max adjuvant in the tank, % w/v",
        "q:cap",
        0.2,
        min_value=0.001,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        help=TIPS["cap"],
    )
    sb.caption(f"= {cap * 10:.3g} g per litre")
    confidence = _w(
        sb.radio,
        "Confidence",
        "q:conf",
        "standard",
        options=["standard", "high"],
        horizontal=True,
        format_func=lambda c: f"{c.capitalize()} ({CONFIDENCE[c][1]})",
        help=TIPS["confidence"],
    )
    return Answers(
        crop=crop,
        beads=beads,
        hairy=hairy,
        stick=stick,
        ai=ai,
        ai_name=ai_name,
        ai_charge=ai_charge,
        product_rate=rate or None,
        rate_unit=rate_unit,
        spray_volume_l_ha=volume or None,
        hardness=hardness,
        droplet_size=drops,
        markets=markets,
        precautionary=precautionary,
        max_k=max_k,
        max_tank_wv=cap,
        confidence=confidence,
    )


def advanced(lib: pd.DataFrame, derived: Scenario) -> tuple[Scenario, bool, bool, bool] | None:
    """Raw controls, prefilled from the answers; returns (scenario, custom, qc, exhaustive)."""
    sig = hashlib.md5(derived.model_dump_json().encode()).hexdigest()[:8]

    def k(name: str) -> str:  # reset to the answer-derived values whenever the answers change
        return f"adv:{sig}:{name}"

    wt, cons0, ai0, sp0 = (
        derived.wetting_target,
        derived.constraints,
        derived.active_ingredient,
        derived.spray,
    )
    with st.sidebar.expander("Advanced", expanded=False):
        st.caption("Raw controls. Any change here switches to custom targets.")
        st.markdown("**Spray details** (time to impact)")
        c1, c2 = st.columns(2)
        height = c1.number_input(
            "Boom height / m", min_value=0.0, value=None, key=k("boom"), placeholder="—"
        )
        speed = c2.number_input(
            "Droplet speed / m/s", min_value=0.0, value=None, key=k("speed"), placeholder="—"
        )
        tti = time_to_impact_ms(height or None, speed or None)
        if tti is not None:
            st.caption(f"Estimated time to impact: **{tti:.0f} ms** (height ÷ speed).")
            if st.button(f"Use {tti:.0f} ms as DST surface age", key=k("use_tti")):
                st.session_state[k("age0")] = round(tti, 1)
        temp = st.number_input(
            "Water temperature / °C",
            -5.0,
            60.0,
            25.0,
            step=1.0,
            key=k("temp"),
            help=f"{NOT_MODELLED}; the physics stays at 25 °C.",
        )
        if warn := temperature_warning(temp):
            st.warning(warn)
        theta = st.number_input(
            "θ of 50 % aqueous acetone / °",
            min_value=0.0,
            max_value=180.0,
            value=None,
            key=k("theta"),
            placeholder="—",
            help=NOT_MODELLED,
        )

        st.markdown("**Wetting target**")
        dst0 = [b for b in wt.bands if b.metric == "dst"]
        n_dst = _w(
            st.number_input, "DST bands", k("n_dst"), len(dst0), min_value=0, max_value=3, step=1
        )
        bands = []
        for i in range(int(n_dst)):
            b = dst0[i] if i < len(dst0) else None
            age0, ge = (
                (b.surface_age_ms, b.lo > 0 and b.hi >= 0.072) if b else (DEFAULT_AGE_MS, False)
            )
            val0 = (b.lo if ge else b.hi) * 1e3 if b else 50.0
            c1, c2, c3 = st.columns([1.2, 0.8, 1.2])
            age = _w(
                c1.number_input,
                f"Age {i + 1} / ms",
                k(f"age{i}"),
                float(age0),
                min_value=1.0,
                max_value=10_000.0,
            )
            op = _w(c2.selectbox, "≤ / ≥", k(f"op{i}"), "≥" if ge else "≤", options=["≤", "≥"])
            val = _w(
                c3.number_input,
                "mN/m",
                k(f"val{i}"),
                float(val0),
                min_value=15.0,
                max_value=72.0,
                step=0.5,
            )
            lo, hi = (val / 1e3, 0.072) if op == "≥" else (0.0, val / 1e3)
            bands.append({"metric": "dst", "surface_age_ms": age, "lo": lo, "hi": hi})
        if st.checkbox("γ_eq band", False, key=k("eq")):
            c1, c2 = st.columns(2)
            lo = c1.number_input("γ_eq ≥ / mN/m", 0.0, 72.0, 0.0, step=0.5, key=k("eqlo"))
            hi = c2.number_input("γ_eq ≤ / mN/m", 0.0, 72.0, 72.0, step=0.5, key=k("eqhi"))
            bands.append({"metric": "gamma_eq", "lo": lo / 1e3, "hi": hi / 1e3})
        z = st.number_input("Robustness z", 0.0, 3.0, float(wt.z), step=0.01, key=k("z"))

        st.markdown("**Ingredients**")
        classes = st.multiselect(
            "Allowed classes", list(get_args(IonicClass)), cons0.allowed_classes, key=k("classes")
        )
        excluded = st.multiselect("Exclude surfactants", sorted(lib["name"]), [], key=k("ex"))
        qc = st.checkbox(
            "Data-quality filter",
            config.EXCLUDE_INCONSISTENT,
            key=k("qc"),
            help="Exclude candidates flagged 'inconsistent source data'.",
        )
        exhaustive = st.checkbox(
            "Exhaustive certificate check (slower)",
            False,
            key=k("exhaustive"),
            help=f"Classify every smaller blend; may hit the {TIME_LIMIT_S} s run limit.",
        )

        st.markdown("**Product**")
        forms = list(get_args(ActiveIngredient.model_fields["formulation"].annotation))
        formulation = st.selectbox(
            "Formulation type", forms, forms.index(ai0.formulation), key=k("form")
        )
        c1, c2 = st.columns([2, 1])
        content = c1.number_input(
            "AI content", min_value=0.0, value=None, key=k("content"), placeholder="—"
        )
        cunit = c2.selectbox("unit", ["g/L", "g/kg"], key=k("cunit"))
        st.caption(AI_NOTE)
        density = st.number_input(
            "Concentrate density / kg/L",
            min_value=0.0,
            value=None,
            key=k("density"),
            placeholder="—",
        )
        load = st.number_input(
            "Max co-formulant load / % w/w",
            min_value=0.0,
            max_value=100.0,
            value=None,
            key=k("load"),
            placeholder="—",
            help="Most surfactant the concentrate can carry; with rate, volume and density it "
            "caps the tank adjuvant.",
        )

    try:
        ai = ai0.model_copy(
            update={
                "formulation": formulation,
                "content": content or None,
                "content_unit": cunit,
                "concentrate_density_kg_l": density or None,
                "max_coformulant_wt_pct": load or None,
            }
        )
        spray = sp0.model_copy(
            update={
                "boom_height_m": height or None,
                "droplet_speed_m_s": speed or None,
                "water_temperature_c": temp,
            }
        )
        cap = cons0.max_total_wv
        in_can = coformulant_tank_cap_wv(ai, spray)
        if in_can is not None:
            st.sidebar.caption(f"In-can load caps the tank adjuvant at {in_can:.3g} % w/v.")
            cap = min(cap, in_can)
        scn = Scenario(
            active_ingredient=ai,
            target_leaf=derived.target_leaf.model_copy(update={"theta_acetone50_deg": theta}),
            spray=spray,
            wetting_target=Spec.model_validate({"bands": bands, "z": z}),
            constraints=cons0.model_copy(
                update={
                    "allowed_classes": classes,
                    "excluded_ids": excluded,
                    "max_total_wv": cap,
                }
            ),
        )
    except ValidationError as err:
        msg = err.errors()[0]["msg"]
        st.sidebar.error("Add at least one band." if not bands else f"Inputs are not valid: {msg}")
        return None
    custom = (scn.wetting_target, scn.constraints) != (
        wt,
        cons0,
    ) or qc != config.EXCLUDE_INCONSISTENT
    return scn, custom, qc, exhaustive


def provenance(lib: pd.DataFrame, names: list[str]) -> str:
    """'measured' or the properties imputed by SurfPro's GNN, per component."""
    rows = lib.set_index("name").loc[names]
    parts = []
    for name, r in rows.iterrows():
        imputed = [label for col, label in PROPS.items() if not r[f"{col}_measured"]]
        parts.append(f"{name}: " + ("measured" if not imputed else "imputed " + ", ".join(imputed)))
    return "; ".join(parts)


def dst_figure(lib: pd.DataFrame, o: opt.Outcome, spec: Spec) -> go.Figure:
    """DST(t) bracket ± σ for the winner and the best (k − 1) blend, with spec windows."""
    fig = go.Figure()
    curves = []
    if o.result is not None:
        top = o.feasible.iloc[0]
        curves.append(
            (f"Winner (k = {o.result.k}): {top['blend']}", top["names"], top["wv"], WINNER)
        )
    ru = o.runner_up
    if ru and ru["k"] > 0:
        label = " + ".join(f"{n} {w:.3g}%" for n, w in zip(ru["names"], ru["wv"], strict=True))
        title = "Best (k − 1)" if o.result is not None else "Closest blend"
        curves.append((f"{title} (k = {ru['k']}): {label}", ru["names"], ru["wv"], RUNNER))

    lows = []
    x = AGES_MS
    for name, names, wv, color in curves:
        c = dst_curves(lib, tuple(names), tuple(float(w) for w in wv), o.sigma_model)
        for bound, dash, what in (
            ("high", "solid", "dst_high (upper bound)"),
            ("low", "dash", "dst_low (lower bound)"),
        ):
            m, s = c[f"{bound}_mean"], c[f"{bound}_std"]
            lows.append(m - s)
            fig.add_trace(
                go.Scatter(
                    x=np.r_[x, x[::-1]],
                    y=np.r_[m + s, (m - s)[::-1]],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.14,
                    line={"width": 0},
                    hoverinfo="skip",
                    legendgroup=name,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=m,
                    mode="lines",
                    line={"color": color, "width": 2, "dash": dash},
                    name=f"{name} · {what}",
                    legendgroup=name,
                    hovertemplate="%{y:.1f} mN/m",
                )
            )
    g0 = float(gamma_water()) * 1e3
    if ru and ru["k"] == 0 and o.result is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(x, g0),
                mode="lines",
                line={"color": RUNNER, "width": 2},
                name=f"Best (k − 1 = 0): water, {g0:.1f} mN/m",
                hovertemplate="%{y:.1f} mN/m",
            )
        )
    y_min = max(
        15.0,
        min([np.min(v) for v in lows] + [b.lo * 1e3 for b in spec.bands if b.lo > 0] + [60.0]) - 3,
    )
    for b in spec.bands:
        lo, hi = max(b.lo * 1e3, y_min), min(b.hi * 1e3, g0 + 2)
        if b.metric == "dst":
            a = b.surface_age_ms
            xs, fill = [a / 1.25, a * 1.25, a * 1.25, a / 1.25, a / 1.25], WINDOW_FILL
        else:
            xs, fill = [x[0], x[-1], x[-1], x[0], x[0]], EQ_FILL
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[lo, lo, hi, hi, lo],
                fill="toself",
                fillcolor=fill,
                mode="lines",
                line={"color": "rgba(137, 135, 129, 0.6)", "width": 1},
                name=f"Spec window: {opt.band_label(b)}"
                + (" (t → ∞)" if b.metric == "gamma_eq" else ""),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        hovermode="x unified",
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
        xaxis={
            "title": "Surface age / ms",
            "type": "log",
            "tickvals": [10, 20, 50, 100, 200, 500, 1000],
        },
        yaxis={"title": "Dynamic surface tension / mN/m", "range": [y_min, g0 + 2]},
    )
    return fig


def reg_badges(names: list[str], flags: dict[str, list], scn: Scenario, lib: pd.DataFrame) -> str:
    """Regulatory badges plus the hard-water badge (recorded, not modelled), per ingredient."""
    classes = dict(zip(lib["name"], lib["ionic_class"], strict=True))
    parts = []
    for n in names:
        hw = hard_water_flag(scn, classes[n])
        parts.append(f"{n}: {badge(flags.get(n, []))}" + (f" · {hw}" if hw else ""))
    return "; ".join(parts)


def apply_answers(a: Answers) -> None:
    """Button callback: write answers back into the question widgets and re-run."""
    ss = st.session_state
    ss["q:crop"] = a.crop.capitalize() if a.crop in DIFFICULT_LEAVES else OTHER_CROP
    ss["q:beads"], ss["q:hairy"] = a.beads, a.hairy
    hard = a.crop in DIFFICULT_LEAVES or a.beads != "no"
    if a.stick:
        ss[f"q:stick:{'hard' if hard else 'easy'}"] = a.stick
    ss["q:ai"], ss["q:ai_name"], ss["q:ai_charge"] = a.ai, a.ai_name, a.ai_charge
    ss["q:rate"], ss["q:rate_unit"], ss["q:volume"] = (
        a.product_rate,
        a.rate_unit,
        a.spray_volume_l_ha,
    )
    ss["q:hardness"], ss["q:drops"] = a.hardness, a.droplet_size
    for m in get_args(Market):
        ss[f"q:mk:{m}"] = m in a.markets
    ss["q:precaution"], ss["q:max_k"], ss["q:cap"], ss["q:conf"] = (
        a.precautionary,
        a.max_k,
        a.max_tank_wv,
        a.confidence,
    )
    ss["autorun"] = True


def show(lib: pd.DataFrame, run: dict) -> None:
    o, scn = run["o"], run["scn"]
    spec, cons, r = scn.wetting_target, scn.constraints, o.result
    flags = regulatory_flags(lib, tuple(cons.markets))
    with st.container(border=True):
        if r is None:
            st.subheader("No recipe meets the target")
            relax = run.get("relax")
            if run["custom"]:
                st.write("These are custom targets: relax them in Advanced.")
            elif relax:
                label, relaxed, ro = relax
                top = ro.feasible.iloc[0]
                st.write(
                    f"One change would work: **{label}** → {top['blend']} at "
                    f"{top['total_wv']:.3g} % w/v."
                )
                st.button(
                    f"Apply: {label}", type="primary", on_click=apply_answers, args=(relaxed,)
                )
            else:
                st.write(
                    "No single change (double the tank limit, lower stick level, one more "
                    "ingredient, one market fewer) finds a recipe."
                )
        else:
            top = o.feasible.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Ingredients", r.k)
            c2.metric("Adjuvant in the tank", f"{top['total_wv']:.3g} % w/v")
            c3.metric("Safety margin", f"{min(r.margins) * 1e3:+.1f} mN/m")
            comp = pd.DataFrame(
                {
                    "surfactant": top["names"],
                    "% w/v": [round(w, 4) for w in top["wv"]],
                    "g per litre": [round(w * 10, 3) for w in top["wv"]],
                    "mM": [round(c, 3) for c in r.blend.root.values()],
                    "regulatory": [
                        reg_badges([n], flags, scn, lib).split(": ", 1)[1] for n in top["names"]
                    ],
                }
            )
            st.dataframe(comp, hide_index=True)
            details = [f"{n}: {t}" for n, ts in r.regulatory_flags.items() for t in ts]
            if details:
                with st.expander("Regulatory flags on the recipe"):
                    for line in details:
                        st.markdown(f"- {line}")
        if o.truncated:
            st.warning(o.truncated)
        st.caption(o.regulatory_note)

    with st.expander("What each input did"):
        st.markdown("\n".join(f"- {line}" for line in input_summary(scn, o, lib)))
        if len(o.excluded):
            st.dataframe(
                o.excluded.rename(columns={"name": "excluded before the search"}), hide_index=True
            )

    st.plotly_chart(dst_figure(lib, o, spec), theme="streamlit")
    st.caption(
        "Solid lines: dst_high (upper bound, used for 'DST ≤' bands); dashed: dst_low (lower "
        "bound). Shading: ± σ (Monte Carlo ⊕ σ_model). Grey boxes: spec windows."
    )

    (st.success if o.certified else st.warning)(o.certificate_plain or o.certificate)
    with st.expander("Full certificate"):
        st.write(o.certificate)

    if r is not None:
        st.subheader("Top feasible recipes")
        tab = o.feasible.head(10).copy()
        tab.insert(0, "#", range(1, len(tab) + 1))
        tab["% w/v"] = tab["wv"].map(lambda w: " + ".join(f"{x:.3g}" for x in w))
        tab["worst margin / mN/m"] = (tab["worst_margin"] * 1e3).round(2)
        tab["provenance"] = tab["names"].map(lambda n: provenance(lib, n))
        tab["regulatory"] = tab["names"].map(lambda n: reg_badges(n, flags, scn, lib))
        cols = [
            "#",
            "blend",
            "% w/v",
            "total_wv",
            "worst margin / mN/m",
            "provenance",
            "regulatory",
        ]
        st.dataframe(tab[cols].rename(columns={"total_wv": "total % w/v"}), hide_index=True)
    if len(o.undetermined):
        st.subheader("Undetermined, worth testing in the lab")
        und = o.undetermined.head(10).copy()
        und["% w/v"] = und["wv"].map(lambda w: " + ".join(f"{x:.3g}" for x in w))
        und["pessimistic margin / mN/m"] = (und["worst_margin"] * 1e3).round(2)
        und["regulatory"] = und["names"].map(lambda n: reg_badges(n, flags, scn, lib))
        st.dataframe(
            und[["blend", "% w/v", "pessimistic margin / mN/m", "regulatory"]], hide_index=True
        )
    st.caption(o.regulatory_note)

    payload = {
        "scenario": scn.model_dump(mode="json"),
        "answers": run["answers"].model_dump(mode="json"),
        "custom_targets": run["custom"],
        "what_this_means": run["line"],
        **opt.outcome_payload(o, spec, cons, "app"),
    }
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download results JSON",
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        file_name="wetopt_results.json",
        mime="application/json",
    )
    c2.download_button(
        "Download scenario YAML",
        yaml.safe_dump(scn.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        file_name="wetopt_scenario.yaml",
        mime="application/x-yaml",
    )


def quality_panel(lib: pd.DataFrame, exclude: bool) -> None:
    """Data-quality exclusions with their reasons (shown before and after a run)."""
    qc = quality_table(lib)
    bad = qc[qc["status"] == INCONSISTENT]
    n_unchecked = int((qc["status"] == "unchecked").sum())
    title = f"Data-quality exclusions: {len(bad)}" + ("" if exclude else " (toggle off: kept)")
    with st.expander(title, expanded=bool(len(bad))):
        if len(bad):
            st.dataframe(
                bad[["name", "reason"]].rename(columns={"name": "surfactant"}), hide_index=True
            )
        st.caption(
            f"Inconsistent source data: the measured pC20 disagrees with the candidate's own CMC, "
            f"γ_CMC and Γ_max by more than {config.QC_SIGMA_MULT:g}·σ_model. {n_unchecked} "
            "candidates have no measured pC20 and are unchecked (kept)."
        )


@st.cache_data
def regulatory_count(lib: pd.DataFrame, markets: tuple[str, ...], policy: str) -> int:
    """Candidates the regulatory policy removes in the selected markets."""
    statuses = regulatory.excluded_statuses(policy)
    flags = regulatory_flags(lib, markets)
    return sum(any(f.status in statuses for f in fl) for fl in flags.values())


def optimiser_page() -> None:
    st.title("Find a recipe")
    st.caption(
        "Answer five questions; the tool finds the adjuvant recipe with the fewest ingredients "
        "that makes the spray stick, and checks that nothing smaller would."
    )
    lib = load_library(str(config.LIBRARY_PATH))
    answers = questions()
    find = st.sidebar.button("Find recipe", type="primary", use_container_width=True)
    adv = advanced(lib, to_scenario(answers))
    if adv is None:
        return
    scn, custom, qc, exhaustive = adv
    n_charge = len(compatibility_exclusions(lib, scn.active_ingredient))
    n_reg = regulatory_count(lib, tuple(scn.constraints.markets), scn.constraints.policy)
    line = what_this_means(answers, custom, n_charge, n_reg, scn)
    st.info(f"**What this means:** {line}")

    autorun = st.session_state.pop("autorun", False) or "run" not in st.session_state
    if find or autorun:
        bar = st.progress(0.0, text="Starting…")
        o = optimise_scenario(
            lib,
            scn,
            exclude_inconsistent=qc,
            exhaustive=exhaustive,
            time_limit_s=TIME_LIMIT_S,
            progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg),
        )
        relax = None
        if o.result is None and not custom:
            with st.spinner("Nothing meets the target; looking for one change that would work…"):
                relax = suggest_relaxation(
                    answers,
                    lambda b: optimise_scenario(
                        lib,
                        to_scenario(b),
                        exclude_inconsistent=qc,
                        exhaustive=False,
                        time_limit_s=RELAX_TIME_S,
                    ),
                )
        st.session_state["run"] = {
            "o": o,
            "scn": scn,
            "answers": answers,
            "custom": custom,
            "qc": qc,
            "exhaustive": exhaustive,
            "line": line,
            "relax": relax,
        }
    run = st.session_state["run"]
    if (run["scn"], run["qc"], run["exhaustive"]) != (scn, qc, exhaustive):
        st.warning("Your answers have changed since this result. Press **Find recipe** to update.")
    quality_panel(lib, qc)
    show(lib, run)


def main() -> None:
    st.set_page_config(page_title="wetopt", layout="wide")
    optimiser_page()


if __name__ == "__main__":
    main()
