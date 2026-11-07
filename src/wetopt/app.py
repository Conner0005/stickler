"""wetopt Streamlit app: set a wetting spec, get the fewest-surfactant blend and its certificate.

Run: uv run streamlit run src/wetopt/app.py
Built on library.py (data/library.parquet), physics.py and optimise.py; no physics of its own.
Display units (mN/m, ms, % w/v, mM) appear only here.
"""

import json
from typing import get_args

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

from wetopt import config
from wetopt import optimise as opt
from wetopt.physics import SurfactantArrays, gamma_water, monte_carlo_dst
from wetopt.schema import Constraints, IonicClass, Market, Spec

EXAMPLES = config.ROOT / "examples"
PROPS = {"cmc": "CMC", "gamma_cmc": "γ_CMC", "gamma_max": "Γ_max", "pc20": "pC20"}
# Series colours: dataviz reference palette slots 1–2 (validated, CVD ΔE 24.7); windows neutral.
WINNER, RUNNER = "#2a78d6", "#eb6834"
WINDOW_FILL = "rgba(137, 135, 129, 0.22)"
EQ_FILL = "rgba(137, 135, 129, 0.12)"
AGES_MS = np.geomspace(10, 1000, 41)


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
def load_presets() -> dict[str, dict]:
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(EXAMPLES.glob("*.yaml"))}


def sidebar(lib: pd.DataFrame) -> tuple[str, Spec, Constraints, bool] | None:
    """Spec and constraints from the sidebar; None if they do not validate."""
    sb = st.sidebar
    presets = load_presets()
    preset = sb.selectbox("Preset", [*presets, "custom"])
    if "tight" not in presets:
        sb.caption(
            "No *tight* preset: the γ_eq scan found no spec where a pair beats every single "
            "surfactant under the current model (docs/physics.md)."
        )
    base = presets.get(preset) or next(iter(presets.values()))
    spec0, cons0 = base["spec"], base["constraints"]

    def key(name: str) -> str:  # widgets reset to the preset's values when the preset changes
        return f"{preset}:{name}"

    sb.subheader("Bands")
    dst0 = [b for b in spec0["bands"] if b["metric"] == "dst"]
    eq0 = next((b for b in spec0["bands"] if b["metric"] == "gamma_eq"), None)
    bands = []
    n_dst = sb.number_input("DST bands (upper limits)", 0, 3, len(dst0), key=key("n_dst"))
    for i in range(int(n_dst)):
        d = dst0[i] if i < len(dst0) else {"surface_age_ms": 200.0, "hi": 0.050}
        c1, c2 = sb.columns(2)
        age = c1.number_input(
            f"Age {i + 1} / ms", 1.0, 10_000.0, float(d["surface_age_ms"]), key=key(f"age{i}")
        )
        hi = c2.number_input(
            "DST ≤ / mN/m", 15.0, 72.0, float(d["hi"]) * 1e3, step=0.5, key=key(f"hi{i}")
        )
        bands.append({"metric": "dst", "surface_age_ms": age, "lo": 0.0, "hi": hi / 1e3})
    if sb.checkbox("γ_eq band", eq0 is not None, key=key("eq")):
        e = eq0 or {"lo": 0.0, "hi": 0.035}
        c1, c2 = sb.columns(2)
        lo = c1.number_input("γ_eq ≥ / mN/m", 0.0, 72.0, e["lo"] * 1e3, step=0.5, key=key("eqlo"))
        hi = c2.number_input(
            "γ_eq ≤ / mN/m", 0.0, 72.0, min(e["hi"], 0.072) * 1e3, step=0.5, key=key("eqhi")
        )
        bands.append({"metric": "gamma_eq", "lo": lo / 1e3, "hi": hi / 1e3})

    sb.subheader("Search")
    max_k = sb.slider("Max k (surfactants)", 1, 3, int(cons0.get("max_k", 3)), key=key("k"))
    cap = sb.number_input(
        "Total loading cap / % w/v",
        0.001,
        5.0,
        float(cons0["max_total_wv"]),
        step=0.01,
        format="%.3f",
        key=key("cap"),
    )
    classes = sb.multiselect(
        "Allowed classes",
        list(get_args(IonicClass)),
        cons0.get("allowed_classes", list(get_args(IonicClass))),
        key=key("classes"),
    )
    excluded = sb.multiselect(
        "Exclude surfactants", sorted(lib["name"]), cons0.get("excluded_ids", []), key=key("ex")
    )
    z = sb.number_input(
        "z (robustness, 1.28 ≈ 90 %)",
        0.0,
        3.0,
        float(spec0.get("z", 1.28)),
        step=0.01,
        key=key("z"),
    )
    with sb.expander("Data quality and regulations"):
        qc = st.checkbox(
            "Exclude 'inconsistent source data'", config.EXCLUDE_INCONSISTENT, key=key("qc")
        )
        markets = st.multiselect(
            "Markets",
            list(get_args(Market)),
            cons0.get("markets", list(get_args(Market))),
            key=key("markets"),
        )
        policies = ["exclude", "strict", "warn"]
        policy = st.selectbox(
            "Policy", policies, policies.index(cons0.get("policy", "exclude")), key=key("policy")
        )
    try:
        spec = Spec.model_validate({"bands": bands, "z": z})
        cons = Constraints(
            max_k=max_k,
            max_total_wv=cap,
            allowed_classes=classes,
            excluded_ids=excluded,
            markets=markets,
            policy=policy,
        )
    except ValidationError as err:
        sb.error(f"Spec is not valid: {err.errors()[0]['msg']}")
        return None
    return preset, spec, cons, qc


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


def show(lib: pd.DataFrame, o: opt.Outcome, spec: Spec, cons: Constraints, preset: str) -> None:
    r = o.result
    with st.container(border=True):
        if r is None:
            st.subheader("No feasible blend")
            st.write(f"Nothing with k ≤ {cons.max_k} meets every band at mean ± z·σ.")
        else:
            top = o.feasible.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Surfactants (k)", r.k)
            c2.metric("Total loading", f"{top['total_wv']:.3g} % w/v")
            c3.metric("Worst-band margin", f"{min(r.margins) * 1e3:+.1f} mN/m")
            comp = pd.DataFrame(
                {
                    "surfactant": top["names"],
                    "% w/v": [round(w, 4) for w in top["wv"]],
                    "mM": [round(c, 3) for c in r.blend.root.values()],
                }
            )
            st.dataframe(comp, hide_index=True)
            for name, flags in r.regulatory_flags.items():
                for f in flags:
                    st.warning(f"{name}: {f}")
        for _, row in o.excluded.iterrows():
            st.caption(f"Excluded {row['name']}: inconsistent source data ({row['reason']}).")
        for line in o.regulatory_removed:
            st.caption(f"Regulatory: {line}.")
        st.caption(o.regulatory_note)

    st.plotly_chart(dst_figure(lib, o, spec), theme="streamlit")
    st.caption(
        "Solid lines: dst_high (upper bound, used for 'DST ≤' bands); dashed: dst_low (lower "
        "bound). Shading: ± σ (Monte Carlo ⊕ σ_model). Grey boxes: spec windows."
    )

    (st.success if o.certified else st.warning)(o.certificate_plain or o.certificate)
    with st.expander("Full certificate"):
        st.write(o.certificate)

    if r is not None:
        st.subheader("Top feasible blends")
        tab = o.feasible.head(10).copy()
        tab.insert(0, "#", range(1, len(tab) + 1))
        tab["% w/v"] = tab["wv"].map(lambda w: " + ".join(f"{x:.3g}" for x in w))
        tab["worst margin / mN/m"] = (tab["worst_margin"] * 1e3).round(2)
        tab["provenance"] = tab["names"].map(lambda n: provenance(lib, n))
        cols = ["#", "blend", "% w/v", "total_wv", "worst margin / mN/m", "provenance"]
        st.dataframe(tab[cols].rename(columns={"total_wv": "total % w/v"}), hide_index=True)
    elif len(o.undetermined):
        st.subheader("Undetermined, worth testing in the lab")
        und = o.undetermined.head(10)
        st.dataframe(und[["blend", "total_wv", "worst_margin"]], hide_index=True)

    payload = opt.outcome_payload(o, spec, cons, preset)
    st.download_button(
        "Download results JSON",
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        file_name=f"wetopt_{preset}.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="wetopt", layout="wide")
    st.title("wetopt: fewest-surfactant spray adjuvant blends")
    lib = load_library(str(config.LIBRARY_PATH))
    parsed = sidebar(lib)
    if parsed is None:
        return
    preset, spec, cons, qc = parsed
    if st.button("Optimise", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        o = opt.optimise(
            lib,
            spec,
            cons,
            exclude_inconsistent=qc,
            progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg),
        )
        st.session_state["run"] = (o, spec, cons, qc, preset)
    run = st.session_state.get("run")
    if run is None:
        st.info(f"{len(lib)} candidates loaded. Set the spec in the sidebar and press Optimise.")
        return
    o, spec_run, cons_run, qc_run, preset_run = run
    if (spec_run, cons_run, qc_run) != (spec, cons, qc):
        st.warning(
            "The sidebar has changed since this result was computed. Press Optimise to update."
        )
    show(lib, o, spec_run, cons_run, preset_run)


if __name__ == "__main__":
    main()
