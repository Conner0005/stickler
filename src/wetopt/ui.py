"""Stickler visual layer: tokens, one stylesheet, and small HTML helpers for Streamlit pages.

Fonts, colours and radii for native widgets come from .streamlit/config.toml; this module only
styles the custom HTML blocks. All dynamic text is escaped.
"""

import html
import math

import streamlit as st

PAPER, INK, LEAF, BLOOM, DROPLET, EMBER = (
    "#F5F7F4",
    "#12261E",
    "#2E6A4C",
    "#DDE6DF",
    "#2A78D6",
    "#EB6834",
)
MUTED = "#4E6259"
DISPLAY = "'Bricolage Grotesque', 'IBM Plex Sans', system-ui, sans-serif"
BODY = "'IBM Plex Sans', system-ui, sans-serif"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1120px; padding-top: 2.5rem; }}
[data-testid="stSidebar"] h2 {{
  font-family: {DISPLAY}; font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
  margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid {BLOOM};
}}
[data-testid="stSidebar"] [data-testid="stHeading"]:first-of-type h2 {{ border-top: 0; }}
.sk {{ font-family: {BODY}; color: {INK}; }}
.sk a {{ color: {LEAF}; text-underline-offset: 3px; }}
.sk-hero-title {{
  font-family: {DISPLAY}; font-weight: 800; font-size: clamp(2.4rem, 5vw, 3.9rem);
  line-height: 1.02; letter-spacing: -0.035em; margin: 0 0 1.1rem; max-width: 14ch;
}}
.sk-lead {{ font-size: 1.15rem; line-height: 1.55; color: {MUTED}; max-width: 46ch; margin: 0; }}
.sk-figure {{ border: 1px solid {BLOOM}; border-radius: 14px; background: #fff; padding: 14px 14px 8px; }}
.sk-figure svg {{ width: 100%; height: auto; display: block; }}
.sk-figcaption {{ font-size: .82rem; color: {MUTED}; margin: .4rem .2rem .2rem; line-height: 1.4; }}
.sk-curve {{ stroke-dasharray: 900; stroke-dashoffset: 900; animation: sk-draw 1.6s ease-out .3s forwards; }}
@keyframes sk-draw {{ to {{ stroke-dashoffset: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .sk-curve {{ animation: none; stroke-dashoffset: 0; }} }}
.sk-points {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  border-top: 1px solid {BLOOM}; border-bottom: 1px solid {BLOOM}; margin: 2.6rem 0 0; }}
.sk-point {{ padding: 1.3rem 1.4rem 1.4rem 0; }}
.sk-point + .sk-point {{ padding-left: 1.4rem; border-left: 1px solid {BLOOM}; }}
.sk-point h3 {{ font-family: {DISPLAY}; font-size: 1.15rem; font-weight: 700; margin: 0 0 .35rem;
  letter-spacing: -0.01em; }}
.sk-point p {{ margin: 0; color: {MUTED}; line-height: 1.5; font-size: .97rem; }}
.sk-h2 {{ font-family: {DISPLAY}; font-weight: 750; font-size: 1.7rem; letter-spacing: -0.02em;
  margin: 2.8rem 0 1rem; }}
.sk-steps {{ list-style: none; padding: 0; margin: 0; counter-reset: sk; display: grid; gap: .2rem; }}
.sk-steps li {{ counter-increment: sk; display: grid; grid-template-columns: 2.6rem 1fr;
  align-items: baseline; padding: .7rem 0; border-bottom: 1px solid {BLOOM}; max-width: 68ch; }}
.sk-steps li::before {{ content: counter(sk); font-family: {DISPLAY}; font-weight: 800;
  font-size: 1.35rem; color: {LEAF}; }}
.sk-steps b {{ font-weight: 600; }}
.sk-credit {{ font-size: .82rem; color: {MUTED}; line-height: 1.5; margin-top: 2.6rem; max-width: 80ch; }}
.sk-page-title {{ font-family: {DISPLAY}; font-weight: 800; font-size: 2.4rem; letter-spacing: -0.03em;
  margin: 0 0 .4rem; }}
.sk-page-lead {{ color: {MUTED}; font-size: 1.05rem; margin: 0 0 1.4rem; max-width: 70ch; line-height: 1.5; }}
.sk-means {{ border-left: 3px solid {LEAF}; background: #fff; border-radius: 0 12px 12px 0;
  padding: .85rem 1.1rem; margin: 0 0 1.2rem; }}
.sk-means h4 {{ font-family: {DISPLAY}; font-size: 1rem; font-weight: 700; margin: 0 0 .35rem; }}
.sk-means ul {{ margin: 0; padding-left: 1.1rem; }}
.sk-means li {{ margin: .12rem 0; line-height: 1.45; }}
.sk-means .sk-custom {{ color: {EMBER}; font-weight: 600; }}
.sk-recipe {{ border: 1.5px solid {INK}; border-radius: 16px; background: #fff; padding: 1.4rem 1.6rem; }}
.sk-recipe h3 {{ font-family: {DISPLAY}; font-weight: 700; font-size: 1rem; color: {MUTED};
  margin: 0 0 .6rem; letter-spacing: 0; }}
.sk-ingredient {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
  gap: .2rem 1.2rem; align-items: baseline; padding: .55rem 0; border-bottom: 1px solid {BLOOM}; }}
.sk-ingredient .name {{ font-family: {DISPLAY}; font-weight: 750; font-size: 1.75rem;
  letter-spacing: -0.02em; }}
.sk-ingredient .dose {{ font-size: 1.05rem; }}
.sk-ingredient .dose small {{ color: {MUTED}; font-size: .9rem; margin-left: .35rem; }}
.sk-ingredient .flags {{ grid-column: 1 / -1; font-size: .85rem; color: {MUTED}; }}
.sk-figures {{ display: flex; flex-wrap: wrap; gap: 0; margin-top: 1rem; }}
.sk-fig {{ padding: .2rem 1.6rem .2rem 0; margin-right: 1.6rem; border-right: 1px solid {BLOOM}; }}
.sk-fig:last-child {{ border-right: 0; margin-right: 0; }}
.sk-fig .v {{ font-family: {DISPLAY}; font-weight: 750; font-size: 1.6rem; letter-spacing: -0.02em; }}
.sk-fig .l {{ font-size: .85rem; color: {MUTED}; }}
.sk-proof {{ display: flex; gap: .7rem; align-items: flex-start; margin-top: 1.1rem; padding-top: 1rem;
  border-top: 1px solid {BLOOM}; line-height: 1.5; }}
.sk-proof svg {{ flex: none; margin-top: .15rem; }}
.sk-proof.open {{ color: {MUTED}; }}
.sk-empty {{ border: 1.5px dashed {MUTED}; border-radius: 16px; padding: 1.3rem 1.6rem; background: #fff; }}
.sk-empty h3 {{ font-family: {DISPLAY}; font-weight: 750; font-size: 1.5rem; margin: 0 0 .4rem; }}
.sk-empty p {{ margin: 0; line-height: 1.5; }}
.sk-prose {{ max-width: 72ch; line-height: 1.6; }}
.sk-prose h3 {{ font-family: {DISPLAY}; font-weight: 700; font-size: 1.3rem; margin: 2rem 0 .5rem; }}
.sk-note {{ background: #fff; border: 1px solid {BLOOM}; border-radius: 12px; padding: .9rem 1.1rem;
  color: {MUTED}; line-height: 1.5; max-width: 80ch; }}
@media (max-width: 640px) {{
  .sk-point + .sk-point {{ padding-left: 0; border-left: 0; border-top: 1px solid {BLOOM}; }}
  .sk-ingredient {{ grid-template-columns: 1fr; }}
  .sk-fig {{ border-right: 0; margin-right: 0; padding-right: 1.2rem; }}
}}
</style>
"""

CHECK = (
    f'<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" '
    f'r="10" fill="{LEAF}"/><path d="M5.5 10.3l3 3 6-6.6" stroke="#fff" stroke-width="2" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
OPEN = (
    f'<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" '
    f'r="9" fill="none" stroke="{MUTED}" stroke-width="2"/><path d="M10 5.5v5.5M10 14v.5" '
    f'stroke="{MUTED}" stroke-width="2" stroke-linecap="round"/></svg>'
)


def e(text: object) -> str:
    return html.escape(str(text))


def inject() -> None:
    """Add the stylesheet once per page run."""
    st.markdown(CSS, unsafe_allow_html=True)


def render(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


def page_header(title: str, lead: str = "") -> str:
    lead_html = f'<p class="sk-page-lead">{e(lead)}</p>' if lead else ""
    return f'<div class="sk"><h1 class="sk-page-title">{e(title)}</h1>{lead_html}</div>'


def dst_hero_svg(ages_ms, gamma_mn_m, target_age_ms: float, target_mn_m: float, label: str) -> str:
    """The hero figure: a recipe's DST curve passing under the target window, water flat at 72."""
    w, h, left, right, top, bottom = 560, 330, 46, 16, 18, 40
    y_lo, y_hi = 25.0, 75.0

    def x(t: float) -> float:
        return left + (math.log10(t) - 1.0) / 2.0 * (w - left - right)

    def y(g: float) -> float:
        g = min(max(g, y_lo), y_hi)
        return top + (y_hi - g) / (y_hi - y_lo) * (h - top - bottom)

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(t):.1f},{y(g):.1f}"
        for i, (t, g) in enumerate(zip(ages_ms, gamma_mn_m, strict=True))
    )
    x0, x1 = x(target_age_ms / 1.3), x(target_age_ms * 1.3)
    grid = "".join(
        f'<line x1="{left}" x2="{w - right}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{BLOOM}"/>'
        f'<text x="{left - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{g:g}</text>'
        for g in (30, 40, 50, 60, 70)
    )
    ticks = "".join(
        f'<text x="{x(t):.1f}" y="{h - bottom + 18}" text-anchor="middle" font-size="11" '
        f'fill="{MUTED}">{t:g}</text>'
        for t in (10, 100, 1000)
    )
    return f"""
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Dynamic surface tension of {e(label)} falls below
{target_mn_m:g} mN/m by {target_age_ms:g} ms, while water stays at 72 mN/m">
  {grid}{ticks}
  <rect x="{x0:.1f}" y="{y(target_mn_m):.1f}" width="{x1 - x0:.1f}" height="{y(y_lo) - y(target_mn_m):.1f}"
        fill="{LEAF}" fill-opacity=".12" stroke="{LEAF}" stroke-opacity=".5" rx="4"/>
  <text x="{x1 + 6:.1f}" y="{y(target_mn_m) + 16:.1f}" font-size="12" fill="{LEAF}" font-weight="600">target</text>
  <text x="{x1 + 6:.1f}" y="{y(target_mn_m) + 31:.1f}" font-size="12" fill="{LEAF}">≤ {target_mn_m:g} mN/m at {target_age_ms:g} ms</text>
  <line x1="{left}" x2="{w - right}" y1="{y(72):.1f}" y2="{y(72):.1f}" stroke="{MUTED}" stroke-dasharray="4 4"/>
  <text x="{w - right}" y="{y(72) - 6:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">water alone, 72 mN/m</text>
  <path class="sk-curve" d="{path}" fill="none" stroke="{DROPLET}" stroke-width="3" stroke-linecap="round"/>
  <text x="{left + 6}" y="{y(gamma_mn_m[0]) - 12:.1f}" font-size="12" fill="{DROPLET}" font-weight="600">{e(label)}</text>
  <text x="{(left + w - right) / 2:.0f}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">surface age (ms, log scale)</text>
  <text x="12" y="{(top + h - bottom) / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 12 {(top + h - bottom) / 2:.0f})" text-anchor="middle">surface tension (mN/m)</text>
</svg>"""


def means_block(rows: list[str], custom: bool) -> str:
    items = "".join(f"<li>{e(r)}</li>" for r in rows)
    note = '<p class="sk-custom">Custom targets from Advanced are in use.</p>' if custom else ""
    return f'<div class="sk sk-means"><h4>What this means</h4>{note}<ul>{items}</ul></div>'


def recipe_block(
    ingredients: list[dict], figures: list[tuple[str, str]], proof: str, certified: bool
) -> str:
    rows = "".join(
        f'<div class="sk-ingredient"><div class="name">{e(i["name"])}</div>'
        f'<div class="dose">{i["wv"]:.3g} % w/v<small>{i["wv"] * 10:.3g} g/L</small>'
        f"<small>{i['mm']:.3g} mM</small></div>"
        f'<div class="flags">{e(i["flags"])}</div></div>'
        for i in ingredients
    )
    figs = "".join(
        f'<div class="sk-fig"><div class="v">{e(v)}</div><div class="l">{e(lbl)}</div></div>'
        for v, lbl in figures
    )
    cls, icon = ("", CHECK) if certified else (" open", OPEN)
    return (
        f'<div class="sk sk-recipe"><h3>Your recipe</h3>{rows}<div class="sk-figures">{figs}</div>'
        f'<div class="sk-proof{cls}">{icon}<div>{e(proof)}</div></div></div>'
    )


def empty_block(title: str, body: str) -> str:
    return f'<div class="sk sk-empty"><h3>{e(title)}</h3><p>{e(body)}</p></div>'


def note(text: str) -> str:
    return f'<div class="sk sk-note">{e(text)}</div>'
