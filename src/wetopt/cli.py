"""wetopt command line. Converts SI to display units (mM, mN/m, µmol/m²) only here."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from wetopt import config
from wetopt import optimise as opt
from wetopt.library import build_library, load_surfpro, select_candidates
from wetopt.schema import Constraints, Spec
from wetopt.validate import data_quality

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """Fewest-surfactant spray adjuvant blends."""


@app.command()
def library(
    path: Annotated[Path, typer.Option(help="Library parquet to read or write.")] = (
        config.LIBRARY_PATH
    ),
    rebuild: Annotated[bool, typer.Option(help="Rebuild from data/raw/surfpro and save.")] = False,
    all_types: Annotated[
        bool, typer.Option(help="Also allow cationic, zwitterionic, gemini (implies --rebuild).")
    ] = False,
) -> None:
    """Show the candidate library; build it from SurfPro first if needed."""
    if rebuild or all_types or not path.exists():
        sel = select_candidates(build_library(*load_surfpro(), all_types=all_types))
        path.parent.mkdir(parents=True, exist_ok=True)
        sel.to_parquet(path, index=False)
        for r in sel.itertuples():
            console.print(f"kept [bold]{r.name}[/]: {r.reason}")
        console.print(f"saved {len(sel)} candidates to {path}\n")
    sel = pd.read_parquet(path)

    table = Table(title=f"wetopt candidate library ({len(sel)})")
    table.add_column("name", no_wrap=True)
    table.add_column("class")
    for col in ["CMC / mM", "γ_CMC / mN/m", "Γ_max / µmol/m²"]:
        table.add_column(col, justify="right")
    table.add_column("M/I: CMC γ Γ pC20", no_wrap=True)
    table.add_column("QC (pC20 check)", no_wrap=True)
    props = ["cmc", "gamma_cmc", "gamma_max", "pc20"]
    qc = data_quality(sel).set_index("name")
    for r in sel.to_dict("records"):
        table.add_row(
            r["name"],
            r["family"],
            f"{r['cmc']:.3g}",  # mol/m³ == mM
            f"{r['gamma_cmc'] * 1e3:.1f}",
            f"{r['gamma_max'] * 1e6:.2f}",
            " ".join("M" if r[f"{p}_measured"] else "I" for p in props),
            qc.loc[r["name"], "status"],
        )
    console.print(table)
    console.print("M = measured (SurfPro literature), I = imputed (SurfPro GNN ensemble)")


def _json_default(x):
    return x.item() if isinstance(x, np.generic) else str(x)


@app.command()
def optimise(
    spec_path: Annotated[Path, typer.Argument(help="YAML with `spec` and `constraints`.")],
    library: Annotated[Path, typer.Option(help="Candidate library parquet.")] = config.LIBRARY_PATH,
    out_dir: Annotated[Path, typer.Option(help="Where to write <spec>.json.")] = (
        config.ROOT / "results"
    ),
    include_inconsistent: Annotated[
        bool, typer.Option(help="Keep candidates flagged 'inconsistent source data'.")
    ] = not config.EXCLUDE_INCONSISTENT,
    markets: Annotated[
        str | None,
        typer.Option(help="Override constraints.markets, e.g. 'EU' or 'EU,GB,US,Global'."),
    ] = None,
) -> None:
    """Fewest-surfactant blend for a spec, top feasible blends, and the k − 1 certificate."""
    data = yaml.safe_load(spec_path.read_text())
    if markets is not None:
        data["constraints"]["markets"] = [m.strip() for m in markets.split(",") if m.strip()]
    spec, cons = Spec.model_validate(data["spec"]), Constraints.model_validate(data["constraints"])
    lib = pd.read_parquet(library)
    o = opt.optimise(lib, spec, cons, exclude_inconsistent=not include_inconsistent)

    console.rule(f"wetopt optimise · {spec_path.name}")
    bands = " · ".join(opt.band_label(b) for b in spec.bands)
    pc = o.pc20
    pc_text = f"pC20 check n = {pc['n']}" + (
        f", RMSE {pc['rmse_mN_m']:.1f} mN/m" if pc["rmse_mN_m"] is not None else ""
    )
    console.print(
        f"Spec: {bands} · z = {spec.z:g} · total ≤ {cons.max_total_wv:g}% w/v · k ≤ {cons.max_k}"
    )
    console.print(f"σ_model = {o.sigma_model * 1e3:.1f} mN/m ({pc_text})")
    for row in o.excluded.itertuples():
        console.print(f"[yellow]Excluded[/] {row.name}: inconsistent source data ({row.reason})")
    if not len(o.excluded):
        console.print("No candidates excluded for data quality.")
    console.print(
        f"Regulatory screen: markets {', '.join(cons.markets) or 'none'}, policy {cons.policy}"
    )
    for line in o.regulatory_removed or ["No candidates excluded for regulatory reasons."]:
        console.print(f"  {line}")
    levels = Table(title="Search by k")
    for c in ["k", "grid blends", "classified (MC)", "feasible", "failed", "undetermined"]:
        levels.add_column(c, justify="right")
    for lv in o.levels:
        levels.add_row(
            *(
                str(lv[c])
                for c in ["k", "grid_blends", "classified", "feasible", "failed", "undetermined"]
            )
        )
    console.print(levels)

    if o.result is not None:
        r = o.result
        parts = []
        for name, c in r.blend.root.items():
            wv = o.feasible.iloc[0]["wv"][list(r.blend.root).index(name)]
            parts.append(f"{name} {wv:.3g}% w/v ({c:.3g} mM)")
        console.print(f"[bold green]Winner (k = {r.k}):[/] " + " + ".join(parts))
        console.print(f"  total {o.feasible.iloc[0]['total_wv']:.3g}% w/v")
        for pred, m in zip(r.predictions, r.margins, strict=True):
            lo, hi, sd = o.feasible.iloc[0][opt.band_label(pred.band)]
            bracket = f"{lo * 1e3:.1f}–{hi * 1e3:.1f} ± {sd * 1e3:.1f} mN/m"
            console.print(f"  {opt.band_label(pred.band)}: {bracket}, margin {m * 1e3:+.1f} mN/m")
        for name, fl in r.regulatory_flags.items():
            for f in fl:
                console.print(f"  [yellow]Flag[/] {name}: {f}")
        top = Table(title="Top feasible blends (best per surfactant set)")
        for c in ["#", "blend", "% w/v", "total % w/v", "worst margin / mN/m"]:
            top.add_column(c, justify="left" if c == "blend" else "right")
        for i, row in enumerate(o.feasible.head(10).itertuples(), 1):
            wv = " + ".join(f"{w:.3g}" for w in row.wv)
            top.add_row(
                str(i), row.blend, wv, f"{row.total_wv:.3g}", f"{row.worst_margin * 1e3:+.1f}"
            )
        console.print(top)
    else:
        console.print("[bold red]No feasible blend.[/]")

    style = "green" if o.certified else "yellow"
    console.print(f"[bold {style}]Certificate:[/] {o.certificate}")
    if len(o.undetermined):
        und = Table(title="Undetermined, worth testing in the lab (nearest to feasible first)")
        for c in ["blend", "% w/v", "optimistic margin / mN/m", "pessimistic margin / mN/m"]:
            und.add_column(c, justify="left" if c == "blend" else "right")
        for row in o.undetermined.head(10).itertuples():
            und.add_row(
                row.blend,
                " + ".join(f"{w:.3g}" for w in row.wv),
                f"{row.worst_optimistic * 1e3:+.1f}",
                f"{row.worst_margin * 1e3:+.1f}",
            )
        console.print(und)

    console.print(f"[dim]{o.regulatory_note}[/]")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec_path.stem}.json"
    payload = {
        "spec_file": str(spec_path),
        "spec": spec.model_dump(),
        "constraints": cons.model_dump(),
        "sigma_model_N_m": o.sigma_model,
        "pc20_check": o.pc20,
        "levels": o.levels,
        "result": o.result.model_dump() if o.result else None,
        "certified": o.certified,
        "certificate": o.certificate,
        "top_feasible": o.feasible.head(10).to_dict("records"),
        "undetermined": o.undetermined.head(50).to_dict("records"),
        "excluded": o.excluded.to_dict("records"),
        "regulatory_removed": o.regulatory_removed,
        "regulatory_note": o.regulatory_note,
        "runtime_s": o.runtime_s,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    console.print(f"{o.runtime_s:.1f} s · wrote {out}")
