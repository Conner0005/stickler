"""wetopt command line. Converts SI to display units (mM, mN/m, µmol/m²) only here."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from wetopt import config
from wetopt.library import build_library, load_surfpro, select_candidates

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
    props = ["cmc", "gamma_cmc", "gamma_max", "pc20"]
    for r in sel.to_dict("records"):
        table.add_row(
            r["name"],
            r["family"],
            f"{r['cmc']:.3g}",  # mol/m³ == mM
            f"{r['gamma_cmc'] * 1e3:.1f}",
            f"{r['gamma_max'] * 1e6:.2f}",
            " ".join("M" if r[f"{p}_measured"] else "I" for p in props),
        )
    console.print(table)
    console.print("M = measured (SurfPro literature), I = imputed (SurfPro GNN ensemble)")
