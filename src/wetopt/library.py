"""Candidate surfactant library from SurfPro (Hödl et al., Digital Discovery 2025, 4, 1176).

Output columns are SI (CLAUDE.md) with two exceptions inherited from SurfPro's error model:
``pc20`` = -log10(C20 / M), and the CMC uncertainty is ``cmc_sigma_log10`` = 1σ of log10(CMC).
Each property P in (cmc, gamma_cmc, gamma_max, pc20) carries ``P_measured`` and ``P_source``.
Units and provenance are documented in docs/data.md.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from wetopt import config

RDLogger.DisableLog("rdApp.*")

# SurfPro column -> (our column, value -> SI, σ scale to SI, σ column, literature ref column)
# Gamma_max is handled in model units (µmol/m²) before conversion; see build_library.
PROPS = {
    "pCMC": ("cmc", lambda p: 10.0 ** (3.0 - p), 1.0, "cmc_sigma_log10", "Ref_CMC"),
    "AW_ST_CMC": ("gamma_cmc", lambda v: v * 1e-3, 1e-3, "gamma_cmc_sigma", "Ref_AW_ST_CMC"),
    "Gamma_max": ("gamma_max", lambda v: v * 1e-6, 1e-6, "gamma_max_sigma", "Ref_Gamma_max"),
    "pC20": ("pc20", lambda v: v, 1.0, "pc20_sigma", "Ref_pC20"),
}
DEFAULT_TYPES = {"anionic", "non-ionic", "sugar-based non-ionic"}
EXTRA_TYPES = {"cationic", "zwitterionic", "gemini cationic", "gemini zwitterionic"}
IONIC_CLASS = {
    "anionic": "anionic",
    "non-ionic": "nonionic",
    "sugar-based non-ionic": "nonionic",
    "cationic": "cationic",
    "gemini cationic": "cationic",
    "zwitterionic": "zwitterionic",
    "gemini zwitterionic": "zwitterionic",
}
IMPUTED_SOURCE = "SurfPro imputed DB (AttentiveFP ensemble), doi:10.1039/D4DD00393D"

_ION = "[OX1-,OX2H1]"
_SO3 = f"[SX4](=O)(=O){_ION}"
# family: (match any of, match none of). First matching family wins; else "other".
FAMILIES = {
    "siloxane": (["[Si]"], []),
    "sulfosuccinate": ([f"[CX3](=O)[CX4]({_SO3})[CX4][CX3](=O)"], []),
    "alkylbenzene sulfonate": (
        [f"[CX4]c1c({_SO3})cccc1", f"[CX4]c1cc({_SO3})ccc1", f"[CX4]c1ccc({_SO3})cc1"],
        [],
    ),
    "alkyl sulfate": ([f"[CX4][OX2]{_SO3}"], []),
    "sugar-based": (
        ["[CX4;R]([OX2H1])@[CX4;R]([OX2H1])", "[CX4;!R]([OX2H1])[CX4;!R]([OX2H1])[CX4;!R][OX2H1]"],
        [],
    ),
    "alcohol ethoxylate": (["[CX4][OX2][CH2][CH2][OX2]"], ["a", "[!#1;!#6;!#8]"]),
}
_PATTERNS = {
    fam: ([Chem.MolFromSmarts(s) for s in anyof], [Chem.MolFromSmarts(s) for s in noneof])
    for fam, (anyof, noneof) in FAMILIES.items()
}
_FLUORINE = Chem.MolFromSmarts("[F]")


def family(mol: Chem.Mol) -> str:
    """First FAMILIES entry whose SMARTS match ``mol``, else 'other'."""
    for fam, (anyof, noneof) in _PATTERNS.items():
        if any(mol.HasSubstructMatch(p) for p in anyof) and not any(
            mol.HasSubstructMatch(p) for p in noneof
        ):
            return fam
    return "other"


def short_name(mol: Chem.Mol) -> str:
    """CnEm for linear alcohol ethoxylates C_nH_2n+1(OCH2CH2)_mOH, else the molecular formula."""
    smiles = Chem.MolToSmiles(mol)
    n_o = sum(a.GetSymbol() == "O" for a in mol.GetAtoms())
    n_c = sum(a.GetSymbol() == "C" for a in mol.GetAtoms())
    m = n_o - 1
    n = n_c - 2 * m
    if m >= 1 and n >= 1 and Chem.CanonSmiles("C" * n + "OCC" * m + "O") == smiles:
        return f"C{n}E{m}"
    return CalcMolFormula(mol)


def load_surfpro(raw_dir=config.SURFPRO_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read SurfPro literature and imputed tables from data/raw/surfpro/data/."""
    files = [raw_dir / "surfpro_literature.csv", raw_dir / "surfpro_imputed.csv"]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing}; run: git clone "
            "https://github.com/BigChemistry-RobotLab/SurfPro data/raw/surfpro"
        )
    return pd.read_csv(files[0]), pd.read_csv(files[1])


def build_library(lit: pd.DataFrame, imp: pd.DataFrame, all_types: bool = False) -> pd.DataFrame:
    """All SurfPro surfactants of the allowed types measured at T_RANGE_C, one row per InChIKey.

    Measured values (literature, median over duplicate InChIKeys) take precedence; missing
    properties are filled from the imputed table.
    """
    types = DEFAULT_TYPES | (EXTRA_TYPES if all_types else set())
    lit = lit[lit["Surfactant_Type"].isin(types) & lit["Temp_Celsius"].between(*config.T_RANGE_C)]
    lit = lit.assign(Gamma_max=lit["Gamma_max"] * 1e6)  # mol/m² -> µmol/m², as in imputed table
    refs = [ref for *_, ref in PROPS.values()]
    num = [*PROPS, "Temp_Celsius", "Molecular_Weight"]
    meas = lit.groupby("InChIKey").agg(
        {
            **{c: "median" for c in num},
            "SMILES": "first",
            "Surfactant_Type": "first",
            **{r: lambda s: "; ".join(s.dropna().astype(str).unique()) for r in refs},
        }
    )
    imp = imp.assign(
        InChIKey=imp["SMILES"].map(dict(zip(lit["SMILES"], lit["InChIKey"], strict=True)))
    )
    stds = [f"{p}_std" for p in PROPS]
    imp = imp.dropna(subset="InChIKey").groupby("InChIKey")[[*PROPS, *stds]].median()
    imp = imp.reindex(meas.index)

    out = pd.DataFrame(index=meas.index)
    out["smiles"] = meas["SMILES"]
    out["surfpro_type"] = meas["Surfactant_Type"]
    out["ionic_class"] = out["surfpro_type"].map(IONIC_CLASS)
    out["temp_c"] = meas["Temp_Celsius"]
    out["mw"] = meas["Molecular_Weight"] * 1e-3  # g/mol -> kg/mol
    for p, (col, to_si, sig_scale, sig_col, ref) in PROPS.items():
        measured = meas[p].notna()
        rmse = config.SURFPRO_TEST_RMSE
        imp_sigma = imp[f"{p}_std"] if rmse is None else rmse[p]
        out[col] = to_si(meas[p].fillna(imp[p]))
        out[sig_col] = np.where(measured, config.MEASURED_SIGMA[p], imp_sigma) * sig_scale
        out[f"{col}_measured"] = measured
        lit_src = ("SurfPro lit. " + meas[ref]).where(meas[ref] != "", "SurfPro lit. (no ref)")
        out[f"{col}_source"] = lit_src.where(measured, IMPUTED_SOURCE)

    mols = out["smiles"].map(Chem.MolFromSmiles)
    out["name"] = mols.map(short_name)
    dup = out["name"].duplicated(keep=False)  # isomers share a formula: number them
    out.loc[dup, "name"] += " #" + (out[dup].groupby("name").cumcount() + 1).astype(str)
    out["family"] = mols.map(family)
    out["fluorinated"] = mols.map(lambda m: m.HasSubstructMatch(_FLUORINE))
    return out.rename_axis("inchikey").reset_index()


def select_candidates(lib: pd.DataFrame, n: int = config.N_CANDIDATES) -> pd.DataFrame:
    """Pick ``n`` demo candidates and record why each was kept in ``reason``.

    Agro-like families (all named FAMILIES, non-fluorinated) are taken round-robin so every
    family is represented; within a family, more measured properties rank first. Remaining
    slots go to other surfactants by the same ranking.
    """
    meas_cols = [f"{col}_measured" for col, *_ in PROPS.values()]
    lib = lib.dropna().assign(n_measured=lib[meas_cols].sum(axis=1))
    lib = lib.assign(agro=(lib["family"] != "other") & ~lib["fluorinated"])
    lib = lib.sort_values(
        ["n_measured", "cmc_measured", "temp_c", "inchikey"],
        ascending=[False, False, True, True],
        key=lambda s: (s - 25).abs() if s.name == "temp_c" else s,
    )
    lib["rank"] = lib.groupby("family").cumcount() + 1
    agro = lib[lib["agro"]].sort_values(["rank", "family"], kind="stable")
    sel = pd.concat([agro, lib[~lib["agro"]]]).head(n).copy()

    def why(r: pd.Series) -> str:
        kind = "agro-like" if r["agro"] else "filler"
        kind += ", fluorinated" if r["fluorinated"] else ""
        imputed = [c.removesuffix("_measured") for c in meas_cols if not r[c]]
        return (
            f"{r['family']} ({kind}), #{r['rank']} in family; {r['n_measured']}/4 measured"
            + (f" (imputed: {', '.join(imputed)})" if imputed else "")
            + f"; {r['temp_c']:g} °C"
        )

    sel["reason"] = sel.apply(why, axis=1)
    return sel.drop(columns=["agro", "rank"]).reset_index(drop=True)
