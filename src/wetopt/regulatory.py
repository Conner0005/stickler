"""Hardcoded regulatory flags for co-formulants (snapshot: data/regulations.yaml).

Not legal advice: every display of flags must show ``Regulations.disclaimer``. Ingredients are
classified by CAS when it is listed, and by structure with small RDKit functions (never by
name), then matched against the rules of the selected markets.
"""

from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem

from wetopt import config

REG_PATH = config.ROOT / "data" / "regulations.yaml"
BLOCKING = frozenset({"not_permitted", "restricted", "eliminated"})
PATTERN_ONLY = "pattern match, confirm by CAS"
PFAS_SPECIFIC = frozenset({"pfos", "pfoa", "pfhxs", "lc_pfca"})


@dataclass(frozen=True)
class Rule:
    id: str
    region: str
    region_label: str
    status: str
    rule: str
    classes: frozenset[str]
    cas: frozenset[str]
    source: str


@dataclass(frozen=True)
class Regulations:
    as_of: str
    disclaimer: str
    class_labels: dict[str, str]
    cas_classes: dict[str, str]
    rules: tuple[Rule, ...]


@cache
def load(path: Path = REG_PATH) -> Regulations:
    """Parse data/regulations.yaml."""
    d = yaml.safe_load(path.read_text())
    rules = tuple(
        Rule(
            id=r["id"],
            region=r["region"],
            region_label=r.get("region_label", r["region"]),
            status=r["status"],
            rule=" ".join(r["rule"].split()),
            classes=frozenset(r.get("classes", [])),
            cas=frozenset(str(c) for c in r.get("cas", [])),
            source=r["source"],
        )
        for r in d["rules"]
    )
    return Regulations(
        str(d["as_of"]),
        " ".join(d["disclaimer"].split()),
        d["class_labels"],
        {str(k): v for k, v in d["cas_classes"].items()},
        rules,
    )


# --- structure tests (RDKit) -------------------------------------------------------------


def _ion(smiles: str) -> Chem.Mol:
    """Largest fragment (counterions dropped)."""
    mol = Chem.MolFromSmiles(smiles)
    return max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())


def _carbon_group(mol: Chem.Mol, start: int, stop: set[int], saturated: bool) -> int | None:
    """Carbon count of the acyclic all-carbon substituent starting at ``start`` (not crossing
    ``stop``); None if it contains a heteroatom, a ring or aromatic atom, or (if ``saturated``)
    a multiple bond."""
    seen, queue = {start}, deque([start])
    while queue:
        atom = mol.GetAtomWithIdx(queue.popleft())
        if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetIsAromatic():
            return None
        for bond in atom.GetBonds():
            nb = bond.GetOtherAtomIdx(atom.GetIdx())
            if nb in stop or nb in seen:
                continue
            if saturated and bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            seen.add(nb)
            queue.append(nb)
    return len(seen)


def _starts_ethoxylate(mol: Chem.Mol, o_idx: int, from_idx: int) -> bool:
    """True if the ether O at ``o_idx`` (reached from ``from_idx``) starts O–CH2–CH2–O…."""
    o = mol.GetAtomWithIdx(o_idx)
    nbs = [a for a in o.GetNeighbors() if a.GetIdx() != from_idx]
    if len(nbs) != 1:
        return False
    c1 = nbs[0]
    if c1.GetSymbol() != "C" or c1.GetTotalNumHs() != 2 or c1.GetIsAromatic():
        return False
    c2s = [a for a in c1.GetNeighbors() if a.GetIdx() != o_idx and a.GetSymbol() == "C"]
    return any(
        c2.GetTotalNumHs() == 2
        and any(a.GetSymbol() == "O" and a.GetIdx() != c1.GetIdx() for a in c2.GetNeighbors())
        for c2 in c2s
    )


def alkylphenol_chains(smiles: str) -> set[int]:
    """Carbon counts of acyclic alkyl substituents on benzene rings that also carry an O
    (phenol/phenolate, or aryl ether whose chain is (CH2CH2O)n)."""
    mol = _ion(smiles)
    found: set[int] = set()
    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if len(ring) != 6 or not all(a.GetSymbol() == "C" and a.GetIsAromatic() for a in atoms):
            continue
        ring_set, has_o, chains = set(ring), False, set()
        for a in atoms:
            for nb in a.GetNeighbors():
                if nb.GetIdx() in ring_set:
                    continue
                if nb.GetSymbol() == "O" and nb.GetDegree() == 1:  # phenol / phenolate
                    has_o = True
                elif nb.GetSymbol() == "O" and _starts_ethoxylate(mol, nb.GetIdx(), a.GetIdx()):
                    has_o = True
                elif nb.GetSymbol() == "C":
                    n = _carbon_group(mol, nb.GetIdx(), ring_set, saturated=True)
                    if n is not None:
                        chains.add(n)
        if has_o:
            found |= chains
    return found


def is_tallow_amine_ethoxylate(smiles: str) -> bool:
    """Non-amide, non-aromatic amine N with exactly one C16–C18 alkyl/alkenyl chain and at
    least one (CH2CH2O)n chain (N–CH2–CH2–O…)."""
    mol = _ion(smiles)
    for n in mol.GetAtoms():
        if n.GetSymbol() != "N" or n.GetIsAromatic():
            continue
        amide = any(
            nb.GetSymbol() == "C"
            and any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(nb).GetSymbol() in ("O", "S")
                for b in nb.GetBonds()
            )
            for nb in n.GetNeighbors()
        )
        if amide:
            continue
        long_chains = sum(
            16 <= (_carbon_group(mol, nb.GetIdx(), {n.GetIdx()}, saturated=False) or 0) <= 18
            for nb in n.GetNeighbors()
        )
        eo = any(
            nb.GetSymbol() == "C"
            and nb.GetTotalNumHs() == 2
            and any(
                c2.GetSymbol() == "C"
                and c2.GetTotalNumHs() == 2
                and any(o.GetSymbol() == "O" for o in c2.GetNeighbors())
                for c2 in nb.GetNeighbors()
                if c2.GetIdx() != n.GetIdx()
            )
            for nb in n.GetNeighbors()
        )
        if long_chains == 1 and eo:
            return True
    return False


def _perfluorinated(atom: Chem.Atom) -> bool:
    """sp3 carbon with no H, at least one F, and only C/F neighbours (besides an anchor)."""
    return (
        atom.GetSymbol() == "C"
        and not atom.GetIsAromatic()
        and atom.GetTotalNumHs() == 0
        and atom.GetHybridization() == Chem.HybridizationType.SP3
        and any(nb.GetSymbol() == "F" for nb in atom.GetNeighbors())
    )


def _perfluoroalkyl_size(mol: Chem.Mol, start: int, anchor: int) -> int | None:
    """Carbons in the perfluoroalkyl group starting at ``start`` (all CF3/CF2/CF), or None."""
    seen, queue = {start}, deque([start])
    while queue:
        atom = mol.GetAtomWithIdx(queue.popleft())
        if not _perfluorinated(atom):
            return None
        for nb in atom.GetNeighbors():
            i = nb.GetIdx()
            if i == anchor or i in seen or nb.GetSymbol() == "F":
                continue
            if nb.GetSymbol() != "C":
                return None
            seen.add(i)
            queue.append(i)
    return len(seen)


def is_pfas(smiles: str) -> bool:
    """OECD (2021): at least one fully fluorinated methyl or methylene carbon (–CF3 / –CF2–),
    i.e. an sp3 C with ≥ 2 F and no H, Cl, Br or I attached."""
    mol = _ion(smiles)
    for a in mol.GetAtoms():
        if (
            a.GetSymbol() != "C"
            or a.GetTotalNumHs()
            or a.GetHybridization() != Chem.HybridizationType.SP3
        ):
            continue
        nbs = [nb.GetSymbol() for nb in a.GetNeighbors()]
        if nbs.count("F") >= 2 and not {"Cl", "Br", "I"} & set(nbs):
            return True
    return False


def pfas_acid_classes(smiles: str) -> set[str]:
    """pfos / pfhxs: C8 / C6 perfluoroalkyl sulfonate; pfoa: C7F15–COO; lc_pfca:
    CnF2n+1–COO with 8 ≤ n ≤ 20 (9–21 total C; SC-12/12). Acid or salt."""
    mol = _ion(smiles)
    out: set[str] = set()
    for a in mol.GetAtoms():
        nbs = list(a.GetNeighbors())
        o_single = [
            n
            for n in nbs
            if n.GetSymbol() == "O"
            and n.GetDegree() == 1
            and n.GetTotalNumHs() + max(0, -n.GetFormalCharge()) >= 1
        ]
        o_double = [
            n
            for n in nbs
            if n.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondType()
            == Chem.BondType.DOUBLE
        ]
        carbons = [n for n in nbs if n.GetSymbol() == "C"]
        if len(carbons) != 1:
            continue
        if a.GetSymbol() == "S" and len(o_double) == 2 and len(o_single) == 1:
            size = _perfluoroalkyl_size(mol, carbons[0].GetIdx(), a.GetIdx())
            out |= {8: {"pfos"}, 6: {"pfhxs"}}.get(size, set())
        elif a.GetSymbol() == "C" and len(o_double) == 1 and len(o_single) == 1:
            size = _perfluoroalkyl_size(mol, carbons[0].GetIdx(), a.GetIdx())
            if size == 7:
                out.add("pfoa")
            elif size is not None and 8 <= size <= 20:
                out.add("lc_pfca")
    return out


def structure_classes(smiles: str) -> dict[str, str]:
    """Classes found by structure → method label."""
    out: dict[str, str] = {}
    chains = alkylphenol_chains(smiles)
    if 9 in chains:
        out["nonylphenol_family"] = "structure"
    if 8 in chains:
        out["octylphenol_family"] = "structure"
    if is_tallow_amine_ethoxylate(smiles):
        out["tallow_amine_ethoxylate"] = f"structure ({PATTERN_ONLY})"
    for c in pfas_acid_classes(smiles):
        out[c] = "structure"
    if is_pfas(smiles):
        out["pfas_any"] = "structure"
    return out


# --- rules -------------------------------------------------------------------------------


def classify(
    smiles: str | None = None, cas: str | None = None, regs: Regulations | None = None
) -> dict[str, str]:
    """Classes of one ingredient → how each was found (listed CAS, structure, or implied)."""
    regs = regs or load()
    classes: dict[str, str] = {}
    if cas and cas in regs.cas_classes:
        classes[regs.cas_classes[cas]] = f"CAS {cas}"
    if smiles:
        for c, how in structure_classes(smiles).items():
            classes.setdefault(c, how)
    if PFAS_SPECIFIC & classes.keys():
        classes.setdefault("pfas_any", "implied by a specific PFAS class")
    return classes


@dataclass(frozen=True)
class Flag:
    market: str  # EU | GB | US | Global
    region: str  # display label
    status: str
    rule: str
    source: str
    label: str  # class label (or the CAS number)
    method: str

    def text(self) -> str:
        where = f"{self.label}; {self.method}"
        return f"{self.region} {self.status}: {self.rule} [{where}] ({self.source})"


def flags_for(
    smiles: str | None,
    cas: str | None,
    markets: list[str],
    regs: Regulations | None = None,
) -> list[Flag]:
    """Rules of the selected markets that this ingredient triggers."""
    regs = regs or load()
    classes = classify(smiles, cas, regs)
    out = []
    for r in regs.rules:
        if r.region not in markets:
            continue
        hit = sorted(r.classes & classes.keys())
        if hit:
            c = hit[0]
            label, how = regs.class_labels[c], classes[c]
            out.append(Flag(r.region, r.region_label, r.status, r.rule, r.source, label, how))
        elif cas and cas in r.cas:
            out.append(
                Flag(
                    r.region, r.region_label, r.status, r.rule, r.source, f"CAS {cas}", "listed CAS"
                )
            )
    return out


def screen(lib: pd.DataFrame, markets: list[str], regs: Regulations | None = None) -> pd.DataFrame:
    """Per-candidate flags for the selected markets: name, flags, blocking markets, labels."""
    regs = regs or load()
    cas_col = lib["cas"] if "cas" in lib else pd.Series([None] * len(lib), index=lib.index)
    rows = []
    for name, smiles, cas in zip(lib["name"], lib["smiles"], cas_col, strict=True):
        fl = flags_for(smiles, cas if isinstance(cas, str) else None, markets, regs)
        blocking = sorted({f.market for f in fl if f.status in BLOCKING})
        rows.append(
            {
                "name": name,
                "flags": fl,
                "blocking": blocking,
                "labels": sorted({f.label for f in fl}),
            }
        )
    return pd.DataFrame(rows, columns=["name", "flags", "blocking", "labels"])


def removal_summary(screened: pd.DataFrame, markets: list[str]) -> list[str]:
    """E.g. '3 candidates excluded for EU: octylphenol and its ethoxylates'."""
    out = []
    for m in markets:
        hit = screened[screened["blocking"].map(lambda b, m=m: m in b)]
        if len(hit):
            labels = sorted(
                {
                    f.label
                    for fl in hit["flags"]
                    for f in fl
                    if f.market == m and f.status in BLOCKING
                }
            )
            out.append(f"{len(hit)} candidates excluded for {m}: {', '.join(labels)}")
    return out
