"""Regulatory flags: structure classification against the hardcoded snapshot."""

import pandas as pd

from tests.test_optimise import DST100, EQ40, toy_library
from wetopt.optimise import optimise
from wetopt.regulatory import PATTERN_ONLY, classify, flags_for, load
from wetopt.schema import Constraints, Spec

ALL = ["EU", "GB", "US", "Global"]
TRITON_X100 = "CC(C)(C)CC(C)(C)c1ccc(OCCOCCOCCOCCOCCOCCOCCOCCOCCO)cc1"  # tert-octylphenol 9 EO
NP9EO = "CCCCCCCCCc1ccc(OCCOCCOCCOCCOCCOCCOCCOCCOCCO)cc1"
C12E5 = "CCCCCCCCCCCCOCCOCCOCCOCCOCCO"
NA_PFOA = "O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[Na+]"
C18_AMINE_2EO = "CCCCCCCCCCCCCCCCCCN(CCOCCO)CCOCCO"
DECYLPHENOL = "CCCCCCCCCCc1ccc(O)cc1"
NP_PO_SULFATES = [  # the three SurfPro candidates (nonylphenol propoxylate sulfates)
    "CCCCCCCCCc1ccc(OCC(C)OCC(C)OCC(C)OS(=O)(=O)[O-])cc1.[Na+]",
    "CCCCCCCCCc1ccc(OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OS(=O)(=O)[O-])cc1.[Na+]",
    "CCCCCCCCCc1ccc(" + "OCC(C)" * 9 + "OS(=O)(=O)[O-])cc1.[Na+]",
]
NP9EO_SULFATE = "CCCCCCCCCc1ccc(OCCOCCOCCOCCOCCOCCOCCOCCOCCOS(=O)(=O)[O-])cc1.[Na+]"
OP_PROPOXYLATE = "CC(C)(C)CC(C)(C)c1ccc(OCC(C)OCC(C)OCC(C)O)cc1"


def hits(smiles):
    return sorted((f.market, f.status) for f in flags_for(smiles, None, ALL))


def test_flags():
    assert hits(TRITON_X100) == [("EU", "not_permitted"), ("GB", "not_permitted")]
    assert hits(NP9EO) == [("EU", "not_permitted"), ("EU", "restricted"), ("GB", "not_permitted")]
    assert hits(C12E5) == []
    assert hits(NA_PFOA) == [("Global", "eliminated"), ("US", "check")]
    assert "lc_pfca" not in classify(NA_PFOA)  # PFOA (C8) is excluded from the LC-PFCA entry
    assert hits(C18_AMINE_2EO) == [("EU", "not_permitted"), ("GB", "not_permitted")]
    assert PATTERN_ONLY in classify(C18_AMINE_2EO)["tallow_amine_ethoxylate"]
    assert hits(DECYLPHENOL) == []
    assert classify(cas="9002-93-1") == {"octylphenol_family": "CAS 9002-93-1"}
    assert load().as_of == "2026-11-07" and "not legal advice" in load().disclaimer


def test_policy_exclude_reports_removed():
    npe = toy_library().iloc[[2]].assign(name="NPE", inchikey="NPE", smiles=NP9EO)
    lib = pd.concat([toy_library(), npe], ignore_index=True)
    sp = Spec.model_validate({"bands": [DST100, EQ40]})
    o = optimise(lib, sp, Constraints(max_k=2, max_total_wv=0.1, markets=["EU"]))
    assert o.regulatory_removed == ["1 candidates excluded for EU: nonylphenol and its ethoxylates"]
    assert "NPE" not in o.result.blend.root and "not legal advice" in o.result.regulatory_note
    warn = optimise(lib, sp, Constraints(max_k=2, max_total_wv=0.1, markets=["EU"], policy="warn"))
    assert warn.regulatory_removed == []


def test_alkylphenol_check_tier():
    check = [("EU", "check"), ("GB", "check")]
    for smi in NP_PO_SULFATES:
        assert hits(smi) == check
    assert hits(NP9EO) == [("EU", "not_permitted"), ("EU", "restricted"), ("GB", "not_permitted")]
    assert hits(NP9EO_SULFATE) == check
    assert hits(OP_PROPOXYLATE) == check


def test_policy_strict_also_excludes_check():
    nps = toy_library().iloc[[2]].assign(name="NPPS", inchikey="NPPS", smiles=NP_PO_SULFATES[0])
    lib = pd.concat([toy_library(), nps], ignore_index=True)
    sp = Spec.model_validate({"bands": [DST100, EQ40]})
    cons = {"max_k": 2, "max_total_wv": 0.1, "markets": ["EU"]}
    assert optimise(lib, sp, Constraints(**cons)).regulatory_removed == []
    strict = optimise(lib, sp, Constraints(**cons, policy="strict")).regulatory_removed
    assert strict == [
        "1 candidates excluded for EU: nonyl/octylphenol derivatives not named in the rules"
    ]
