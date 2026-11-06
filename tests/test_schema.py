"""Smoke test: every schema model loads from YAML and round-trips."""

import pytest
import yaml

from wetopt.schema import (
    Band,
    Blend,
    Constraints,
    Prediction,
    Quantity,
    Result,
    Spec,
    Surfactant,
)

# Schema fixture, NOT data: mw is computed from the SMILES; every other value is
# a placeholder flagged imputed.
SURFACTANT = """
id: fixture_anionic
name: schema fixture (not real data)
smiles: CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]
ionic_class: anionic
mw: {value: 0.288385, sigma: 0.0, source: "RDKit Descriptors.MolWt of smiles"}
cmc: {value: 8.0, sigma: 1.0, imputed: true}
gamma_cmc: {value: 0.039, sigma: 0.002, imputed: true}
gamma_max: {value: 3.0e-6, sigma: 0.5e-6, imputed: true}
pc20: {value: 2.5, sigma: 0.2, imputed: true}
"""

BAND = "{metric: dst, surface_age_ms: 100, lo: 0.0, hi: 0.045}"

CASES = {
    Quantity: "{value: 8.0, sigma: 1.0, imputed: true}",
    Surfactant: SURFACTANT,
    Band: BAND,
    Spec: f"bands: [{BAND}, {{metric: gamma_eq, lo: 0.022, hi: 0.035}}]\nz: 1.28",
    Constraints: "max_k: 3\nmax_total_wv: 0.5\nallowed_classes: [anionic, nonionic]\n"
    "excluded_ids: [fixture_cationic]",
    Blend: "{fixture_anionic: 2.0, fixture_nonionic: 0.5}",
    Prediction: f"band: {BAND}\nmean: 0.041\nstd: 0.002",
    Result: f"""
blend: {{fixture_anionic: 2.0}}
k: 1
predictions: [{{band: {BAND}, mean: 0.041, std: 0.002}}]
margins: [0.00144]
certificate: fixture
""",
}


@pytest.mark.parametrize(("model", "text"), CASES.items(), ids=lambda x: getattr(x, "__name__", ""))
def test_model_loads_from_yaml(model, text):
    obj = model.model_validate(yaml.safe_load(text))
    assert model.model_validate(obj.model_dump()) == obj
