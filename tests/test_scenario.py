"""Five-group scenario schema and the input rules (no physics)."""

import json

import pandas as pd
import pytest
import yaml

from wetopt import config
from wetopt.scenario import (
    coformulant_tank_cap_wv,
    compatibility_exclusions,
    optimise_scenario,
    tank_concentration_g_l,
    temperature_warning,
    time_to_impact_ms,
)
from wetopt.schema import ActiveIngredient, Scenario, SprayConditions

CEREAL = config.ROOT / "examples" / "cereal.yaml"


def test_schema_round_trip():
    scn = Scenario.model_validate(yaml.safe_load(CEREAL.read_text()))
    assert scn.active_ingredient.name == "azoxystrobin" and scn.target_leaf.preset == "wheat"
    assert scn.spray.spray_volume_l_ha == 200 and scn.spray.boom_height_m == 0.5
    again = Scenario.model_validate(yaml.safe_load(yaml.safe_dump(scn.model_dump(mode="json"))))
    assert again == scn == Scenario.model_validate_json(scn.model_dump_json())
    legacy = {"spec": scn.wetting_target.model_dump(), "constraints": scn.constraints.model_dump()}
    assert Scenario.model_validate(legacy).wetting_target == scn.wetting_target


def test_cationic_ai_excludes_anionic_surfactants():
    lib = pd.read_parquet(config.LIBRARY_PATH)
    ai = ActiveIngredient(name="paraquat dichloride", ionic_character="cationic")
    ex = compatibility_exclusions(lib, ai)
    assert set(ex["name"]) == set(lib.loc[lib["ionic_class"] == "anionic", "name"]) and len(ex)
    assert (ex["reason"] == "charge incompatibility with paraquat dichloride").all()
    assert compatibility_exclusions(
        lib, ai.model_copy(update={"ionic_character": "nonionic"})
    ).empty


def test_input_maths_and_warnings():
    assert tank_concentration_g_l(250.0, 1.0, 200.0) == pytest.approx(1.25)  # g/L × L/ha ÷ L/ha
    assert tank_concentration_g_l(250.0, None, 200.0) is None
    assert time_to_impact_ms(0.5, 5.0) == pytest.approx(100.0)  # 0.5 m ÷ 5 m/s = 100 ms
    assert temperature_warning(25.0) is None and temperature_warning(20.0) is None
    assert "outside the 20–30 °C" in temperature_warning(12.0)
    spray = SprayConditions(spray_volume_l_ha=200, product_rate=1.0)
    ai = ActiveIngredient(concentrate_density_kg_l=1.1, max_coformulant_wt_pct=10)
    assert coformulant_tank_cap_wv(ai, spray) == pytest.approx(0.055)  # 0.1 × 1.1 kg ÷ 200 L


def test_cereal_result_unchanged():
    scn = Scenario.model_validate(yaml.safe_load(CEREAL.read_text()))
    o = optimise_scenario(pd.read_parquet(config.LIBRARY_PATH), scn)
    top = o.feasible.iloc[0]
    assert o.result.k == 1 and top["blend"] == "C14H28O6 #1"
    assert top["total_wv"] == pytest.approx(0.161, abs=0.002) and o.certified
    json.dumps(scn.model_dump(mode="json"))
