"""Plain-language answers → technical targets, Custom detection, and the relaxation suggestion."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from wetopt import config
from wetopt.questions import (
    OTHER_AI,
    OTHER_CROP,
    Answers,
    hard_to_wet,
    stick_level,
    suggest_relaxation,
    to_scenario,
    what_this_means,
)
from wetopt.scenario import optimise_scenario


def _hi(a: Answers) -> float:
    return to_scenario(a).wetting_target.bands[0].hi


def test_answers_map_to_technical_targets():
    s = to_scenario(Answers())  # wheat, azoxystrobin, EU, standard
    band = s.wetting_target.bands[0]
    assert (band.metric, band.surface_age_ms, band.lo, band.hi) == ("dst", 200.0, 0.0, 0.050)
    assert s.wetting_target.z == 1.28 and s.target_leaf.hard_to_wet
    assert s.constraints.markets == ["EU"] and s.constraints.policy == "exclude"
    assert (s.constraints.max_k, s.constraints.max_total_wv) == (3, 0.2)
    easy = Answers(crop=OTHER_CROP, beads="no")
    assert not hard_to_wet(easy) and stick_level(easy) == "gentle" and _hi(easy) == 0.055
    assert hard_to_wet(Answers(crop=OTHER_CROP, beads="yes"))
    assert hard_to_wet(Answers(crop=OTHER_CROP, beads="not sure"))
    assert _hi(Answers(stick="strong")) == 0.045 and _hi(Answers(stick="gentle")) == 0.055
    assert to_scenario(Answers(confidence="high")).wetting_target.z == 1.64
    s = to_scenario(
        Answers(
            ai="paraquat dichloride",
            markets=["EU", "GB"],
            precautionary=True,
            max_k=2,
            max_tank_wv=0.3,
            hardness="hard",
            product_rate=2.0,
            spray_volume_l_ha=150,
        )
    )
    assert s.active_ingredient.ionic_character == "cationic" and s.constraints.policy == "strict"
    assert (s.constraints.max_k, s.constraints.max_total_wv) == (2, 0.3)
    assert s.constraints.markets == ["EU", "GB"] and s.spray.water_hardness == "hard"
    assert (s.spray.product_rate, s.spray.spray_volume_l_ha) == (2.0, 150)
    other = to_scenario(Answers(ai=OTHER_AI, ai_name="X-cide", ai_charge="anionic"))
    assert (other.active_ingredient.name, other.active_ingredient.ionic_character) == (
        "X-cide",
        "anionic",
    )
    line = what_this_means(Answers(), custom=False, n_charge=0, n_regulatory=2)
    assert line.startswith("Wheat is hard to wet → target DST ≤ 50 mN/m at 200 ms")
    assert "all surfactant classes allowed" in line and "EU → 2 ingredients excluded" in line
    assert line.endswith("Confidence 90 %")


def test_relaxation_finds_a_feasible_recipe():
    lib = pd.read_parquet(config.LIBRARY_PATH)

    def run(b: Answers):
        return optimise_scenario(lib, to_scenario(b), exhaustive=False)

    tight = Answers(max_k=1, max_tank_wv=0.1)  # wheat / Standard at 0.1 % w/v: infeasible
    assert run(tight).result is None
    label, relaxed, o = suggest_relaxation(tight, run)
    assert label == "Double the tank limit to 0.2 % w/v" and relaxed.max_tank_wv == 0.2
    assert o.result is not None and o.result.k == 1


def _render() -> None:
    from wetopt.app import optimiser_page

    optimiser_page()


def test_form_line_and_custom_switch():
    at = AppTest.from_function(_render, default_timeout=120).run()
    assert not at.exception
    assert [h.value for h in at.sidebar.header] == [
        "1. What are you spraying onto?",
        "2. How strongly should the spray stick?",
        "3. What's in the tank?",
        "4. Where will it be sold?",
        "5. Preferences",
    ]
    assert at.sidebar.button[0].label == "Find recipe"
    assert "What this means:** Wheat is hard to wet" in at.info[0].value
    assert at.session_state["run"]["o"].result.k == 1  # default opens on the cereal result
    z = next(n for n in at.sidebar.number_input if n.label == "Robustness z")
    z.set_value(1.5).run()
    assert "Custom targets" in at.info[0].value
