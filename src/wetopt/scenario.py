"""Scenario inputs → model inputs, and a plain summary of what each input did.

No physics here. These are recorded, not modelled: the AI's own effect on surface tension, leaf
hairiness, θ of 50 % aqueous acetone, water temperature and hardness, and droplet size class.
"""

import pandas as pd

from wetopt import config
from wetopt import optimise as opt
from wetopt.schema import ActiveIngredient, Scenario, SprayConditions

NOT_MODELLED = "recorded, not modelled"
AI_PRESETS = {  # name → ionic character (project brief)
    "glyphosate isopropylamine salt": "anionic",
    "paraquat dichloride": "cationic",
    "azoxystrobin": "nonionic",
}
AI_NOTE = (
    "The AI's own effect on surface tension is not modelled — measure with the AI in the tank "
    "and use the calibration upload to capture it."
)
DIFFICULT_LEAVES = [  # hard to wet (Baur & Pontzen 2007 list of difficult-to-wet crops)
    "wheat",
    "barley",
    "maize",
    "rice",
    "millet",
    "oilseed rape",
    "cabbage",
    "soybean",
    "banana",
]
LEAF_BAND = {"metric": "dst", "surface_age_ms": 200.0, "lo": 0.0, "hi": 0.050}
LEAF_BAND_NOTE = "DST(200 ms) ≤ 50 mN/m: about 50 % adhesion on barley (Baur & Pontzen 2007)"
DEFAULT_AGE_MS = 200.0
DROPLET_TIP = "Lower DST at ~20 ms gives finer droplets and more drift."
OPPOSITE = {"anionic": "cationic", "cationic": "anionic"}
CHARGE_REASON = "charge incompatibility with"


def leaf_bands(preset: str) -> list[dict]:
    """Wetting-target bands prefilled by a leaf preset (difficult-to-wet crops); none for Custom."""
    return [dict(LEAF_BAND)] if preset in DIFFICULT_LEAVES else []


def tank_concentration_g_l(
    content: float | None, rate: float | None, volume_l_ha: float | None
) -> float | None:
    """AI concentration in the spray tank, g/L: content (g/L or g/kg) × product rate (L/ha or
    kg/ha) ÷ spray volume (L/ha). None if any input is missing."""
    if content is None or rate is None or volume_l_ha is None:
        return None
    return content * rate / volume_l_ha


def coformulant_tank_cap_wv(ai: ActiveIngredient, spray: SprayConditions) -> float | None:
    """Most surfactant an in-can co-formulant can put in the tank, % w/v:
    load (w/w) × product mass rate (kg/ha; L/ha × density) ÷ spray volume (L/ha) × 100.
    None if any input is missing."""
    rate, volume, load = spray.product_rate, spray.spray_volume_l_ha, ai.max_coformulant_wt_pct
    if rate is None or volume is None or load is None:
        return None
    if spray.product_rate_unit == "L/ha":
        if ai.concentrate_density_kg_l is None:
            return None
        rate = rate * ai.concentrate_density_kg_l
    return load / 100 * rate / volume * 100


def time_to_impact_ms(height_m: float | None, speed_m_s: float | None) -> float | None:
    """Estimated time from nozzle to leaf, ms: boom height ÷ droplet speed."""
    if height_m is None or speed_m_s is None:
        return None
    return height_m / speed_m_s * 1e3


def temperature_warning(t_c: float) -> str | None:
    """Warn outside the 20–30 °C range of the source data (the physics stays at 25 °C)."""
    lo, hi = config.T_RANGE_C
    if lo <= t_c <= hi:
        return None
    return (
        f"Water at {t_c:g} °C is outside the {lo:g}–{hi:g} °C range of the source data; "
        "predictions are for 25 °C."
    )


def compatibility_exclusions(lib: pd.DataFrame, ai: ActiveIngredient) -> pd.DataFrame:
    """Surfactants of the opposite charge to an ionic AI (columns name, reason)."""
    opposite = OPPOSITE.get(ai.ionic_character)
    names = lib.loc[lib["ionic_class"] == opposite, "name"] if opposite else lib["name"].iloc[:0]
    who = ai.name or f"the {ai.ionic_character} AI"
    return pd.DataFrame({"name": names.to_numpy(), "reason": f"{CHARGE_REASON} {who}"})


def hard_water_flag(scn: Scenario, ionic_class: str) -> str:
    """Badge for anionic surfactants in hard water (recorded, not modelled)."""
    if scn.spray.water_hardness == "hard" and ionic_class == "anionic":
        return "💧 hard water: anionic"
    return ""


def optimise_scenario(lib: pd.DataFrame, scn: Scenario, **kwargs) -> opt.Outcome:
    """Run the optimiser on a scenario; the AI compatibility rule removes candidates first."""
    compat = compatibility_exclusions(lib, scn.active_ingredient)
    return opt.optimise(lib, scn.wetting_target, scn.constraints, extra_excluded=compat, **kwargs)


def input_summary(scn: Scenario, o: opt.Outcome, lib: pd.DataFrame) -> list[str]:
    """One line per input group saying what it did to the run."""
    ai, leaf, spray, cons = scn.active_ingredient, scn.target_leaf, scn.spray, scn.constraints
    lines = []

    n_charge = (
        int(o.excluded["reason"].str.startswith(CHARGE_REASON).sum()) if len(o.excluded) else 0
    )
    opposite = OPPOSITE.get(ai.ionic_character)
    tank = tank_concentration_g_l(ai.content, spray.product_rate, spray.spray_volume_l_ha)
    ai_line = f"AI: {ai.name or 'unnamed'} ({ai.ionic_character}, {ai.formulation}); "
    ai_line += (
        f"excluded {n_charge} {opposite} surfactants (charge incompatibility)"
        if opposite
        else "no charge exclusions"
    )
    ai_line += (
        f"; tank AI {tank:.3g} g/L ({tank / 10:.3g} % w/v)"
        if tank is not None
        else "; tank AI not computed (needs content, product rate and spray volume)"
    )
    lines.append(ai_line + "; its own effect on surface tension is not modelled.")

    leaf_line = f"Leaf: {leaf.preset}"
    leaf_line += f"; target prefilled: {LEAF_BAND_NOTE}" if leaf.preset in DIFFICULT_LEAVES else ""
    if leaf.hairy is not None or leaf.theta_acetone50_deg is not None:
        leaf_line += f"; hairy leaf / θ(50 % acetone) {NOT_MODELLED}"
    lines.append(leaf_line + ".")

    ages = sorted({b.surface_age_ms for b in scn.wetting_target.bands if b.metric == "dst"})
    tti = time_to_impact_ms(spray.boom_height_m, spray.droplet_speed_m_s)
    spray_line = "Spray: surface age " + (
        ", ".join(f"{a:g} ms" for a in ages) if ages else "none (no DST band)"
    )
    if tti is not None:
        spray_line += (
            f"; estimated time to impact {tti:.0f} ms "
            f"({spray.boom_height_m:g} m ÷ {spray.droplet_speed_m_s:g} m/s)"
        )
    elif spray.boom_height_m is not None:
        spray_line += "; time to impact needs a droplet speed"
    lines.append(spray_line + ".")

    water = (
        f"Water and droplets: {spray.water_temperature_c:g} °C, {spray.water_hardness} water, "
        f"{spray.droplet_size} droplets, {NOT_MODELLED}"
    )
    warn = temperature_warning(spray.water_temperature_c)
    n_hard = int((lib["ionic_class"] == "anionic").sum()) if spray.water_hardness == "hard" else 0
    lines.append(
        water
        + (f"; {warn}" if warn else "")
        + (f"; hard-water badge on {n_hard} anionic candidates" if n_hard else "")
        + "."
    )

    bands = "; ".join(opt.band_label(b) for b in scn.wetting_target.bands)
    lines.append(f"Wetting target: {bands}; z = {scn.wetting_target.z:g}.")

    n_qc = len(o.excluded) - n_charge
    lines.append(
        f"Constraints: k ≤ {cons.max_k}, total ≤ {cons.max_total_wv:g} % w/v; "
        f"data quality excluded {n_qc}."
    )
    if o.regulatory_removed:
        lines += [f"Markets: {line}." for line in o.regulatory_removed]
    else:
        lines.append(f"Markets {', '.join(cons.markets) or 'none'} ({cons.policy}): excluded 0.")
    return lines
