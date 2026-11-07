"""Plain-language answers → technical targets (a Scenario), the one-line translation, and the
single relaxation that would make an infeasible spec work.

No physics: this only maps answers onto the five-group Scenario schema.
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from wetopt import optimise as opt
from wetopt.scenario import AI_PRESETS, DIFFICULT_LEAVES, OPPOSITE
from wetopt.schema import (
    ActiveIngredient,
    AICharge,
    Constraints,
    Market,
    Scenario,
    Spec,
    SprayConditions,
    TargetLeaf,
)

OTHER_CROP = "Other crop or weed"
OTHER_AI = "Other"
AGE_MS = 200.0
STICK = {"gentle": 0.055, "standard": 0.050, "strong": 0.045}  # DST(200 ms) upper limit, N/m
STICK_ORDER = ["gentle", "standard", "strong"]
STICK_SOURCE = {
    "standard": "published anchor: about 50 % adhesion on barley (Baur & Pontzen 2007)",
    "gentle": "tool assumption around the published anchor",
    "strong": "tool assumption around the published anchor",
}
CONFIDENCE = {"standard": (1.28, "about 90 %"), "high": (1.64, "about 95 %")}

TIPS = {
    "crop": "Some leaves are waxy and hard to wet; the spray must stick to them faster.",
    "beads": "If drops bead up and roll off, the leaf is water-repellent (hard to wet).",
    "hairy": "Hint only: hairy leaves shatter droplets, so consider the Strong stick level.",
    "stick": "Sets the dynamic surface tension (DST) target: how fast the droplet's surface "
    "tension drops after it forms; lower sticks better.",
    "ai": "An ionic active ingredient can clash with surfactants of the opposite charge.",
    "rate": "From your product label: how much product goes on per hectare.",
    "volume": "From your product label: how much water the product is sprayed in.",
    "hardness": "Hard water can weaken anionic surfactants; recorded, not modelled.",
    "droplets": "Recorded, not modelled. Lower DST at ~20 ms gives finer droplets and more drift.",
    "markets": "Ingredients banned or restricted where you sell are left out.",
    "precaution": "Also leave out ingredients that are only flagged 'check' (strict policy).",
    "max_k": "The tool still finds the fewest ingredients; this is the most you would accept.",
    "cap": "The most adjuvant (surfactant) you are willing to add to the spray tank.",
    "confidence": "How sure the prediction must be: Standard ≈ 90 %, High ≈ 95 %.",
}


class Answers(BaseModel):
    """The five plain-language questions."""

    crop: str = "wheat"
    beads: Literal["yes", "not sure", "no"] = "not sure"  # only asked for OTHER_CROP
    hairy: bool = False
    stick: Literal["gentle", "standard", "strong"] | None = None  # None → default by wettability
    ai: str = "azoxystrobin"  # an AI_PRESETS name or OTHER_AI
    ai_name: str = ""
    ai_charge: AICharge = "nonionic"
    product_rate: float | None = None
    rate_unit: Literal["L/ha", "kg/ha"] = "L/ha"
    spray_volume_l_ha: float | None = None
    hardness: Literal["soft", "medium", "hard"] = "medium"
    droplet_size: Literal["fine", "medium", "coarse"] = "medium"
    markets: list[Market] = ["EU"]
    precautionary: bool = False
    max_k: int = Field(3, ge=1, le=3)
    max_tank_wv: float = Field(0.2, gt=0)
    confidence: Literal["standard", "high"] = "standard"


def hard_to_wet(a: Answers) -> bool:
    """Listed crops are hard to wet; for another crop or weed, beading (or not sure) means hard."""
    return a.crop in DIFFICULT_LEAVES or a.beads != "no"


def stick_level(a: Answers) -> str:
    """Chosen stick level, defaulting to Standard (hard to wet) or Gentle (easy to wet)."""
    return a.stick or ("standard" if hard_to_wet(a) else "gentle")


def ai_identity(a: Answers) -> tuple[str, str]:
    if a.ai in AI_PRESETS:
        return a.ai, AI_PRESETS[a.ai]
    return a.ai_name.strip() or "other AI", a.ai_charge


def to_scenario(a: Answers) -> Scenario:
    """The technical targets implied by the answers."""
    name, charge = ai_identity(a)
    return Scenario(
        active_ingredient=ActiveIngredient(name=name, ionic_character=charge),
        target_leaf=TargetLeaf(
            preset=a.crop, hard_to_wet=hard_to_wet(a), hairy=True if a.hairy else None
        ),
        spray=SprayConditions(
            spray_volume_l_ha=a.spray_volume_l_ha,
            product_rate=a.product_rate,
            product_rate_unit=a.rate_unit,
            water_hardness=a.hardness,
            droplet_size=a.droplet_size,
        ),
        wetting_target=Spec.model_validate(
            {
                "bands": [
                    {
                        "metric": "dst",
                        "surface_age_ms": AGE_MS,
                        "lo": 0.0,
                        "hi": STICK[stick_level(a)],
                    }
                ],
                "z": CONFIDENCE[a.confidence][0],
            }
        ),
        constraints=Constraints(
            max_k=a.max_k,
            max_total_wv=a.max_tank_wv,
            markets=a.markets,
            policy="strict" if a.precautionary else "exclude",
        ),
    )


def what_this_means(
    a: Answers, custom: bool, n_charge: int, n_regulatory: int, scn: Scenario | None = None
) -> str:
    """One line translating the answers into targets (or 'Custom targets' after manual edits)."""
    if custom and scn is not None:
        bands = "; ".join(opt.band_label(b) for b in scn.wetting_target.bands)
        head = f"Custom targets (set in Advanced): {bands}, z = {scn.wetting_target.z:g}"
    else:
        level, crop = stick_level(a), a.crop if a.crop != OTHER_CROP else "This leaf"
        wet = "hard" if hard_to_wet(a) else "easy"
        default = level == ("standard" if wet == "hard" else "gentle")
        why = f"{crop.capitalize()} is {wet} to wet" + ("" if default else f", {level} stick")
        head = f"{why} → target DST ≤ {STICK[level] * 1e3:g} mN/m at {AGE_MS:g} ms"
    name, charge = ai_identity(a)
    opposite = OPPOSITE.get(charge)
    ai = f"{name.capitalize()} is {charge} → " + (
        f"{n_charge} {opposite} surfactants excluded"
        if opposite
        else "all surfactant classes allowed"
    )
    mk = (
        f"{'/'.join(a.markets)}{' (precautionary)' if a.precautionary else ''} → "
        f"{n_regulatory} ingredient{'s' if n_regulatory != 1 else ''} excluded"
        if a.markets
        else "No market selected → no regulatory screen"
    )
    conf = f"Confidence {CONFIDENCE[a.confidence][1].removeprefix('about ')}"
    return " · ".join([head, ai, mk] + ([] if custom and scn is not None else [conf]))


def relaxations(a: Answers) -> list[tuple[str, Answers]]:
    """Single relaxations, in the order to try: double the tank limit, step the stick level
    down, allow one more ingredient, untick a market."""
    out = [
        (
            f"Double the tank limit to {2 * a.max_tank_wv:g} % w/v",
            a.model_copy(update={"max_tank_wv": 2 * a.max_tank_wv}),
        )
    ]
    i = STICK_ORDER.index(stick_level(a))
    if i > 0:
        lower = STICK_ORDER[i - 1]
        out.append(
            (
                f"Step the stick level down to {lower.capitalize()}",
                a.model_copy(update={"stick": lower}),
            )
        )
    if a.max_k < 3:
        out.append(
            (f"Allow {a.max_k + 1} ingredients", a.model_copy(update={"max_k": a.max_k + 1}))
        )
    for m in a.markets:
        rest = [x for x in a.markets if x != m]
        out.append((f"Untick {m}", a.model_copy(update={"markets": rest})))
    return out


def suggest_relaxation(
    a: Answers, run: Callable[[Answers], opt.Outcome]
) -> tuple[str, Answers, opt.Outcome] | None:
    """The first single relaxation (in relaxations() order) whose run finds a feasible recipe."""
    for label, relaxed in relaxations(a):
        o = run(relaxed)
        if o.result is not None:
            return label, relaxed, o
    return None
