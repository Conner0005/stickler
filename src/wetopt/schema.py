"""Pydantic v2 data models for wetopt.

Units: every field is SI unless its name carries a unit suffix
(``surface_age_ms`` -> ms, ``max_total_wv`` -> % w/v). Concentrations are mol/m³
(numerically equal to mM), surface tensions N/m, surface excess mol/m²,
molar mass kg/mol.
"""

from typing import Literal, Self, get_args

from pydantic import BaseModel, Field, PositiveFloat, RootModel, model_validator

IonicClass = Literal["anionic", "cationic", "nonionic", "zwitterionic"]
Metric = Literal["dst", "gamma_eq"]


class Quantity(BaseModel):
    """A property value with 1σ uncertainty and provenance.

    Needs a ``source`` (citation, DOI or method) or ``imputed=True``;
    an imputed value must carry σ > 0.
    """

    value: float
    sigma: float = Field(ge=0)
    source: str | None = None
    imputed: bool = False

    @model_validator(mode="after")
    def _provenance(self) -> Self:
        if not (self.source or self.imputed):
            raise ValueError("value needs a source or imputed=True")
        if self.imputed and self.sigma <= 0:
            raise ValueError("imputed value needs sigma > 0")
        return self


class Surfactant(BaseModel):
    """One surfactant and its interfacial properties at T = 298.15 K."""

    id: str
    name: str
    smiles: str
    ionic_class: IonicClass
    mw: Quantity = Field(description="molar mass, kg/mol")
    cmc: Quantity = Field(description="critical micelle concentration, mol/m³")
    gamma_cmc: Quantity = Field(description="surface tension at the CMC, N/m")
    gamma_max: Quantity = Field(description="maximum surface excess Γ_max, mol/m²")
    pc20: Quantity | None = Field(
        None,
        description=(
            "pC20 = -log10(C20 / (mol/L)), C20 = conc. lowering γ by 20 mN/m; literature "
            "convention kept (Rosen & Kunjappu, Surfactants and Interfacial Phenomena, "
            "4th ed., 2012)"
        ),
    )

    @model_validator(mode="after")
    def _positive(self) -> Self:
        for name in ("mw", "cmc", "gamma_cmc", "gamma_max"):
            if getattr(self, name).value <= 0:
                raise ValueError(f"{name} must be > 0")
        return self


class Band(BaseModel):
    """Acceptance band lo < metric < hi, in N/m. ``dst`` needs a surface age."""

    metric: Metric
    surface_age_ms: float | None = Field(None, gt=0, description="surface age for dst, ms")
    lo: float
    hi: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.metric == "dst") != (self.surface_age_ms is not None):
            raise ValueError("surface_age_ms is required for dst and only for dst")
        if self.lo >= self.hi:
            raise ValueError("lo must be < hi")
        return self

    @property
    def surface_age_s(self) -> float | None:
        """Surface age in s (SI) for physics code."""
        return None if self.surface_age_ms is None else self.surface_age_ms * 1e-3


class Spec(BaseModel):
    """Wetting spec: all bands must hold at mean ± z·σ."""

    bands: list[Band] = Field(min_length=1)
    z: float = Field(1.28, ge=0, description="robustness margin in σ; 1.28 ≈ 90 % one-sided")


class Constraints(BaseModel):
    """Search constraints on candidate blends."""

    max_k: int = Field(3, ge=1, description="max number of surfactants in a blend")
    max_total_wv: float = Field(gt=0, description="cap on total surfactant, % w/v")
    allowed_classes: list[IonicClass] = list(get_args(IonicClass))
    excluded_ids: list[str] = []


class Blend(RootModel[dict[str, PositiveFloat]]):
    """Surfactant id -> concentration in mol/m³ (= mM)."""


class Prediction(BaseModel):
    """Model prediction for one band, N/m."""

    band: Band
    mean: float
    std: float = Field(ge=0)


class Result(BaseModel):
    """Optimiser output: minimal blend, per-band predictions and margins, certificate."""

    blend: Blend
    k: int = Field(ge=0)
    predictions: list[Prediction]
    margins: list[float] = Field(description="robust margin per prediction, N/m; ≥ 0 = band met")
    certificate: str

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.k != len(self.blend.root):
            raise ValueError("k must equal the number of blend components")
        if len(self.margins) != len(self.predictions):
            raise ValueError("need one margin per prediction")
        return self
