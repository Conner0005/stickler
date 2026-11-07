"""Fewest-surfactant blend search with a no-smaller-blend certificate.

For k = 1..max_k: enumerate subsets, score a log grid of loadings (% w/v) with the nominal
model, classify with Monte Carlo σ, refine the best ~20 subsets with bounded Powell, and stop at
the first k with a feasible blend. Band logic (z from the Spec, σ = MC std ⊕ σ_model):
- feasible: every band holds on its pessimistic side. 'DST ≤ hi' uses dst_high + zσ ≤ hi,
  'DST ≥ lo' uses dst_low − zσ ≥ lo, and γ_eq uses mean ± zσ.
- failed: some band fails even on its optimistic side (dst_low − zσ > hi or dst_high + zσ < lo).
- undetermined: neither. These are listed as worth testing; they block the certificate.
No monotonicity is assumed: γ can rise with concentration above the mixed CMC.
"""

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wetopt import config, regulatory
from wetopt.physics import (
    DSTStats,
    SurfactantArrays,
    dst_bracket,
    gamma_water,
    monte_carlo_dst,
    szyszkowski_K,
)
from wetopt.schema import Band, Blend, Constraints, Prediction, Result, Spec
from wetopt.validate import INCONSISTENT, data_quality, pc20_check

GRID_LEVELS = 10  # log-grid levels per component
GRID_SPAN = 1e-3  # lowest level as a fraction of the loading cap
N_REFINE = 20  # subsets refined per k
DELTA = 5e-5  # N/m (0.05 mN/m) margin buffer while refining
WV_TO_KG_M3 = 10.0  # 1 % w/v = 10 kg/m³
SCREEN_CHUNK = 300_000  # blends per nominal-model chunk
MC_CHUNK = 2_000  # blends per Monte Carlo chunk
MAX_MC = 20_000  # at max_k, Monte Carlo at most this many screen-passing blends (lowest loading)


@dataclass(frozen=True)
class Problem:
    names: np.ndarray
    ionic: np.ndarray
    mw: np.ndarray  # kg/mol
    props: SurfactantArrays
    K: np.ndarray  # nominal Langmuir constants, m³/mol
    gamma0: float
    ages: np.ndarray  # s; inf = equilibrium
    band_age: np.ndarray  # column of ``ages`` for each band
    bands: tuple[Band, ...]
    lo: np.ndarray  # N/m
    hi: np.ndarray  # N/m
    z: float
    cap: float  # % w/v
    max_k: int
    sigma_model: float  # N/m
    n_draws: int
    allowed: np.ndarray  # indices of searchable candidates (arrays above span the whole library)


def make_problem(
    lib: pd.DataFrame,
    spec: Spec,
    cons: Constraints,
    sigma_model: float,
    n_draws: int = 200,
    drop: frozenset[str] = frozenset(),
) -> tuple[Problem, pd.DataFrame]:
    """Precompute nominal model inputs for the whole library; search only allowed candidates.

    Arrays span the whole library so each surfactant keeps the same Monte Carlo draws whatever
    is excluded (by allowed_classes, excluded_ids or ``drop`` names).
    """
    ex = set(cons.excluded_ids) | set(drop)
    keep = (
        lib["ionic_class"].isin(cons.allowed_classes)
        & ~lib["name"].isin(ex)
        & ~lib["inchikey"].isin(ex)
    ).to_numpy()
    lib = lib.reset_index(drop=True)
    props = SurfactantArrays.from_library(lib)
    g0 = float(gamma_water())
    band_ages = [b.surface_age_s if b.metric == "dst" else np.inf for b in spec.bands]
    ages = np.unique(band_ages)
    problem = Problem(
        names=lib["name"].to_numpy(),
        ionic=lib["ionic_class"].to_numpy(),
        mw=lib["mw"].to_numpy(float),
        props=props,
        K=szyszkowski_K(props.cmc, props.gamma_cmc, props.gamma_max, props.n, g0),
        gamma0=g0,
        ages=ages,
        band_age=np.searchsorted(ages, band_ages),
        bands=tuple(spec.bands),
        lo=np.array([b.lo for b in spec.bands]),
        hi=np.array([b.hi for b in spec.bands]),
        z=spec.z,
        cap=cons.max_total_wv,
        max_k=cons.max_k,
        sigma_model=sigma_model,
        n_draws=n_draws,
        allowed=np.flatnonzero(keep),
    )
    return problem, lib[keep]


def subsets(p: Problem, k: int) -> np.ndarray:
    """All k-subsets of candidates, skipping anionic + cationic pairs; shape (n, k)."""
    rows = [
        c
        for c in itertools.combinations(p.allowed, k)
        if not {"anionic", "cationic"} <= {p.ionic[i] for i in c}
    ]
    return np.array(rows, dtype=int).reshape(-1, k)


def log_grid(k: int, cap: float, levels: int = GRID_LEVELS) -> np.ndarray:
    """Per-component loadings (% w/v) on a log grid from GRID_SPAN·cap to cap, Σ ≤ cap."""
    g = np.array(list(itertools.product(np.geomspace(GRID_SPAN * cap, cap, levels), repeat=k)))
    return g[g.sum(1) <= cap * (1 + 1e-9)]


def conc(p: Problem, idx: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """% w/v -> mol/m³: C = 10·w/M."""
    return WV_TO_KG_M3 * wv / p.mw[idx]


def mean_bracket(p: Problem, idx: np.ndarray, wv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nominal-parameter (dst_low, dst_high), each (B, len(ages)); t = inf is γ_eq."""
    pr = p.props
    return dst_bracket(
        p.ages,
        conc(p, idx, wv),
        pr.cmc[idx],
        p.K[idx],
        pr.gamma_max[idx],
        pr.n[idx],
        pr.D[idx],
        p.gamma0,
    )


def band_margins(p: Problem, low, low_std, high, high_std) -> tuple[np.ndarray, np.ndarray]:
    """(pessimistic, optimistic) margins per band, N/m, shape (B, n_bands); ≥ 0 = pass.

    pessimistic = min(hi − (dst_high + zσ_high), (dst_low − zσ_low) − lo)
    optimistic  = min(hi − (dst_low − zσ_low), (dst_high + zσ_high) − lo)
    At t = ∞ both bounds equal γ_eq, so the same formulas give γ_eq ± zσ.
    """
    low = np.asarray(low)
    lm, ls, hm, hs = (
        np.broadcast_to(a, low.shape)[:, p.band_age] for a in (low, low_std, high, high_std)
    )
    pess = np.minimum(p.hi - (hm + p.z * hs), (lm - p.z * ls) - p.lo)
    opt = np.minimum(p.hi - (lm - p.z * ls), (hm + p.z * hs) - p.lo)
    return pess, opt


def mc_eval(p: Problem, idx: np.ndarray, wv: np.ndarray) -> tuple[DSTStats, np.ndarray, np.ndarray]:
    """Monte Carlo stats and (pessimistic, optimistic) margins for blends, in chunks."""
    parts = [
        monte_carlo_dst(
            p.ages,
            idx[s : s + MC_CHUNK],
            conc(p, idx[s : s + MC_CHUNK], wv[s : s + MC_CHUNK]),
            p.props,
            n_draws=p.n_draws,
            sigma_model=p.sigma_model,
        )
        for s in range(0, len(idx), MC_CHUNK)
    ]
    stats = DSTStats(*(np.concatenate(a) for a in zip(*parts, strict=True)))
    return stats, *band_margins(p, *stats)


def _refine(p: Problem, idx: np.ndarray, wv0: np.ndarray, stats: DSTStats, goal: str) -> np.ndarray:
    """Bounded Powell on log10 loadings, with σ and the Monte Carlo − nominal mean offset
    frozen at the seed.

    goal 'loading': min Σw/cap + 100·Σ relu(DELTA − margin_b)/(mN/m); goal 'optimistic':
    max the worst optimistic margin. Both penalise Σw > cap.
    """
    low_std, high_std = stats.low_std[None], stats.high_std[None]
    low0, high0 = mean_bracket(p, idx[None], wv0[None])
    d_low, d_high = stats.low_mean[None] - low0, stats.high_mean[None] - high0

    def f(x: np.ndarray) -> float:
        wv = 10.0**x
        low, high = mean_bracket(p, idx[None], wv[None])
        pess, opt = band_margins(p, low + d_low, low_std, high + d_high, high_std)
        over = 100.0 * max(0.0, wv.sum() / p.cap - 1.0)
        if goal == "loading":
            return wv.sum() / p.cap + 1e5 * np.maximum(DELTA - pess, 0.0).sum() + over
        return -1e3 * opt.min() + over

    bounds = [(np.log10(GRID_SPAN * p.cap), np.log10(p.cap))] * len(idx)
    opts = {"xtol": 1e-3, "ftol": 1e-9, "maxfev": 300}
    res = minimize(f, np.log10(wv0), method="Powell", bounds=bounds, options=opts)
    wv = 10.0**res.x
    return wv * min(1.0, p.cap / wv.sum())


@dataclass
class Level:
    """Every blend of one k that was classified with Monte Carlo σ."""

    k: int
    n_grid: int
    complete: bool  # every grid blend was Monte Carlo classified
    idx: np.ndarray
    wv: np.ndarray
    refined: np.ndarray
    stats: DSTStats
    pess: np.ndarray
    opt: np.ndarray

    @property
    def feasible(self) -> np.ndarray:
        return self.pess.min(1) >= 0

    @property
    def failed(self) -> np.ndarray:
        return self.opt.min(1) < 0

    @property
    def undetermined(self) -> np.ndarray:
        return ~self.feasible & ~self.failed


def _best_per_subset(idx: np.ndarray, score: np.ndarray, n: int) -> np.ndarray:
    """Row indices of the best-scoring row (lowest score) of each subset, top n subsets."""
    order = np.lexsort((score,))
    _, first = np.unique(np.sort(idx[order], axis=1), axis=0, return_index=True)
    rows = order[first]
    return rows[np.argsort(score[rows])][:n]


def search_level(p: Problem, k: int) -> Level | None:
    """Grid, Monte Carlo classification and refinement for one k."""
    S, G = subsets(p, k), log_grid(k, p.cap)
    if len(S) == 0:
        return None
    pess_s = np.empty((len(S), len(G)))
    step = max(1, SCREEN_CHUNK // len(G))
    for s in range(0, len(S), step):
        idx = np.repeat(S[s : s + step], len(G), axis=0)
        wv = np.tile(G, (len(S[s : s + step]), 1))
        low, high = mean_bracket(p, idx, wv)
        pe, _ = band_margins(p, low, p.sigma_model, high, p.sigma_model)
        pess_s[s : s + step] = pe.min(1).reshape(-1, len(G))

    complete = k < p.max_k  # a certificate for this k may be needed later
    if complete:
        si, gi = np.divmod(np.arange(pess_s.size), len(G))
    else:  # σ_model ≤ σ, so a blend failing this screen cannot be feasible
        si, gi = np.nonzero(pess_s >= 0)
        keep = np.argsort(G[gi].sum(1), kind="stable")[:MAX_MC]
        si, gi = si[keep], gi[keep]
        if len(si) == 0:  # nothing passes the screen: seed refinement from the best screen points
            best = np.argsort(-pess_s.max(1))[:N_REFINE]
            si, gi = best, pess_s[best].argmax(1)
    idx, wv = S[si], G[gi]
    stats, pess, opt = mc_eval(p, idx, wv)
    level = Level(k, len(S) * len(G), complete, idx, wv, np.zeros(len(idx), bool), stats, pess, opt)

    # Refine: lowest loading among feasible, else closest to feasible. For infeasible levels
    # below max_k, also raise the optimistic margin to find undetermined blends between grid points.
    loading = wv.sum(1)
    if level.feasible.any():
        score = np.where(level.feasible, loading / p.cap - 1e-3 * pess.min(1), np.inf)
        goals = [("loading", score)]
    else:
        goals = [("loading", -pess.min(1))] + ([("optimistic", -opt.min(1))] if complete else [])
    new_idx, new_wv = [], []
    for goal, score in goals:
        for r in _best_per_subset(idx, score, N_REFINE):
            if np.isfinite(score[r]):
                seed = DSTStats(*(a[r] for a in stats))
                w = _refine(p, idx[r], wv[r], seed, goal)
                # Backtrack along the log-loading line in case Monte Carlo rejects the optimum.
                alphas = (1.0, 0.75, 0.5, 0.25) if goal == "loading" else (1.0,)
                for a in alphas:
                    new_idx.append(idx[r])
                    new_wv.append(wv[r] ** (1 - a) * w**a)
    if new_idx:
        ri, rw = np.array(new_idx), np.array(new_wv)
        rs, rp, ro = mc_eval(p, ri, rw)
        level = Level(
            k,
            level.n_grid,
            complete,
            np.vstack([idx, ri]),
            np.vstack([wv, rw]),
            np.r_[level.refined, np.ones(len(ri), bool)],
            DSTStats(*(np.concatenate([a, b]) for a, b in zip(stats, rs, strict=True))),
            np.vstack([pess, rp]),
            np.vstack([opt, ro]),
        )
    return level


def band_label(b: Band) -> str:
    """'DST(200 ms) ≤ 50 mN/m' style label (I/O units); lo = 0 and hi ≥ 72 mN/m are open ends."""
    what = f"DST({b.surface_age_ms:g} ms)" if b.metric == "dst" else "γ_eq"
    parts = [f"≥ {b.lo * 1e3:g}"] if b.lo > 0 else []
    parts += [f"≤ {b.hi * 1e3:g}"] if b.hi < 0.072 else []
    return f"{what} {' and '.join(parts)} mN/m"


def _blend_text(p: Problem, idx, wv) -> str:
    return " + ".join(f"{p.names[i]} {w:.3g}%" for i, w in zip(idx, wv, strict=True))


def _water_certificate(p: Problem) -> str:
    g = np.full((1, len(p.ages)), p.gamma0)
    _, opt = band_margins(p, g, p.sigma_model, g, p.sigma_model)
    b = int(opt[0].argmin())
    if opt[0, b] >= 0:
        return ""
    return (
        f"k = 0: water alone fails {band_label(p.bands[b])} even on the optimistic side "
        f"(γ0 = {p.gamma0 * 1e3:.1f} mN/m)."
    )


def certificate(p: Problem, prev: Level | None, k: int, n_candidates: int) -> tuple[str, bool]:
    """Certificate text for k − 1 and whether it holds."""
    if k == 1:
        text = _water_certificate(p)
        return (
            (text, True) if text else ("No certificate: water alone is not shown to fail.", False)
        )
    assert prev is not None
    who = "single surfactant" if k == 2 else f"{k - 1}-surfactant blend"
    if not prev.complete:
        return f"No certificate: the k = {k - 1} level was not exhaustively classified.", False
    if prev.undetermined.any():
        return (
            f"No certificate: {int(prev.undetermined.sum())} {who}s are undetermined "
            "(the DST bracket straddles a band); they are worth testing in the lab.",
            False,
        )
    r = int(prev.opt.min(1).argmax())  # closest to passing
    b = int(prev.opt[r].argmin())
    band, j, z, s = p.bands[b], p.band_age[b], p.z, prev.stats
    lo_m, lo_s, hi_m, hi_s = s.low_mean[r, j], s.low_std[r, j], s.high_mean[r, j], s.high_std[r, j]
    dst = band.metric == "dst"
    if band.hi - (lo_m - z * lo_s) <= (hi_m + z * hi_s) - band.lo:  # upper limit fails
        val, sd, bound, cmp, lim = lo_m, lo_s, lo_m - z * lo_s, ">", band.hi
        side = "optimistic (dst_low)" if dst else "γ_eq"
    else:  # lower limit fails
        val, sd, bound, cmp, lim = hi_m, hi_s, hi_m + z * hi_s, "<", band.lo
        side = "optimistic (dst_high)" if dst else "γ_eq"
    n_ref = int(prev.refined.sum())
    return (
        f"No {who} meets {band_label(band)} within {p.cap:g}% w/v (z = {z:g}): all "
        f"{len(prev.idx)} {who}s checked ({prev.n_grid} grid points over {n_candidates} candidates "
        f"+ {n_ref} refined, Monte Carlo σ ⊕ σ_model {p.sigma_model * 1e3:.1f} mN/m) fail at least "
        f"one band on its optimistic side. Closest: {_blend_text(p, prev.idx[r], prev.wv[r])}, "
        f"{side} {val * 1e3:.1f} ± {sd * 1e3:.1f} mN/m; {val * 1e3:.1f} "
        f"{'−' if cmp == '>' else '+'} {z:g}σ = {bound * 1e3:.1f} {cmp} {lim * 1e3:g} mN/m."
    ), True


def _rows(p: Problem, level: Level, rows: np.ndarray) -> pd.DataFrame:
    s, out = level.stats, []
    for r in rows:
        rec = {
            "blend": " + ".join(p.names[i] for i in level.idx[r]),
            "wv": [float(w) for w in level.wv[r]],
            "total_wv": float(level.wv[r].sum()),
            "worst_margin": float(level.pess[r].min()),
            "worst_optimistic": float(level.opt[r].min()),
            "refined": bool(level.refined[r]),
        }
        for b, band in enumerate(p.bands):
            j = p.band_age[b]
            rec[band_label(band)] = (
                float(s.low_mean[r, j]),
                float(s.high_mean[r, j]),
                float(max(s.low_std[r, j], s.high_std[r, j])),
            )
        out.append(rec)
    return pd.DataFrame(out)


def _closest_first(und: pd.DataFrame) -> pd.DataFrame:
    """Undetermined blends, best per surfactant set, nearest to provable feasibility first."""
    if not len(und):
        return und
    und = und.sort_values("worst_margin", ascending=False)
    return und.drop_duplicates("blend").reset_index(drop=True)


@dataclass
class Outcome:
    result: Result | None
    feasible: pd.DataFrame  # best blend per subset at the winning k, ranked
    undetermined: pd.DataFrame  # worth testing in the lab
    certificate: str
    certified: bool
    sigma_model: float
    pc20: dict
    levels: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)  # name, reason
    regulatory_removed: list[str] = field(default_factory=list)
    regulatory_note: str = ""  # disclaimer + snapshot date; show wherever flags appear


def optimise(
    lib: pd.DataFrame,
    spec: Spec,
    cons: Constraints,
    n_draws: int = 200,
    sigma_model: float | None = None,
    exclude_inconsistent: bool = config.EXCLUDE_INCONSISTENT,
) -> Outcome:
    """Find the fewest-surfactant feasible blend, then the lowest loading; certify k − 1.

    σ_model defaults to max(config.SIGMA_MODEL, pC20-check RMSE over all fully measured
    candidates) (config value if there are none). Candidates flagged INCONSISTENT by
    data_quality are excluded unless ``exclude_inconsistent`` is False.
    """
    t0 = time.perf_counter()
    check = pc20_check(lib)
    if sigma_model is None:
        sigma_model = max(config.SIGMA_MODEL, check.rmse) if check.n else config.SIGMA_MODEL
    qc = data_quality(lib, config.QC_SIGMA_MULT * sigma_model)
    excluded = qc.loc[qc["status"] == INCONSISTENT, ["name", "reason"]]
    if not exclude_inconsistent:
        excluded = excluded.iloc[:0]
    regs = regulatory.load()
    screened = regulatory.screen(lib, list(cons.markets), regs)
    blocking = screened["blocking"].map(bool)
    blocked = screened[blocking] if cons.policy == "exclude" else screened.iloc[:0]
    removed = regulatory.removal_summary(blocked, list(cons.markets))
    note = f"Regulatory snapshot as of {regs.as_of}. {regs.disclaimer}"
    reg_flags = dict(zip(screened["name"], screened["flags"], strict=True))
    drop = frozenset(excluded["name"]) | frozenset(blocked["name"])
    p, cands = make_problem(lib, spec, cons, sigma_model, n_draws, drop)
    levels: list[Level] = []
    winner_level = None
    for k in range(1, p.max_k + 1):
        level = search_level(p, k)
        if level is None:
            break
        levels.append(level)
        if level.feasible.any():
            winner_level = level
            break

    summaries = [
        {
            "k": lv.k,
            "grid_blends": lv.n_grid,
            "classified": len(lv.idx),
            "complete": lv.complete,
            "feasible": int(lv.feasible.sum()),
            "failed": int(lv.failed.sum()),
            "undetermined": int(lv.undetermined.sum()),
        }
        for lv in levels
    ]
    pc20 = {"n": check.n, "rmse_mN_m": None, "bias_mN_m": None}
    if check.n:
        pc20.update(rmse_mN_m=check.rmse * 1e3, bias_mN_m=check.bias * 1e3)
    if winner_level is None:
        frames = [_rows(p, lv, np.flatnonzero(lv.undetermined)) for lv in levels]
        und = pd.concat(frames) if frames else pd.DataFrame()
        und = _closest_first(und)
        text = f"No blend with k ≤ {p.max_k} is feasible under the model."
        done = [lv for lv in levels if lv.complete and lv.failed.all()]
        certified = bool(done)
        if done:  # certify the largest k at which every blend fails
            text += " " + certificate(p, done[-1], done[-1].k + 1, len(cands))[0]
        runtime = time.perf_counter() - t0
        return Outcome(
            None,
            pd.DataFrame(),
            und,
            text,
            certified,
            sigma_model,
            pc20,
            summaries,
            runtime,
            excluded,
            removed,
            note,
        )

    lv, k = winner_level, winner_level.k
    feas = np.flatnonzero(lv.feasible)
    order = feas[np.lexsort((-lv.pess[feas].min(1), lv.wv[feas].sum(1)))]
    best = _best_per_subset(lv.idx[order], np.arange(len(order)), len(order))
    ranked = order[np.sort(best)]
    prev = levels[-2] if k > 1 else None
    text, certified = certificate(p, prev, k, len(cands))
    und = _closest_first(
        _rows(p, prev, np.flatnonzero(prev.undetermined)) if prev is not None else pd.DataFrame()
    )

    w = ranked[0]
    preds = []
    for b, band in enumerate(p.bands):
        j, s = p.band_age[b], lv.stats
        hi_side = p.hi[b] - (s.high_mean[w, j] + p.z * s.high_std[w, j])
        lo_side = (s.low_mean[w, j] - p.z * s.low_std[w, j]) - p.lo[b]
        use_high = hi_side <= lo_side
        mean = s.high_mean[w, j] if use_high else s.low_mean[w, j]
        std = s.high_std[w, j] if use_high else s.low_std[w, j]
        preds.append(Prediction(band=band, mean=float(mean), std=float(std)))
    blend = {
        p.names[i]: float(c) for i, c in zip(lv.idx[w], conc(p, lv.idx[w], lv.wv[w]), strict=True)
    }
    result = Result(
        blend=Blend(blend),
        k=k,
        predictions=preds,
        margins=[float(m) for m in lv.pess[w]],
        certificate=text,
        regulatory_flags={name: [f.text() for f in reg_flags[name]] for name in blend},
        regulatory_removed=removed,
        regulatory_note=note,
    )
    return Outcome(
        result,
        _rows(p, lv, ranked),
        und,
        text,
        certified,
        sigma_model,
        pc20,
        summaries,
        time.perf_counter() - t0,
        excluded,
        removed,
        note,
    )
