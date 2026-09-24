"""
Physiology-based case sampler for the vitals domain.

Design rule (this is the whole point)
-------------------------------------
Every vital is drawn INDEPENDENTLY of the others. There are no syndrome
templates and no archetypes, so there is no recurring pattern for the student
model to memorise. The clinical picture of a case is whatever the deterministic
NEWS2/qSOFA rules say it is -- it is never decided in advance.

The ONLY coupling between vitals is physical validity:
    diastolic < systolic, and pulse pressure within a survivable range.
That is physics, not a clinical pattern, so it does not create an archetype.

Tier balance is achieved by rejection sampling against a quota, never by
steering the vitals towards a desired answer.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, asdict, field
from typing import Iterator

import yaml
from pathlib import Path

from guardrails.clinical_rules import (
    calculate_news2, calculate_qsofa, determine_alert_level,
)
from distill.regions import Region, get as get_region

_TH = yaml.safe_load(open(Path("guardrails/thresholds.yaml")))
_PL = _TH["plausibility"]

# Per-vital proposal: a mixture of a "common" band and the full plausible range.
# Applied INDEPENDENTLY per vital, so the joint distribution stays a product and
# no cross-vital pattern is introduced. The mixture exists only so that rare
# tiers are reachable in sensible time.
_PROPOSAL = {
    "systolic_bp":  dict(common=(95, 160),  full=(_PL["systolic_bp"]["min"],  _PL["systolic_bp"]["max"])),
    "diastolic_bp": dict(common=(55, 95),   full=(_PL["diastolic_bp"]["min"], _PL["diastolic_bp"]["max"])),
    "heart_rate":   dict(common=(50, 120),  full=(_PL["heart_rate"]["min"],   _PL["heart_rate"]["max"])),
    "spo2":         dict(common=(90, 100),  full=(_PL["spo2"]["min"],         _PL["spo2"]["max"])),
}
_COMMON_WEIGHT = 0.55

MIN_PULSE_PRESSURE = 15
MAX_PULSE_PRESSURE = 120


@dataclass
class Case:
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    spo2: int
    tier: str = ""
    news2: dict = field(default_factory=dict)
    qsofa: dict = field(default_factory=dict)
    region: str | None = None          # held-out region this case falls in, if any
    case_id: str = ""

    def values(self) -> tuple:
        return (self.systolic_bp, self.diastolic_bp, self.heart_rate, self.spo2)


def _draw(rng: random.Random, vital: str) -> int:
    p = _PROPOSAL[vital]
    lo, hi = p["common"] if rng.random() < _COMMON_WEIGHT else p["full"]
    return int(rng.uniform(lo, hi + 1))


def physically_valid(sbp: int, dbp: int, hr: int, spo2: int) -> bool:
    """Physical validity only -- never a clinical pattern."""
    if not (_PL["systolic_bp"]["min"]  <= sbp  <= _PL["systolic_bp"]["max"]):  return False
    if not (_PL["diastolic_bp"]["min"] <= dbp  <= _PL["diastolic_bp"]["max"]): return False
    if not (_PL["heart_rate"]["min"]   <= hr   <= _PL["heart_rate"]["max"]):   return False
    if not (_PL["spo2"]["min"]         <= spo2 <= _PL["spo2"]["max"]):         return False
    pp = sbp - dbp
    return MIN_PULSE_PRESSURE <= pp <= MAX_PULSE_PRESSURE


def score(sbp: int, dbp: int, hr: int, spo2: int) -> tuple[str, dict, dict]:
    """The deterministic rules decide the tier. We never choose it."""
    n = calculate_news2(sbp, dbp, hr, spo2)
    q = calculate_qsofa(sbp, hr, spo2)
    lvl = determine_alert_level(n, q)
    return lvl.value if hasattr(lvl, "value") else str(lvl), n.model_dump(), q.model_dump()


def make_case(sbp: int, dbp: int, hr: int, spo2: int, held_out: Region | None) -> Case:
    tier, n, q = score(sbp, dbp, hr, spo2)
    region = held_out.name if (held_out and held_out.contains(sbp, dbp, hr, spo2)) else None
    cid = hashlib.sha256(f"{sbp}|{dbp}|{hr}|{spo2}".encode()).hexdigest()[:12]
    return Case(sbp, dbp, hr, spo2, tier, n, q, region, cid)


def sample(
    n_per_tier: int,
    seed: int,
    held_out_region: str | None = None,
    collect_held_out: int = 0,
    max_draws: int = 20_000_000,
) -> tuple[list[Case], list[Case]]:
    """
    Returns (train_cases, held_out_cases).

    train_cases    : balanced across tiers, EXCLUDING the held-out region entirely.
    held_out_cases : drawn only from inside the held-out region, balanced across
                     whatever tiers occur there.
    """
    rng = random.Random(seed)
    region = get_region(held_out_region) if held_out_region else None

    tiers = ["NORMAL", "MEDIUM", "HIGH", "CRITICAL"]
    train: dict[str, list[Case]] = {t: [] for t in tiers}
    held:  dict[str, list[Case]] = {t: [] for t in tiers}
    seen: set[tuple] = set()

    per_held = collect_held_out  # per tier inside the region (best effort)
    draws = 0
    while draws < max_draws:
        draws += 1
        sbp  = _draw(rng, "systolic_bp")
        dbp  = _draw(rng, "diastolic_bp")
        hr   = _draw(rng, "heart_rate")
        spo2 = _draw(rng, "spo2")
        if not physically_valid(sbp, dbp, hr, spo2):
            continue
        key = (sbp, dbp, hr, spo2)
        if key in seen:
            continue
        c = make_case(sbp, dbp, hr, spo2, region)

        if c.region is not None:
            if per_held and len(held[c.tier]) < per_held:
                seen.add(key); held[c.tier].append(c)
        else:
            if len(train[c.tier]) < n_per_tier:
                seen.add(key); train[c.tier].append(c)

        if all(len(train[t]) >= n_per_tier for t in tiers) and (
            not per_held or all(len(held[t]) >= per_held for t in tiers)
        ):
            break

    # Some tiers are intrinsically rare (HIGH needs total 5-6 with no single 3
    # and qSOFA < 2). Report what was achieved rather than spinning forever.
    shortfall = {t: n_per_tier - len(train[t]) for t in tiers if len(train[t]) < n_per_tier}
    held_short = {t: per_held - len(held[t]) for t in tiers if per_held and len(held[t]) < per_held}
    if shortfall or held_short:
        import warnings
        warnings.warn(
            f"quota not met after {draws:,} draws -- train short {shortfall}, "
            f"held-out short {held_short}. Raise max_draws or lower the quota.",
            RuntimeWarning,
        )

    tr = [c for t in tiers for c in train[t]]
    hl = [c for t in tiers for c in held[t]]
    rng.shuffle(tr); rng.shuffle(hl)
    return tr, hl
