"""
Held-out regions of the physiological space.

Why this exists
---------------
The previous pipeline generated cases from 7 syndrome templates and then split
those same cases into train/val. The validation set was therefore drawn from the
SAME generator as training, so a high validation score mostly measured how well
the model had memorised the generator -- not whether it understood physiology.

The fix used here: carve out a REGION of the vital-sign space, never show it
during training, and test only inside it. A model that learned physiology will
interpolate into the hidden region. A model that memorised examples will not.

Each region is a plain predicate over the four vitals. Regions are deliberately
simple boxes so that "what was hidden" is unambiguous and reportable.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Region:
    name: str
    description: str
    clinical_meaning: str
    predicate: Callable[[float, float, float, float], bool]

    def contains(self, sbp: float, dbp: float, hr: float, spo2: float) -> bool:
        return self.predicate(sbp, dbp, hr, spo2)


# ── The catalogue ─────────────────────────────────────────────────────────────
# Keep these MUTUALLY EXCLUSIVE where possible, so a rotation holds out one
# clinical concept at a time.

REGIONS = {
    "shock_corner": Region(
        name="shock_corner",
        description="SBP <= 100 AND HR >= 110",
        clinical_meaning=(
            "Low blood pressure together with a fast heart rate -- the classic "
            "compensated-shock picture. The heart beats faster to defend a "
            "falling blood pressure. Clinically the most important corner to "
            "get right, so it is the default held-out region."
        ),
        predicate=lambda sbp, dbp, hr, spo2: sbp <= 100 and hr >= 110,
    ),
    "hypertensive_brady": Region(
        name="hypertensive_brady",
        description="SBP >= 180 AND HR <= 55",
        clinical_meaning=(
            "High blood pressure with a slow heart rate. Can indicate raised "
            "intracranial pressure (part of Cushing's reflex)."
        ),
        predicate=lambda sbp, dbp, hr, spo2: sbp >= 180 and hr <= 55,
    ),
    "silent_hypoxia": Region(
        name="silent_hypoxia",
        description="SpO2 <= 92 AND HR <= 100 AND SBP >= 110",
        clinical_meaning=(
            "Low oxygen while the other vitals still look reassuring. Easy for "
            "a template-driven model to miss, because nothing else looks wrong."
        ),
        predicate=lambda sbp, dbp, hr, spo2: spo2 <= 92 and hr <= 100 and sbp >= 110,
    ),
    "wide_pulse_pressure": Region(
        name="wide_pulse_pressure",
        description="(SBP - DBP) >= 80",
        clinical_meaning=(
            "A very wide gap between systolic and diastolic pressure. Seen in "
            "aortic regurgitation, sepsis and thyrotoxicosis."
        ),
        predicate=lambda sbp, dbp, hr, spo2: (sbp - dbp) >= 80,
    ),
}

# shock_corner is deliberately NOT the default: measurement showed it is 100%
# CRITICAL by construction (SBP<=100 gives a qSOFA point, HR>=110 gives another,
# qSOFA 2 forces CRITICAL). Holding it out would test a single tier only.
# wide_pulse_pressure spans all four tiers with good volume, so it is a real test.
DEFAULT_HELD_OUT = "wide_pulse_pressure"


def get(name: str) -> Region:
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; available: {sorted(REGIONS)}")
    return REGIONS[name]
