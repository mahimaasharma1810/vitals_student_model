"""
Evaluation sets, built deterministically.

Four sets, each with a DIFFERENT purpose and its OWN pass threshold. They are
never pooled into one score -- pooling is exactly how the previous pipeline
turned an 8.0% failure rate on hard cases into a 3.6% "pass" by adding 200 easy
rows to the denominator.

  held_out     the hidden region, never trained on   -> did it generalise?
  boundary     values sitting exactly on a scoring threshold -> off-by-one errors
  robustness   physiologically impossible values     -> does it stay sane?
  indist       same distribution as training         -> reference only, NOT quality

`gold` (clinician-written cases) is deliberately absent: no clinician is
available yet. When one is, add it here as a fifth set with its own threshold.
"""
from __future__ import annotations
import yaml
from pathlib import Path

from distill.sampler import make_case, physically_valid

_TH = yaml.safe_load(open(Path("guardrails/thresholds.yaml")))

# A neutral, unambiguously normal patient. When probing one vital we hold the
# others here so that the case tests ONE boundary at a time.
NEUTRAL = dict(systolic_bp=120, diastolic_bp=70, heart_rate=75, spo2=98)


def boundary_cases() -> list:
    """Every NEWS2/qSOFA threshold, probed at value-1, value, value+1."""
    n = _TH["news2"]; q = _TH["qsofa"]
    probes: list[tuple[str, int]] = []
    for vital, keys in (
        ("systolic_bp",  n["systolic_bp"]),
        ("diastolic_bp", n["diastolic_bp"]),
        ("heart_rate",   n["heart_rate"]),
        ("spo2",         n["spo2"]),
    ):
        for k, v in keys.items():
            if isinstance(v, (int, float)):
                probes.append((vital, int(v)))
    # qSOFA thresholds too -- they can flip the tier on their own
    probes += [("systolic_bp", int(q["sbp_threshold"])),
               ("heart_rate",  int(q["hr_threshold"])),
               ("spo2",        int(q["spo2_threshold"]))]

    seen, out = set(), []
    for vital, v in probes:
        for delta in (-1, 0, 1):
            vals = dict(NEUTRAL); vals[vital] = v + delta
            key = tuple(vals[k] for k in ("systolic_bp", "diastolic_bp", "heart_rate", "spo2"))
            if key in seen: continue
            if not physically_valid(*key): continue
            seen.add(key)
            out.append(make_case(*key, None))
    return out


def robustness_cases() -> list:
    """
    Values that cannot occur in a living patient, or that sit outside the
    plausibility bounds. Reported SEPARATELY and never mixed into a clinical
    score: failing to narrate SpO2 110 sensibly is not a clinical failure.

    In the live agent these are rejected by the input validator before the model
    is reached, so this set measures defence in depth, not routine behaviour.
    """
    raw = [
        (120,  80,  75, 110),   # impossible SpO2
        (120,  80,  75, 105),   # impossible SpO2
        (45,   95,  15,  98),   # HR 15, diastolic above systolic
        (300, 100,  75,  98),   # extreme systolic
        (40,   20, 300,  50),   # every value at a plausibility bound
        (120,  80, 300,  98),   # impossible HR
        (90,  110,  75,  98),   # diastolic above systolic
        (250,  40,  45,  99),   # extreme pulse pressure
    ]
    return [make_case(*v, None) for v in raw]


SETS = {
    "boundary":   boundary_cases,
    "robustness": robustness_cases,
}
