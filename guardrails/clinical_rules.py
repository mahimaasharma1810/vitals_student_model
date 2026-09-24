"""
Clinical scoring engine — NEWS2 + qSOFA.

Rule-based, fully deterministic. Runs before the LLM and independently
of it. If the LLM is unavailable, these scores alone drive the alert level.

References:
- NEWS2: Royal College of Physicians, 2017
- qSOFA / Sepsis-3: Singer et al., JAMA 2016;315(8):801-810
"""

import yaml
from pathlib import Path
from functools import lru_cache

from vitals.schemas import NEWS2Breakdown, QSOFABreakdown, AlertLevel, RecommendedAction


@lru_cache
def _load_thresholds() -> dict:
    path = Path(__file__).parent / "thresholds.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ── NEWS2 ─────────────────────────────────────────────────────────────────────

def _score_sbp(sbp: float, t: dict) -> int:
    if sbp <= t["score_3_low"]:
        return 3
    if sbp <= t["score_2_low"]:
        return 2
    if sbp <= t["score_1_low"]:
        return 1
    if sbp <= t["normal_high"]:
        return 0
    return 3  # ≥ score_3_high


def _score_dbp(dbp: float, t: dict) -> int:
    if dbp <= t["score_3_low"]:
        return 3
    if dbp <= t["score_2_low"]:
        return 2
    if dbp <= t["normal_high"]:
        return 0
    if dbp <= t["score_2_high"]:
        return 2
    return 3  # ≥ score_3_high


def _score_hr(hr: float, t: dict) -> int:
    if hr <= t["score_3_low"]:
        return 3
    if hr <= t["score_1_low"]:
        return 1
    if hr <= t["normal_high"]:
        return 0
    if hr <= t["score_1_high"]:
        return 1
    if hr <= t["score_2_high"]:
        return 2
    return 3  # ≥ score_3_high


def _score_spo2(spo2: float, t: dict) -> int:
    if spo2 <= t["score_3"]:
        return 3
    if spo2 <= t["score_2_high"]:
        return 2
    if spo2 <= t["score_1_high"]:
        return 1
    return 0  # ≥ normal


def calculate_news2(
    systolic_bp: float,
    diastolic_bp: float,
    heart_rate: float,
    spo2: float,
) -> NEWS2Breakdown:
    t = _load_thresholds()["news2"]

    sbp_score = _score_sbp(systolic_bp, t["systolic_bp"])
    dbp_score = _score_dbp(diastolic_bp, t["diastolic_bp"])
    hr_score = _score_hr(heart_rate, t["heart_rate"])
    spo2_score = _score_spo2(spo2, t["spo2"])

    total = sbp_score + dbp_score + hr_score + spo2_score
    any_critical = any(s == 3 for s in [sbp_score, dbp_score, hr_score, spo2_score])

    return NEWS2Breakdown(
        sbp_score=sbp_score,
        dbp_score=dbp_score,
        heart_rate_score=hr_score,
        spo2_score=spo2_score,
        total_score=total,
        any_single_critical=any_critical,
    )


# ── qSOFA ─────────────────────────────────────────────────────────────────────

def calculate_qsofa(
    systolic_bp: float,
    heart_rate: float,
    spo2: float,
) -> QSOFABreakdown:
    t = _load_thresholds()["qsofa"]

    sbp_flag = systolic_bp <= t["sbp_threshold"]
    hr_flag = heart_rate > t["hr_threshold"]
    spo2_flag = spo2 < t["spo2_threshold"]

    score = sum([sbp_flag, hr_flag, spo2_flag])
    high_risk = score >= t["high_risk_score"]

    return QSOFABreakdown(
        sbp_flag=sbp_flag,
        hr_flag=hr_flag,
        spo2_flag=spo2_flag,
        score=score,
        high_risk=high_risk,
    )


# ── Alert level from rule scores ──────────────────────────────────────────────

def determine_alert_level(
    news2: NEWS2Breakdown,
    qsofa: QSOFABreakdown,
) -> AlertLevel:
    """
    Determines alert level from NEWS2 and qSOFA scores.
    qSOFA high risk overrides to CRITICAL regardless of NEWS2 total.
    Any single NEWS2 score of 3 also triggers CRITICAL.
    """
    if news2.any_single_critical or news2.total_score >= 7 or qsofa.high_risk:
        return AlertLevel.CRITICAL
    if news2.total_score >= 5:
        return AlertLevel.HIGH
    if news2.total_score >= 3:
        return AlertLevel.MEDIUM
    return AlertLevel.NORMAL


def determine_recommended_action(alert_level: AlertLevel) -> RecommendedAction:
    return {
        AlertLevel.CRITICAL: RecommendedAction.IMMEDIATE_RESPONSE,
        AlertLevel.HIGH: RecommendedAction.URGENT_REVIEW,
        AlertLevel.MEDIUM: RecommendedAction.CLINICIAN_REVIEW,
        AlertLevel.NORMAL: RecommendedAction.OBSERVATION,
    }[alert_level]
