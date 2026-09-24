"""
Tests for guardrails and clinical scoring engine.
Run with: pytest tests/test_guardrails.py -v
"""

import pytest
from datetime import datetime, timezone
from vitals.schemas import VitalsSnapshotInput, AlertLevel
from guardrails.clinical_rules import (
    calculate_news2, calculate_qsofa, determine_alert_level,
)
from guardrails.input_guardrails import validate_input
from guardrails.output_guardrails import validate_llm_output


def make_snapshot(**kwargs):
    defaults = dict(
        patient_id="test-patient",
        systolic_bp=120, diastolic_bp=80,
        heart_rate=72, spo2=98,
        snapshot_timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return VitalsSnapshotInput(**defaults)


# ── NEWS2 scoring ─────────────────────────────────────────────────────────────

def test_news2_all_normal():
    s = calculate_news2(120, 80, 72, 98)
    assert s.total_score == 0
    assert not s.any_single_critical

def test_news2_critical_sbp():
    s = calculate_news2(88, 60, 72, 98)
    assert s.sbp_score == 3
    assert s.any_single_critical

def test_news2_critical_hr():
    s = calculate_news2(120, 80, 140, 98)
    assert s.heart_rate_score == 3
    assert s.any_single_critical

def test_news2_critical_spo2():
    s = calculate_news2(120, 80, 72, 90)
    assert s.spo2_score == 3
    assert s.any_single_critical

def test_news2_high_total():
    # TC-05: multi-vital deterioration
    s = calculate_news2(85, 52, 136, 90)
    assert s.total_score >= 7
    assert s.any_single_critical


# ── qSOFA ─────────────────────────────────────────────────────────────────────

def test_qsofa_normal():
    q = calculate_qsofa(120, 72, 98)
    assert q.score == 0
    assert not q.high_risk

def test_qsofa_high_risk():
    # TC-04: SBP=88, HR=96, SpO2=91%
    q = calculate_qsofa(88, 96, 91)
    assert q.sbp_flag
    assert q.hr_flag
    assert q.spo2_flag
    assert q.score == 3
    assert q.high_risk


# ── Alert level determination ─────────────────────────────────────────────────

def test_alert_level_normal():
    news2 = calculate_news2(120, 80, 72, 98)
    qsofa = calculate_qsofa(120, 72, 98)
    assert determine_alert_level(news2, qsofa) == AlertLevel.NORMAL

def test_alert_level_critical_from_single_vital():
    news2 = calculate_news2(88, 60, 72, 98)  # SBP score 3
    qsofa = calculate_qsofa(88, 72, 98)
    assert determine_alert_level(news2, qsofa) == AlertLevel.CRITICAL

def test_alert_level_critical_from_qsofa():
    news2 = calculate_news2(102, 65, 95, 93)  # total ~4 = HIGH, but qSOFA≥2
    qsofa = calculate_qsofa(102, 95, 93)
    assert qsofa.high_risk
    assert determine_alert_level(news2, qsofa) == AlertLevel.CRITICAL


# ── Input guardrails ──────────────────────────────────────────────────────────

def test_input_valid_normal():
    result = validate_input(make_snapshot())
    assert result.valid
    assert not result.immediate_critical

def test_input_rejects_impossible_hr():
    with pytest.raises(Exception):
        make_snapshot(heart_rate=500)

def test_input_rejects_sbp_less_than_dbp():
    with pytest.raises(Exception):
        make_snapshot(systolic_bp=70, diastolic_bp=80)

def test_input_immediate_critical_for_critical_vitals():
    result = validate_input(make_snapshot(systolic_bp=85, diastolic_bp=55, heart_rate=140, spo2=89))
    assert result.valid
    assert result.immediate_critical


# ── Output guardrails ─────────────────────────────────────────────────────────

def test_output_valid():
    snapshot = make_snapshot(systolic_bp=88, heart_rate=95, spo2=91)
    llm_out = {
        "risk_level": "CRITICAL",
        "confidence": 0.85,
        "reasoning": "Patient has systolic BP of 88 mmHg which is critically low, heart rate of 95 bpm, and SpO2 of 91%.",
        "sepsis_risk_flag": True,
        "contributing_factors": ["hypotension", "tachycardia"],
        "recommended_action": "immediate_response",
        "limitations": "Temperature and respiratory rate not available.",
        "disclaimer": "CLINICAL DECISION SUPPORT ONLY.",
    }
    result = validate_llm_output(llm_out, AlertLevel.CRITICAL, snapshot)
    assert result.valid

def test_output_rejects_prohibited_content():
    snapshot = make_snapshot()
    llm_out = {
        "risk_level": "CRITICAL",
        "confidence": 0.9,
        "reasoning": "SpO2 98, administer norepinephrine 0.1 mcg/kg/min",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.NORMAL, snapshot)
    assert not result.valid

def test_output_rejects_ungrounded_reasoning():
    snapshot = make_snapshot(systolic_bp=88)
    llm_out = {
        "risk_level": "HIGH",
        "confidence": 0.7,
        "reasoning": "The patient appears to be at risk based on general clinical assessment.",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.HIGH, snapshot)
    assert not result.valid

def test_output_rejects_large_contradiction():
    snapshot = make_snapshot(systolic_bp=88, heart_rate=140, spo2=88)
    llm_out = {
        "risk_level": "NORMAL",
        "confidence": 0.8,
        "reasoning": "Systolic BP 88, heart rate 140, SpO2 88 — patient seems fine.",
        "sepsis_risk_flag": False,
    }
    result = validate_llm_output(llm_out, AlertLevel.CRITICAL, snapshot)
    assert not result.valid  # NORMAL vs CRITICAL = 3-tier contradiction


# ══════════════════════════════════════════════════════════════════════════════
# Regression tests added 2026-09-20 for three guardrail defects.
# These are written to FAIL against the pre-fix code and pass after.
# ══════════════════════════════════════════════════════════════════════════════

def _out(reasoning, risk_level="MEDIUM", confidence=0.8):
    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "reasoning": reasoning,
        "sepsis_risk_flag": False,
        "contributing_factors": [],
        "recommended_action": "clinician_review",
        "limitations": "none",
        "disclaimer": "CLINICAL DECISION SUPPORT ONLY.",
    }


# ── Bug 1 + 2: prohibited-content patterns missed inflected forms ─────────────

PROHIBITED_PHRASES = [
    "Consider intubation if this worsens.",
    "The patient may need to be intubated.",
    "Prepare to intubate the patient.",
    "Prepare for defibrillation.",
    "The patient was defibrillated.",
    "Consider prescribing antibiotics.",
    "Antibiotics were prescribed yesterday.",
    "Review the prescription.",
    "Fluids are being administered.",
    "Fluids were administered.",
    "Recommend administration of fluids.",
    "An infusion should be started.",
    "Fluids were infused overnight.",
    "Repeated boluses were given.",
    "Give 500 mcg of the agent.",
    "Give 2 g of the agent.",
    "Give 10 units now.",
]


@pytest.mark.parametrize("phrase", PROHIBITED_PHRASES)
def test_prohibited_content_catches_inflected_forms(phrase):
    """Word-stem forms must be blocked, not just the exact dictionary form."""
    snapshot = make_snapshot()
    # "72" is an exact-integer vital match, so it is grounded under BOTH the old
    # and new grounding rules. That isolates this test to the prohibited check:
    # without it the test would pass vacuously on a grounding failure.
    result = validate_llm_output(_out(f"Heart rate is 72 bpm. {phrase}"),
                                 AlertLevel.MEDIUM, snapshot)
    assert not result.valid, f"prohibited phrase was NOT blocked: {phrase!r}"
    assert not result.checks["prohibited"].passed, (
        f"blocked, but not by the prohibited check: {result.rejection_reason}")


def test_prohibited_content_still_allows_safe_reasoning():
    """The stem widening must not start blocking ordinary clinical language."""
    snapshot = make_snapshot()
    safe = ("SpO2 is 98% and heart rate is 72 bpm. Findings are stable and "
            "continued observation is appropriate.")
    result = validate_llm_output(_out(safe), AlertLevel.MEDIUM, snapshot)
    assert result.valid, f"safe reasoning was wrongly blocked: {result.rejection_reason}"


# ── Bug 3: grounding rejected naturally-written vitals ────────────────────────

@pytest.mark.parametrize("reasoning", [
    "SpO2 of 98% is within normal limits.",
    "SpO2 of 98 % is within normal limits.",
    "SpO2 of 98 is within normal limits.",
    "Oxygen saturation 98.0% recorded.",
    "Heart rate of 72 bpm is normal.",
    "Systolic BP of 120 mmHg is normal.",
])
def test_grounding_accepts_natural_vital_formats(reasoning):
    """A real vital written the way a clinician writes it must count as grounded."""
    snapshot = make_snapshot(systolic_bp=120, diastolic_bp=80, heart_rate=72, spo2=98)
    result = validate_llm_output(_out(reasoning), AlertLevel.MEDIUM, snapshot)
    assert result.valid, f"wrongly rejected as ungrounded: {result.rejection_reason}"


@pytest.mark.parametrize("reasoning", [
    "SpO2 of 88% indicates hypoxia.",            # wrong value (actual 98)
    "Heart rate of 140 bpm is concerning.",      # wrong value (actual 72)
    "Patient appears generally unwell.",         # no number at all
    "Saturation is 1980 on the monitor.",        # 98 embedded in a larger number
    "SpO2 of 98.5% recorded.",                   # near-miss, not the real value
])
def test_grounding_still_rejects_numbers_that_are_not_the_real_vitals(reasoning):
    """SAFETY: loosening the format must not loosen which VALUES are accepted."""
    snapshot = make_snapshot(systolic_bp=120, diastolic_bp=80, heart_rate=72, spo2=98)
    result = validate_llm_output(_out(reasoning), AlertLevel.MEDIUM, snapshot)
    assert not result.valid, f"ungrounded reasoning wrongly accepted: {reasoning!r}"


# ── Per-check breakdown ───────────────────────────────────────────────────────

def test_breakdown_present_and_complete_on_success():
    snapshot = make_snapshot()
    result = validate_llm_output(_out("SpO2 is 98% and stable."), AlertLevel.MEDIUM, snapshot)
    assert result.valid
    assert set(result.checks) == {"schema", "grounding", "prohibited", "contradiction"}
    assert all(c.passed for c in result.checks.values())


def test_breakdown_evaluates_every_check_even_when_one_fails():
    """No early return: a schema failure must not hide the other three results."""
    snapshot = make_snapshot()
    broken = {"risk_level": "MEDIUM", "confidence": 0.8}   # missing reasoning + flag
    result = validate_llm_output(broken, AlertLevel.MEDIUM, snapshot)
    assert not result.valid
    assert set(result.checks) == {"schema", "grounding", "prohibited", "contradiction"}
    assert not result.checks["schema"].passed
    assert result.checks["schema"].reason


def test_breakdown_isolates_which_check_failed():
    snapshot = make_snapshot()
    r = validate_llm_output(_out("SpO2 is 98%. Consider intubation."), AlertLevel.MEDIUM, snapshot)
    assert not r.valid
    assert not r.checks["prohibited"].passed
    assert r.checks["schema"].passed
    assert r.checks["grounding"].passed


def test_backward_compatible_fields_preserved():
    """Existing callers (output_validator node, older tests) must keep working."""
    snapshot = make_snapshot(systolic_bp=88, heart_rate=95, spo2=91)
    r = validate_llm_output(_out("Systolic BP 88 mmHg, SpO2 91%.", "HIGH", 0.4),
                            AlertLevel.HIGH, snapshot)
    assert r.valid is True
    assert r.confidence == 0.4
    assert r.flag_for_human_review is True          # below 0.6 threshold
    assert r.rejection_reason is None
    assert isinstance(r.sanitized_reasoning, str)
    assert r.disclaimer
