"""
Input guardrails — first gate in the agent pipeline.

Responsibilities:
1. Schema validation (Pydantic handles this upstream)
2. Physiological plausibility check (values that are schema-valid but impossible)
3. SBP > DBP cross-field check
4. Immediate CRITICAL bypass — if any vital is in critical zone, flag for
   LLM bypass so patient safety is not gated on LLM availability
5. Patient ID presence check (non-empty string)

All rejections are logged to the audit trail by the caller.
"""

from dataclasses import dataclass
from vitals.schemas import VitalsSnapshotInput, AlertLevel
from guardrails.clinical_rules import calculate_news2, calculate_qsofa, determine_alert_level


@dataclass
class InputValidationResult:
    valid: bool
    rejection_reason: str | None  # set when valid=False
    immediate_critical: bool       # True → bypass LLM, go straight to CRITICAL alert
    news2_preview: object | None   # pre-computed for efficiency (reused downstream)
    qsofa_preview: object | None


def validate_input(payload: VitalsSnapshotInput) -> InputValidationResult:
    """
    Runs all input guardrail checks on an incoming vitals snapshot.
    Pydantic field validators on VitalsSnapshotInput handle range and
    cross-field checks — this layer handles logic that goes beyond schema.
    """

    # Patient ID must not be blank
    if not payload.patient_id or not payload.patient_id.strip():
        return InputValidationResult(
            valid=False,
            rejection_reason="patient_id must not be empty",
            immediate_critical=False,
            news2_preview=None,
            qsofa_preview=None,
        )

    # Pre-compute clinical scores
    news2 = calculate_news2(
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        heart_rate=payload.heart_rate,
        spo2=payload.spo2,
    )
    qsofa = calculate_qsofa(
        systolic_bp=payload.systolic_bp,
        heart_rate=payload.heart_rate,
        spo2=payload.spo2,
    )

    # Determine if we should bypass the LLM for immediate patient safety
    alert_level = determine_alert_level(news2, qsofa)
    immediate_critical = alert_level == AlertLevel.CRITICAL

    return InputValidationResult(
        valid=True,
        rejection_reason=None,
        immediate_critical=immediate_critical,
        news2_preview=news2,
        qsofa_preview=qsofa,
    )
