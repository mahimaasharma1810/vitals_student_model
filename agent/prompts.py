"""
MedGemma prompt templates.

Design principles:
- Structured output only (JSON schema enforced)
- No drug names, doses, or treatment orders in output
- Explicit limitations documented for unavailable signals
- Confidence must reflect uncertainty, not be inflated
"""

SYSTEM_PROMPT = """\
You are MedGemma, a clinical decision support AI for post-operative patient monitoring.
Your role is to assess the risk of sepsis and cardiovascular deterioration based on
vital signs and patient history.

STRICT RULES:
1. You MUST respond ONLY with a valid JSON object matching the schema below.
2. You MUST NOT recommend specific drugs, doses, or treatments.
3. You MUST reference the actual vital sign values in your reasoning.
4. Your confidence score must honestly reflect uncertainty.
5. You are DECISION SUPPORT ONLY — never a replacement for clinical judgment.

OUTPUT SCHEMA (respond with ONLY this JSON, no other text):
{
  "risk_level": "CRITICAL | HIGH | MEDIUM | NORMAL",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<plain English explanation referencing actual vital values>",
  "sepsis_risk_flag": <true | false>,
  "contributing_factors": ["<factor from patient history>", ...],
  "recommended_action": "immediate_response | urgent_review | clinician_review | observation",
  "limitations": "<note any missing signals e.g. temperature, respiratory rate, mental status>",
  "disclaimer": "CLINICAL DECISION SUPPORT ONLY. This output must not replace clinical judgment."
}
"""

USER_PROMPT_TEMPLATE = """\
PATIENT ID: {patient_id}

CURRENT VITAL SIGNS (snapshot timestamp: {timestamp}):
- Systolic BP:    {systolic_bp} mmHg
- Diastolic BP:   {diastolic_bp} mmHg
- Heart Rate:     {heart_rate} bpm
- SpO2:           {spo2}%

RULE-BASED CLINICAL SCORES:
- NEWS2 Score:    {news2_total} (SBP:{sbp_score} DBP:{dbp_score} HR:{hr_score} SpO2:{spo2_score})
- qSOFA Score:    {qsofa_score}/3 (SBP≤100:{sbp_flag} HR>90:{hr_flag} SpO2<94%:{spo2_flag})
- Rule Alert:     {rule_alert_level}

PATIENT EHR CONTEXT:
{ehr_context}

Based on the vital signs, clinical scores, and patient history above, provide your
structured risk assessment as JSON.
"""


def build_user_prompt(
    snapshot,
    news2,
    qsofa,
    rule_alert_level: str,
    ehr_context: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        patient_id=snapshot.patient_id,
        timestamp=snapshot.snapshot_timestamp.isoformat(),
        systolic_bp=snapshot.systolic_bp,
        diastolic_bp=snapshot.diastolic_bp,
        heart_rate=snapshot.heart_rate,
        spo2=snapshot.spo2,
        news2_total=news2.total_score,
        sbp_score=news2.sbp_score,
        dbp_score=news2.dbp_score,
        hr_score=news2.heart_rate_score,
        spo2_score=news2.spo2_score,
        qsofa_score=qsofa.score,
        sbp_flag=qsofa.sbp_flag,
        hr_flag=qsofa.hr_flag,
        spo2_flag=qsofa.spo2_flag,
        rule_alert_level=rule_alert_level,
        ehr_context=ehr_context,
    )
