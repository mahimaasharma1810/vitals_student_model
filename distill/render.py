"""
Render a sampled Case into exactly the prompt the AGENT sends at runtime.

Why this file matters
---------------------
The previous distillation trained the student on a DIFFERENT format from the one
`agent/` actually uses:

    old training input : "Systolic BP: 116 mmHg (NORMAL - within reference ...)"
    agent runtime input: "PATIENT ID: ...\\nCURRENT VITAL SIGNS (snapshot ...)"

    old training output: prose, "Clinical Assessment: ... Primary Concern: ..."
    agent runtime needs: JSON with 8 required fields

A student trained on the old format cannot be dropped into the agent at all.
So this module imports the agent's OWN prompt builder and schema. If the agent's
prompt ever changes, the training data changes with it, and the two can never
drift apart silently.

EHR policy (decided 2026-09-22): option (a) -- always "No EHR data available."
The live EHR store holds only 8 documents across 5 patients, which is far too
little to distil retrieval behaviour honestly. `contributing_factors` is
therefore derived from the vitals themselves. This is a documented limitation,
not an oversight.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.nodes.llm_analyzer import OUTPUT_SCHEMA
from vitals.schemas import NEWS2Breakdown, QSOFABreakdown

NO_EHR = "No EHR data available."

# Fixed values so that prompts are reproducible. The agent varies these at
# runtime; the student must not key its answer on them, so they are held
# constant during training rather than randomised into spurious signal.
TRAIN_PATIENT_ID = "TRAIN-0000"
TRAIN_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

REQUIRED_FIELDS = tuple(OUTPUT_SCHEMA["required"])


def render_user(case, patient_id: str = TRAIN_PATIENT_ID,
                timestamp: datetime = TRAIN_TIMESTAMP,
                ehr_context: str = NO_EHR) -> str:
    """Exactly what agent/nodes/llm_analyzer.py will send at inference time."""
    snapshot = SimpleNamespace(
        patient_id=patient_id,
        snapshot_timestamp=timestamp,
        systolic_bp=case.systolic_bp,
        diastolic_bp=case.diastolic_bp,
        heart_rate=case.heart_rate,
        spo2=case.spo2,
    )
    return build_user_prompt(
        snapshot=snapshot,
        news2=NEWS2Breakdown(**case.news2),
        qsofa=QSOFABreakdown(**case.qsofa),
        rule_alert_level=case.tier,
        ehr_context=ehr_context,
    )


def render_system() -> str:
    return SYSTEM_PROMPT
