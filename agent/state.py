"""
LangGraph agent state — shared across all nodes in the pipeline.
"""

from __future__ import annotations
from typing import Optional, Any
from typing_extensions import TypedDict

from vitals.schemas import (
    VitalsSnapshotInput,
    AlertLevel,
    RecommendedAction,
    NEWS2Breakdown,
    QSOFABreakdown,
)


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    snapshot: VitalsSnapshotInput
    actor_role: str                    # API caller role

    # ── Node 1: Input Validator ────────────────────────────────────────────────
    input_valid: bool
    rejection_reason: Optional[str]
    immediate_critical: bool           # bypass LLM if True

    # ── Node 2: EHR Retriever ──────────────────────────────────────────────────
    ehr_context: str                   # retrieved EHR text (de-identified)
    ehr_available: bool

    # ── Node 3: Clinical Scorer ────────────────────────────────────────────────
    news2: Optional[NEWS2Breakdown]
    qsofa: Optional[QSOFABreakdown]
    rule_alert_level: Optional[AlertLevel]
    rule_recommended_action: Optional[RecommendedAction]

    # ── Node 4: LLM Analyzer ──────────────────────────────────────────────────
    llm_raw_output: Optional[dict]     # raw JSON from MedGemma
    llm_available: bool
    llm_error: Optional[str]

    # ── Node 5: Output Validator ───────────────────────────────────────────────
    llm_output_valid: bool
    sanitized_reasoning: Optional[str]
    llm_confidence: Optional[float]
    flag_for_human_review: bool
    output_rejection_reason: Optional[str]
    contributing_factors: Optional[list[str]]
    llm_limitations: Optional[str]
    sepsis_risk_flag: bool

    # ── Node 6: Alert Router ───────────────────────────────────────────────────
    final_alert_level: Optional[AlertLevel]
    final_recommended_action: Optional[RecommendedAction]
    rule_based_only: bool              # True when LLM was unavailable or invalid

    # ── Node 7: Audit Logger ───────────────────────────────────────────────────
    alert_id: Optional[str]
    snapshot_id: Optional[str]
    audit_logged: bool
