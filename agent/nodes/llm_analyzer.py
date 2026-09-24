"""
LLM Analyzer node — calls MedGemma via Ollama REST API.

Falls back gracefully if Ollama is unavailable. The system continues
operating on rule-based scores alone in fallback mode.
"""

from __future__ import annotations
import json
import logging
import re
import httpx

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _strip_thinking_tokens(text: str) -> str:
    text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    text = re.sub(r"<unused\d+>", "", text)
    return text.strip()


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from the response text."""
    text = _strip_thinking_tokens(text)
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "NORMAL"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "sepsis_risk_flag": {"type": "boolean"},
        "contributing_factors": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {
            "type": "string",
            "enum": ["immediate_response", "urgent_review", "clinician_review", "observation"],
        },
        "limitations": {"type": "string"},
        "disclaimer": {"type": "string"},
    },
    "required": ["risk_level", "confidence", "reasoning", "sepsis_risk_flag",
                 "contributing_factors", "recommended_action", "limitations", "disclaimer"],
}


def llm_analyzer_node(state: AgentState) -> AgentState:
    snapshot = state["snapshot"]
    news2 = state["news2"]
    qsofa = state["qsofa"]
    rule_level = state["rule_alert_level"]

    user_prompt = build_user_prompt(
        snapshot=snapshot,
        news2=news2,
        qsofa=qsofa,
        rule_alert_level=rule_level.value,
        ehr_context=state.get("ehr_context", "No EHR data available."),
    )

    try:
        with httpx.Client(timeout=settings.OLLAMA_TIMEOUT) as client:
            response = client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": OUTPUT_SCHEMA,
                },
            )
            response.raise_for_status()
            raw_content = response.json()["message"]["content"]
            parsed = _extract_json(raw_content)

            if parsed is None:
                logger.error(
                    "LLM DEGRADED — MedGemma replied but the response could not be "
                    "parsed as JSON. Falling back to RULE-BASED ONLY for patient %s. "
                    "Raw response (first 300 chars): %r",
                    snapshot.patient_id,
                    raw_content[:300],
                )
                return {
                    **state,
                    "llm_raw_output": None,
                    "llm_available": True,
                    "llm_error": "Could not parse JSON from LLM response.",
                }

            return {
                **state,
                "llm_raw_output": parsed,
                "llm_available": True,
                "llm_error": None,
            }

    except Exception as e:
        logger.error(
            "LLM UNAVAILABLE — MedGemma call failed (%s at %s, model=%s). "
            "Falling back to RULE-BASED ONLY for patient %s. "
            "Rule-based scoring still protects CRITICAL cases, but no contextual "
            "EHR reasoning is being applied to this snapshot.",
            type(e).__name__,
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_MODEL,
            snapshot.patient_id,
            exc_info=True,
        )
        return {
            **state,
            "llm_raw_output": None,
            "llm_available": False,
            "llm_error": str(e)[:200],
        }
