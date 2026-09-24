"""
Consistency gate for teacher outputs.

Principle carried over from the existing pipeline: THE RULES DECIDE, THE MODEL
ONLY EXPLAINS. A teacher answer is rejected if it contradicts the deterministic
NEWS2/qSOFA verdict, invents numbers, or breaks the agent's output contract.

Rejected rows are kept in rejects.jsonl with a reason, so the acceptance rate is
auditable rather than silently absorbed.
"""
from __future__ import annotations
import re
import yaml
from pathlib import Path

from agent.nodes.llm_analyzer import OUTPUT_SCHEMA
from guardrails.clinical_rules import determine_recommended_action
from vitals.schemas import AlertLevel

_TH = yaml.safe_load(open(Path("guardrails/thresholds.yaml")))

# Numbers that may legitimately appear in reasoning even though they are not the
# patient's own values: the published thresholds the scores are defined against.
# NEWS2/qSOFA thresholds plus the textbook reference points a clinician would
# normally quote ("normal diastolic is under 80", "hypertension above 140").
# Quoting a published reference value is NOT fabrication. Anything the model was
# actually given in the prompt is allowed too -- see _prompt_numbers().
_ALLOWED_CONSTANTS = {
    0,1,2,3,4,5,6,7,8,9,10,12,15,18,20,22,
    40,50,51,55,60,65,70,75,80,85,89,90,91,92,93,94,95,96,97,98,99,100,
    105,109,110,111,120,125,130,131,140,150,160,180,200,219,220,
}

_DRUG_HINT = re.compile(
    r"\b(mg|mcg|ml|dose|dosage|administer|prescrib|infusion|bolus|"
    r"noradrenaline|norepinephrine|adrenaline|dopamine|dobutamine|antibiotic|"
    r"vancomycin|ceftriaxone|piperacillin|morphine|fentanyl|insulin)\b", re.I)


def _prompt_numbers(user_prompt: str) -> set[int]:
    """Every number the model was shown. Repeating these can never be fabrication."""
    return {int(n) for n in re.findall(r"\b\d{1,3}\b", user_prompt or "")}


def check(case, obj: dict | None, raw: str, user_prompt: str = "") -> tuple[bool, list[str], list[str]]:
    """Returns (accepted, reasons_for_rejection, notes).

    `notes` records observations that are NOT grounds for rejection -- kept so
    they can be measured later instead of silently discarding teacher output.
    """
    bad: list[str] = []
    notes: list[str] = []

    if obj is None:
        return False, ["not_json"], notes

    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    if missing:
        bad.append(f"missing_fields:{','.join(missing)}")

    # 1. must not contradict the deterministic verdict
    if obj.get("risk_level") != case.tier:
        bad.append(f"risk_level_contradiction:{obj.get('risk_level')}!={case.tier}")

    # 2. recommended action must match the rule mapping
    try:
        expected = determine_recommended_action(AlertLevel(case.tier))
        expected = expected.value if hasattr(expected, "value") else str(expected)
        if obj.get("recommended_action") != expected:
            bad.append(f"action_mismatch:{obj.get('recommended_action')}!={expected}")
    except Exception:
        bad.append("action_uncheckable")

    # 3. sepsis flag. agent/nodes/output_validator.py accepts the MODEL's value
    #    (qSOFA high_risk is only the default when the model omits it), so a
    #    disagreement is not a contract violation. Record it, do not reject.
    if isinstance(obj.get("sepsis_risk_flag"), bool):
        if obj["sepsis_risk_flag"] != bool(case.qsofa.get("high_risk")):
            notes.append("sepsis_flag_differs_from_qsofa")
    else:
        bad.append("sepsis_flag_not_bool")

    # 4. confidence sane
    c = obj.get("confidence")
    if not isinstance(c, (int, float)) or not (0.0 <= float(c) <= 1.0):
        bad.append("confidence_out_of_range")

    # 5. no fabricated numbers in the narrative
    allowed = set(_ALLOWED_CONSTANTS) | _prompt_numbers(user_prompt) | {
        int(case.systolic_bp), int(case.diastolic_bp),
        int(case.heart_rate), int(case.spo2),
        int(case.news2.get("total_score", -1)), int(case.qsofa.get("score", -1)),
    }
    text = " ".join(str(obj.get(k, "")) for k in ("reasoning", "limitations"))
    nums = {int(n) for n in re.findall(r"\b\d{1,3}\b", text)}
    fabricated = sorted(nums - allowed)
    if fabricated:
        bad.append(f"fabricated_numbers:{fabricated[:5]}")

    # 6. prohibited content -- the agent forbids drugs/doses/treatments
    whole = " ".join(str(v) for v in obj.values())
    if _DRUG_HINT.search(whole):
        bad.append("prohibited_treatment_content")

    # 7. disclaimer must be present and non-trivial
    if len(str(obj.get("disclaimer", ""))) < 20:
        bad.append("disclaimer_missing")

    return (not bad), bad, notes
