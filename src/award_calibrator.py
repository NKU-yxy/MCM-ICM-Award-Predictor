"""Lightweight post-LLM award calibration for the public web reviewer.

The DeepSeek rubric remains the expensive paper reader. This module only uses
the returned score plus already extracted PDF evidence to produce a stricter,
stable O/F/M/H/S distribution without loading the heavier local ML model.
"""

from __future__ import annotations

import math
from typing import Any, Dict


AWARD_KEYS = ["O", "F", "M", "H", "S"]
CALIBRATION_VERSION = "calibrated_v1"


def calibrate_rubric_award(
    rubric: Dict[str, Any],
    *,
    metadata: Dict[str, Any],
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of ``rubric`` with calibrated award fields.

    The public UI expects the final fields to stay at
    ``award_prediction``/``probabilities``. Raw LLM values are preserved for
    audit under ``raw_award_prediction`` and ``raw_probabilities``.
    """

    calibrated = dict(rubric or {})
    raw_award = str(calibrated.get("award_prediction", "S/U") or "S/U")
    raw_probs = dict(calibrated.get("probabilities") or {})
    raw_score = _number(calibrated.get("score"), 0.0)

    calibrated["raw_award_prediction"] = raw_award
    calibrated["raw_probabilities"] = _normalize_percent_dict(raw_probs)
    calibrated["calibration_version"] = CALIBRATION_VERSION

    if calibrated.get("status") != "ok":
        probabilities = {"O": 0, "F": 0, "M": 0, "H": 0, "S": 100}
        calibrated.update(
            {
                "award_prediction": "S/U",
                "probabilities": probabilities,
                "calibrated_score": 0,
                "calibration_note": "AI rubric unavailable; assigned conservative S/U.",
            }
        )
        return calibrated

    details = calibrated.get("details") or {}
    evidence = _evidence_summary(metadata, structure, details)
    penalty, reasons = _evidence_penalty(evidence)
    calibrated_score = max(0.0, min(100.0, raw_score - penalty))
    probabilities = _calibrated_probabilities(calibrated_score, evidence)
    award = _award_from_probabilities(probabilities)

    calibrated.update(
        {
            "award_prediction": "S/U" if award == "S" else award,
            "probabilities": probabilities,
            "calibrated_score": int(round(calibrated_score)),
            "calibration_note": _calibration_note(penalty, reasons),
        }
    )
    return calibrated


def _evidence_summary(
    metadata: Dict[str, Any],
    structure: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    page_count = int(_number(metadata.get("page_count"), 0))
    image_count = int(_number(metadata.get("image_count"), 0))
    figure_captions = int(_number(structure.get("figure_caption_count"), 0))
    table_count = int(_number(structure.get("table_count"), 0))
    visual_total = max(image_count, figure_captions) + table_count

    return {
        "page_count": page_count,
        "visual_total": visual_total,
        "ref_count": int(_number(metadata.get("ref_count"), 0)),
        "formula_count": int(_number(structure.get("formula_count"), 0)),
        "completeness": float(_number(structure.get("structure_completeness"), 0.0)),
        "has_sensitivity": bool(structure.get("has_sensitivity_analysis")),
        "has_validation": bool(structure.get("has_model_validation")),
        "has_assumption": bool(structure.get("has_assumption_justification")),
        "has_model_comparison": bool(structure.get("has_model_comparison")),
        "has_error_analysis": bool(structure.get("has_error_analysis")),
        "content_score": _number(details.get("content_score"), 0.0),
        "conclusion_score": _number(details.get("conclusion_score"), 0.0),
        "visual_score": _number(details.get("visual_score"), 0.0),
    }


def _evidence_penalty(evidence: Dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []

    if evidence["page_count"] and evidence["page_count"] < 12:
        penalty += 6.0
        reasons.append("paper is very short")
    elif evidence["page_count"] and evidence["page_count"] < 18:
        penalty += 3.0
        reasons.append("page count is below typical finalist/O range")

    if evidence["visual_total"] < 5:
        penalty += 5.0
        reasons.append("visual/table evidence is sparse")
    elif evidence["visual_total"] < 10:
        penalty += 3.0
        reasons.append("visual/table evidence is limited")

    if evidence["completeness"] < 0.50:
        penalty += 5.0
        reasons.append("core paper structure is incomplete")
    elif evidence["completeness"] < 0.75:
        penalty += 2.5
        reasons.append("paper structure is not fully complete")

    if not evidence["has_sensitivity"]:
        penalty += 2.0
        reasons.append("missing sensitivity analysis")
    if not evidence["has_validation"]:
        penalty += 2.0
        reasons.append("missing model validation")
    if not (evidence["has_error_analysis"] or evidence["has_model_comparison"]):
        penalty += 1.5
        reasons.append("missing error analysis or model comparison")

    return min(penalty, 10.0), reasons


def _calibrated_probabilities(score: float, evidence: Dict[str, Any]) -> Dict[str, int]:
    centers = {"O": 96.0, "F": 90.0, "M": 82.0, "H": 69.0, "S": 50.0}
    scales = {"O": 3.0, "F": 4.0, "M": 5.5, "H": 7.0, "S": 11.0}
    weights = {
        award: math.exp(-0.5 * ((score - centers[award]) / scales[award]) ** 2)
        for award in AWARD_KEYS
    }

    probs = _normalize_float(weights)
    probs = _apply_score_caps(probs, score, evidence)
    return _round_percentages(probs)


def _apply_score_caps(
    probs: Dict[str, float],
    score: float,
    evidence: Dict[str, Any],
) -> Dict[str, float]:
    if score < 65:
        caps = {"O": 0.0, "F": 0.005, "M": 0.08}
    elif score < 75:
        caps = {"O": 0.002, "F": 0.02, "M": 0.20}
    elif score < 80:
        caps = {"O": 0.005, "F": 0.04, "M": 0.34}
    elif score < 88:
        caps = {"O": 0.02, "F": 0.10}
    elif score < 94:
        caps = {"O": 0.05}
    else:
        caps = {}

    if not _o_ready(evidence):
        caps["O"] = min(caps.get("O", 1.0), 0.08)
    if not _finalist_ready(evidence):
        caps["F"] = min(caps.get("F", 1.0), 0.16)

    adjusted = dict(probs)
    excess = 0.0
    for award, cap in caps.items():
        if adjusted.get(award, 0.0) > cap:
            excess += adjusted[award] - cap
            adjusted[award] = cap

    targets = ["M", "H", "S"]
    if score < 80:
        targets = ["H", "S"]
    target_total = sum(adjusted.get(key, 0.0) for key in targets)
    if excess > 0:
        if target_total <= 0:
            for key in targets:
                adjusted[key] = adjusted.get(key, 0.0) + excess / len(targets)
        else:
            for key in targets:
                adjusted[key] = adjusted.get(key, 0.0) + excess * adjusted.get(key, 0.0) / target_total
    return _normalize_float(adjusted)


def _o_ready(evidence: Dict[str, Any]) -> bool:
    return (
        evidence["page_count"] >= 20
        and evidence["visual_total"] >= 14
        and evidence["completeness"] >= 0.80
        and evidence["has_sensitivity"]
        and evidence["has_validation"]
        and evidence["content_score"] >= 26
        and evidence["conclusion_score"] >= 12
    )


def _finalist_ready(evidence: Dict[str, Any]) -> bool:
    return (
        evidence["page_count"] >= 18
        and evidence["visual_total"] >= 10
        and evidence["completeness"] >= 0.70
        and (evidence["has_sensitivity"] or evidence["has_validation"])
        and evidence["content_score"] >= 23
    )


def _calibration_note(penalty: float, reasons: list[str]) -> str:
    if penalty <= 0:
        return "Calibrated with stricter local award thresholds; no evidence penalty applied."
    reason_text = "; ".join(reasons[:3])
    return (
        f"Calibrated with stricter local award thresholds; "
        f"{penalty:.1f} point evidence penalty applied ({reason_text})."
    )


def _award_from_probabilities(probabilities: Dict[str, int]) -> str:
    return max(AWARD_KEYS, key=lambda key: probabilities.get(key, 0))


def _normalize_percent_dict(values: Dict[str, Any]) -> Dict[str, int]:
    raw = {key: max(_number(values.get("S/U" if key == "S" and "S/U" in values else key), 0.0), 0.0) for key in AWARD_KEYS}
    total = sum(raw.values())
    if total <= 0:
        return {"O": 0, "F": 0, "M": 0, "H": 0, "S": 100}
    if total <= 1.5:
        raw = {key: value * 100.0 for key, value in raw.items()}
    return _round_percentages(_normalize_float(raw))


def _normalize_float(values: Dict[str, float]) -> Dict[str, float]:
    clean = {key: max(float(values.get(key, 0.0)), 0.0) for key in AWARD_KEYS}
    total = sum(clean.values())
    if total <= 0:
        return {"O": 0.0, "F": 0.0, "M": 0.0, "H": 0.0, "S": 1.0}
    return {key: clean[key] / total for key in AWARD_KEYS}


def _round_percentages(probs: Dict[str, float]) -> Dict[str, int]:
    raw = {key: max(float(probs.get(key, 0.0)), 0.0) * 100.0 for key in AWARD_KEYS}
    rounded = {key: int(round(raw[key])) for key in AWARD_KEYS}
    diff = 100 - sum(rounded.values())
    if diff:
        adjust_key = max(AWARD_KEYS, key=lambda key: raw[key])
        rounded[adjust_key] = max(0, rounded[adjust_key] + diff)
    return rounded


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
