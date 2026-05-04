"""Lightweight post-LLM award calibration for the public web reviewer.

The DeepSeek rubric remains the expensive paper reader. This module only uses
the returned score plus already extracted PDF evidence to produce a stricter,
stable O/F/M/H/S distribution without loading the heavier local ML model.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from src.award_prior import get_overall_average_prior


AWARD_KEYS = ["O", "F", "M", "H", "S"]
CALIBRATION_VERSION = "calibrated_v4_conservative_single"
RAW_PROBABILITY_WEIGHT = 0.15
OFFICIAL_PRIOR_POWER = 0.08
SCORE_CENTERS = {"O": 97.0, "F": 88.0, "M": 76.0, "H": 63.0, "S": 48.0}
SCORE_SCALES = {"O": 3.0, "F": 5.0, "M": 8.0, "H": 8.0, "S": 12.0}
OFFICIAL_PRIORS = get_overall_average_prior()


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
    normalized_raw_probs = _normalize_percent_dict(raw_probs)
    raw_score = _number(calibrated.get("score"), 0.0)

    calibrated["raw_award_prediction"] = raw_award
    calibrated["raw_probabilities"] = normalized_raw_probs
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
    probabilities = _calibrated_probabilities(
        calibrated_score,
        evidence,
        raw_probabilities=normalized_raw_probs if raw_probs else None,
    )
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
    visual_evidence_count = int(_number(metadata.get("visual_evidence_count"), -1))
    figure_captions = int(_number(structure.get("figure_caption_count"), 0))
    table_count = int(_number(structure.get("table_count"), 0))
    visual_total = (
        visual_evidence_count
        if visual_evidence_count >= 0
        else max(image_count, figure_captions) + table_count
    )

    return {
        "page_count": page_count,
        "abstract_word_count": int(_number(metadata.get("abstract_word_count"), 0)),
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
        "length_score": _number(details.get("length_style_score"), 0.0),
        "conclusion_score": _number(details.get("conclusion_score"), 0.0),
        "visual_score": _number(details.get("visual_score"), 0.0),
    }


def _evidence_penalty(evidence: Dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []

    if evidence["page_count"] and evidence["page_count"] < 12:
        penalty += 4.0
        reasons.append("paper is very short")
    elif evidence["page_count"] and evidence["page_count"] < 18:
        penalty += 2.0
        reasons.append("page count is below typical finalist/O range")

    if evidence["visual_total"] < 5:
        penalty += 3.0
        reasons.append("visual/table evidence is sparse")
    elif evidence["visual_total"] < 10:
        penalty += 2.0
        reasons.append("visual/table evidence is limited")
    elif evidence["visual_total"] < 14:
        penalty += 1.0
        reasons.append("visual/table evidence is below O-sample range")

    if evidence["completeness"] < 0.50:
        penalty += 3.0
        reasons.append("core paper structure is incomplete")
    elif evidence["completeness"] < 0.75:
        penalty += 1.25
        reasons.append("paper structure is not fully complete")

    if not evidence["has_sensitivity"]:
        amount = 1.0 if evidence["conclusion_score"] < 9 else 0.35
        if amount > 0:
            penalty += amount
            reasons.append("missing explicit sensitivity analysis")
    if not evidence["has_validation"]:
        if evidence["content_score"] < 23 or evidence["conclusion_score"] < 9:
            amount = 1.25
        elif evidence["content_score"] < 26 or evidence["conclusion_score"] < 12:
            amount = 0.5
        else:
            amount = 0.0
        if amount > 0:
            penalty += amount
            reasons.append("missing explicit model validation")
    if not (evidence["has_error_analysis"] or evidence["has_model_comparison"]):
        if evidence["conclusion_score"] < 9:
            amount = 1.0
        elif evidence["conclusion_score"] < 11 and evidence["content_score"] < 24:
            amount = 0.5
        else:
            amount = 0.0
        if amount > 0:
            penalty += amount
            reasons.append("missing explicit error analysis or model comparison")

    return min(penalty, 4.0), reasons


def _calibrated_probabilities(
    score: float,
    evidence: Dict[str, Any],
    *,
    raw_probabilities: Dict[str, int] | None = None,
) -> Dict[str, int]:
    probability_score = _probability_score(score, evidence)
    score_probs = _score_probabilities(probability_score)
    if raw_probabilities:
        raw_probs = {key: raw_probabilities.get(key, 0) / 100.0 for key in AWARD_KEYS}
        score_weight = 1.0 - RAW_PROBABILITY_WEIGHT
        probs = _normalize_float(
            {
                award: score_weight * score_probs.get(award, 0.0)
                + RAW_PROBABILITY_WEIGHT * raw_probs.get(award, 0.0)
                for award in AWARD_KEYS
            }
        )
    else:
        probs = score_probs

    probs = _apply_score_caps(probs, probability_score, evidence)
    return _round_percentages(probs)


def _probability_score(score: float, evidence: Dict[str, Any]) -> float:
    """Use O-sample evidence to map probabilities, without inflating reported score."""
    if score < 65:
        return score

    strength = _o_sample_evidence_strength(evidence)
    if strength >= 8:
        return min(100.0, score + 8.0)
    if strength >= 7:
        return min(100.0, score + 6.0)
    if strength >= 6:
        return min(100.0, score + 4.0)
    return score


def _o_sample_evidence_strength(evidence: Dict[str, Any]) -> int:
    abstract_words = evidence.get("abstract_word_count", 0)
    checks = [
        evidence["page_count"] >= 20,
        360 <= abstract_words <= 650,
        evidence["visual_total"] >= 18,
        evidence["content_score"] >= 24,
        evidence["visual_score"] >= 18,
        evidence["conclusion_score"] >= 10,
        evidence["completeness"] >= 0.80,
        evidence["has_sensitivity"] or evidence["has_validation"],
    ]
    return sum(1 for ok in checks if ok)


def _score_probabilities(score: float) -> Dict[str, float]:
    weights = {
        award: math.exp(-0.5 * ((score - SCORE_CENTERS[award]) / SCORE_SCALES[award]) ** 2)
        * max(OFFICIAL_PRIORS.get(award, 1e-6), 1e-6) ** OFFICIAL_PRIOR_POWER
        for award in AWARD_KEYS
    }
    return _normalize_float(weights)


def _apply_score_caps(
    probs: Dict[str, float],
    score: float,
    evidence: Dict[str, Any],
) -> Dict[str, float]:
    evidence_strength = _o_sample_evidence_strength(evidence)
    if score < 65:
        caps = {"O": 0.0, "F": 0.005, "M": 0.15}
        targets = ["H", "S"]
    elif score < 75:
        caps = {
            "O": 0.002,
            "F": 0.020,
            "M": 0.55 if evidence_strength >= 6 else 0.25,
        }
        targets = ["M", "H", "S"] if evidence_strength >= 6 else ["H", "S"]
    elif score < 85:
        caps = {
            "O": 0.005,
            "F": 0.08 if evidence_strength >= 6 else 0.04,
            "M": 0.90,
        }
        targets = ["M", "H", "S"]
    elif score < 90:
        caps = {
            "O": 0.010,
            "F": 0.25 if evidence_strength >= 6 else 0.08,
            "M": 0.90,
        }
        targets = ["M", "H", "S"]
    elif score < 94:
        caps = {"O": 0.030, "F": 0.45, "M": 0.90}
        targets = ["F", "M", "H"]
    else:
        caps = {"O": 0.050, "F": 0.55}
        targets = ["F", "M", "H"]

    if score < 95 or not _o_ready(evidence):
        caps["O"] = min(caps.get("O", 1.0), 0.03)
    if score < 90 or not _finalist_ready(evidence):
        caps["F"] = min(caps.get("F", 1.0), 0.08)

    adjusted = dict(probs)
    excess = 0.0
    capped_awards = set()
    for award, cap in caps.items():
        if adjusted.get(award, 0.0) > cap:
            excess += adjusted[award] - cap
            adjusted[award] = cap
            capped_awards.add(award)

    targets = [key for key in targets if key not in capped_awards]
    if not targets:
        targets = [key for key in ["M", "H", "S"] if key not in capped_awards]

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
    abstract_words = evidence.get("abstract_word_count", 0)
    return (
        evidence["page_count"] >= 22
        and 360 <= abstract_words <= 650
        and evidence["visual_total"] >= 20
        and evidence["completeness"] >= 0.85
        and evidence["has_sensitivity"]
        and (evidence["has_validation"] or evidence["has_error_analysis"] or evidence["has_model_comparison"])
        and evidence["content_score"] >= 28
        and evidence["length_score"] >= 16
        and evidence["conclusion_score"] >= 13
        and evidence["visual_score"] >= 21
    )


def _finalist_ready(evidence: Dict[str, Any]) -> bool:
    return (
        evidence["page_count"] >= 20
        and evidence["visual_total"] >= 18
        and evidence["completeness"] >= 0.80
        and _o_sample_evidence_strength(evidence) >= 6
        and evidence["content_score"] >= 25
        and evidence["visual_score"] >= 18
        and evidence["conclusion_score"] >= 11
    )


def _calibration_note(penalty: float, reasons: list[str]) -> str:
    parts = ["Calibrated with v4 conservative single-paper thresholds"]
    if penalty > 0:
        reason_text = "; ".join(reasons[:3])
        parts.append(f"{penalty:.1f} point evidence penalty applied ({reason_text})")
    else:
        parts.append("no evidence penalty applied")
    return "; ".join(parts) + "."


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
