"""
Conservative MCM/ICM award probability model.

The estimator combines:
- official COMAP contest priors,
- similarity/OC-SVM quality evidence,
- ordinal score thresholds,
- light structural/topic adjustments,
- hard guards that prevent ordinary papers from being promoted to F/O.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.award_prior import (
    get_award_prior,
    get_competition_intensity,
    get_problem_profile,
)
from src.ordinal_calibrator import OrdinalCalibrator

logger = logging.getLogger(__name__)

AWARDS = ["O", "F", "M", "H", "S"]


class EnhancedAwardEstimator:
    """Estimate O/F/M/H/S probabilities from calibrated paper quality signals."""

    def __init__(
        self,
        o_similarities: np.ndarray = None,
        stats: dict = None,
        ordinal_calibrator: OrdinalCalibrator = None,
        ensemble_weight_score: float = 0.65,
    ):
        self.stats = stats or {}
        self.o_similarities = o_similarities
        self.award_distributions = {}
        self.ordinal_calibrator = ordinal_calibrator
        self.ensemble_weight_score = ensemble_weight_score
        if o_similarities is not None and len(o_similarities) > 3:
            self._fit_distributions(o_similarities)

    def _fit_distributions(self, o_similarities: np.ndarray):
        o_mean = float(np.mean(o_similarities))
        o_std = float(max(np.std(o_similarities), 0.01))
        self.award_distributions = {
            "O": {"mean": o_mean, "std": o_std},
            "F": {"mean": o_mean - 0.35 * o_std, "std": o_std * 1.20},
            "M": {"mean": o_mean - 1.10 * o_std, "std": o_std * 1.55},
            "H": {"mean": o_mean - 2.00 * o_std, "std": o_std * 2.10},
            "S": {"mean": o_mean - 3.20 * o_std, "std": o_std * 2.80},
        }
        logger.info("Conservative probability model fitted")

    def estimate_probabilities(
        self,
        similarity: float,
        problem: str = "A",
        year: int = 2025,
        score: float = None,
        structure_info: dict = None,
        full_text: str = None,
        aspect_scores: dict = None,
    ) -> Dict[str, float]:
        if not self.award_distributions and score is None:
            return self._normalize(get_award_prior(problem, year))

        score_probs = self._ordinal_probabilities(score, aspect_scores)
        bayesian_probs = self._bayesian_with_official_priors(
            similarity=similarity,
            problem=problem,
            year=year,
            structure_info=structure_info,
            full_text=full_text,
        )

        w_score = float(np.clip(self.ensemble_weight_score, 0.0, 1.0))
        probs = {
            award: w_score * score_probs.get(award, 0.0)
            + (1.0 - w_score) * bayesian_probs.get(award, 0.0)
            for award in AWARDS
        }
        quality = self._quality_score(score, aspect_scores) if score is not None else None
        return self._apply_high_award_guard(self._normalize(probs), quality)

    def _ordinal_probabilities(self, score: float, aspect_scores: dict = None) -> Dict[str, float]:
        if score is None:
            return self._normalize(get_award_prior("A", 2025))
        quality = self._quality_score(score, aspect_scores)
        if self.ordinal_calibrator is not None and self.ordinal_calibrator.fitted:
            probs = self.ordinal_calibrator.predict_proba(quality)
        else:
            probs = self._manual_ordinal_probabilities(quality)
        return self._apply_high_award_guard(probs, quality)

    @staticmethod
    def _quality_score(score: float, aspect_scores: dict = None) -> float:
        if aspect_scores:
            vals = [aspect_scores.get(k, score) for k in ["abstract", "figures", "modeling"]]
            return float(0.55 * score + 0.45 * np.mean(vals))
        return float(score)

    def _manual_ordinal_probabilities(self, quality: float) -> Dict[str, float]:
        thresholds = {
            "O": (96.0, 3.0),
            "F": (90.0, 4.0),
            "M": (78.0, 5.5),
            "H": (64.0, 6.5),
        }
        p_ge_o = norm.cdf((quality - thresholds["O"][0]) / thresholds["O"][1])
        p_ge_f = max(norm.cdf((quality - thresholds["F"][0]) / thresholds["F"][1]), p_ge_o)
        p_ge_m = max(norm.cdf((quality - thresholds["M"][0]) / thresholds["M"][1]), p_ge_f)
        p_ge_h = max(norm.cdf((quality - thresholds["H"][0]) / thresholds["H"][1]), p_ge_m)
        return self._normalize(
            {
                "O": p_ge_o,
                "F": p_ge_f - p_ge_o,
                "M": p_ge_m - p_ge_f,
                "H": p_ge_h - p_ge_m,
                "S": 1.0 - p_ge_h,
            }
        )

    def _bayesian_with_official_priors(
        self,
        similarity: float,
        problem: str,
        year: int,
        structure_info: dict = None,
        full_text: str = None,
    ) -> Dict[str, float]:
        if not self.award_distributions:
            return self._normalize(get_award_prior(problem, year))

        priors = get_award_prior(problem, year)
        likelihoods = self._compute_similarity_likelihoods(similarity)
        probs = {award: priors.get(award, 0.0) * likelihoods.get(award, 1e-12) for award in AWARDS}
        probs = self._normalize(probs)

        if structure_info:
            probs = self._adjust_by_structure(probs, structure_info, problem)
        if full_text:
            probs = self._adjust_by_problem_fit(probs, full_text, problem)
        probs = self._adjust_by_competition(probs, problem)
        return self._normalize(probs)

    def _compute_similarity_likelihoods(self, similarity: float) -> Dict[str, float]:
        likelihoods = {}
        for award, params in self.award_distributions.items():
            likelihoods[award] = max(
                float(norm.pdf(similarity, loc=params["mean"], scale=params["std"])),
                1e-30,
            )
        return likelihoods

    @staticmethod
    def _gaussian_optimal(value, optimal, spread):
        z = (value - optimal) / max(spread, 1.0)
        return float(np.exp(-0.5 * z * z))

    def _adjust_by_structure(
        self, probs: Dict[str, float], structure_info: dict, problem: str
    ) -> Dict[str, float]:
        profile = get_problem_profile(problem)
        indicators = profile.get("key_indicators", {})
        total_words = max(structure_info.get("total_word_count", 10000), 1)

        completeness = structure_info.get("structure_completeness", 0.5)
        formula_per_k = structure_info.get("formula_count", 0) / total_words * 1000
        visual_total = structure_info.get("figure_caption_count", 0) + structure_info.get("table_count", 0)
        citation_per_k = structure_info.get("citation_count", 0) / total_words * 1000

        advanced = np.mean(
            [
                bool(structure_info.get("has_sensitivity_analysis")),
                bool(structure_info.get("has_model_validation")),
                bool(structure_info.get("has_strengths_weaknesses")),
            ]
        )
        deep_quality = np.mean(
            [
                bool(structure_info.get("has_assumption_justification")),
                bool(structure_info.get("has_model_comparison")),
                bool(structure_info.get("has_error_analysis")),
                bool(structure_info.get("has_dimensional_analysis")),
            ]
        )

        quality = (
            0.25 * completeness
            + 0.15 * self._gaussian_optimal(formula_per_k, 18, 12) * indicators.get("formula_weight", 1.0)
            + 0.15 * self._gaussian_optimal(visual_total, 20, 12) * indicators.get("figure_weight", 1.0)
            + 0.10 * self._gaussian_optimal(citation_per_k, 12, 8)
            + 0.15 * advanced * indicators.get("sensitivity_weight", 1.0)
            + 0.20 * deep_quality
        )
        quality = float(np.clip(quality, 0.0, 1.0))
        bias = (quality - 0.5) * 0.18
        weights = {"O": 1.8, "F": 1.3, "M": 0.4, "H": -0.4, "S": -1.0}
        return self._normalize({k: probs[k] * max(0.05, 1.0 + bias * weights[k]) for k in AWARDS})

    def _adjust_by_problem_fit(
        self, probs: Dict[str, float], full_text: str, problem: str
    ) -> Dict[str, float]:
        profile = get_problem_profile(problem)
        keywords = profile.get("preferred_keywords", [])
        if not keywords:
            return probs
        lower = full_text.lower()
        total_words = max(len(lower.split()), 1)
        hits = sum(lower.count(kw.lower()) for kw in keywords)
        fit_score = min((hits / total_words * 1000) / 8.0, 1.0)
        bias = (fit_score - 0.35) * 0.08
        weights = {"O": 1.3, "F": 1.0, "M": 0.3, "H": -0.2, "S": -0.8}
        return self._normalize({k: probs[k] * max(0.05, 1.0 + bias * weights[k]) for k in AWARDS})

    def _adjust_by_competition(self, probs: Dict[str, float], problem: str) -> Dict[str, float]:
        intensity = get_competition_intensity(problem)
        adjustment = 0.05 * (1.0 - intensity)
        weights = {"O": 1.5, "F": 1.2, "M": 0.4, "H": -0.2, "S": -0.6}
        return self._normalize({k: probs[k] * max(0.05, 1.0 + adjustment * weights[k]) for k in AWARDS})

    def _apply_high_award_guard(self, probs: Dict[str, float], score: float = None) -> Dict[str, float]:
        if score is None:
            return self._normalize(probs)

        if score < 65:
            caps = {"O": 0.001, "F": 0.005, "M": 0.08}
            targets = ["H", "S"]
        elif score < 75:
            caps = {"O": 0.003, "F": 0.015, "M": 0.20}
            targets = ["H", "S"]
        elif score < 85:
            caps = {"O": 0.010, "F": 0.050}
            targets = ["M", "H", "S"]
        elif score < 92:
            caps = {"O": 0.030, "F": 0.120}
            targets = ["M", "H", "S"]
        elif score < 96:
            caps = {"O": 0.080, "F": 0.220}
            targets = ["M", "H", "S"]
        else:
            caps = {"O": 0.250, "F": 0.350}
            targets = ["M", "H", "S"]

        adjusted = dict(probs)
        excess = 0.0
        for award, cap in caps.items():
            if adjusted.get(award, 0.0) > cap:
                excess += adjusted[award] - cap
                adjusted[award] = cap

        if excess > 0:
            target_total = sum(adjusted.get(k, 0.0) for k in targets)
            if target_total <= 0:
                for k in targets:
                    adjusted[k] = adjusted.get(k, 0.0) + excess / len(targets)
            else:
                for k in targets:
                    adjusted[k] = adjusted.get(k, 0.0) + excess * adjusted.get(k, 0.0) / target_total
        return self._normalize(adjusted)

    @staticmethod
    def _normalize(probs: Dict[str, float]) -> Dict[str, float]:
        clean = {award: max(float(probs.get(award, 0.0)), 0.0) for award in AWARDS}
        total = sum(clean.values())
        if total <= 0:
            return {"O": 0.0, "F": 0.0, "M": 0.0, "H": 0.0, "S": 1.0}
        return {award: clean[award] / total for award in AWARDS}

    def _default_probabilities(self, problem: str = "A", year: int = 2025) -> Dict[str, float]:
        return self._normalize(get_award_prior(problem, year))

    def get_parameters(self) -> dict:
        params = {
            "award_distributions": self.award_distributions,
            "version": "v4_conservative",
            "ensemble_weight_score": self.ensemble_weight_score,
        }
        if self.ordinal_calibrator is not None:
            params["ordinal_calibrator"] = self.ordinal_calibrator.get_params()
        return params

    def load_parameters(self, params: dict):
        self.award_distributions = params.get("award_distributions", {})
        self.ensemble_weight_score = params.get("ensemble_weight_score", 0.65)
        cal_params = params.get("ordinal_calibrator")
        if cal_params:
            self.ordinal_calibrator = OrdinalCalibrator()
            self.ordinal_calibrator.load_params(cal_params)

    @staticmethod
    def get_award_description(probs: Dict[str, float]) -> str:
        best_award = max(probs, key=probs.get)
        best_prob = probs[best_award]
        award_names = {
            "O": "Outstanding Winner",
            "F": "Finalist",
            "M": "Meritorious Winner",
            "H": "Honorable Mention",
            "S": "Successful Participant",
        }
        if best_prob > 0.6:
            confidence = "high confidence"
        elif best_prob > 0.4:
            confidence = "medium confidence"
        else:
            confidence = "low confidence"
        return f"Most likely award: {award_names.get(best_award, best_award)} ({confidence})"

    @staticmethod
    def format_probabilities(probs: Dict[str, float], problem: str = None) -> str:
        names = {
            "O": "Outstanding (O)",
            "F": "Finalist    (F)",
            "M": "Meritorious (M)",
            "H": "Honorable   (H)",
            "S": "Successful  (S)",
        }
        lines = []
        if problem:
            lines.append(f"  [Problem {problem}, official-prior calibrated]")
        max_prob = max(probs.values()) if probs else 0
        for award in AWARDS:
            prob = probs.get(award, 0.0)
            marker = " <==" if prob == max_prob else ""
            bar = "#" * int(prob * 40)
            lines.append(f"  {names[award]} {prob*100:5.1f}% {bar}{marker}")
        return "\n".join(lines)

    @staticmethod
    def compute_quality_tier(probs: Dict[str, float]) -> Tuple[str, str]:
        expected_rank = (
            probs.get("O", 0) * 5
            + probs.get("F", 0) * 4
            + probs.get("M", 0) * 3
            + probs.get("H", 0) * 2
            + probs.get("S", 0) * 1
        )
        if expected_rank >= 4.2:
            return "顶尖水平", "*"
        if expected_rank >= 3.5:
            return "优秀水平", "*"
        if expected_rank >= 2.8:
            return "良好水平", "+"
        if expected_rank >= 2.0:
            return "中等水平", "+"
        return "有待提升", "-"


class AwardProbabilityEstimator(EnhancedAwardEstimator):
    """Backward-compatible alias."""

    pass
