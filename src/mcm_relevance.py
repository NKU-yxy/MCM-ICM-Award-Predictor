"""
Lightweight MCM/ICM relevance detection.

This module deliberately avoids neural models. It uses TF-IDF word/char
features, TruncatedSVD and LogisticRegression so the deployed service can run
within small Render instances without runtime model downloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer


AWARD_ZERO_PROBS = {"O": 0.0, "F": 0.0, "M": 0.0, "H": 0.0, "S": 0.0}


NEGATIVE_TEXT_SAMPLES = [
    """
    Artemis mission status report. The launch vehicle, crew capsule, and lunar
    orbit operations are reviewed for engineering readiness. The document
    focuses on propulsion, thermal protection, avionics, and mission assurance.
    """,
    """
    Annual financial report. This filing describes revenue recognition,
    consolidated balance sheets, cash flows, audit controls, risk factors, and
    management discussion for shareholders and regulators.
    """,
    """
    Clinical case report. A patient presented with fever, abnormal laboratory
    results, and imaging findings. The discussion covers diagnosis, treatment,
    follow-up, and ethical approval.
    """,
    """
    Software architecture manual for a distributed web service. The report
    discusses API gateways, databases, authentication, observability, and cloud
    deployment incidents.
    """,
    """
    Legal memorandum. The analysis reviews statutory interpretation, precedent,
    jurisdiction, evidentiary standards, liability, and remedies requested by
    the parties.
    """,
    """
    Literature review of Renaissance art history. The paper discusses visual
    composition, patronage, archival evidence, conservation, and historical
    interpretation.
    """,
    """
    Astronomy observation proposal. The document requests telescope time to
    measure stellar spectra, exoplanet transits, instrument calibration, and
    photometric uncertainty.
    """,
    """
    Chemistry laboratory report. The experiment measures reaction yield,
    infrared spectra, melting point, titration curves, and safety precautions.
    """,
    """
    Machine learning benchmark paper for image classification. It describes a
    dataset, convolutional architecture, ablation study, hyperparameters, and
    top-1 accuracy, but it is not a contest solution report.
    """,
    """
    Public policy white paper. The document recommends municipal budget
    changes, stakeholder engagement, legislative steps, and implementation
    timelines without mathematical contest modeling requirements.
    """,
    """
    NASA mission architecture report. This technical document reviews Artemis
    launch windows, orbital insertion, spacecraft subsystems, lunar surface
    operations, propulsion margins, risk posture, avionics, communications,
    mission assurance, and engineering readiness. It contains tables, figures,
    schedules, acronyms, requirements, and quantitative analysis, but it is an
    aerospace program report rather than a MCM or ICM contest solution.
    """,
    """
    Academic machine learning paper. The manuscript proposes a transformer
    variant, compares baselines on benchmark datasets, reports precision,
    recall, ablation studies, computational cost, and limitations. Although it
    has abstract, introduction, methods, experiments, results, and references,
    it is a research article and does not answer a COMAP Problem A-F.
    """,
    """
    Engineering technical assessment. The report evaluates reliability,
    materials, thermal loads, instrumentation, finite element simulations,
    validation tests, manufacturing tolerances, and maintenance plans for an
    industrial system. The document is technical and mathematical but not an
    undergraduate mathematical modeling contest paper.
    """,
    """
    Corporate annual report. The filing includes audited financial statements,
    revenue, balance sheets, cash flow, risk factors, management discussion,
    governance disclosures, market outlook, and shareholder information. This
    PDF is structured and data-rich but should never receive a MCM/ICM award.
    """,
    """
    Public policy implementation report. The document describes stakeholder
    interviews, policy options, regulatory constraints, implementation
    timelines, budget scenarios, and social impact. It may include charts and
    recommendations, but it is not a COMAP contest submission and has no team
    control number.
    """,
]


def normalize_text(text: str, max_chars: int = 60000) -> str:
    """Normalize extracted PDF text for lightweight vector models."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:max_chars]


class LightweightSemanticEncoder:
    """TF-IDF + SVD semantic encoder used by both scoring and rejection."""

    def __init__(
        self,
        n_components: int = 128,
        max_features: int = 30000,
        params: Optional[dict] = None,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.vectorizer: Optional[FeatureUnion] = None
        self.svd: Optional[TruncatedSVD] = None
        self.normalizer: Optional[Normalizer] = None
        self.fitted = False
        self.output_dim = n_components
        if params:
            self.load_params(params)

    def _build_vectorizer(self) -> FeatureUnion:
        word_features = max(self.max_features // 2, 1000)
        char_features = max(self.max_features // 2, 1000)
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95,
                        max_features=word_features,
                        sublinear_tf=True,
                        lowercase=True,
                        stop_words="english",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        max_df=0.98,
                        max_features=char_features,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )

    def fit(self, texts: Iterable[str]):
        clean_texts = [normalize_text(t) for t in texts if normalize_text(t)]
        if len(clean_texts) < 4:
            raise ValueError("Need at least four non-empty texts to fit semantic encoder")

        self.vectorizer = self._build_vectorizer()
        x_tfidf = self.vectorizer.fit_transform(clean_texts)

        n_components = min(self.n_components, x_tfidf.shape[0] - 1, x_tfidf.shape[1] - 1)
        n_components = max(2, int(n_components))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.normalizer = Normalizer(copy=False)

        x_lsa = self.svd.fit_transform(x_tfidf)
        self.normalizer.fit(x_lsa)
        self.output_dim = n_components
        self.fitted = True
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if not self.fitted or self.vectorizer is None or self.svd is None:
            raise RuntimeError("LightweightSemanticEncoder is not fitted")
        clean_texts = [normalize_text(t) for t in texts]
        x_tfidf = self.vectorizer.transform(clean_texts)
        x_lsa = self.svd.transform(x_tfidf)
        if self.normalizer is not None:
            x_lsa = self.normalizer.transform(x_lsa)
        return np.asarray(x_lsa, dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) < 10:
            return np.zeros(self.output_dim, dtype=np.float32)
        return self.transform([text])[0]

    def get_params(self) -> dict:
        return {
            "n_components": self.n_components,
            "max_features": self.max_features,
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "normalizer": self.normalizer,
            "fitted": self.fitted,
            "output_dim": self.output_dim,
        }

    def load_params(self, params: dict):
        self.n_components = params.get("n_components", 128)
        self.max_features = params.get("max_features", 30000)
        self.vectorizer = params.get("vectorizer")
        self.svd = params.get("svd")
        self.normalizer = params.get("normalizer")
        self.fitted = params.get("fitted", False)
        self.output_dim = params.get("output_dim", self.n_components)


@dataclass
class RelevanceSignals:
    semantic_prob: float
    direct_signal: float
    structure_signal: float
    problem_signal: float

    @property
    def score(self) -> float:
        return float(
            np.clip(
                0.65 * self.semantic_prob
                + 0.25 * self.direct_signal
                + 0.08 * self.problem_signal
                + 0.02 * self.structure_signal,
                0.0,
                1.0,
            )
        )


class MCMRelevanceDetector:
    """Reject PDFs that do not look like MCM/ICM contest solution papers."""

    def __init__(
        self,
        encoder: Optional[LightweightSemanticEncoder] = None,
        classifier: Optional[LogisticRegression] = None,
        threshold: float = 0.75,
        semantic_floor: float = 0.55,
        problem_floor_without_marker: float = 0.35,
        require_mcm_marker_when_problem_weak: bool = True,
        params: Optional[dict] = None,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.threshold = threshold
        self.semantic_floor = semantic_floor
        self.problem_floor_without_marker = problem_floor_without_marker
        self.require_mcm_marker_when_problem_weak = require_mcm_marker_when_problem_weak
        self.fitted = classifier is not None
        if params:
            self.load_params(params, encoder=encoder)

    def fit(
        self,
        positive_texts: List[str],
        negative_texts: Optional[List[str]] = None,
        encoder: Optional[LightweightSemanticEncoder] = None,
    ):
        negative_texts = negative_texts or NEGATIVE_TEXT_SAMPLES
        positive_texts = [normalize_text(t) for t in positive_texts if normalize_text(t)]
        negative_texts = [normalize_text(t) for t in negative_texts if normalize_text(t)]
        if len(positive_texts) < 3 or len(negative_texts) < 3:
            raise ValueError("Need at least three positive and negative texts")

        self.encoder = encoder or LightweightSemanticEncoder()
        if not self.encoder.fitted:
            self.encoder.fit(positive_texts + negative_texts)

        x_pos = self.encoder.transform(positive_texts)
        x_neg = self.encoder.transform(negative_texts)
        x = np.vstack([x_pos, x_neg])
        y = np.array([1] * len(x_pos) + [0] * len(x_neg))

        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.classifier.fit(x, y)
        self.fitted = True
        return self

    def semantic_probability(self, text: str) -> float:
        if not self.fitted or self.encoder is None or self.classifier is None:
            return 0.5
        vec = self.encoder.encode(text).reshape(1, -1)
        return float(self.classifier.predict_proba(vec)[0, 1])

    def evaluate(
        self,
        full_text: str,
        structure: Optional[dict] = None,
        metadata: Optional[dict] = None,
        detection: Optional[dict] = None,
    ) -> Dict:
        text = normalize_text(full_text)
        structure = structure or {}
        metadata = metadata or {}
        detection = detection or {}

        if len(text) < 500:
            return {
                "is_mcm": False,
                "mcm_relevance": 0.0,
                "semantic_prob": 0.0,
                "direct_signal": 0.0,
                "structure_signal": 0.0,
                "problem_signal": 0.0,
                "rejection_reason": "PDF文本不可解析，无法评奖",
            }

        signals = RelevanceSignals(
            semantic_prob=self.semantic_probability(text),
            direct_signal=self._direct_contest_signal(text),
            structure_signal=self._structure_signal(structure, metadata),
            problem_signal=self._problem_signal(detection),
        )
        relevance = signals.score

        rejection_reason = ""
        if signals.semantic_prob < self.semantic_floor:
            rejection_reason = "PDF语义与MCM/ICM论文不相关，不予评奖"
        elif (
            self.require_mcm_marker_when_problem_weak
            and signals.direct_signal <= 0.0
            and signals.problem_signal < self.problem_floor_without_marker
        ):
            rejection_reason = "缺少MCM/ICM/COMAP或明确题目信号，不予评奖"
        elif relevance < self.threshold:
            rejection_reason = "MCM/ICM相关性不足，不予评奖"

        is_mcm = not rejection_reason

        return {
            "is_mcm": bool(is_mcm),
            "mcm_relevance": relevance,
            "semantic_prob": signals.semantic_prob,
            "direct_signal": signals.direct_signal,
            "structure_signal": signals.structure_signal,
            "problem_signal": signals.problem_signal,
            "rejection_reason": rejection_reason,
        }

    @staticmethod
    def _direct_contest_signal(text: str) -> float:
        lower = text.lower()
        patterns = [
            r"\bmcm\b",
            r"\bicm\b",
            r"\bcomap\b",
            r"mathematical contest in modeling",
            r"interdisciplinary contest in modeling",
            r"\bproblem\s+[a-f]\b",
            r"\bteam\s*#?\s*\d{5,7}\b",
            r"\bcontrol\s+number\b",
            r"\bsummary\s+sheet\b",
            r"\bpage\s+\d+\s+of\s+\d+\b",
        ]
        hits = sum(1 for pat in patterns if re.search(pat, lower))
        return float(np.clip(hits / 4.0, 0.0, 1.0))

    @staticmethod
    def _structure_signal(structure: dict, metadata: dict) -> float:
        completeness = float(structure.get("structure_completeness", 0.0))
        page_count = float(metadata.get("page_count", 0) or 0)
        word_count = float(structure.get("total_word_count", 0) or 0)

        section_bonus = np.mean(
            [
                bool(structure.get("has_abstract")),
                bool(structure.get("has_introduction")),
                bool(structure.get("has_methodology")),
                bool(structure.get("has_results")),
                bool(structure.get("has_conclusion")),
                bool(structure.get("has_references")),
            ]
        )
        page_signal = 1.0 if 6 <= page_count <= 30 else 0.35 if page_count >= 3 else 0.0
        word_signal = 1.0 if word_count >= 2500 else min(word_count / 2500.0, 1.0)
        modeling_signal = np.mean(
            [
                bool(structure.get("has_sensitivity_analysis")),
                bool(structure.get("has_model_validation")),
                bool(structure.get("has_assumption_justification")),
                bool(structure.get("has_error_analysis")),
            ]
        )
        return float(
            np.clip(
                0.35 * completeness
                + 0.25 * section_bonus
                + 0.20 * page_signal
                + 0.10 * word_signal
                + 0.10 * modeling_signal,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _problem_signal(detection: dict) -> float:
        problem = detection.get("problem")
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if problem not in set("ABCDEF"):
            return 0.0
        method = detection.get("detection_method", "")
        if method in {"path", "text_reference"}:
            return float(np.clip(confidence, 0.0, 1.0))
        return float(np.clip(confidence * 0.75, 0.0, 1.0))

    def get_params(self) -> dict:
        return {
            "classifier": self.classifier,
            "threshold": self.threshold,
            "semantic_floor": self.semantic_floor,
            "problem_floor_without_marker": self.problem_floor_without_marker,
            "require_mcm_marker_when_problem_weak": self.require_mcm_marker_when_problem_weak,
            "fitted": self.fitted,
        }

    def load_params(self, params: dict, encoder: Optional[LightweightSemanticEncoder] = None):
        self.encoder = encoder or self.encoder
        self.classifier = params.get("classifier")
        self.threshold = params.get("threshold", 0.75)
        self.semantic_floor = params.get("semantic_floor", 0.55)
        self.problem_floor_without_marker = params.get("problem_floor_without_marker", 0.35)
        self.require_mcm_marker_when_problem_weak = params.get(
            "require_mcm_marker_when_problem_weak", True
        )
        self.fitted = params.get("fitted", self.classifier is not None)


def make_non_mcm_result(reason: str = "非美赛PDF，不予评奖", relevance: Optional[dict] = None) -> Dict:
    """Create a uniform successful rejection payload."""
    relevance = relevance or {}
    return {
        "success": True,
        "is_mcm": False,
        "message": "非美赛PDF，不予评奖",
        "rejection_reason": reason,
        "probabilities": AWARD_ZERO_PROBS.copy(),
        "score": 0.0,
        "aspect_scores": {},
        "aspect_details": {},
        "similarity": 0.0,
        "problem": None,
        "contest": None,
        "year": None,
        "quality_tier": "非美赛PDF",
        "emoji": "",
        "description": reason,
        "mcm_relevance": float(relevance.get("mcm_relevance", 0.0) or 0.0),
        "metadata": {},
    }
