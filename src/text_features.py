"""
Lightweight text feature extraction for MCM/ICM PDF scoring.

Runtime deliberately avoids sentence-transformers. Semantic vectors come from
the TF-IDF + SVD encoder fitted by scripts/train_scoring_model.py and stored in
models/scoring_model.pkl.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mcm_relevance import LightweightSemanticEncoder, normalize_text
from src.utils import (
    calculate_avg_sentence_length,
    calculate_readability_score,
    calculate_technical_term_density,
    calculate_vocabulary_diversity,
    check_abstract_structure,
    count_academic_phrases,
    count_sentences,
    has_numerical_results,
    load_config,
)


class TextFeatureExtractor:
    """Extract semantic, statistical and structure-aware text features."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        semantic_encoder_params: Optional[dict] = None,
        semantic_encoder: Optional[LightweightSemanticEncoder] = None,
    ):
        self.config = load_config(config_path)
        self.text_config = self.config["text_features"]

        if semantic_encoder is not None:
            self.semantic_encoder = semantic_encoder
        elif semantic_encoder_params:
            self.semantic_encoder = LightweightSemanticEncoder(params=semantic_encoder_params)
        else:
            self.semantic_encoder = None

        default_dim = int(self.text_config.get("embedding_dim", 128))
        self.embedding_dim = (
            int(self.semantic_encoder.output_dim)
            if self.semantic_encoder is not None and self.semantic_encoder.fitted
            else default_dim
        )

    def extract(self, abstract: str, full_text: str = None, structure: dict = None) -> Dict:
        max_length = int(self.text_config.get("max_abstract_length", 2000))
        abstract = normalize_text(abstract or "", max_chars=max_length)
        full_text = full_text or abstract
        structure = structure or {}

        semantic_source = (abstract + "\n" + normalize_text(full_text, max_chars=6000)).strip()
        semantic_features = self._extract_semantic_features(semantic_source)
        statistical_features = self._extract_statistical_features(abstract)
        structural_features = self._extract_structural_features(abstract, full_text, structure)

        feature_vector = np.concatenate(
            [semantic_features, statistical_features, structural_features]
        ).astype(np.float32)

        return {
            "semantic_features": semantic_features,
            "statistical_features": statistical_features,
            "structural_features": structural_features,
            "feature_vector": feature_vector,
            "feature_dim": len(feature_vector),
        }

    def _extract_semantic_features(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self.semantic_encoder is not None and self.semantic_encoder.fitted:
            vec = self.semantic_encoder.encode(text).astype(np.float32)
            self.embedding_dim = len(vec)
            return vec

        return self._hashed_semantic_features(text, self.embedding_dim)

    @staticmethod
    def _hashed_semantic_features(text: str, dim: int) -> np.ndarray:
        """Deterministic no-fit fallback for legacy models."""
        vec = np.zeros(dim, dtype=np.float32)
        words = [w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()]
        for word in words:
            if not word:
                continue
            digest = hashlib.md5(word.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    @staticmethod
    def _extract_statistical_features(text: str) -> np.ndarray:
        features = []

        word_count = len(text.split())
        features.append(min(word_count / 500.0, 1.0))

        sentence_count = count_sentences(text)
        features.append(min(sentence_count / 50.0, 1.0))

        avg_sent_len = calculate_avg_sentence_length(text)
        features.append(min(avg_sent_len / 50.0, 1.0))

        features.append(1.0 if has_numerical_results(text) else 0.0)
        features.append(calculate_technical_term_density(text))
        features.append(check_abstract_structure(text))

        return np.array(features, dtype=np.float32)

    def _extract_structural_features(self, abstract: str, full_text: str, structure: dict) -> np.ndarray:
        features = []

        features.append(calculate_readability_score(abstract))

        text_sample = " ".join(full_text.split()[:5000])
        features.append(calculate_vocabulary_diversity(text_sample))
        features.append(count_academic_phrases(full_text))

        total_words = max(structure.get("total_word_count", len(full_text.split())), 1)
        features.append(structure.get("structure_completeness", 0.5))

        formula_count = structure.get("formula_count", 0)
        formula_per_k = formula_count / total_words * 1000
        features.append(self._gaussian_optimal(formula_per_k, optimal=18, spread=12))

        table_count = structure.get("table_count", 0)
        features.append(self._gaussian_optimal(table_count, optimal=6, spread=5))

        figure_count = structure.get("figure_caption_count", 0)
        features.append(self._gaussian_optimal(figure_count, optimal=15, spread=10))

        citation_count = structure.get("citation_count", 0)
        citation_per_k = citation_count / total_words * 1000
        features.append(self._gaussian_optimal(citation_per_k, optimal=12, spread=8))

        features.append(self._gaussian_optimal(total_words, optimal=12000, spread=5000))

        advanced_count = structure.get("advanced_section_count", 0)
        quality_count = structure.get("quality_section_count", 0)
        features.append(min((advanced_count * 0.7 + quality_count * 1.0) / 6.0, 1.0))

        avg_para_len = structure.get("avg_paragraph_length", 0)
        features.append(self._gaussian_optimal(avg_para_len, optimal=120, spread=60))

        section_count = structure.get("section_count", 0)
        features.append(min(section_count / 6.0, 1.0))

        features.append(1.0 if structure.get("has_assumption_justification") else 0.0)
        features.append(1.0 if structure.get("has_model_comparison") else 0.0)
        features.append(1.0 if structure.get("has_error_analysis") else 0.0)
        features.append(1.0 if structure.get("has_dimensional_analysis") else 0.0)
        features.append(self._compute_coherence(full_text))
        features.append(self._paragraph_variety(full_text))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _gaussian_optimal(value: float, optimal: float, spread: float) -> float:
        z = (value - optimal) / max(spread, 1.0)
        return float(np.exp(-0.5 * z * z))

    def _compute_coherence(self, full_text: str) -> float:
        paragraphs = [
            p.strip()
            for p in full_text.split("\n\n")
            if p.strip() and len(p.strip()) > 100
        ]
        if len(paragraphs) < 3:
            return 0.5

        sample_paras = paragraphs[:8]
        try:
            if self.semantic_encoder is not None and self.semantic_encoder.fitted:
                embeddings = self.semantic_encoder.transform(sample_paras)
            else:
                embeddings = np.vstack(
                    [self._hashed_semantic_features(p, self.embedding_dim) for p in sample_paras]
                )

            similarities = []
            for i in range(len(embeddings) - 1):
                denom = max(np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]), 1e-8)
                similarities.append(float(np.dot(embeddings[i], embeddings[i + 1]) / denom))
            avg_sim = np.mean(similarities)
            std_sim = np.std(similarities) if len(similarities) > 1 else 0.0
            return float(np.clip(avg_sim * (1.0 - std_sim), 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _paragraph_variety(full_text: str) -> float:
        paragraphs = [
            p.strip()
            for p in full_text.split("\n\n")
            if p.strip() and len(p.strip()) > 30
        ]
        if len(paragraphs) <= 3:
            return 0.0
        para_lens = [len(p.split()) for p in paragraphs]
        para_mean = np.mean(para_lens)
        para_std = np.std(para_lens)
        return float(min(para_std / max(para_mean, 1), 1.0))

    def batch_extract(self, abstracts: List[str]) -> np.ndarray:
        features = [self.extract(abstract)["feature_vector"] for abstract in abstracts]
        return np.vstack(features)

    def get_feature_names(self) -> List[str]:
        semantic_names = [f"semantic_{i}" for i in range(self.embedding_dim)]
        statistical_names = [
            "word_count_norm",
            "sentence_count_norm",
            "avg_sentence_length_norm",
            "has_numerical_results",
            "technical_term_density",
            "structure_score",
        ]
        structural_names = [
            "readability_score",
            "vocabulary_diversity",
            "academic_phrase_density",
            "structure_completeness",
            "formula_quality",
            "table_quality",
            "figure_quality",
            "citation_quality",
            "word_count_quality",
            "advanced_quality_score",
            "paragraph_quality",
            "section_count_norm",
            "has_assumption_justification",
            "has_model_comparison",
            "has_error_analysis",
            "has_dimensional_analysis",
            "logical_coherence_score",
            "paragraph_variety_score",
        ]
        return semantic_names + statistical_names + structural_names


def extract_text_features(text: str, config_path: str = "config.yaml") -> np.ndarray:
    extractor = TextFeatureExtractor(config_path)
    return extractor.extract(text)["feature_vector"]


def main():
    sample = """
    We develop a mathematical model for a contest problem, state assumptions,
    estimate parameters, run sensitivity analysis, and provide conclusions.
    """
    extractor = TextFeatureExtractor()
    result = extractor.extract(sample)
    print(f"text feature dimension: {result['feature_vector'].shape[0]}")


if __name__ == "__main__":
    main()
