"""Train the lightweight MCM/ICM award scoring model.

The deployed service intentionally avoids Torch, torchvision and online model
downloads. Training fits three small components and stores them in
models/scoring_model.pkl:

- TF-IDF + TruncatedSVD semantic encoder
- LogisticRegression MCM/ICM relevance rejector
- O-award boundary scorer with OC-SVM quality evidence
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_fusion import fuse_features
from src.image_features import ImageFeatureExtractor
from src.mcm_relevance import (
    MCMRelevanceDetector,
    LightweightSemanticEncoder,
    NEGATIVE_TEXT_SAMPLES,
    normalize_text,
)
from src.ocsvm_scorer import OCSVMScorer
from src.pdf_parser import PDFParser, extract_paper_content
from src.probability_model_v2 import EnhancedAwardEstimator as AwardProbabilityEstimator
from src.text_features import TextFeatureExtractor
from src.utils import ensure_dir, load_config, parse_problem_path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_year_weight(year: int, current_year: int = 2025) -> float:
    """Weight recent official papers more heavily when computing the O centroid."""
    gap = max(current_year - int(year), 0)
    weight = np.exp(-0.35 * gap)
    if gap >= 8:
        weight *= 0.5
    return float(np.clip(weight, 0.03, 1.0))


def iter_o_award_pdfs(data_dir: Path) -> List[Path]:
    """Return local O-award PDFs from data/raw/YYYY/MCM_A/*_O.pdf style folders."""
    if not data_dir.exists():
        return []
    pdfs = [
        p
        for p in data_dir.rglob("*.pdf")
        if p.is_file() and p.stem.upper().endswith("_O")
    ]
    return sorted(pdfs)


def select_training_pdfs(
    pdf_paths: Iterable[Path],
    min_year: int = 2024,
    max_papers_per_bucket: int = 0,
) -> List[Path]:
    """Keep training bounded by year/problem so slow outlier PDFs cannot dominate."""
    selected_by_bucket: Dict[Tuple[int, str, str], List[Path]] = defaultdict(list)
    for pdf_path in pdf_paths:
        info = _path_metadata(pdf_path)
        year = int(info.get("year", 0) or 0)
        if year < min_year:
            continue
        key = (year, str(info.get("contest", "MCM")), str(info.get("problem", "A")).upper())
        if max_papers_per_bucket and len(selected_by_bucket[key]) >= max_papers_per_bucket:
            continue
        selected_by_bucket[key].append(pdf_path)

    selected: List[Path] = []
    for key in sorted(selected_by_bucket):
        selected.extend(sorted(selected_by_bucket[key]))
    return selected


def _path_metadata(pdf_path: Path) -> Dict:
    info = parse_problem_path(str(pdf_path))
    if info.get("contest") and info.get("problem") and info.get("year"):
        return info

    fallback = {"year": 2024, "contest": "MCM", "problem": "A"}
    for part in pdf_path.parts:
        if part.isdigit() and len(part) == 4:
            fallback["year"] = int(part)
        if "_" in part:
            contest, _, problem = part.partition("_")
            if contest in {"MCM", "ICM"} and problem:
                fallback["contest"] = contest
                fallback["problem"] = problem[:1].upper()
    return fallback


def _text_for_semantic(content: Dict) -> str:
    abstract = content.get("abstract", "") or ""
    full_text = content.get("full_text", "") or ""
    return normalize_text(f"{abstract}\n\n{full_text}", max_chars=60000)


def extract_text_only(pdf_path: Path, parser: PDFParser) -> Dict:
    """Extract text/structure without loading images, for fitting semantic models."""
    try:
        doc = fitz.open(str(pdf_path))
        full_text = parser._extract_full_text(doc)
        abstract = parser.extract_abstract(doc)
        metadata = parser.extract_metadata(doc, str(pdf_path))
        structure = parser._analyze_paper_structure(full_text, doc)
        doc.close()
        return {
            "success": True,
            "abstract": abstract,
            "full_text": full_text,
            "metadata": metadata,
            "structure": structure,
        }
    except Exception as exc:
        logger.warning("Failed text extraction for %s: %s", pdf_path, exc)
        return {
            "success": False,
            "abstract": "",
            "full_text": "",
            "metadata": {},
            "structure": {},
        }


def load_positive_texts(pdf_paths: Iterable[Path]) -> Tuple[List[str], List[Path]]:
    parser = PDFParser()
    texts: List[str] = []
    kept_paths: List[Path] = []
    for pdf_path in pdf_paths:
        logger.info("Reading text: %s", pdf_path)
        content = extract_text_only(pdf_path, parser)
        text = _text_for_semantic(content)
        if content.get("success") and len(text) >= 500:
            texts.append(text)
            kept_paths.append(pdf_path)
        else:
            logger.warning("Skipping unreadable or too-short PDF: %s", pdf_path)
    return texts, kept_paths


def load_local_negative_texts() -> List[str]:
    """Use obvious non-MCM local PDFs as rejector validation/training examples."""
    candidates: List[Path] = []
    for root in [Path("data/non_mcm"), Path("data/non-mcm"), Path("data/negative")]:
        if root.exists():
            candidates.extend(root.rglob("*.pdf"))

    artemis = Path("..") / "Artemis.pdf"
    if artemis.exists():
        candidates.append(artemis)

    parser = PDFParser()
    texts: List[str] = []
    for pdf_path in sorted(set(candidates)):
        content = extract_text_only(pdf_path, parser)
        text = _text_for_semantic(content)
        if len(text) >= 500:
            logger.info("Using local non-MCM negative sample: %s", pdf_path)
            texts.append(text)
    return texts


def fit_semantic_and_relevance_models(
    positive_texts: List[str],
    config: Dict,
) -> Tuple[LightweightSemanticEncoder, MCMRelevanceDetector, List[str]]:
    text_cfg = config.get("text_features", {})
    rel_cfg = config.get("mcm_relevance", {})

    negative_texts = NEGATIVE_TEXT_SAMPLES + load_local_negative_texts()
    encoder = LightweightSemanticEncoder(
        n_components=int(text_cfg.get("embedding_dim", 128)),
        max_features=int(text_cfg.get("max_tfidf_features", 30000)),
    )
    encoder.fit(positive_texts + negative_texts)

    detector = MCMRelevanceDetector(
        threshold=float(rel_cfg.get("threshold", 0.75)),
        semantic_floor=float(rel_cfg.get("semantic_floor", 0.55)),
        problem_floor_without_marker=float(rel_cfg.get("problem_floor_without_marker", 0.35)),
        require_mcm_marker_when_problem_weak=bool(
            rel_cfg.get("require_mcm_marker_when_problem_weak", True)
        ),
    )
    detector.fit(positive_texts, negative_texts, encoder=encoder)

    pos_scores = [detector.evaluate(t)["mcm_relevance"] for t in positive_texts[: min(20, len(positive_texts))]]
    pos_results = [detector.evaluate(t) for t in positive_texts[: min(20, len(positive_texts))]]
    neg_results = [detector.evaluate(t) for t in negative_texts]
    neg_scores = [r["mcm_relevance"] for r in neg_results]
    logger.info(
        "MCM relevance fitted: positive sample mean=%.3f, negative mean=%.3f, negative max=%.3f",
        float(np.mean(pos_scores)) if pos_scores else 0.0,
        float(np.mean(neg_scores)) if neg_scores else 0.0,
        float(np.max(neg_scores)) if neg_scores else 0.0,
    )
    logger.info(
        "MCM relevance pass rates: positive_sample=%s/%s, negative=%s/%s",
        sum(1 for r in pos_results if r.get("is_mcm")),
        len(pos_results),
        sum(1 for r in neg_results if r.get("is_mcm")),
        len(neg_results),
    )
    return encoder, detector, negative_texts


def load_o_award_features(
    pdf_paths: Iterable[Path],
    semantic_encoder: LightweightSemanticEncoder,
) -> Tuple[np.ndarray, List[Dict]]:
    text_extractor = TextFeatureExtractor(semantic_encoder=semantic_encoder)
    image_extractor = ImageFeatureExtractor()

    features_list: List[np.ndarray] = []
    metadata_list: List[Dict] = []

    for pdf_file in pdf_paths:
        info = _path_metadata(pdf_file)
        year = int(info.get("year", 2024))
        contest = str(info.get("contest", "MCM"))
        problem = str(info.get("problem", "A")).upper()

        try:
            logger.info("Extracting features: %s/%s_%s/%s", year, contest, problem, pdf_file.name)
            content = extract_paper_content(str(pdf_file))
            if not content.get("success", False):
                logger.warning("Skipping failed parse: %s", pdf_file)
                continue

            abstract = content.get("abstract", "") or content.get("full_text", "")[:2000]
            full_text = content.get("full_text", "") or abstract
            if len(normalize_text(full_text)) < 500:
                logger.warning("Skipping too-short text: %s", pdf_file)
                continue

            structure_info = content.get("structure", {})
            text_result = text_extractor.extract(
                abstract,
                full_text=full_text,
                structure=structure_info,
            )
            image_result = image_extractor.extract(content.get("images", []))

            pdf_metadata = content.get("metadata", {})
            page_count = int(pdf_metadata.get("page_count", 20) or 20)
            ref_count = int(pdf_metadata.get("ref_count", 15) or 15)

            fused_feat = fuse_features(
                text_features=text_result["feature_vector"],
                image_features=image_result["feature_vector"],
                year=year,
                contest=contest,
                problem=problem,
                page_count=page_count,
                ref_count=ref_count,
            )

            features_list.append(fused_feat.astype(np.float32))
            metadata_list.append(
                {
                    "year": year,
                    "contest": contest,
                    "problem": problem,
                    "filename": pdf_file.name,
                    "path": str(pdf_file),
                    "weight": get_year_weight(year),
                    "text_semantic": text_result["semantic_features"],
                    "text_stats": text_result["statistical_features"],
                    "text_structural": text_result["structural_features"],
                    "image_features": image_result["feature_vector"],
                    "abstract_length": len(abstract),
                    "full_text_length": len(full_text),
                    "image_count": image_result.get("image_count", 0),
                    "page_count": page_count,
                    "ref_count": ref_count,
                    "structure": structure_info,
                }
            )
        except Exception as exc:
            logger.exception("Feature extraction failed for %s: %s", pdf_file, exc)

    if not features_list:
        return np.array([]), []

    features = np.vstack(features_list)
    logger.info("Loaded %s O-award feature vectors, dim=%s", len(features), features.shape[1])
    return features, metadata_list


def compute_weighted_centroid(features: np.ndarray, metadata: List[Dict]) -> np.ndarray:
    weights = np.array([m["weight"] for m in metadata], dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    centroid = np.average(features, axis=0, weights=weights)

    year_weights = defaultdict(float)
    for meta in metadata:
        year_weights[meta["year"]] += meta["weight"]
    logger.info("Year weight distribution:")
    for year in sorted(year_weights.keys(), reverse=True):
        logger.info("  %s: %.2f", year, year_weights[year])
    return centroid.astype(np.float32)


def compute_statistics(features: np.ndarray, centroid: np.ndarray) -> Dict:
    similarities = cosine_similarity(features, centroid.reshape(1, -1)).flatten()
    distances = np.linalg.norm(features - centroid, axis=1)
    stats = {
        "similarity_mean": float(np.mean(similarities)),
        "similarity_std": float(np.std(similarities)),
        "similarity_min": float(np.min(similarities)),
        "similarity_max": float(np.max(similarities)),
        "similarity_median": float(np.median(similarities)),
        "similarity_q25": float(np.percentile(similarities, 25)),
        "similarity_q75": float(np.percentile(similarities, 75)),
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
    }
    logger.info(
        "O similarity stats: mean=%.4f std=%.4f range=[%.4f, %.4f]",
        stats["similarity_mean"],
        stats["similarity_std"],
        stats["similarity_min"],
        stats["similarity_max"],
    )
    return stats


def _compute_score_from_similarity(similarity: float, stats: Dict) -> float:
    sim_mean = stats["similarity_mean"]
    sim_std = max(stats["similarity_std"], 1e-6)
    sim_min = stats["similarity_min"]
    sim_max = stats["similarity_max"]

    if similarity >= sim_max:
        score = 100.0
    elif similarity >= sim_mean:
        score = 85.0 + 15.0 * (similarity - sim_mean) / max(sim_max - sim_mean, 1e-6)
    elif similarity >= sim_mean - 2.0 * sim_std:
        score = 50.0 + 35.0 * (similarity - (sim_mean - 2.0 * sim_std)) / (2.0 * sim_std)
    else:
        threshold = sim_mean - 2.0 * sim_std
        score = 50.0 * (similarity - sim_min) / max(threshold - sim_min, 1e-6)
    return float(np.clip(score, 0.0, 100.0))


def evaluate_validation_set(
    name: str,
    features: np.ndarray,
    centroid: np.ndarray,
    stats: Dict,
    metadata: List[Dict],
) -> np.ndarray:
    similarities = cosine_similarity(features, centroid.reshape(1, -1)).flatten()
    scores = np.array([_compute_score_from_similarity(sim, stats) for sim in similarities])

    logger.info("%s set: n=%s mean_score=%.1f median=%.1f min=%.1f max=%.1f",
                name, len(scores), scores.mean(), np.median(scores), scores.min(), scores.max())
    for sim, score, meta in zip(similarities[:10], scores[:10], metadata[:10]):
        logger.info("  %s: sim=%.4f score=%.1f", meta["filename"], sim, score)
    return scores


def compute_aspect_statistics(metadata: List[Dict]) -> Dict:
    aspect_stats: Dict[str, Dict] = {}

    sem_list = [m["text_semantic"] for m in metadata if isinstance(m.get("text_semantic"), np.ndarray)]
    img_list = [m["image_features"] for m in metadata if isinstance(m.get("image_features"), np.ndarray)]
    struct_list = [m["text_structural"] for m in metadata if isinstance(m.get("text_structural"), np.ndarray)]

    def _centroid_stats(rows: List[np.ndarray]) -> Dict:
        mat = np.vstack(rows)
        centroid = np.mean(mat, axis=0)
        sims = cosine_similarity(mat, centroid.reshape(1, -1)).flatten()
        return {
            "centroid": centroid,
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "sim_min": float(np.min(sims)),
            "sim_max": float(np.max(sims)),
        }

    if sem_list:
        aspect_stats["abstract"] = _centroid_stats(sem_list)
        aspect_stats["abstract"]["avg_length"] = float(np.mean([m.get("abstract_length", 0) for m in metadata]))

    if img_list:
        aspect_stats["figures"] = _centroid_stats(img_list)
        aspect_stats["figures"]["avg_image_count"] = float(np.mean([m.get("image_count", 0) for m in metadata]))

    if struct_list:
        aspect_stats["modeling"] = _centroid_stats(struct_list)
        aspect_stats["modeling"].update(
            {
                "avg_formula_count": float(np.mean([m.get("structure", {}).get("formula_count", 0) for m in metadata])),
                "avg_citation_count": float(np.mean([m.get("structure", {}).get("citation_count", 0) for m in metadata])),
                "avg_completeness": float(np.mean([m.get("structure", {}).get("structure_completeness", 0.5) for m in metadata])),
                "avg_advanced_sections": float(np.mean([m.get("structure", {}).get("advanced_section_count", 0) for m in metadata])),
                "avg_page_count": float(np.mean([m.get("page_count", 20) for m in metadata])),
                "avg_ref_count": float(np.mean([m.get("ref_count", 0) for m in metadata])),
            }
        )

    return aspect_stats


def save_model(
    centroid: np.ndarray,
    stats: Dict,
    metadata: List[Dict],
    save_path: str,
    semantic_encoder: LightweightSemanticEncoder,
    relevance_detector: MCMRelevanceDetector,
    prob_model_params: Dict,
    aspect_stats: Dict,
    ocsvm_scorer: OCSVMScorer,
):
    model_data = {
        "centroid": centroid,
        "stats": stats,
        "metadata": [
            {
                k: v
                for k, v in m.items()
                if k
                not in {
                    "text_semantic",
                    "text_stats",
                    "text_structural",
                    "image_features",
                    "structure",
                }
            }
            for m in metadata
        ],
        "train_date": datetime.now().isoformat(),
        "n_papers": len(metadata),
        "feature_dim": int(centroid.shape[0]),
        "semantic_encoder": semantic_encoder.get_params(),
        "mcm_relevance_model": relevance_detector.get_params(),
        "prob_model_params": prob_model_params,
        "aspect_stats": aspect_stats,
    }
    if ocsvm_scorer is not None and ocsvm_scorer.fitted:
        model_data["ocsvm_scorer"] = ocsvm_scorer.get_params()

    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("Model saved to %s", save_path)


def main():
    logger.info("=" * 72)
    logger.info("Training lightweight MCM/ICM award predictor")
    logger.info("=" * 72)

    config = load_config()
    data_dir = Path(config.get("data", {}).get("raw_dir", "data/raw"))
    all_pdf_paths = iter_o_award_pdfs(data_dir)
    training_cfg = config.get("training_data", {})
    pdf_paths = select_training_pdfs(
        all_pdf_paths,
        min_year=int(training_cfg.get("min_year", 2024)),
        max_papers_per_bucket=int(training_cfg.get("max_papers_per_bucket", 0) or 0),
    )
    if not pdf_paths:
        logger.error("No local O-award PDFs found under %s", data_dir)
        return

    logger.info("Found %s local O-award PDFs, using %s for training", len(all_pdf_paths), len(pdf_paths))
    positive_texts, usable_paths = load_positive_texts(pdf_paths)
    if len(positive_texts) < 5:
        logger.error("Not enough readable O-award PDFs to train")
        return

    semantic_encoder, relevance_detector, negative_texts = fit_semantic_and_relevance_models(
        positive_texts,
        config,
    )

    features, metadata = load_o_award_features(usable_paths, semantic_encoder)
    if len(features) == 0:
        logger.error("No feature vectors extracted")
        return

    year_counts = defaultdict(int)
    for meta in metadata:
        year_counts[meta["year"]] += 1
    logger.info("Training corpus:")
    logger.info("  total: %s O-award papers", len(features))
    logger.info("  negative relevance examples: %s", len(negative_texts))
    for year in sorted(year_counts.keys(), reverse=True):
        logger.info("  %s: %s papers (weight %.2f)", year, year_counts[year], get_year_weight(year))

    n_samples = len(features)
    if n_samples < 15:
        train_features = val_features = test_features = features
        train_metadata = val_metadata = test_metadata = metadata
    else:
        indices = np.arange(n_samples)
        trainval_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.18, random_state=42)
        train_features = features[train_idx]
        val_features = features[val_idx]
        test_features = features[test_idx]
        train_metadata = [metadata[i] for i in train_idx]
        val_metadata = [metadata[i] for i in val_idx]
        test_metadata = [metadata[i] for i in test_idx]

    centroid = compute_weighted_centroid(train_features, train_metadata)
    stats = compute_statistics(train_features, centroid)

    train_similarities = cosine_similarity(train_features, centroid.reshape(1, -1)).flatten()
    prob_estimator = AwardProbabilityEstimator(
        train_similarities,
        stats,
        ensemble_weight_score=float(config.get("ordinal", {}).get("ensemble_weight_score", 0.65)),
    )
    prob_model_params = prob_estimator.get_parameters()

    ocsvm_scorer = OCSVMScorer(
        pca_components=int(config.get("ocsvm", {}).get("pca_components", 50)),
        nu=float(config.get("ocsvm", {}).get("nu", 0.05)),
        gamma=config.get("ocsvm", {}).get("gamma", "scale"),
        kernel=config.get("ocsvm", {}).get("kernel", "rbf"),
    )
    train_ocsvm_scores = ocsvm_scorer.fit(train_features)
    logger.info(
        "OC-SVM train scores: mean=%.1f std=%.1f min=%.1f max=%.1f",
        train_ocsvm_scores.mean(),
        train_ocsvm_scores.std(),
        train_ocsvm_scores.min(),
        train_ocsvm_scores.max(),
    )

    val_scores = evaluate_validation_set("Validation", val_features, centroid, stats, val_metadata)
    test_scores = evaluate_validation_set("Test", test_features, centroid, stats, test_metadata)

    aspect_stats = compute_aspect_statistics(train_metadata)
    for aspect, st in aspect_stats.items():
        logger.info("%s aspect centroid dim=%s", aspect, np.asarray(st["centroid"]).shape[0])

    model_path = "models/scoring_model.pkl"
    save_model(
        centroid=centroid,
        stats=stats,
        metadata=train_metadata,
        save_path=model_path,
        semantic_encoder=semantic_encoder,
        relevance_detector=relevance_detector,
        prob_model_params=prob_model_params,
        aspect_stats=aspect_stats,
        ocsvm_scorer=ocsvm_scorer,
    )

    logger.info("=" * 72)
    logger.info("Training complete")
    logger.info("Model path: %s", model_path)
    logger.info("Feature dimension: %s", centroid.shape[0])
    logger.info("Validation mean score: %.1f", val_scores.mean())
    logger.info("Test mean score: %.1f", test_scores.mean())
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
