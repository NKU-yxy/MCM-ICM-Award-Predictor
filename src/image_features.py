"""
Lightweight image/statistical feature extraction.

The deployed predictor does not load CNN backbones. It keeps chart/layout
signals that are useful for MCM/ICM papers and cheap to compute with Pillow and
NumPy.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


class ImageFeatureExtractor:
    """Extract cheap figure quality features from parsed PDF images."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.image_config = self.config["image_features"]
        self.feature_dim = 18

    def extract(self, images: List[Image.Image]) -> Dict:
        statistical_features = self._extract_statistical_features(images)
        return {
            "deep_features": np.array([], dtype=np.float32),
            "statistical_features": statistical_features,
            "feature_vector": statistical_features,
            "feature_dim": len(statistical_features),
            "image_count": len(images or []),
        }

    def _extract_statistical_features(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros(self.feature_dim, dtype=np.float32)

        features = []

        image_count = len(images)
        features.append(min(image_count / 100.0, 1.0))

        widths = [img.width for img in images]
        heights = [img.height for img in images]
        aspect_ratios = [w / h if h > 0 else 1.0 for w, h in zip(widths, heights)]
        resolutions = [w * h for w, h in zip(widths, heights)]

        avg_width = np.mean(widths)
        features.append(min(avg_width / 2000.0, 1.0))

        avg_height = np.mean(heights)
        features.append(min(avg_height / 2000.0, 1.0))

        features.append(float(np.mean(aspect_ratios)))

        resolution_std = np.std(resolutions)
        features.append(min(resolution_std / 1000000.0, 1.0))

        color_richness_list = []
        for img in images[:10]:
            try:
                img_small = img.resize((50, 50))
                colors = img_small.getcolors(maxcolors=10000)
                if colors:
                    color_richness_list.append(len(colors))
            except Exception:
                pass
        avg_color_richness = np.mean(color_richness_list) if color_richness_list else 0
        features.append(min(avg_color_richness / 2500.0, 1.0))

        sample_images = images[:20]
        edge_densities = []
        saturations = []
        contrasts = []
        whitespace_ratios = []
        is_color_list = []
        has_colorbar_count = 0
        text_region_ratios = []

        for img in sample_images:
            try:
                analysis = self._analyze_single_image(img)
                edge_densities.append(analysis["edge_density"])
                saturations.append(analysis["avg_saturation"])
                contrasts.append(analysis["contrast"])
                whitespace_ratios.append(analysis["whitespace_ratio"])
                is_color_list.append(analysis["is_color"])
                if analysis["has_colorbar_hint"]:
                    has_colorbar_count += 1
                text_region_ratios.append(analysis["text_region_ratio"])
            except Exception:
                pass

        n_analyzed = max(len(edge_densities), 1)

        chart_threshold = 0.15
        chart_count = sum(1 for ed in edge_densities if ed > chart_threshold)
        chart_ratio = chart_count / n_analyzed
        features.append(chart_ratio)

        avg_edge_density = np.mean(edge_densities) if edge_densities else 0
        features.append(min(avg_edge_density, 1.0))

        avg_saturation = np.mean(saturations) if saturations else 0
        features.append(float(avg_saturation))

        color_ratio = sum(is_color_list) / n_analyzed if is_color_list else 0
        features.append(float(color_ratio))

        avg_contrast = np.mean(contrasts) if contrasts else 0
        features.append(min(avg_contrast / 128.0, 1.0))

        if len(resolutions) > 1:
            res_cv = np.std(resolutions) / max(np.mean(resolutions), 1)
            size_consistency = max(0.0, 1.0 - res_cv)
        else:
            size_consistency = 1.0
        features.append(float(size_consistency))

        avg_whitespace = np.mean(whitespace_ratios) if whitespace_ratios else 0
        features.append(float(avg_whitespace))

        professional_score = (
            0.25 * chart_ratio
            + 0.20 * min(avg_edge_density / 0.2, 1.0)
            + 0.15 * min(avg_contrast / 128.0, 1.0)
            + 0.15 * size_consistency
            + 0.15 * avg_whitespace
            + 0.10 * (1.0 - abs(avg_saturation - 0.3))
        )
        features.append(min(float(professional_score), 1.0))

        features.append(min(has_colorbar_count / n_analyzed, 1.0))

        avg_text_ratio = np.mean(text_region_ratios) if text_region_ratios else 0
        features.append(float(avg_text_ratio))

        if len(edge_densities) > 1:
            visual_diversity = min((np.std(edge_densities) + np.std(saturations)) / 0.3, 1.0)
        else:
            visual_diversity = 0.0
        features.append(float(visual_diversity))

        high_res_count = sum(1 for w, h in zip(widths, heights) if w > 400 or h > 300)
        features.append(high_res_count / max(len(images), 1))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _analyze_single_image(img: Image.Image) -> dict:
        img_small = img.convert("RGB").resize((100, 100))
        pixels = np.array(img_small, dtype=np.float32)

        gray = np.mean(pixels, axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_density = float(((np.mean(dx) + np.mean(dy)) / 2.0) / 255.0)

        r, g, b = pixels[:, :, 0] / 255, pixels[:, :, 1] / 255, pixels[:, :, 2] / 255
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        sat = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
        avg_saturation = float(np.mean(sat))

        contrast = float(np.std(gray))
        whitespace_ratio = float(np.mean(np.all(pixels > 240, axis=2)))
        is_color = float(avg_saturation > 0.08)

        has_colorbar_hint = False
        try:
            right_strip = pixels[:, -10:, :]
            right_v_std = np.std(np.mean(right_strip, axis=1), axis=0)
            right_h_std = np.std(np.mean(right_strip, axis=0), axis=0)
            has_colorbar_hint = bool(np.mean(right_v_std) > 20 and np.mean(right_h_std) < 30)
        except Exception:
            pass

        text_region_ratio = 0.0
        if edge_density > 0.1 and avg_saturation < 0.15:
            binary = (gray > 128).astype(float)
            transitions = np.abs(np.diff(binary, axis=1)).sum() + np.abs(np.diff(binary, axis=0)).sum()
            text_region_ratio = min(transitions / 10000.0, 1.0)

        return {
            "edge_density": edge_density,
            "avg_saturation": avg_saturation,
            "contrast": contrast,
            "whitespace_ratio": whitespace_ratio,
            "is_color": is_color,
            "has_colorbar_hint": has_colorbar_hint,
            "text_region_ratio": text_region_ratio,
        }

    def batch_extract(self, images_list: List[List[Image.Image]]) -> np.ndarray:
        return np.vstack([self.extract(images)["feature_vector"] for images in images_list])

    def get_feature_names(self) -> List[str]:
        return [
            "image_count_norm",
            "avg_width_norm",
            "avg_height_norm",
            "avg_aspect_ratio",
            "resolution_std_norm",
            "avg_color_richness_norm",
            "chart_ratio",
            "avg_edge_density",
            "avg_saturation",
            "color_mode_ratio",
            "avg_contrast",
            "size_consistency",
            "avg_whitespace_ratio",
            "professional_score",
            "has_colorbar",
            "avg_text_region_ratio",
            "visual_diversity",
            "high_res_ratio",
        ]


def extract_image_features(images: List, config_path: str = "config.yaml") -> np.ndarray:
    extractor = ImageFeatureExtractor(config_path)
    return extractor.extract(images)["feature_vector"]


def main():
    extractor = ImageFeatureExtractor()
    result = extractor.extract([])
    print(f"image feature dimension: {result['feature_vector'].shape[0]}")


if __name__ == "__main__":
    main()
