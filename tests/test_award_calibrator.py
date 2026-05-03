import unittest

from src.award_calibrator import CALIBRATION_VERSION, calibrate_rubric_award


def _rubric(score, details=None, probabilities=None):
    return {
        "status": "ok",
        "score": score,
        "award_prediction": "M",
        "probabilities": probabilities or {"O": 0, "F": 5, "M": 75, "H": 18, "S": 2},
        "details": details
        or {
            "content_score": 25,
            "length_style_score": 17,
            "visual_score": 20,
            "conclusion_score": 12,
            "writing_score": 8,
        },
        "comments": "",
    }


def _metadata(**overrides):
    data = {
        "page_count": 25,
        "abstract_word_count": 451,
        "visual_evidence_count": 22,
        "ref_count": 12,
        "image_count": 16,
    }
    data.update(overrides)
    return data


def _structure(**overrides):
    data = {
        "figure_caption_count": 16,
        "table_count": 5,
        "formula_count": 30,
        "structure_completeness": 0.90,
        "has_sensitivity_analysis": True,
        "has_model_validation": True,
        "has_assumption_justification": True,
        "has_model_comparison": True,
        "has_error_analysis": True,
    }
    data.update(overrides)
    return data


class AwardCalibratorTests(unittest.TestCase):
    def test_o_like_score_82_is_not_demoted_to_h(self):
        calibrated = calibrate_rubric_award(
            _rubric(82, probabilities={"O": 0, "F": 10, "M": 76, "H": 13, "S": 1}),
            metadata=_metadata(),
            structure=_structure(has_model_validation=False),
        )

        self.assertEqual(calibrated["calibration_version"], CALIBRATION_VERSION)
        self.assertIn(calibrated["award_prediction"], {"M", "F"})
        self.assertGreaterEqual(calibrated["probabilities"]["M"], calibrated["probabilities"]["H"])

    def test_strong_score_87_allows_finalist_and_nonzero_o(self):
        calibrated = calibrate_rubric_award(
            _rubric(
                87,
                details={
                    "content_score": 28,
                    "length_style_score": 18,
                    "visual_score": 22,
                    "conclusion_score": 13,
                    "writing_score": 9,
                },
                probabilities={"O": 1, "F": 20, "M": 70, "H": 9, "S": 0},
            ),
            metadata=_metadata(),
            structure=_structure(),
        )

        self.assertEqual(calibrated["award_prediction"], "F")
        self.assertGreater(calibrated["probabilities"]["O"], 0)

    def test_weak_score_60_stays_s_or_h(self):
        calibrated = calibrate_rubric_award(
            _rubric(
                60,
                details={
                    "content_score": 17,
                    "length_style_score": 12,
                    "visual_score": 10,
                    "conclusion_score": 7,
                    "writing_score": 7,
                },
                probabilities={"O": 0, "F": 0, "M": 0, "H": 30, "S": 70},
            ),
            metadata=_metadata(
                page_count=10,
                abstract_word_count=220,
                visual_evidence_count=4,
                ref_count=2,
                image_count=2,
            ),
            structure=_structure(
                figure_caption_count=2,
                table_count=0,
                formula_count=3,
                structure_completeness=0.45,
                has_sensitivity_analysis=False,
                has_model_validation=False,
                has_assumption_justification=False,
                has_model_comparison=False,
                has_error_analysis=False,
            ),
        )

        self.assertIn(calibrated["award_prediction"], {"S/U", "H"})
        self.assertLessEqual(calibrated["probabilities"]["M"], 12)

    def test_missing_validation_alone_does_not_block_m_or_f(self):
        calibrated = calibrate_rubric_award(
            _rubric(
                82,
                details={
                    "content_score": 27,
                    "length_style_score": 17,
                    "visual_score": 20,
                    "conclusion_score": 13,
                    "writing_score": 8,
                },
                probabilities={"O": 0, "F": 10, "M": 76, "H": 13, "S": 1},
            ),
            metadata=_metadata(),
            structure=_structure(has_model_validation=False),
        )

        self.assertGreaterEqual(calibrated["calibrated_score"], 82)
        self.assertIn(calibrated["award_prediction"], {"M", "F"})


if __name__ == "__main__":
    unittest.main()
