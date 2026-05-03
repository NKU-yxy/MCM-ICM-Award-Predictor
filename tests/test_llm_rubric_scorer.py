import unittest
from pathlib import Path

from src.llm_rubric_scorer import DeepSeekRubricScorer


class DeepSeekRubricScorerPromptTests(unittest.TestCase):
    def test_problem_type_guidance_is_injected_for_icm_f(self):
        scorer = DeepSeekRubricScorer(prompt_path=str(Path("改进后prompt.txt")))

        prompt = scorer._build_prompt(
            abstract="This paper designs a data-supported intervention plan with SMART metrics.",
            full_text="References\n[1] Example reference.\n",
            structure={
                "figure_caption_count": 10,
                "table_count": 3,
                "table_caption_count": 2,
                "pymupdf_table_count": 3,
                "citation_count": 1,
                "formula_count": 5,
                "structure_completeness": 1.0,
                "has_sensitivity_analysis": True,
                "has_model_validation": False,
                "has_strengths_weaknesses": True,
                "has_assumption_justification": True,
                "has_model_comparison": False,
                "has_error_analysis": True,
            },
            image_result={},
            image_count=14,
            page_count=25,
            ref_count=1,
            pdf_metadata={
                "abstract_word_count": 12,
                "visual_evidence_count": 18,
                "filtered_image_count": 14,
                "raw_image_count": 14,
                "rendered_vector_figure_count": 0,
            },
            problem="F",
            contest="ICM",
            year=2024,
        )

        self.assertNotIn("{problem_type_guidance}", prompt)
        self.assertIn("ICM Problem F", prompt)
        self.assertIn("Logical Framework Approach", prompt)


if __name__ == "__main__":
    unittest.main()
