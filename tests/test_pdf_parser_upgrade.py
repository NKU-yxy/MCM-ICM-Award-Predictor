import unittest

import fitz

from src.pdf_parser import PDFParser


def _make_pdf(text: str, *, draw_table: bool = False) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(54, 54, 540, 760), text, fontsize=10)

    if draw_table:
        x_values = [72, 172, 272, 372]
        y_values = [420, 450, 480, 510]
        for y in y_values:
            page.draw_line((x_values[0], y), (x_values[-1], y), width=0.8)
        for x in x_values:
            page.draw_line((x, y_values[0]), (x, y_values[-1]), width=0.8)
        for row, y in enumerate([438, 468, 498]):
            for col, x in enumerate([88, 188, 288]):
                page.insert_text((x, y), f"R{row}C{col}", fontsize=8)

    return doc


class PDFParserUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.parser = PDFParser("config.yaml")

    def test_summary_sheet_first_page_is_abstract(self):
        body = " ".join(
            f"The model records quantitative result {i} and validates the decision chain."
            for i in range(12)
        )
        doc = _make_pdf(
            "MCM/ICM Summary Sheet\n"
            "Team Control Number 1234567\n"
            "Problem Chosen: C\n"
            "2024\n"
            "A Network Optimization Model for Resource Allocation\n"
            f"{body}\n"
        )
        try:
            info = self.parser.extract_abstract_info(doc)
        finally:
            doc.close()

        self.assertEqual(info["method"], "summary_sheet_first_page")
        self.assertGreaterEqual(info["word_count"], 50)
        self.assertIn("Network Optimization Model", info["text"])
        self.assertNotIn("Team Control Number", info["text"])
        self.assertNotIn("Problem Chosen", info["text"])

    def test_problem_statement_does_not_trigger_summary_sheet_fallback(self):
        doc = _make_pdf(
            "2024 ICM\n"
            "Problem E: Sustainability of Property Insurance\n"
            "Extreme-weather events are becoming a crisis for property owners and insurers.\n"
            "Develop a model for insurance companies to determine if they should underwrite policies.\n"
        )
        try:
            info = self.parser.extract_abstract_info(doc)
        finally:
            doc.close()

        self.assertEqual(info["method"], "not_found")
        self.assertEqual(info["word_count"], 0)

    def test_abstract_stops_at_keywords_and_joins_hyphenation(self):
        body = (
            "This paper develops a network opti-\n"
            "mization model for allocation. "
            + " ".join(
                f"The model records quantitative result {i} and validates the decision chain."
                for i in range(8)
            )
        )
        doc = _make_pdf(
            f"Abstract\n{body}\nKeywords: network, allocation\n"
            "Introduction\nThis section should not enter the abstract."
        )
        try:
            info = self.parser.extract_abstract_info(doc)
        finally:
            doc.close()

        self.assertIn("optimization model", info["text"])
        self.assertNotIn("Keywords", info["text"])
        self.assertNotIn("Introduction", info["text"])
        self.assertGreaterEqual(info["word_count"], 50)
        self.assertGreaterEqual(info["confidence"], 0.5)

    def test_caption_dedup_and_pymupdf_table_count(self):
        doc = _make_pdf(
            "Abstract\n"
            + " ".join(f"Evidence sentence {i} supports the model." for i in range(20))
            + "\nIntroduction\n"
            "Figure 1: Model framework\n"
            "Fig. 2(a): Sensitivity curve\n"
            "Fig. 2(a): Sensitivity curve duplicate\n"
            "Table 3: Parameters\n"
            "Tab. 3: Parameters duplicate\n",
            draw_table=True,
        )
        try:
            full_text = self.parser._extract_full_text(doc)
            structure = self.parser._analyze_paper_structure(full_text, doc)
            metadata = {
                "filtered_image_count": 0,
                "rendered_vector_figure_count": 0,
                "raw_image_count": 0,
            }
            self.parser._attach_visual_evidence_counts(metadata, structure)
        finally:
            doc.close()

        self.assertEqual(structure["figure_caption_count"], 2)
        self.assertEqual(structure["table_caption_count"], 1)
        self.assertGreaterEqual(structure["pymupdf_table_count"], 1)
        self.assertEqual(
            structure["table_count"],
            max(structure["table_caption_count"], structure["pymupdf_table_count"]),
        )
        self.assertEqual(metadata["visual_evidence_count"], metadata["image_count"])
        self.assertEqual(metadata["raw_image_count"], 0)

    def test_reference_count_uses_full_reference_section(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(54, 54, 540, 760),
            "Abstract\n"
            + " ".join(f"Evidence sentence {i} supports the model." for i in range(20))
            + "\nConclusion\nThe model closes with quantified decisions.",
            fontsize=10,
        )
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(54, 54, 540, 760),
            "References\n"
            "[1] Smith, J. A network model. Journal of Modeling, 2020.\n"
            "[2] Brown, K. Optimization under uncertainty. Proceedings, 2021.\n",
            fontsize=10,
        )
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(54, 54, 540, 760),
            "[3] Green, L. Scenario analysis for policy. Journal of Systems, 2022.\n"
            "Appendix\nThis material should not be counted as references.",
            fontsize=10,
        )
        try:
            full_text = self.parser._extract_full_text(doc)
            metadata = self.parser.extract_metadata(
                doc,
                "data/raw/2024/ICM_F/2400000_O.pdf",
                full_text=full_text,
            )
        finally:
            doc.close()

        self.assertEqual(metadata["ref_count"], 3)

    def test_reference_count_supports_author_year_entries(self):
        text = (
            "References\n"
            "Smith, J. 2020. A network model for allocation. Journal of Modeling, 12(3), 1-9.\n"
            "Brown and Green 2021. Optimization under uncertainty. Proceedings of ICM.\n"
            "Lee, K. 2022. Scenario analysis for policy. doi:10.1000/example\n"
        )

        self.assertEqual(self.parser._count_references(text), 3)


if __name__ == "__main__":
    unittest.main()
