import unittest
from unittest.mock import patch

from fastapi import HTTPException

import web_app


class _Client:
    host = "203.0.113.10"


class _Request:
    def __init__(self, user_agent: str):
        self.headers = {"User-Agent": user_agent}
        self.client = _Client()


class _FakeDoc:
    needs_pass = False
    is_encrypted = False

    def __init__(self, pages: int = 1):
        self.pages = pages
        self.closed = False

    def __len__(self):
        return self.pages

    def close(self):
        self.closed = True


class WebSecurityTests(unittest.TestCase):
    def test_secure_rate_limit_defaults_are_enabled(self):
        self.assertEqual(web_app.RATE_LIMIT_MAX_REQUESTS, 5)
        self.assertEqual(web_app.RATE_LIMIT_DAILY_MAX_REQUESTS, 20)
        self.assertEqual(web_app.GLOBAL_DAILY_MAX_REQUESTS, 5000)

    def test_rate_limit_identity_is_not_user_agent_bypassable(self):
        first = web_app.RateLimitMiddleware._client_identity(_Request("Browser A"))
        second = web_app.RateLimitMiddleware._client_identity(_Request("Browser B"))

        self.assertEqual(first, second)

    def test_uploaded_pdf_validation_accepts_small_pdf(self):
        with patch("web_app.fitz.open", return_value=_FakeDoc(pages=1)):
            self.assertEqual(web_app._validate_uploaded_pdf(web_app.TEMP_DIR / "ok.pdf"), 1)

    def test_uploaded_pdf_validation_rejects_non_pdf(self):
        with patch("web_app.fitz.open", side_effect=RuntimeError("bad pdf")):
            with self.assertRaises(HTTPException) as ctx:
                web_app._validate_uploaded_pdf(web_app.TEMP_DIR / "bad.pdf")

        self.assertEqual(ctx.exception.status_code, 400)

    def test_template_mentions_v3_update(self):
        html = (web_app.TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("v3 已更新", html)
        self.assertIn("改善参考文献数量计算", html)
        self.assertIn("v3 版本近期评分趋势", html)
        self.assertIn("v2 版本奖项判断分布", html)
        self.assertIn("v1 版本奖项判断分布", html)

    def test_v3_migration_preserves_v1_v2_and_resets_v3(self):
        saved = {
            "total_predictions": 8,
            "today_predictions": 2,
            "today_date": "2026-05-04",
            "calibration_version": "calibrated_v2_official_prior",
            "legacy_award_counts": {"S": 5},
            "current_version_award_counts": {"M": 2, "H": 1},
            "recent_scores": [{"score": 80, "problem": "C", "timestamp": "2026-05-04T10:00"}],
        }

        migrated = web_app._migrate_saved_stats(saved)

        self.assertEqual(migrated["version_award_counts"]["v1"]["S"], 5)
        self.assertEqual(migrated["version_award_counts"]["v2"]["M"], 2)
        self.assertEqual(migrated["version_award_counts"]["v2"]["H"], 1)
        self.assertEqual(sum(migrated["version_award_counts"]["v3"].values()), 0)
        self.assertEqual(sum(migrated["current_version_award_counts"].values()), 0)
        self.assertEqual(migrated["current_version_recent_scores"], [])


if __name__ == "__main__":
    unittest.main()
