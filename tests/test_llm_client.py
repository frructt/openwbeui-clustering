from __future__ import annotations

import unittest

from src.clients.llm_client import _clean_json_payload, normalize_enrichment_payload


class LLMClientTests(unittest.TestCase):
    def test_clean_json_payload_accepts_trailing_text(self) -> None:
        payload = (
            '{"topic_title":"SQL","topic_description":"desc","suspected_jtbd":"jtbd",'
            '"suspected_pain_points":"pain","suspected_business_value":"value",'
            '"suspected_product_action":"action","confidence_note":"note"}\n\nExtra commentary'
        )
        parsed = _clean_json_payload(payload)
        self.assertEqual(parsed["topic_title"], "SQL")
        self.assertEqual(parsed["suspected_product_action"], "action")

    def test_clean_json_payload_accepts_markdown_fence(self) -> None:
        payload = """```json
{"topic_title":"Excel","topic_description":"desc","suspected_jtbd":"jtbd","suspected_pain_points":"pain","suspected_business_value":"value","suspected_product_action":"action","confidence_note":"note"}
```"""
        parsed = _clean_json_payload(payload)
        self.assertEqual(parsed["topic_title"], "Excel")

    def test_normalize_enrichment_payload_flattens_lists(self) -> None:
        payload = normalize_enrichment_payload(
            {
                "topic_title": "SQL",
                "topic_description": "desc",
                "suspected_jtbd": ["one", "two"],
                "suspected_pain_points": ["timeout", "bad ux"],
                "suspected_business_value": {"level": "high"},
                "suspected_product_action": True,
                "confidence_note": None,
            }
        )
        self.assertEqual(payload["suspected_jtbd"], "one; two")
        self.assertEqual(payload["suspected_pain_points"], "timeout; bad ux")
        self.assertEqual(payload["suspected_business_value"], '{"level": "high"}')
        self.assertEqual(payload["suspected_product_action"], "True")


if __name__ == "__main__":
    unittest.main()
