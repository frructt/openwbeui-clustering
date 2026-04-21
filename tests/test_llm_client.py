from __future__ import annotations

import unittest

from src.clients.llm_client import _clean_json_payload


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


if __name__ == "__main__":
    unittest.main()
