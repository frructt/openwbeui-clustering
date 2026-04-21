from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.io_utils import write_dataframe
from src.stages.llm_enrichment import run_llm_enrichment
from tests.helpers import build_test_config


class FailingProvider:
    def enrich_topic(self, topic_payload: dict[str, object]) -> dict[str, object]:
        raise ValueError("synthetic parse failure")


class ListProvider:
    def enrich_topic(self, topic_payload: dict[str, object]) -> dict[str, object]:
        return {
            "topic_title": "sql",
            "topic_description": "desc",
            "suspected_jtbd": ["draft query", "debug result"],
            "suspected_pain_points": ["errors", "low confidence"],
            "suspected_business_value": {"level": "high"},
            "suspected_product_action": ["improve prompts", "review routing"],
            "confidence_note": None,
        }


class LLMEnrichmentStageTests(unittest.TestCase):
    def test_llm_enrichment_falls_back_per_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp), llm_enabled=True)
            summary = pd.DataFrame(
                [
                    {
                        "topic_id": 1,
                        "topic_title_auto": "sql",
                        "topic_keywords": "sql, query",
                        "n_units": 5,
                        "n_users": 3,
                        "n_chats": 2,
                    }
                ]
            )
            examples = pd.DataFrame(
                [
                    {
                        "topic_id": 1,
                        "unit_id": "u1",
                        "chat_uuid": "c1",
                        "user": "ivan",
                        "timestamp": "2026-01-01 10:00:00",
                        "text": "Напиши SQL запрос",
                    }
                ]
            )
            write_dataframe(summary, config.artifacts.topic_summary_base_path)
            write_dataframe(examples, config.artifacts.topic_examples_base_path)

            result = run_llm_enrichment(config, provider=FailingProvider())
            self.assertEqual(result.loc[0, "topic_title"], "sql")
            self.assertIn("fallback", str(result.loc[0, "confidence_note"]).lower())

    def test_llm_enrichment_normalizes_nested_values_before_parquet_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp), llm_enabled=True)
            summary = pd.DataFrame(
                [
                    {
                        "topic_id": 2,
                        "topic_title_auto": "analytics",
                        "topic_keywords": "sql, analytics",
                        "n_units": 8,
                        "n_users": 4,
                        "n_chats": 3,
                    }
                ]
            )
            examples = pd.DataFrame(
                [
                    {
                        "topic_id": 2,
                        "unit_id": "u2",
                        "chat_uuid": "c2",
                        "user": "olga",
                        "timestamp": "2026-01-01 11:00:00",
                        "text": "Построй аналитику по SQL",
                    }
                ]
            )
            write_dataframe(summary, config.artifacts.topic_summary_base_path)
            write_dataframe(examples, config.artifacts.topic_examples_base_path)

            result = run_llm_enrichment(config, provider=ListProvider())
            self.assertEqual(result.loc[0, "suspected_jtbd"], "draft query; debug result")
            self.assertEqual(result.loc[0, "suspected_business_value"], '{"level": "high"}')
            stored = pd.read_parquet(config.artifacts.llm_enrichment_path)
            self.assertEqual(stored.loc[0, "suspected_pain_points"], "errors; low confidence")


if __name__ == "__main__":
    unittest.main()
