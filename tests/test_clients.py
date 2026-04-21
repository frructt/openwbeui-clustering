from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.clients.embedding_client import HashingEmbeddingClient, materialize_embeddings
from src.clients.llm_client import MockLLMClient
from tests.helpers import build_test_config


class ClientTests(unittest.TestCase):
    def test_hashing_embeddings_are_deterministic(self) -> None:
        provider = HashingEmbeddingClient(dimensions=16)
        first = provider.embed_texts(["sql запрос", "excel отчет"])
        second = provider.embed_texts(["sql запрос", "excel отчет"])
        self.assertEqual(first.shape, (2, 16))
        self.assertTrue((first == second).all())

    def test_materialize_embeddings_matches_unit_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            units = pd.DataFrame({"unit_id": ["u1", "u2"], "text": ["sql запрос", "excel отчет"]})
            embeddings = materialize_embeddings(units, config)
            self.assertEqual(embeddings.shape[0], len(units))

    def test_mock_llm_returns_required_keys(self) -> None:
        payload = {"topic_id": 1, "topic_keywords": "sql, запрос", "examples": ["Напиши SQL"]}
        response = MockLLMClient().enrich_topic(payload)
        self.assertIn("topic_title", response)
        self.assertIn("suspected_product_action", response)


if __name__ == "__main__":
    unittest.main()

