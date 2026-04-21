from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.clients.embedding_client import (
    EmbeddingRequestError,
    HashingEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    build_embedding_cache_key,
    materialize_embeddings,
    prepare_text_for_embedding,
    resolve_embedding_texts,
)
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

    def test_prepare_text_for_embedding_truncates_and_sanitizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            config.embeddings.max_text_chars = 5
            prepared = prepare_text_for_embedding("ab\x00cdefghi", config.embeddings)
            self.assertEqual(prepared, "ab cd")

    def test_embedding_cache_key_depends_on_prepared_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            config.embeddings.max_text_chars = 4
            first = build_embedding_cache_key("abcdef", config.embeddings)
            second = build_embedding_cache_key("abcdZZ", config.embeddings)
            self.assertEqual(first, second)

    def test_openai_client_splits_batch_on_http_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            client = OpenAICompatibleEmbeddingClient(config.embeddings)

            def fake_request(payload):
                batch = payload["input"]
                if len(batch) > 1:
                    raise EmbeddingRequestError("bad request", status_code=400, response_body="too many items in batch")
                return {"data": [{"index": 0, "embedding": [float(len(batch[0]))]}]}

            with mock.patch.object(client, "_request", side_effect=fake_request):
                embeddings = client.embed_texts(["sql", "excel", "jira"])
            self.assertEqual(embeddings.shape, (3, 1))

    def test_resolve_embedding_texts_prefers_modeling_text(self) -> None:
        frame = pd.DataFrame(
            {
                "text": ["raw traceback /tmp/a.json"],
                "modeling_text": ["traceback path"],
            }
        )
        resolved = resolve_embedding_texts(frame)
        self.assertEqual(resolved.iloc[0], "traceback path")


if __name__ == "__main__":
    unittest.main()
