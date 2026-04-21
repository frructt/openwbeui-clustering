from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stages.ingest import run_ingest
from src.stages.preprocess import build_modeling_text, preprocess_user_messages
from tests.helpers import build_test_config


class PreprocessTests(unittest.TestCase):
    def test_preprocess_keeps_only_user_messages_and_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            normalized = run_ingest(config)
            user_messages = preprocess_user_messages(normalized, config)

            self.assertTrue((user_messages["role"] == "user").all())
            self.assertNotIn("спасибо", {text.lower() for text in user_messages["message"].tolist()})
            self.assertIn("message_len_chars", user_messages.columns)
            self.assertIn("week", user_messages.columns)
            self.assertIn("message_for_modeling", user_messages.columns)
            self.assertIn("message_modeling_len_tokens", user_messages.columns)
            self.assertTrue((user_messages["message_len_chars"] >= config.processing.min_chars).all())

    def test_preprocess_deduplicates_repeated_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            normalized = run_ingest(config)
            user_messages = preprocess_user_messages(normalized, config)

            duplicated = user_messages[
                user_messages["message"].str.contains("сводную таблицу в Excel", case=False, na=False)
            ]
            self.assertEqual(len(duplicated), 1)

    def test_build_modeling_text_reduces_structured_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            text = (
                "Traceback error id=12345 path=/tmp/a.json "
                '{"user_id":"550e8400-e29b-41d4-a716-446655440000","url":"https://example.com"}'
            )
            modeled = build_modeling_text(text, config)
            self.assertIn("traceback", modeled)
            self.assertIn("error", modeled)
            self.assertIn("path", modeled)
            self.assertIn("url", modeled)
            self.assertIn("uuid", modeled)
            self.assertNotIn("12345", modeled)


if __name__ == "__main__":
    unittest.main()
