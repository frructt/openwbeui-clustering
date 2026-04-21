from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stages.ingest import run_ingest
from src.stages.preprocess import preprocess_user_messages
from src.stages.unit_builder import build_analysis_units
from tests.helpers import build_test_config


class UnitBuilderTests(unittest.TestCase):
    def test_message_mode_creates_one_unit_per_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp), units_mode="message")
            normalized = run_ingest(config)
            user_messages = preprocess_user_messages(normalized, config)
            units, mapping = build_analysis_units(user_messages, config)

            self.assertEqual(len(units), len(user_messages))
            self.assertEqual(len(mapping), len(user_messages))
            self.assertTrue((units["n_messages"] == 1).all())
            self.assertIn("modeling_text", units.columns)
            self.assertIn("modeling_len_tokens", units.columns)

    def test_merged_messages_mode_merges_adjacent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp), units_mode="merged_messages")
            normalized = run_ingest(config)
            user_messages = preprocess_user_messages(normalized, config)
            units, mapping = build_analysis_units(user_messages, config)

            self.assertLess(len(units), len(user_messages))
            merged_sql = units[units["chat_uuid"] == "chat-1"]
            self.assertTrue((merged_sql["n_messages"] >= 1).all())
            self.assertEqual(mapping["source_row_id"].nunique(), len(user_messages))
            self.assertTrue((merged_sql["modeling_len_tokens"] > 0).all())


if __name__ == "__main__":
    unittest.main()
