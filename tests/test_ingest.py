from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.schemas import ValidationError
from src.stages.ingest import build_ingest_report, normalize_input_frame, run_ingest
from tests.helpers import FIXTURE_PATH, build_test_config


class IngestStageTests(unittest.TestCase):
    def test_run_ingest_normalizes_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            frame = run_ingest(config)
            self.assertEqual(
                list(frame.columns),
                ["source_row_id", "timestamp", "user", "chat_uuid", "chat_title", "role", "message"],
            )
            report = build_ingest_report(frame)
            self.assertEqual(report["total_rows"], 13)
            self.assertEqual(report["user_rows"], 12)

    def test_normalize_input_frame_raises_on_bad_timestamp(self) -> None:
        bad = pd.read_csv(FIXTURE_PATH)
        bad.loc[0, "Время"] = "not-a-date"
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            with self.assertRaises(ValidationError):
                normalize_input_frame(bad, config)

    def test_normalize_input_frame_raises_on_missing_required_column(self) -> None:
        bad = pd.read_csv(FIXTURE_PATH).drop(columns=["message"])
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp))
            with self.assertRaises(ValidationError):
                normalize_input_frame(bad, config)


if __name__ == "__main__":
    unittest.main()

