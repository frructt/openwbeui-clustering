from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline import PipelineRunner
from tests.helpers import build_test_config


class PipelineSmokeTests(unittest.TestCase):
    def test_full_pipeline_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_test_config(Path(tmp), units_mode="merged_messages", topic_backend="simple", llm_enabled=True)
            runner = PipelineRunner(config)
            runner.run_all()

            summary = pd.read_csv(config.artifacts.topic_summary_path)
            examples = pd.read_csv(config.artifacts.topic_examples_path)
            trends = pd.read_csv(config.artifacts.topic_trends_path)

            self.assertFalse(summary.empty)
            self.assertFalse(examples.empty)
            self.assertFalse(trends.empty)
            self.assertIn("topic_title_llm", summary.columns)
            self.assertTrue(Path(config.artifacts.insight_report_path).exists())


if __name__ == "__main__":
    unittest.main()

