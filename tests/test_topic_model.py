from __future__ import annotations

import unittest

from src.stages.topic_model import _resolve_vectorizer_min_df


class TopicModelTests(unittest.TestCase):
    def test_integer_min_df_is_scaled_for_small_corpora(self) -> None:
        resolved = _resolve_vectorizer_min_df(5, 36)
        self.assertIsInstance(resolved, float)
        self.assertGreater(resolved, 0.0)
        self.assertLessEqual(resolved, 1.0)

    def test_min_df_one_stays_one(self) -> None:
        self.assertEqual(_resolve_vectorizer_min_df(1, 10), 1)

    def test_float_min_df_is_bounded(self) -> None:
        resolved = _resolve_vectorizer_min_df(0.0, 5)
        self.assertGreater(resolved, 0.0)
        self.assertLessEqual(resolved, 1.0)


if __name__ == "__main__":
    unittest.main()
