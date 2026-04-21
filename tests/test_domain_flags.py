from __future__ import annotations

import unittest

import pandas as pd

from src.domain_flags import apply_domain_flags


class DomainFlagsTests(unittest.TestCase):
    def test_apply_domain_flags_detects_expected_domains(self) -> None:
        texts = pd.Series(
            [
                "Напиши SQL select * from sales",
                "Вот ссылка https://example.com и Traceback с ошибкой",
                "Сделай pptx презентацию по Jira и Confluence",
            ]
        )
        flags = apply_domain_flags(texts)
        self.assertTrue(flags.loc[0, "has_sql"])
        self.assertTrue(flags.loc[1, "has_link"])
        self.assertTrue(flags.loc[1, "has_error_terms"])
        self.assertTrue(flags.loc[2, "has_document_terms"])
        self.assertTrue(flags.loc[2, "has_jira_terms"])
        self.assertTrue(flags.loc[2, "has_confluence_terms"])


if __name__ == "__main__":
    unittest.main()

