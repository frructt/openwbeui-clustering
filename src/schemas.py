"""Shared schemas and constants."""

from __future__ import annotations

from dataclasses import dataclass


class PipelineError(RuntimeError):
    """Base error raised by the pipeline."""


class ValidationError(PipelineError):
    """Raised when the input data does not satisfy required constraints."""


CANONICAL_COLUMNS = (
    "source_row_id",
    "timestamp",
    "user",
    "chat_uuid",
    "chat_title",
    "role",
    "message",
)

DOMAIN_FLAG_COLUMNS = (
    "has_code",
    "has_sql",
    "has_link",
    "has_excel_terms",
    "has_jira_terms",
    "has_confluence_terms",
    "has_grafana_terms",
    "has_litellm_terms",
    "has_openwebui_terms",
    "has_1c_terms",
    "has_error_terms",
    "has_document_terms",
)

TOPIC_SUMMARY_SHARE_COLUMNS = (
    "has_code_share",
    "has_sql_share",
    "has_excel_share",
    "has_jira_share",
    "has_confluence_share",
    "has_grafana_share",
    "has_litellm_share",
    "has_openwebui_share",
    "has_1c_share",
    "has_error_terms_share",
    "has_document_terms_share",
)


@dataclass(frozen=True)
class StageResult:
    """A small wrapper used by stage functions."""

    stage_name: str
    records: int
    output_path: str | None = None

