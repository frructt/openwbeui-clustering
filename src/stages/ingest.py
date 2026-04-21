"""Input loading, normalization, and ingest report generation."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import AppConfig
from src.io_utils import read_table, write_dataframe, write_json
from src.schemas import CANONICAL_COLUMNS, ValidationError


logger = logging.getLogger(__name__)


def _rename_mapping(config: AppConfig) -> dict[str, str]:
    return {
        config.input.time_column: "timestamp",
        config.input.user_column: "user",
        config.input.chat_uuid_column: "chat_uuid",
        config.input.chat_title_column: "chat_title",
        config.input.role_column: "role",
        config.input.message_column: "message",
    }


def normalize_input_frame(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    mapping = _rename_mapping(config)
    missing = [source for source in mapping if source not in df.columns]
    if missing:
        raise ValidationError(f"Missing required input columns: {missing}")

    normalized = df.rename(columns=mapping).copy()
    normalized = normalized[list(mapping.values())]
    normalized.insert(0, "source_row_id", range(len(normalized)))
    normalized["chat_title"] = normalized["chat_title"].fillna("").astype(str)
    normalized["user"] = normalized["user"].fillna("").astype(str).str.strip()
    normalized["chat_uuid"] = normalized["chat_uuid"].fillna("").astype(str).str.strip()
    normalized["role"] = normalized["role"].fillna("").astype(str).str.strip()
    normalized["message"] = normalized["message"].fillna("").astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", format="mixed")
    invalid_ts = normalized["timestamp"].isna()
    if invalid_ts.any():
        invalid_rows = normalized.loc[invalid_ts, "source_row_id"].tolist()[:10]
        raise ValidationError(f"Failed to parse timestamp for rows: {invalid_rows}")

    if normalized["chat_uuid"].eq("").any():
        invalid_rows = normalized.loc[normalized["chat_uuid"].eq(""), "source_row_id"].tolist()[:10]
        raise ValidationError(f"Empty chat_uuid detected for rows: {invalid_rows}")

    normalized = normalized[["source_row_id", *CANONICAL_COLUMNS[1:]]]
    return normalized


def build_ingest_report(df: pd.DataFrame) -> dict[str, int]:
    return {
        "total_rows": int(len(df)),
        "user_rows": int(df["role"].eq("user").sum()),
        "assistant_rows": int(df["role"].eq("assistant").sum()),
        "empty_message_rows": int(df["message"].astype(str).str.strip().eq("").sum()),
        "unique_users": int(df["user"].nunique(dropna=True)),
        "unique_chats": int(df["chat_uuid"].nunique(dropna=True)),
    }


def run_ingest(config: AppConfig) -> pd.DataFrame:
    logger.info("Running ingest stage from %s", config.input.path)
    raw = read_table(
        path=config.input.path,
        file_format=config.input.format,
        encoding=config.input.encoding,
        sheet_name=config.input.sheet_name,
    )
    normalized = normalize_input_frame(raw, config)
    write_dataframe(normalized, config.artifacts.raw_normalized_path)
    write_json(build_ingest_report(normalized), config.artifacts.ingest_report_path)
    logger.info("Ingest completed with %s rows", len(normalized))
    return normalized
