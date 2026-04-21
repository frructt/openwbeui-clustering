"""Analysis unit builders for message and merged_messages modes."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.artifacts import stable_text_hash
from src.config import AppConfig
from src.domain_flags import aggregate_flags, flag_names
from src.io_utils import read_dataframe, write_dataframe
from src.schemas import ValidationError


logger = logging.getLogger(__name__)


def _build_unit_record(group: pd.DataFrame, mode: str) -> dict[str, object]:
    ordered = group.sort_values("timestamp")
    source_ids = ordered["source_row_id"].astype(str).tolist()
    text = "\n".join(ordered["message"].tolist())
    flags = aggregate_flags(ordered, flag_names())
    unit_id = f"{mode}-{stable_text_hash('|'.join(source_ids))[:16]}"
    return {
        "unit_id": unit_id,
        "chat_uuid": ordered["chat_uuid"].iloc[0],
        "user": ordered["user"].iloc[0],
        "chat_title": ordered["chat_title"].iloc[0],
        "start_ts": ordered["timestamp"].min(),
        "end_ts": ordered["timestamp"].max(),
        "timestamp": ordered["timestamp"].min(),
        "n_messages": int(len(ordered)),
        "text": text,
        "text_len_chars": len(text),
        "text_len_words": len(text.split()),
        "date": ordered["date"].iloc[0],
        "week": ordered["week"].iloc[0],
        "month": ordered["month"].iloc[0],
        **flags,
    }


def _build_message_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = df.copy()
    units["unit_id"] = units["source_row_id"].map(lambda value: f"message-{int(value):08d}")
    units["start_ts"] = units["timestamp"]
    units["end_ts"] = units["timestamp"]
    units["n_messages"] = 1
    units["text"] = units["message"]
    units["text_len_chars"] = units["message_len_chars"]
    units["text_len_words"] = units["message_len_words"]
    unit_columns = [
        "unit_id",
        "chat_uuid",
        "user",
        "chat_title",
        "start_ts",
        "end_ts",
        "timestamp",
        "n_messages",
        "text",
        "text_len_chars",
        "text_len_words",
        "date",
        "week",
        "month",
        *flag_names(),
    ]
    mappings = units[["unit_id", "source_row_id"]].copy()
    return units[unit_columns].copy(), mappings


def _iter_merge_groups(df: pd.DataFrame, max_gap_minutes: int, max_messages_per_unit: int) -> Iterable[pd.DataFrame]:
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    for (_, _), chat_df in df.groupby(["chat_uuid", "user"], sort=False):
        ordered = chat_df.sort_values("timestamp")
        current_rows: list[pd.Series] = []
        previous_ts = None
        for _, row in ordered.iterrows():
            row_ts = row["timestamp"]
            if not current_rows:
                current_rows = [row]
                previous_ts = row_ts
                continue
            should_flush = (
                row_ts - previous_ts > max_gap
                or len(current_rows) >= max_messages_per_unit
            )
            if should_flush:
                yield pd.DataFrame(current_rows)
                current_rows = [row]
            else:
                current_rows.append(row)
            previous_ts = row_ts
        if current_rows:
            yield pd.DataFrame(current_rows)


def _build_merged_mode(df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for group in _iter_merge_groups(df, config.units.max_gap_minutes, config.units.max_messages_per_unit):
        unit = _build_unit_record(group, mode="merged")
        unit_rows.append(unit)
        for source_row_id in group["source_row_id"].tolist():
            mapping_rows.append({"unit_id": unit["unit_id"], "source_row_id": source_row_id})
    return pd.DataFrame(unit_rows), pd.DataFrame(mapping_rows)


def build_analysis_units(df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.units.mode == "message":
        return _build_message_mode(df)
    if config.units.mode == "merged_messages":
        return _build_merged_mode(df, config)
    raise ValidationError(f"Unsupported units.mode value: {config.units.mode}")


def run_unit_builder(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    messages = read_dataframe(config.artifacts.user_messages_path)
    units, mappings = build_analysis_units(messages, config)
    write_dataframe(units, config.artifacts.analysis_units_path)
    write_dataframe(mappings, config.artifacts.unit_message_map_path)
    logger.info("Built %s analysis units in %s mode", len(units), config.units.mode)
    return units, mappings
