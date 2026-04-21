"""Filtering and text cleanup."""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.config import AppConfig
from src.domain_flags import apply_domain_flags, flag_names
from src.io_utils import read_dataframe, write_dataframe


logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")


def normalize_message_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text).replace("\r", "\n")).strip()


def preprocess_user_messages(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    logger.info("Preprocessing %s normalized rows", len(df))
    result = df.copy()
    result["message_raw"] = result["message"].astype(str)
    result["message"] = result["message_raw"].map(normalize_message_text)

    result = result[result["role"].str.lower() == config.processing.role_keep.lower()].copy()
    result = result[result["message"].ne("")]
    result = result[result["message"].str.len() >= config.processing.min_chars]

    if config.processing.drop_trivial_messages and config.processing.trivial_messages:
        trivial = {normalize_message_text(item).lower() for item in config.processing.trivial_messages}
        result = result[~result["message"].str.lower().isin(trivial)]

    if config.processing.deduplicate:
        result = result.drop_duplicates(subset=["chat_uuid", "timestamp", "user", "message"], keep="first")

    result["message_len_chars"] = result["message"].str.len()
    result["message_len_words"] = result["message"].str.split().map(len)
    result["date"] = result["timestamp"].dt.strftime("%Y-%m-%d")
    iso_week = result["timestamp"].dt.isocalendar()
    result["week"] = iso_week["year"].astype(str) + "-W" + iso_week["week"].astype(str).str.zfill(2)
    result["month"] = result["timestamp"].dt.strftime("%Y-%m")

    flags = apply_domain_flags(result["message"])
    for column in flag_names():
        result[column] = flags[column].astype(bool)

    return result.reset_index(drop=True)


def run_preprocess(config: AppConfig) -> pd.DataFrame:
    normalized = read_dataframe(config.artifacts.raw_normalized_path)
    user_messages = preprocess_user_messages(normalized, config)
    write_dataframe(user_messages, config.artifacts.user_messages_path)
    logger.info("Preprocess stage kept %s user messages", len(user_messages))
    return user_messages

