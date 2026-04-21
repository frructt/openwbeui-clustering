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
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ t]\d{1,2}:\d{2}(?::\d{2})?)?\b")
NUM_RE = re.compile(r"\b\d{4,}\b")
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[\w./\\-]+")
MODELING_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_+#.-]{1,}")
MODELING_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "к",
    "ко",
    "из",
    "за",
    "для",
    "не",
    "что",
    "как",
    "это",
    "или",
    "а",
    "но",
    "у",
    "же",
    "ли",
    "мне",
    "нужно",
    "можно",
    "надо",
    "есть",
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "your",
    "have",
    "please",
}


def normalize_message_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text).replace("\r", "\n")).strip()


def build_modeling_text(text: str, config: AppConfig) -> str:
    normalized = normalize_message_text(text).lower()
    if not config.processing.build_modeling_text:
        return normalized

    prepared = normalized
    prepared = URL_RE.sub(" url ", prepared)
    prepared = EMAIL_RE.sub(" email ", prepared)
    prepared = PATH_RE.sub(" path ", prepared)
    prepared = UUID_RE.sub(" uuid ", prepared)
    prepared = HEX_RE.sub(" hex ", prepared)
    prepared = DATE_RE.sub(" date ", prepared)
    prepared = NUM_RE.sub(" num ", prepared)
    prepared = prepared.replace("```", " ")
    prepared = re.sub(r"[{}\[\]<>=\"'`:,;|\\/*]+", " ", prepared)
    prepared = WHITESPACE_RE.sub(" ", prepared).strip()

    tokens: list[str] = []
    for token in MODELING_TOKEN_RE.findall(prepared):
        cleaned = token.strip("._-")
        if len(cleaned) < 2:
            continue
        if cleaned.isdigit():
            continue
        if config.processing.modeling_strip_stopwords and cleaned in MODELING_STOPWORDS:
            continue
        tokens.append(cleaned)

    if config.processing.modeling_max_tokens:
        tokens = tokens[: config.processing.modeling_max_tokens]

    if not tokens:
        fallback_tokens = MODELING_TOKEN_RE.findall(normalized)[: max(config.processing.modeling_max_tokens, 1)]
        tokens = [token for token in fallback_tokens if len(token) >= 2] or ["empty"]
    return " ".join(tokens)


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
    result["message_for_modeling"] = result["message"].map(lambda text: build_modeling_text(text, config))
    result["message_modeling_len_tokens"] = result["message_for_modeling"].str.split().map(len)
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
