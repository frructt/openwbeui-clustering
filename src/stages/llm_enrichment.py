"""LLM enrichment for topic-level summaries."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.artifacts import stable_text_hash
from src.clients.llm_client import BaseLLMProvider, RESPONSE_KEYS, build_llm_provider
from src.config import AppConfig
from src.io_utils import ensure_dir, read_dataframe, read_json, write_dataframe, write_json


logger = logging.getLogger(__name__)


def _topic_cache_key(summary_row: pd.Series, examples: pd.DataFrame) -> str:
    payload = {
        "topic_id": int(summary_row["topic_id"]),
        "topic_keywords": summary_row["topic_keywords"],
        "n_units": int(summary_row["n_units"]),
        "examples": examples["text"].tolist(),
    }
    return stable_text_hash(str(payload))


def run_llm_enrichment(config: AppConfig, provider: BaseLLMProvider | None = None) -> pd.DataFrame:
    summary = read_dataframe(config.artifacts.topic_summary_base_path)
    examples = read_dataframe(config.artifacts.topic_examples_base_path)
    if not config.llm.enabled:
        disabled = summary[["topic_id"]].copy()
        for key in RESPONSE_KEYS:
            disabled[key] = None
        write_dataframe(disabled, config.artifacts.llm_enrichment_path)
        logger.info("LLM enrichment disabled; wrote placeholder output")
        return disabled

    active_provider = build_llm_provider(config, provider)
    cache_dir = ensure_dir(config.artifacts.llm_cache_dir)
    rows: list[dict[str, object]] = []

    for _, summary_row in summary.iterrows():
        topic_id = int(summary_row["topic_id"])
        if topic_id == -1:
            rows.append({"topic_id": topic_id, **{key: None for key in RESPONSE_KEYS}})
            continue
        topic_examples = examples[examples["topic_id"] == topic_id].head(config.llm.max_examples_per_topic)
        payload = {
            "topic_id": topic_id,
            "topic_keywords": summary_row["topic_keywords"],
            "topic_title_auto": summary_row["topic_title_auto"],
            "n_units": int(summary_row["n_units"]),
            "n_users": int(summary_row["n_users"]),
            "n_chats": int(summary_row["n_chats"]),
            "examples": topic_examples["text"].tolist(),
        }
        cache_key = _topic_cache_key(summary_row, topic_examples)
        cache_path = Path(cache_dir) / f"{cache_key}.json"
        if cache_path.exists():
            enrichment = read_json(cache_path)
        else:
            enrichment = active_provider.enrich_topic(payload)
            write_json(enrichment, cache_path)
        rows.append({"topic_id": topic_id, **enrichment})

    result = pd.DataFrame(rows)
    write_dataframe(result, config.artifacts.llm_enrichment_path)
    logger.info("LLM enrichment completed for %s topics", len(result))
    return result

