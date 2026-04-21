"""Artifact path handling and run metadata helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.io_utils import ensure_dir, ensure_parent_dir, write_json


def stable_text_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def config_digest(config: AppConfig) -> str:
    payload = asdict(config)
    payload["config_path"] = str(config.config_path) if config.config_path else None
    return stable_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def ensure_artifact_directories(config: AppConfig) -> None:
    artifacts = config.artifacts
    file_paths = [
        artifacts.raw_normalized_path,
        artifacts.ingest_report_path,
        artifacts.user_messages_path,
        artifacts.analysis_units_path,
        artifacts.unit_message_map_path,
        artifacts.embeddings_path,
        artifacts.topic_assignments_path,
        artifacts.topic_summary_base_path,
        artifacts.topic_examples_base_path,
        artifacts.topic_domain_breakdown_base_path,
        artifacts.topic_trends_base_path,
        artifacts.llm_enrichment_path,
        artifacts.run_manifest_path,
        artifacts.topic_summary_path,
        artifacts.topic_examples_path,
        artifacts.topic_trends_path,
        artifacts.topic_domain_breakdown_path,
        artifacts.insight_report_path,
        artifacts.topic_sizes_figure_path,
        artifacts.topic_growth_figure_path,
    ]
    for path in file_paths:
        ensure_parent_dir(path)
    for directory in (artifacts.embedding_cache_dir, artifacts.topic_model_dir, artifacts.llm_cache_dir):
        ensure_dir(directory)


def build_run_manifest(config: AppConfig) -> dict[str, Any]:
    return {
        "config_path": str(config.config_path) if config.config_path else None,
        "config_digest": config_digest(config),
        "input_path": config.input.path,
        "input_format": config.input.format,
        "embedding_provider": config.embeddings.provider,
        "embedding_model": config.embeddings.model,
        "topic_backend": config.topic_model.backend,
        "llm_enabled": config.llm.enabled,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model if config.llm.enabled else None,
    }


def write_run_manifest(config: AppConfig, extra: dict[str, Any] | None = None) -> Path:
    manifest = build_run_manifest(config)
    if extra:
        manifest.update(extra)
    return write_json(manifest, config.artifacts.run_manifest_path)
