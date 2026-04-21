"""Config loading and typed accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class InputConfig:
    path: str
    format: str = "csv"
    time_column: str = "Время"
    user_column: str = "Пользователь"
    chat_uuid_column: str = "Chat UUID"
    chat_title_column: str = "Чат"
    role_column: str = "role"
    message_column: str = "message"
    encoding: str = "utf-8"
    sheet_name: str | int | None = 0


@dataclass(slots=True)
class ProcessingConfig:
    role_keep: str = "user"
    min_chars: int = 10
    drop_trivial_messages: bool = True
    trivial_messages: list[str] = field(default_factory=list)
    deduplicate: bool = True
    build_modeling_text: bool = True
    modeling_max_tokens: int = 400
    modeling_strip_stopwords: bool = True


@dataclass(slots=True)
class UnitsConfig:
    mode: str = "merged_messages"
    max_messages_per_unit: int = 2
    max_gap_minutes: int = 15


@dataclass(slots=True)
class EmbeddingsConfig:
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    model: str = "Qwen/Qwen3-Embedding-8B"
    batch_size: int = 64
    timeout_sec: int = 120
    max_retries: int = 5
    hash_dimensions: int = 64
    max_text_chars: int | None = 12000
    strip_null_bytes: bool = True


@dataclass(slots=True)
class TopicModelConfig:
    backend: str = "bertopic"
    umap_n_neighbors: int = 15
    umap_n_components: int = 10
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_method: str = "eom"
    vectorizer_ngram_min: int = 1
    vectorizer_ngram_max: int = 2
    vectorizer_min_df: int = 5
    nr_topics: int | None = None
    representative_examples: int = 12
    keyword_top_n: int = 10
    reduce_outliers: bool = True
    reduce_outliers_strategy: str = "embeddings"


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = True
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "dummy"
    model: str = "gpt-oss-120b"
    timeout_sec: int = 120
    max_retries: int = 5
    max_examples_per_topic: int = 12
    temperature: float = 0.0


@dataclass(slots=True)
class ReportingConfig:
    top_n_topics: int = 30
    top_n_examples: int = 15
    generate_figures: bool = True
    topic_growth_periods: int = 4


@dataclass(slots=True)
class ArtifactsConfig:
    raw_normalized_path: str = "data/interim/raw_normalized.parquet"
    ingest_report_path: str = "data/interim/ingest_quality_report.json"
    user_messages_path: str = "data/interim/user_messages.parquet"
    analysis_units_path: str = "data/interim/analysis_units.parquet"
    unit_message_map_path: str = "data/interim/unit_message_map.parquet"
    embeddings_path: str = "data/interim/embeddings.npy"
    embedding_cache_dir: str = "data/interim/embedding_cache"
    topic_model_dir: str = "data/interim/topic_model"
    topic_assignments_path: str = "data/interim/topic_assignments.parquet"
    topic_summary_base_path: str = "data/interim/topic_summary_base.parquet"
    topic_examples_base_path: str = "data/interim/topic_examples_base.parquet"
    topic_domain_breakdown_base_path: str = "data/interim/topic_domain_breakdown_base.parquet"
    topic_trends_base_path: str = "data/interim/topic_trends_by_week_base.parquet"
    llm_enrichment_path: str = "data/interim/topic_llm_enrichment.parquet"
    llm_cache_dir: str = "data/interim/llm_cache"
    run_manifest_path: str = "data/interim/run_manifest.json"
    topic_summary_path: str = "reports/tables/topic_summary.csv"
    topic_examples_path: str = "reports/tables/topic_examples.csv"
    topic_trends_path: str = "reports/tables/topic_trends_by_week.csv"
    topic_domain_breakdown_path: str = "reports/tables/topic_domain_breakdown.csv"
    insight_report_path: str = "reports/insights/insight_report.md"
    topic_sizes_figure_path: str = "reports/figures/topic_sizes.png"
    topic_growth_figure_path: str = "reports/figures/topic_growth.png"


@dataclass(slots=True)
class AppConfig:
    input: InputConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    units: UnitsConfig = field(default_factory=UnitsConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    topic_model: TopicModelConfig = field(default_factory=TopicModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    config_path: Path | None = None


def _build_dataclass(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return dataclass_type(**payload)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = AppConfig(
        input=_build_dataclass(InputConfig, raw.get("input")),
        processing=_build_dataclass(ProcessingConfig, raw.get("processing")),
        units=_build_dataclass(UnitsConfig, raw.get("units")),
        embeddings=_build_dataclass(EmbeddingsConfig, raw.get("embeddings")),
        topic_model=_build_dataclass(TopicModelConfig, raw.get("topic_model")),
        llm=_build_dataclass(LLMConfig, raw.get("llm")),
        reporting=_build_dataclass(ReportingConfig, raw.get("reporting")),
        artifacts=_build_dataclass(ArtifactsConfig, raw.get("artifacts")),
        config_path=config_path,
    )
    return config
