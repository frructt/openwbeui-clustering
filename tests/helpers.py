"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

from src.config import (
    AppConfig,
    ArtifactsConfig,
    EmbeddingsConfig,
    InputConfig,
    LLMConfig,
    ProcessingConfig,
    ReportingConfig,
    TopicModelConfig,
    UnitsConfig,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_dialogs.csv"


def build_test_config(
    base_dir: Path,
    *,
    units_mode: str = "message",
    topic_backend: str = "simple",
    llm_enabled: bool = True,
) -> AppConfig:
    artifacts = ArtifactsConfig(
        raw_normalized_path=str(base_dir / "data" / "interim" / "raw_normalized.parquet"),
        ingest_report_path=str(base_dir / "data" / "interim" / "ingest_quality_report.json"),
        user_messages_path=str(base_dir / "data" / "interim" / "user_messages.parquet"),
        analysis_units_path=str(base_dir / "data" / "interim" / "analysis_units.parquet"),
        unit_message_map_path=str(base_dir / "data" / "interim" / "unit_message_map.parquet"),
        embeddings_path=str(base_dir / "data" / "interim" / "embeddings.npy"),
        embedding_cache_dir=str(base_dir / "data" / "interim" / "embedding_cache"),
        topic_model_dir=str(base_dir / "data" / "interim" / "topic_model"),
        topic_assignments_path=str(base_dir / "data" / "interim" / "topic_assignments.parquet"),
        topic_summary_base_path=str(base_dir / "data" / "interim" / "topic_summary_base.parquet"),
        topic_examples_base_path=str(base_dir / "data" / "interim" / "topic_examples_base.parquet"),
        topic_domain_breakdown_base_path=str(base_dir / "data" / "interim" / "topic_domain_breakdown_base.parquet"),
        topic_trends_base_path=str(base_dir / "data" / "interim" / "topic_trends_by_week_base.parquet"),
        llm_enrichment_path=str(base_dir / "data" / "interim" / "topic_llm_enrichment.parquet"),
        llm_cache_dir=str(base_dir / "data" / "interim" / "llm_cache"),
        run_manifest_path=str(base_dir / "data" / "interim" / "run_manifest.json"),
        topic_summary_path=str(base_dir / "reports" / "tables" / "topic_summary.csv"),
        topic_examples_path=str(base_dir / "reports" / "tables" / "topic_examples.csv"),
        topic_trends_path=str(base_dir / "reports" / "tables" / "topic_trends_by_week.csv"),
        topic_domain_breakdown_path=str(base_dir / "reports" / "tables" / "topic_domain_breakdown.csv"),
        insight_report_path=str(base_dir / "reports" / "insights" / "insight_report.md"),
        topic_sizes_figure_path=str(base_dir / "reports" / "figures" / "topic_sizes.png"),
        topic_growth_figure_path=str(base_dir / "reports" / "figures" / "topic_growth.png"),
    )
    return AppConfig(
        input=InputConfig(path=str(FIXTURE_PATH), format="csv"),
        processing=ProcessingConfig(
            role_keep="user",
            min_chars=5,
            drop_trivial_messages=True,
            trivial_messages=["ок", "спасибо"],
            deduplicate=True,
        ),
        units=UnitsConfig(mode=units_mode, max_messages_per_unit=2, max_gap_minutes=20),
        embeddings=EmbeddingsConfig(
            provider="hashing",
            model="hashing-test",
            batch_size=16,
            timeout_sec=3,
            max_retries=1,
            hash_dimensions=32,
        ),
        topic_model=TopicModelConfig(
            backend=topic_backend,
            vectorizer_min_df=1,
            representative_examples=3,
            keyword_top_n=5,
            hdbscan_min_cluster_size=2,
            hdbscan_min_samples=1,
        ),
        llm=LLMConfig(
            enabled=llm_enabled,
            provider="mock",
            model="mock-llm",
            timeout_sec=3,
            max_retries=1,
            max_examples_per_topic=3,
            temperature=0.0,
        ),
        reporting=ReportingConfig(top_n_topics=10, top_n_examples=3, generate_figures=False, topic_growth_periods=2),
        artifacts=artifacts,
    )

