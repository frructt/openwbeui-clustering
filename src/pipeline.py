"""Pipeline orchestration."""

from __future__ import annotations

import logging

from src.artifacts import ensure_artifact_directories, write_run_manifest
from src.clients.embedding_client import BaseEmbeddingProvider, materialize_embeddings
from src.clients.llm_client import BaseLLMProvider
from src.config import AppConfig
from src.io_utils import read_json
from src.stages.ingest import run_ingest
from src.stages.llm_enrichment import run_llm_enrichment
from src.stages.preprocess import run_preprocess
from src.stages.reporting import run_reporting
from src.stages.topic_model import run_topic_model
from src.stages.unit_builder import run_unit_builder


logger = logging.getLogger(__name__)


class PipelineRunner:
    """Top-level orchestration wrapper."""

    def __init__(
        self,
        config: AppConfig,
        embedding_provider: BaseEmbeddingProvider | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        self.config = config
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        ensure_artifact_directories(config)

    def ingest(self):
        result = run_ingest(self.config)
        write_run_manifest(self.config, {"last_stage": "ingest", "normalized_rows": len(result)})
        return result

    def preprocess(self):
        result = run_preprocess(self.config)
        write_run_manifest(self.config, {"last_stage": "preprocess", "user_messages": len(result)})
        return result

    def units(self):
        units, mappings = run_unit_builder(self.config)
        write_run_manifest(self.config, {"last_stage": "units", "analysis_units": len(units)})
        return units, mappings

    def embed(self):
        from src.io_utils import read_dataframe

        units = read_dataframe(self.config.artifacts.analysis_units_path)
        embeddings = materialize_embeddings(units, self.config, self.embedding_provider)
        write_run_manifest(self.config, {"last_stage": "embed", "embeddings_shape": list(embeddings.shape)})
        return embeddings

    def topics(self):
        result = run_topic_model(self.config)
        write_run_manifest(
            self.config,
            {"last_stage": "topics", "n_topics": int(result["summary"]["topic_id"].nunique())},
        )
        return result

    def enrich(self):
        result = run_llm_enrichment(self.config, self.llm_provider)
        write_run_manifest(self.config, {"last_stage": "enrich", "llm_topics": len(result)})
        return result

    def report(self):
        ingest_report = None
        try:
            ingest_report = read_json(self.config.artifacts.ingest_report_path)
        except FileNotFoundError:
            logger.warning("Ingest report not found; final markdown report will omit dataset counters")
        result = run_reporting(self.config, ingest_report=ingest_report)
        write_run_manifest(self.config, {"last_stage": "report"})
        return result

    def run_all(self):
        self.ingest()
        self.preprocess()
        self.units()
        self.embed()
        self.topics()
        self.enrich()
        return self.report()

