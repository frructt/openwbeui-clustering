"""Embedding-based topic modeling stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.io_utils import load_numpy, read_dataframe, write_dataframe, write_json
from src.metrics import (
    build_auto_title,
    build_topic_domain_breakdown,
    build_topic_summary,
    build_topic_trends,
    infer_simple_topic_label,
    select_representative_examples,
    top_keywords_from_texts,
)
from src.schemas import ValidationError


logger = logging.getLogger(__name__)


def _save_model_metadata(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    write_json(payload, target / "metadata.json")


def _simple_topics(units: pd.DataFrame, embeddings: np.ndarray, config: AppConfig) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, str]]:
    working = units.copy().reset_index(drop=True)
    working["_simple_label"] = working.apply(infer_simple_topic_label, axis=1)
    label_order = {label: index for index, label in enumerate(sorted(working["_simple_label"].unique()))}
    working["topic_id"] = working["_simple_label"].map(label_order).astype(int)
    keywords: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    for label, topic_id in label_order.items():
        topic_texts = working.loc[working["topic_id"] == topic_id, "text"].tolist()
        topic_keywords = top_keywords_from_texts(topic_texts, config.topic_model.keyword_top_n)
        keywords[topic_id] = topic_keywords
        titles[topic_id] = build_auto_title(topic_keywords, label)
    _save_model_metadata(
        config.artifacts.topic_model_dir,
        {
            "backend": "simple",
            "label_order": label_order,
            "n_topics": len(label_order),
        },
    )
    return working[["unit_id", "topic_id"]].copy(), keywords, titles


def _bertopic_topics(units: pd.DataFrame, embeddings: np.ndarray, config: AppConfig) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, str]]:
    try:
        import hdbscan
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic dependencies are not installed. Install requirements.txt or switch topic_model.backend to `simple`."
        ) from exc

    vectorizer = CountVectorizer(
        ngram_range=(config.topic_model.vectorizer_ngram_min, config.topic_model.vectorizer_ngram_max),
        min_df=config.topic_model.vectorizer_min_df,
        token_pattern=r"(?u)\b\w+\b",
    )
    umap_model = UMAP(
        n_neighbors=config.topic_model.umap_n_neighbors,
        n_components=config.topic_model.umap_n_components,
        min_dist=config.topic_model.umap_min_dist,
        metric=config.topic_model.umap_metric,
        random_state=42,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.topic_model.hdbscan_min_cluster_size,
        min_samples=config.topic_model.hdbscan_min_samples,
        cluster_selection_method=config.topic_model.hdbscan_cluster_selection_method,
        prediction_data=True,
    )
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=clusterer,
        nr_topics=config.topic_model.nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )
    documents = units["text"].astype(str).tolist()
    topics, _ = topic_model.fit_transform(documents, embeddings)
    assignments = pd.DataFrame({"unit_id": units["unit_id"], "topic_id": topics})

    keywords: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    for topic_id in sorted(assignments["topic_id"].unique()):
        if int(topic_id) == -1:
            keywords[int(topic_id)] = top_keywords_from_texts(
                units.loc[assignments["topic_id"] == topic_id, "text"].tolist(),
                config.topic_model.keyword_top_n,
            )
            titles[int(topic_id)] = "outliers"
            continue
        topic_terms = topic_model.get_topic(int(topic_id)) or []
        topic_keywords = [term for term, _ in topic_terms[: config.topic_model.keyword_top_n]]
        keywords[int(topic_id)] = topic_keywords
        titles[int(topic_id)] = build_auto_title(topic_keywords, f"topic_{int(topic_id)}")

    try:
        topic_model.save(config.artifacts.topic_model_dir, serialization="pickle")
    except Exception as exc:  # pragma: no cover - best effort save
        logger.warning("Failed to serialize BERTopic model; writing metadata instead: %s", exc)
        _save_model_metadata(
            config.artifacts.topic_model_dir,
            {
                "backend": "bertopic",
                "n_topics": int(len(set(topics))),
                "error": str(exc),
            },
        )
    return assignments, keywords, titles


def run_topic_model(config: AppConfig) -> dict[str, pd.DataFrame]:
    units = read_dataframe(config.artifacts.analysis_units_path).reset_index(drop=True)
    embeddings = load_numpy(config.artifacts.embeddings_path)
    if len(units) != len(embeddings):
        raise ValidationError("The number of embeddings does not match the number of analysis units")

    if config.topic_model.backend == "simple":
        assignments, topic_keywords, topic_titles = _simple_topics(units, embeddings, config)
    elif config.topic_model.backend == "bertopic":
        assignments, topic_keywords, topic_titles = _bertopic_topics(units, embeddings, config)
    else:
        raise ValidationError(f"Unsupported topic_model.backend value: {config.topic_model.backend}")

    write_dataframe(assignments, config.artifacts.topic_assignments_path)

    units_with_topics = units.merge(assignments, on="unit_id", how="inner")
    units_with_topics["_embedding_index"] = np.arange(len(units_with_topics))
    summary = build_topic_summary(units_with_topics, topic_keywords, topic_titles)
    domain_breakdown = build_topic_domain_breakdown(units_with_topics)
    trends = build_topic_trends(units_with_topics)
    examples = select_representative_examples(units_with_topics, embeddings, config.topic_model.representative_examples)

    write_dataframe(summary, config.artifacts.topic_summary_base_path)
    write_dataframe(domain_breakdown, config.artifacts.topic_domain_breakdown_base_path)
    write_dataframe(trends, config.artifacts.topic_trends_base_path)
    write_dataframe(examples, config.artifacts.topic_examples_base_path)

    logger.info("Topic modeling produced %s topics", summary["topic_id"].nunique())
    return {
        "assignments": assignments,
        "summary": summary,
        "domain_breakdown": domain_breakdown,
        "trends": trends,
        "examples": examples,
    }

