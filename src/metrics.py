"""Topic summary, keyword extraction, and reporting helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.domain_flags import flag_names


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_+#.-]{2,}")
STOPWORDS = {
    "это",
    "как",
    "что",
    "для",
    "или",
    "если",
    "мне",
    "нужно",
    "можно",
    "нужен",
    "есть",
    "надо",
    "with",
    "from",
    "into",
    "that",
    "this",
    "have",
    "please",
    "help",
    "user",
    "assistant",
    "the",
    "and",
}

FLAG_TO_SHARE = {
    "has_code": "has_code_share",
    "has_sql": "has_sql_share",
    "has_excel_terms": "has_excel_share",
    "has_jira_terms": "has_jira_share",
    "has_confluence_terms": "has_confluence_share",
    "has_grafana_terms": "has_grafana_share",
    "has_litellm_terms": "has_litellm_share",
    "has_openwebui_terms": "has_openwebui_share",
    "has_1c_terms": "has_1c_share",
    "has_error_terms": "has_error_terms_share",
    "has_document_terms": "has_document_terms_share",
}


def top_keywords_from_texts(texts: Iterable[str], top_n: int = 10) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(str(text).lower()):
            if token not in STOPWORDS and not token.isdigit():
                counts[token] += 1
    return [token for token, _ in counts.most_common(top_n)]


def build_auto_title(keywords: list[str], fallback: str) -> str:
    if keywords:
        return ", ".join(keywords[:3])
    return fallback


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    safe_denominator = np.clip(matrix_norm * max(vector_norm, 1e-12), 1e-12, None)
    return matrix.dot(vector) / safe_denominator


def select_representative_examples(
    units_with_topics: pd.DataFrame,
    embeddings: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for topic_id, group in units_with_topics.groupby("topic_id", dropna=False):
        indices = group["_embedding_index"].to_numpy()
        if len(indices) == 0:
            continue
        topic_vectors = embeddings[indices]
        centroid = topic_vectors.mean(axis=0)
        similarities = _cosine_similarity(topic_vectors, centroid)
        ranked = group.assign(_similarity=similarities).sort_values("_similarity", ascending=False).head(top_n)
        rows.append(
            ranked.loc[:, ["topic_id", "unit_id", "chat_uuid", "user", "timestamp", "text"]].copy()
        )
    if not rows:
        return pd.DataFrame(columns=["topic_id", "unit_id", "chat_uuid", "user", "timestamp", "text"])
    result = pd.concat(rows, ignore_index=True)
    result["timestamp"] = result["timestamp"].astype(str)
    return result


def build_topic_summary(
    units_with_topics: pd.DataFrame,
    topic_keywords: dict[int, list[str]],
    topic_titles: dict[int, str],
) -> pd.DataFrame:
    total_units = max(len(units_with_topics), 1)
    records: list[dict[str, object]] = []
    for topic_id, group in units_with_topics.groupby("topic_id", dropna=False):
        record: dict[str, object] = {
            "topic_id": int(topic_id),
            "topic_title_auto": topic_titles.get(int(topic_id), f"topic_{int(topic_id)}"),
            "topic_keywords": ", ".join(topic_keywords.get(int(topic_id), [])),
            "n_units": int(len(group)),
            "n_users": int(group["user"].nunique()),
            "n_chats": int(group["chat_uuid"].nunique()),
            "share_of_total": len(group) / total_units,
            "avg_len_chars": float(group["text_len_chars"].mean()),
        }
        for flag, share_name in FLAG_TO_SHARE.items():
            record[share_name] = float(group[flag].mean()) if flag in group else 0.0
        records.append(record)
    summary = pd.DataFrame(records).sort_values(["n_units", "topic_id"], ascending=[False, True]).reset_index(drop=True)
    return summary


def build_topic_domain_breakdown(units_with_topics: pd.DataFrame) -> pd.DataFrame:
    breakdown_records: list[dict[str, object]] = []
    for topic_id, group in units_with_topics.groupby("topic_id", dropna=False):
        record: dict[str, object] = {"topic_id": int(topic_id)}
        for flag, share_name in FLAG_TO_SHARE.items():
            record[share_name] = float(group[flag].mean()) if flag in group else 0.0
        breakdown_records.append(record)
    return pd.DataFrame(breakdown_records).sort_values("topic_id").reset_index(drop=True)


def build_topic_trends(units_with_topics: pd.DataFrame) -> pd.DataFrame:
    trends = (
        units_with_topics.groupby(["week", "topic_id"], dropna=False)
        .size()
        .reset_index(name="n_units")
        .sort_values(["week", "n_units"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return trends


def calculate_growth_score(topic_week_df: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    if topic_week_df.empty:
        return pd.DataFrame(columns=["topic_id", "growth_score"])
    tail = (
        topic_week_df.sort_values("week")
        .groupby("topic_id", group_keys=False)
        .tail(periods)
    )
    scores = []
    for topic_id, group in tail.groupby("topic_id", dropna=False):
        ordered = group.sort_values("week")
        values = ordered["n_units"].tolist()
        if len(values) < 2:
            score = 0.0
        else:
            deltas = [current - previous for previous, current in zip(values, values[1:])]
            score = float(sum(deltas) / max(len(deltas), 1))
        scores.append({"topic_id": int(topic_id), "growth_score": score})
    return pd.DataFrame(scores).sort_values(["growth_score", "topic_id"], ascending=[False, True]).reset_index(drop=True)


def infer_simple_topic_label(row: pd.Series) -> str:
    ordered_labels = [
        ("sql", "has_sql"),
        ("excel", "has_excel_terms"),
        ("jira", "has_jira_terms"),
        ("confluence", "has_confluence_terms"),
        ("grafana", "has_grafana_terms"),
        ("litellm", "has_litellm_terms"),
        ("openwebui", "has_openwebui_terms"),
        ("1c", "has_1c_terms"),
        ("documents", "has_document_terms"),
        ("code", "has_code"),
        ("errors", "has_error_terms"),
    ]
    for label, column in ordered_labels:
        if bool(row.get(column, False)):
            return label
    return "general"


def ensure_all_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in flag_names():
        if column not in result.columns:
            result[column] = False
    return result

