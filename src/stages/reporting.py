"""Final report and figure generation."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import AppConfig
from src.io_utils import read_dataframe, write_dataframe, write_markdown
from src.metrics import calculate_growth_score


logger = logging.getLogger(__name__)


def _topic_display_title(row: pd.Series) -> str:
    llm_title = row.get("topic_title_llm")
    if pd.notna(llm_title) and str(llm_title).strip():
        return str(llm_title)
    return str(row.get("topic_title_auto", f"topic_{row.get('topic_id')}"))


def _render_markdown_report(
    summary: pd.DataFrame,
    trends: pd.DataFrame,
    examples: pd.DataFrame,
    ingest_report: dict[str, object] | None,
    config: AppConfig,
) -> str:
    top_by_volume = summary.head(config.reporting.top_n_topics)
    top_by_users = summary.sort_values("n_users", ascending=False).head(config.reporting.top_n_topics)
    growth = calculate_growth_score(trends, config.reporting.topic_growth_periods).head(config.reporting.top_n_topics)
    error_topics = summary.sort_values("has_error_terms_share", ascending=False).head(config.reporting.top_n_topics)

    lines = [
        "# Insight Report",
        "",
        "## Dataset",
        "",
    ]
    if ingest_report:
        lines.extend(
            [
                f"- Total input rows: {ingest_report.get('total_rows', 'n/a')}",
                f"- User rows: {ingest_report.get('user_rows', 'n/a')}",
                f"- Assistant rows: {ingest_report.get('assistant_rows', 'n/a')}",
                f"- Unique users: {ingest_report.get('unique_users', 'n/a')}",
                f"- Unique chats: {ingest_report.get('unique_chats', 'n/a')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Pipeline Result",
            "",
            f"- Analysis units: {int(summary['n_units'].sum()) if not summary.empty else 0}",
            f"- Topics: {int(summary['topic_id'].nunique()) if not summary.empty else 0}",
            "",
            "## Top Topics By Volume",
            "",
        ]
    )
    for _, row in top_by_volume.iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])}: {_topic_display_title(row)} "
            f"(units={int(row['n_units'])}, users={int(row['n_users'])}, keywords={row['topic_keywords']})"
        )

    lines.extend(["", "## Top Topics By User Reach", ""])
    for _, row in top_by_users.iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])}: {_topic_display_title(row)} "
            f"(users={int(row['n_users'])}, units={int(row['n_units'])})"
        )

    lines.extend(["", "## Fastest Growing Topics", ""])
    for _, row in growth.iterrows():
        match = summary[summary["topic_id"] == row["topic_id"]]
        if match.empty:
            continue
        topic = match.iloc[0]
        lines.append(
            f"- Topic {int(row['topic_id'])}: {_topic_display_title(topic)} "
            f"(growth_score={row['growth_score']:.2f})"
        )

    lines.extend(["", "## Topics With The Highest Error Signal", ""])
    for _, row in error_topics.iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])}: {_topic_display_title(row)} "
            f"(error_share={row['has_error_terms_share']:.2%})"
        )

    domain_columns = [
        ("has_code_share", "Code"),
        ("has_sql_share", "SQL"),
        ("has_excel_share", "Excel"),
        ("has_jira_share", "Jira"),
        ("has_confluence_share", "Confluence"),
        ("has_grafana_share", "Grafana"),
        ("has_litellm_share", "LiteLLM"),
        ("has_openwebui_share", "OpenWebUI"),
    ]
    lines.extend(["", "## Domain Highlights", ""])
    for column, label in domain_columns:
        if column not in summary.columns:
            continue
        topic = summary.sort_values(column, ascending=False).head(1)
        if topic.empty or float(topic.iloc[0][column]) <= 0:
            continue
        row = topic.iloc[0]
        lines.append(
            f"- {label}: topic {int(row['topic_id'])} ({_topic_display_title(row)}) "
            f"with share {row[column]:.2%}"
        )

    lines.extend(["", "## Product Hypotheses", ""])
    enriched = summary[summary["suspected_product_action"].notna()]
    for _, row in enriched.head(config.reporting.top_n_topics).iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])}: {row['suspected_product_action']} "
            f"(JTBD: {row['suspected_jtbd']})"
        )

    lines.extend(["", "## Representative Examples", ""])
    for _, row in examples.head(config.reporting.top_n_examples).iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])} | {row['user']} | {row['timestamp']}: {row['text']}"
        )
    return "\n".join(lines) + "\n"


def _write_figures(summary: pd.DataFrame, growth: pd.DataFrame, config: AppConfig) -> None:
    if not config.reporting.generate_figures:
        logger.info("Figure generation disabled")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local env
        logger.warning("matplotlib is not installed; skipping figures: %s", exc)
        return

    top_sizes = summary.head(min(15, len(summary)))
    if not top_sizes.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_sizes["topic_id"].astype(str), top_sizes["n_units"])
        ax.set_title("Topic Sizes")
        ax.set_xlabel("Analysis units")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(config.artifacts.topic_sizes_figure_path, dpi=150)
        plt.close(fig)

    if not growth.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(growth["topic_id"].astype(str), growth["growth_score"])
        ax.set_title("Topic Growth Score")
        ax.set_xlabel("Average weekly delta")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(config.artifacts.topic_growth_figure_path, dpi=150)
        plt.close(fig)


def run_reporting(config: AppConfig, ingest_report: dict[str, object] | None = None) -> dict[str, pd.DataFrame]:
    summary = read_dataframe(config.artifacts.topic_summary_base_path)
    examples = read_dataframe(config.artifacts.topic_examples_base_path)
    trends = read_dataframe(config.artifacts.topic_trends_base_path)
    domain_breakdown = read_dataframe(config.artifacts.topic_domain_breakdown_base_path)
    try:
        llm_enrichment = read_dataframe(config.artifacts.llm_enrichment_path)
    except FileNotFoundError:
        llm_enrichment = summary[["topic_id"]].copy()
        llm_enrichment["topic_title"] = None
        llm_enrichment["topic_description"] = None
        llm_enrichment["suspected_jtbd"] = None
        llm_enrichment["suspected_pain_points"] = None
        llm_enrichment["suspected_business_value"] = None
        llm_enrichment["suspected_product_action"] = None
        llm_enrichment["confidence_note"] = None
        logger.warning("LLM enrichment artifact not found; continuing with auto-generated topic titles only")

    final_summary = summary.merge(llm_enrichment, on="topic_id", how="left")
    final_summary["topic_title_llm"] = final_summary["topic_title"].fillna(final_summary["topic_title_auto"])
    final_summary = final_summary.drop(columns=["topic_title"], errors="ignore")

    final_examples = examples.loc[:, ["topic_id", "chat_uuid", "user", "timestamp", "text"]].copy()
    final_examples["timestamp"] = final_examples["timestamp"].astype(str)

    write_dataframe(final_summary, config.artifacts.topic_summary_path)
    write_dataframe(final_examples, config.artifacts.topic_examples_path)
    write_dataframe(trends, config.artifacts.topic_trends_path)
    write_dataframe(domain_breakdown, config.artifacts.topic_domain_breakdown_path)

    growth = calculate_growth_score(trends, config.reporting.topic_growth_periods)
    markdown = _render_markdown_report(final_summary, trends, final_examples, ingest_report, config)
    write_markdown(markdown, config.artifacts.insight_report_path)
    _write_figures(final_summary, growth, config)

    logger.info("Reporting artifacts were written to reports/")
    return {
        "summary": final_summary,
        "examples": final_examples,
        "trends": trends,
        "domain_breakdown": domain_breakdown,
    }
