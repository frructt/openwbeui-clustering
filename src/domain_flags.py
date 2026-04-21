"""Domain flag heuristics and regex registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.schemas import DOMAIN_FLAG_COLUMNS


@dataclass(frozen=True)
class FlagSpec:
    name: str
    pattern: str


FLAG_SPECS: tuple[FlagSpec, ...] = (
    FlagSpec("has_code", r"```|def\s+\w+\(|class\s+\w+|function\s+\w+|import\s+\w+|console\.log|Traceback|Exception|<[^>]+>|SELECT\s+.+\s+FROM"),
    FlagSpec("has_sql", r"\bselect\b|\bfrom\b|\bwhere\b|\bjoin\b|\bgroup by\b|\border by\b|\bwith\b|\binsert into\b|\bupdate\b|\bdelete from\b|\bsql\b"),
    FlagSpec("has_link", r"https?://|www\."),
    FlagSpec("has_excel_terms", r"\bexcel\b|\bxlsx\b|\bcsv\b|сводн\w+|впр|vlookup|power query|power pivot|формул\w+"),
    FlagSpec("has_jira_terms", r"\bjira\b|тикет\w+|issue\b|epic\b|story points?\b|backlog\b"),
    FlagSpec("has_confluence_terms", r"\bconfluence\b|конфлюенс|wiki\b|knowledge base"),
    FlagSpec("has_grafana_terms", r"\bgrafana\b|дашборд\w+|panel\b|prometheus\b|loki\b"),
    FlagSpec("has_litellm_terms", r"\blitellm\b|router\b.+model|fallbacks?\b"),
    FlagSpec("has_openwebui_terms", r"\bopenwebui\b|open webui|чат-интерфейс|webui\b"),
    FlagSpec("has_1c_terms", r"\b1c\b|1с|предприятие\b|конфигураци\w+\s+1с"),
    FlagSpec("has_error_terms", r"ошибк\w+|не работает|не получается|проблем\w+|доступ\w+ нет|exception|failed|traceback|сломал\w+"),
    FlagSpec("has_document_terms", r"документ\w+|письм\w+|презентаци\w+|pptx\b|docx\b|word\b|pdf\b|слайд\w+"),
)


def flag_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in FLAG_SPECS)


def validate_flag_registry() -> None:
    registry = flag_names()
    missing = sorted(set(DOMAIN_FLAG_COLUMNS) - set(registry))
    if missing:
        raise ValueError(f"Missing domain flag specs: {missing}")


def apply_domain_flags(texts: pd.Series) -> pd.DataFrame:
    validate_flag_registry()
    normalized = texts.fillna("").astype(str)
    result: dict[str, pd.Series] = {}
    for spec in FLAG_SPECS:
        result[spec.name] = normalized.str.contains(spec.pattern, case=False, regex=True, na=False)
    return pd.DataFrame(result, index=texts.index)


def aggregate_flags(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> dict[str, bool]:
    selected = list(columns or flag_names())
    return {column: bool(frame[column].any()) for column in selected}

