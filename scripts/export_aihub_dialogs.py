#!/usr/bin/env python3
"""Export AI-Hub chat messages from Grafana into the pipeline input format."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv


LOG = logging.getLogger("export_aihub_dialogs")

DEFAULT_GRAFANA_URL = "https://sam-monitoring.bia-tech.ru"
DEFAULT_DATASOURCE_UID = "dey5wt106rvuoc"
GRAFANA_DATASOURCE_TYPE = "grafana-postgresql-datasource"
EXPORT_COLUMNS = ("Время", "Пользователь", "Chat UUID", "Чат", "role", "message")


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    @property
    def start_epoch(self) -> float:
        return self.start.timestamp()

    @property
    def end_epoch(self) -> float:
        return self.end.timestamp()


@dataclass(frozen=True)
class MessageCursor:
    sort_value: str
    message_id: str


@dataclass(slots=True)
class GrafanaRequestError(RuntimeError):
    status_code: int | None = None
    response_body: str = ""

    def __str__(self) -> str:
        if self.status_code is None:
            return "Grafana request failed"
        if _is_recovery_conflict(self.response_body):
            return f"Grafana request failed with HTTP {self.status_code}: replica recovery conflict"
        return f"Grafana request failed with HTTP {self.status_code}"


def parse_time(value: str, now: datetime) -> datetime:
    """Parse Grafana-like relative time or an ISO-8601 value into UTC."""

    if value == "now":
        return now

    relative = re.fullmatch(r"now-(\d+)([smhdw])", value)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        delta = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }[unit]
        return now - delta

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid time {value!r}; use now[-Nunit] or ISO-8601") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def make_time_range(time_from: str, time_to: str) -> TimeRange:
    now = datetime.now(timezone.utc)
    start = parse_time(time_from, now)
    end = parse_time(time_to, now)
    if start >= end:
        raise ValueError("The start of the time range must be before its end")
    return TimeRange(start=start, end=end)


def sql_literal(value: str) -> str:
    """Build a safe PostgreSQL string literal for Grafana rawSql."""

    if "\x00" in value:
        raise ValueError("Filter contains a forbidden NUL character")
    return "'" + value.replace("'", "''") + "'"


def _epoch_sql(expression: str) -> str:
    return f"""CASE
    WHEN {expression}::numeric >= 100000000000000000
        THEN {expression}::numeric / 1000000000.0
    WHEN {expression}::numeric >= 100000000000000
        THEN {expression}::numeric / 1000000.0
    WHEN {expression}::numeric >= 100000000000
        THEN {expression}::numeric / 1000.0
    ELSE {expression}::numeric
END"""


def _numeric_literal(value: str) -> str:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Cursor contains an invalid numeric value") from error
    if not parsed.is_finite():
        raise ValueError("Cursor contains a non-finite numeric value")
    return format(parsed, "f")


def build_sql(
    time_range: TimeRange,
    user: str | None,
    chat: str | None,
    limit: int,
    cursor: MessageCursor | None = None,
) -> str:
    """Build a message-level OpenWebUI query with keyset pagination."""

    if limit < 1:
        raise ValueError("limit must be greater than zero")

    filters = ["TRUE"]
    if user is not None:
        filters.append(f"u.name = {sql_literal(user)}")
    if chat is not None:
        filters.append(f"c.title = {sql_literal(chat)}")
    where = "\n        AND ".join(filters)

    cursor_condition = "TRUE"
    if cursor is not None:
        sort_value = _numeric_literal(cursor.sort_value)
        cursor_condition = f"""(
    message_sort_value > {sort_value}
    OR (
        message_sort_value = {sort_value}
        AND message_id > {sql_literal(cursor.message_id)}
    )
)"""

    message_epoch = _epoch_sql("m.created_at")
    return f"""WITH raw_messages AS (
    SELECT
        m.id::text AS message_id,
        c.id::text AS chat_id,
        c.title AS chat_title,
        u.name AS user_name,
        m.role AS role,
        m.content AS content,
        m.created_at::numeric AS message_sort_value,
        {message_epoch} AS message_epoch
    FROM chat_message m
    JOIN chat c ON m.chat_id = c.id
    JOIN \"user\" u ON c.user_id = u.id
    WHERE
        {where}
)
SELECT
    message_id AS \"message_id\",
    message_sort_value::text AS \"message_sort_value\",
    message_epoch AS \"message_epoch\",
    chat_id AS \"chat_id\",
    chat_title AS \"chat_title\",
    user_name AS \"user_name\",
    role AS \"role\",
    content AS \"content\"
FROM raw_messages
WHERE
    message_epoch >= {time_range.start_epoch:.6f}
    AND message_epoch < {time_range.end_epoch:.6f}
    AND {cursor_condition}
ORDER BY message_sort_value, message_id
LIMIT {int(limit)}
;"""


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
    max_attempts: int = 4,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=dict(headers), method="POST")
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RuntimeError("HTTP API returned a non-object JSON value")
            return decoded
        except HTTPError as error:
            response_body = error.read(2048).decode("utf-8", errors="replace")
            last_error = GrafanaRequestError(
                status_code=error.code,
                response_body=response_body,
            )
            if not _is_retryable_request_error(last_error):
                raise last_error from error
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            last_error = error

        if attempt + 1 < max_attempts:
            delay = 2**attempt + random.uniform(0, 0.5)
            LOG.warning("Grafana request failed; retrying in %.2fs", delay)
            time.sleep(delay)

    if last_error is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Grafana request failed without an error")
    raise last_error


def _is_recovery_conflict(response_body: str) -> bool:
    normalized = response_body.casefold()
    return "sqlstate 40001" in normalized or "conflict with recovery" in normalized


def _is_retryable_request_error(error: GrafanaRequestError) -> bool:
    return error.status_code in {429, 500, 502, 503, 504} or _is_recovery_conflict(error.response_body)


def parse_grafana_rows(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from Grafana table and data-frame response formats."""

    results = response.get("results")
    if not isinstance(results, Mapping):
        raise RuntimeError("Grafana response has no results")
    result = results.get("A")
    if not isinstance(result, Mapping):
        raise RuntimeError("Grafana response has no result A")
    if result.get("error"):
        raise RuntimeError(f"Grafana datasource error: {result['error']}")

    rows: list[dict[str, Any]] = []
    tables = result.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, Mapping):
                continue
            columns = table.get("columns", [])
            names = [
                column.get("text", column.get("name", ""))
                if isinstance(column, Mapping)
                else str(column)
                for column in columns
            ]
            for values in table.get("rows", []):
                if isinstance(values, list):
                    rows.append(dict(zip(names, values)))

    frames = result.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            schema = frame.get("schema")
            data = frame.get("data")
            if not isinstance(schema, Mapping) or not isinstance(data, Mapping):
                continue
            fields = schema.get("fields", [])
            names = [
                field.get("name", "") if isinstance(field, Mapping) else str(field)
                for field in fields
            ]
            values = data.get("values")
            if isinstance(values, list):
                row_count = max(
                    (len(column) for column in values if isinstance(column, list)),
                    default=0,
                )
                for row_index in range(row_count):
                    row: dict[str, Any] = {}
                    for column_index, name in enumerate(names):
                        column = values[column_index] if column_index < len(values) else []
                        row[name] = column[row_index] if row_index < len(column) else None
                    rows.append(row)
                continue

            frame_rows = data.get("rows")
            if isinstance(frame_rows, list):
                for values_row in frame_rows:
                    if isinstance(values_row, list):
                        rows.append(dict(zip(names, values_row)))
    return rows


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None

    numeric: float | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
    elif isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value):
        numeric = float(value)

    if numeric is not None:
        if numeric >= 100000000000000000:
            numeric /= 1000000000.0
        elif numeric >= 100000000000000:
            numeric /= 1000000.0
        elif numeric >= 100000000000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def build_export_frame(
    rows: Iterable[Mapping[str, Any]],
    time_range: TimeRange,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate direct message rows and map them to the ingest input format."""

    result_rows: list[dict[str, Any]] = []
    stats = {
        "messages_received": 0,
        "messages_exported": 0,
        "messages_outside_range": 0,
        "messages_without_timestamp": 0,
        "messages_with_invalid_chat_id": 0,
    }

    for row in rows:
        stats["messages_received"] += 1
        chat_id = str(row.get("chat_id", "")).strip()
        if not chat_id:
            stats["messages_with_invalid_chat_id"] += 1
            continue

        timestamp = parse_timestamp(row.get("message_epoch"))
        if timestamp is None:
            stats["messages_without_timestamp"] += 1
            continue
        if timestamp < time_range.start or timestamp >= time_range.end:
            stats["messages_outside_range"] += 1
            continue
        result_rows.append(
            {
                "Время": timestamp.isoformat().replace("+00:00", "Z"),
                "Пользователь": str(row.get("user_name", "")),
                "Chat UUID": chat_id,
                "Чат": str(row.get("chat_title", "")),
                "role": str(row.get("role", "")),
                "message": content_to_text(row.get("content")),
            }
        )
        stats["messages_exported"] += 1

    return pd.DataFrame(result_rows, columns=EXPORT_COLUMNS), stats


def _cursor_from_row(row: Mapping[str, Any]) -> MessageCursor:
    sort_value = str(row.get("message_sort_value", "")).strip()
    message_id = str(row.get("message_id", "")).strip()
    if not sort_value or not message_id:
        raise RuntimeError("Grafana page is missing a message pagination cursor")
    _numeric_literal(sort_value)
    return MessageCursor(sort_value=sort_value, message_id=message_id)


def fetch_message_rows(
    grafana_url: str,
    datasource_uid: str,
    time_range: TimeRange,
    token: str,
    timeout: float,
    user: str | None,
    chat: str | None,
    page_size: int,
    max_messages: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: MessageCursor | None = None
    while True:
        if max_messages:
            remaining = max_messages - len(rows)
            if remaining <= 0:
                break
            limit = min(page_size, remaining)
        else:
            limit = page_size

        payload = {
            "queries": [
                {
                    "refId": "A",
                    "datasource": {"uid": datasource_uid, "type": GRAFANA_DATASOURCE_TYPE},
                    "rawSql": build_sql(time_range, user, chat, limit, cursor),
                    "format": "table",
                    "intervalMs": 1000,
                    "maxDataPoints": limit,
                }
            ],
            "from": str(int(time_range.start.timestamp() * 1000)),
            "to": str(int(time_range.end.timestamp() * 1000)),
        }
        page = parse_grafana_rows(
            post_json(
                grafana_url.rstrip("/") + "/api/ds/query",
                payload,
                {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "X-Grafana-Org-Id": "1",
                    "User-Agent": "aihub-dialog-exporter/1.0",
                },
                timeout,
            )
        )
        rows.extend(page)
        LOG.info("Fetched page: rows=%d total=%d", len(page), len(rows))
        if len(page) < limit:
            break
        cursor = _cursor_from_row(page[-1])
    return rows


def write_export(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif suffix == ".csv":
        frame.to_csv(output, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Output format must be .parquet or .csv")


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to the dotenv file")
    parser.add_argument("--grafana-url", default=DEFAULT_GRAFANA_URL)
    parser.add_argument("--datasource-uid", default=DEFAULT_DATASOURCE_UID)
    parser.add_argument("--from", dest="time_from", default="now-30d")
    parser.add_argument("--to", dest="time_to", default="now")
    parser.add_argument("--user", default=None)
    parser.add_argument("--chat", default=None)
    parser.add_argument("--page-size", type=int, default=500, help="Messages per Grafana page")
    parser.add_argument("--max-messages", type=int, default=0, help="Total message limit; 0 exports all")
    parser.add_argument(
        "--max-dialogs",
        dest="max_messages",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("data/raw/aihub_dialogs.parquet"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")
    if args.max_messages < 0:
        parser.error("--max-messages cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(args.env_file, override=False)
    token = os.getenv("GRAFANA_TOKEN", "").strip()
    if not token:
        LOG.error("GRAFANA_TOKEN was not found in %s", args.env_file)
        return 2

    try:
        time_range = make_time_range(args.time_from, args.time_to)
        raw_rows = fetch_message_rows(
            args.grafana_url,
            args.datasource_uid,
            time_range,
            token,
            args.timeout,
            args.user,
            args.chat,
            args.page_size,
            args.max_messages,
        )
        exported, stats = build_export_frame(raw_rows, time_range)
        write_export(exported, args.output)
        report_path = args.report or args.output.with_name(f"{args.output.stem}_export_report.json")
        report = {
            "from": time_range.start.isoformat(),
            "to": time_range.end.isoformat(),
            "output": str(args.output),
            **stats,
        }
        write_report(report, report_path)
        LOG.info("Exported %d messages to %s", len(exported), args.output)
        LOG.info("Export report written to %s", report_path)
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        LOG.error("Export failed: %s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
