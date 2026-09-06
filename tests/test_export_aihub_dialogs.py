from __future__ import annotations

import io
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pandas as pd

from src.config import AppConfig, InputConfig
from src.stages.ingest import normalize_input_frame
from scripts.export_aihub_dialogs import (
    EXPORT_COLUMNS,
    GrafanaRequestError,
    MessageCursor,
    TimeRange,
    build_export_frame,
    build_sql,
    fetch_message_rows,
    main,
    parse_grafana_rows,
    post_json,
    write_export,
)


def message_row(message_id: str, sort_value: str, epoch: float) -> dict[str, object]:
    return {
        "message_id": message_id,
        "message_sort_value": sort_value,
        "message_epoch": epoch,
        "chat_id": "chat-1",
        "chat_title": "Title",
        "user_name": "User",
        "role": "user",
        "content": "Text",
    }


def grafana_table_response(rows: list[dict[str, object]]) -> dict[str, object]:
    columns = [
        "message_id",
        "message_sort_value",
        "message_epoch",
        "chat_id",
        "chat_title",
        "user_name",
        "role",
        "content",
    ]
    return {
        "results": {
            "A": {
                "tables": [
                    {
                        "columns": [{"text": column} for column in columns],
                        "rows": [[row.get(column) for column in columns] for row in rows],
                    }
                ]
            }
        }
    }


class GrafanaExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time_range = TimeRange(
            start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            end=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )

    def test_build_sql_uses_chat_message_and_keyset_cursor(self) -> None:
        query = build_sql(
            self.time_range,
            "O'Reilly",
            "chat title",
            limit=10,
            cursor=MessageCursor(sort_value="1780272000", message_id="message-2"),
        )

        self.assertIn("FROM chat_message m", query)
        self.assertIn("u.name = 'O''Reilly'", query)
        self.assertIn("c.title = 'chat title'", query)
        self.assertIn("message_sort_value > 1780272000", query)
        self.assertIn("message_id > 'message-2'", query)
        self.assertIn("ORDER BY message_sort_value, message_id", query)
        self.assertNotIn("OFFSET", query)
        self.assertNotIn("jsonb_array_elements", query)

    def test_parse_grafana_table_and_frame_formats(self) -> None:
        table_rows = parse_grafana_rows(
            {"results": {"A": {"tables": [{"columns": [{"text": "message_id"}], "rows": [["m1"]]}]}}}
        )
        frame_rows = parse_grafana_rows(
            {
                "results": {
                    "A": {
                        "frames": [
                            {
                                "schema": {"fields": [{"name": "message_id"}]},
                                "data": {"values": [["m2"]]},
                            }
                        ]
                    }
                }
            }
        )

        self.assertEqual(table_rows, [{"message_id": "m1"}])
        self.assertEqual(frame_rows, [{"message_id": "m2"}])

    def test_build_export_frame_keeps_only_timestamped_messages_in_range(self) -> None:
        rows = [
            message_row("m1", "1780272000", 1780272000),
            message_row("m2", "1782864000", 1782864000),
            {**message_row("m3", "", 0), "message_epoch": None},
        ]

        exported, stats = build_export_frame(rows, self.time_range)

        self.assertEqual(list(exported.columns), list(EXPORT_COLUMNS))
        self.assertEqual(len(exported), 1)
        self.assertEqual(stats["messages_received"], 3)
        self.assertEqual(stats["messages_exported"], 1)
        self.assertEqual(stats["messages_outside_range"], 1)
        self.assertEqual(stats["messages_without_timestamp"], 1)

    def test_fetch_message_rows_uses_keyset_without_duplicates(self) -> None:
        first_page = [
            message_row("m1", "1780272000", 1780272000),
            message_row("m2", "1780272000", 1780272000),
        ]
        second_page = [message_row("m3", "1780272001", 1780272001)]
        responses = [grafana_table_response(first_page), grafana_table_response(second_page)]

        with patch("scripts.export_aihub_dialogs.post_json", side_effect=responses) as post:
            rows = fetch_message_rows(
                "https://grafana.example",
                "datasource",
                self.time_range,
                "token",
                timeout=1,
                user=None,
                chat=None,
                page_size=2,
                max_messages=0,
            )

        self.assertEqual([row["message_id"] for row in rows], ["m1", "m2", "m3"])
        self.assertEqual(post.call_count, 2)
        first_sql = post.call_args_list[0].args[1]["queries"][0]["rawSql"]
        second_sql = post.call_args_list[1].args[1]["queries"][0]["rawSql"]
        self.assertIn("AND TRUE", first_sql)
        self.assertIn("message_id > 'm2'", second_sql)

    def test_post_json_retries_replica_recovery_error(self) -> None:
        recovery_error = HTTPError(
            "https://grafana.example",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"SQLSTATE 40001: conflict with recovery"}'),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"

        with patch("scripts.export_aihub_dialogs.urlopen", side_effect=[recovery_error, response]) as open_mock, patch(
            "scripts.export_aihub_dialogs.random.uniform", return_value=0
        ), patch("scripts.export_aihub_dialogs.time.sleep") as sleep:
            payload = post_json("https://grafana.example", {}, {}, timeout=1)

        self.assertEqual(payload, {})
        self.assertEqual(open_mock.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_post_json_does_not_retry_permanent_http_400(self) -> None:
        permanent_error = HTTPError(
            "https://grafana.example",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"syntax error"}'),
        )

        with patch("scripts.export_aihub_dialogs.urlopen", side_effect=permanent_error) as open_mock:
            with self.assertRaises(GrafanaRequestError):
                post_json("https://grafana.example", {}, {}, timeout=1)

        self.assertEqual(open_mock.call_count, 1)

    def test_write_export_supports_parquet_and_csv(self) -> None:
        frame = pd.DataFrame(
            [["2026-06-01T00:00:00Z", "User", "chat-1", "Title", "user", "Text"]],
            columns=EXPORT_COLUMNS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "export.parquet"
            csv_path = Path(tmp) / "export.csv"
            write_export(frame, parquet_path)
            write_export(frame, csv_path)

            self.assertEqual(list(pd.read_parquet(parquet_path).columns), list(EXPORT_COLUMNS))
            self.assertEqual(list(pd.read_csv(csv_path, encoding="utf-8-sig").columns), list(EXPORT_COLUMNS))

    def test_exported_rows_are_compatible_with_ingest(self) -> None:
        exported, _ = build_export_frame(
            [message_row("m1", "1780272000", 1780272000)],
            self.time_range,
        )

        normalized = normalize_input_frame(exported, AppConfig(input=InputConfig(path="unused")))

        self.assertEqual(
            list(normalized.columns),
            ["source_row_id", "timestamp", "user", "chat_uuid", "chat_title", "role", "message"],
        )

    def test_main_reads_grafana_token_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GRAFANA_TOKEN=dotenv-token\n", encoding="utf-8")
            output_path = Path(tmp) / "export.parquet"
            report_path = Path(tmp) / "report.json"
            captured_token: list[str] = []

            def fake_fetch(*args, **kwargs):
                captured_token.append(args[3])
                return []

            with patch.dict(os.environ, {}, clear=True), patch(
                "scripts.export_aihub_dialogs.fetch_message_rows", side_effect=fake_fetch
            ), patch("scripts.export_aihub_dialogs.write_export"), patch(
                "scripts.export_aihub_dialogs.write_report"
            ):
                result = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--from",
                        "2026-06-01",
                        "--to",
                        "2026-07-01",
                        "--output",
                        str(output_path),
                        "--report",
                        str(report_path),
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(captured_token, ["dotenv-token"])


if __name__ == "__main__":
    unittest.main()
