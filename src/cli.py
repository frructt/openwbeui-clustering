"""Command line interface."""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.logging_utils import configure_logging
from src.pipeline import PipelineRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline topic analysis pipeline for AI hub dialogues")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "ingest", "preprocess", "units", "embed", "topics", "enrich", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    config = load_config(args.config)
    runner = PipelineRunner(config)

    if args.command == "run":
        runner.run_all()
    elif args.command == "ingest":
        runner.ingest()
    elif args.command == "preprocess":
        runner.preprocess()
    elif args.command == "units":
        runner.units()
    elif args.command == "embed":
        runner.embed()
    elif args.command == "topics":
        runner.topics()
    elif args.command == "enrich":
        runner.enrich()
    elif args.command == "report":
        runner.report()
    else:  # pragma: no cover - guarded by argparse
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
