# Repository Guidelines

## Project Structure & Module Organization

This repository contains an offline topic-analysis pipeline for OpenWebUI dialogue exports. Application code is in `src/`: `pipeline.py` coordinates execution, `stages/` contains ingest, preprocessing, embedding, topic modeling, enrichment, and reporting stages, and `clients/` wraps OpenAI-compatible services. YAML run settings live in `configs/`; use `configs/smoke.yaml` for local, deterministic work. Tests live in `tests/`, with shared fixtures and config helpers in `tests/helpers.py`. `scripts/export_aihub_dialogs.py` exports source data from Grafana. Treat `data/` and `reports/` as generated artifacts unless a task explicitly requires updating a checked-in example.

## Build, Test, and Development Commands

Create an isolated Python 3.11+ environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- `python -m pytest` runs the complete test suite.
- `python -m src.cli run --config configs/smoke.yaml` runs the complete pipeline without external model services.
- `python -m src.cli ingest --config configs/default.yaml` runs one stage; replace `ingest` with `preprocess`, `units`, `embed`, `topics`, `enrich`, or `report` as needed.
- `python scripts/export_aihub_dialogs.py --help` documents the separate Grafana exporter.

## Coding Style & Naming Conventions

Write typed Python with four-space indentation and standard-library-first imports. Use `snake_case` for modules, functions, variables, and YAML keys; use `PascalCase` for classes, such as `PipelineRunner`. Keep each pipeline responsibility in its relevant module or `src/stages/` file rather than adding orchestration to the CLI. Match existing explicit validation, error handling, and dataclass/schema patterns. No formatter, linter, or type-checker is currently configured; preserve the surrounding code's style and keep changes focused.

## Testing Guidelines

Tests are discovered by `pytest` and are implemented as `unittest.TestCase` classes. Name files `tests/test_<area>.py` and methods `test_<behavior>`. Use `TemporaryDirectory`, `tests.helpers.build_test_config`, and the sample fixture to keep tests isolated from local data and live endpoints. Add or update a focused regression test for every behavior change, then run `python -m pytest`; use the smoke configuration for pipeline-level coverage.

## Commit, Pull Request, and Configuration Guidance

Use concise, imperative commit subjects consistent with recent history, for example `Improve topic modeling robustness`. Keep commits narrowly scoped. Pull requests should explain the behavior change, identify modified configuration or output artifacts, link the relevant issue when available, and list validation performed. Include screenshots only for report or figure changes. Copy `.env.example` to `.env` for local credentials; never commit tokens, API keys, dialogue exports containing sensitive data, or endpoint-specific secrets.
