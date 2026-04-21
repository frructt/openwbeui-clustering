# AI Hub Dialogues

Offline pipeline for thematic analysis of OpenWebUI dialogue exports.

## Input format

The default input contract is a CSV with columns:

`Время | Пользователь | Chat UUID | Чат | role | message`

The pipeline normalizes them to the internal schema:

`timestamp | user | chat_uuid | chat_title | role | message`

## Main capabilities

- ingest CSV / Parquet / XLSX exports;
- filter and preprocess `role=user` messages;
- build analysis units in `message` or `merged_messages` mode;
- create embeddings via an OpenAI-compatible endpoint;
- run topic modeling with BERTopic on precomputed embeddings;
- enrich topics with an OpenAI-compatible LLM;
- save reusable artifacts, tables, figures, and a Markdown insight report.

## Model endpoints

Default production-like config assumes:

- embeddings: `Qwen/Qwen3-Embedding-8B` on `http://localhost:8000/v1`
- LLM enrichment: `gpt-oss-120b` on `http://localhost:8001/v1`

These values live in [configs/default.yaml](/Users/agornostaev/Documents/ai-hub-dialouges/configs/default.yaml:1) and can be changed without editing code.

For local smoke runs without model access there is [configs/smoke.yaml](/Users/agornostaev/Documents/ai-hub-dialouges/configs/smoke.yaml:1), which uses deterministic hashing embeddings, a simple topic backend, and a mock LLM provider.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want BERTopic topic modeling, install the full dependency set from `requirements.txt`. The smoke configuration works without real model endpoints, but still expects the Python libraries listed above.

## Usage

Full pipeline:

```bash
python -m src.cli run --config configs/default.yaml
```

Stage-by-stage:

```bash
python -m src.cli ingest --config configs/default.yaml
python -m src.cli preprocess --config configs/default.yaml
python -m src.cli units --config configs/default.yaml
python -m src.cli embed --config configs/default.yaml
python -m src.cli topics --config configs/default.yaml
python -m src.cli enrich --config configs/default.yaml
python -m src.cli report --config configs/default.yaml
```

Smoke run without vLLM endpoints:

```bash
python -m src.cli run --config configs/smoke.yaml
```

## Output artifacts

Main intermediate artifacts:

- `data/interim/raw_normalized.parquet`
- `data/interim/user_messages.parquet`
- `data/interim/analysis_units.parquet`
- `data/interim/unit_message_map.parquet`
- `data/interim/embeddings.npy`
- `data/interim/topic_assignments.parquet`
- `data/interim/topic_llm_enrichment.parquet`

Final outputs:

- `reports/tables/topic_summary.csv`
- `reports/tables/topic_examples.csv`
- `reports/tables/topic_trends_by_week.csv`
- `reports/tables/topic_domain_breakdown.csv`
- `reports/insights/insight_report.md`
- `reports/figures/topic_sizes.png`
- `reports/figures/topic_growth.png`

## Interpreting results

- `topic_summary.csv` is the main table for topic volume, user reach, domain flags, and LLM labels.
- `topic_examples.csv` contains representative examples for manual review.
- `topic_trends_by_week.csv` helps identify growing themes.
- `insight_report.md` is a compact management-friendly summary with top themes, pain signals, and product hypotheses.

## Notes

- `Чат` is treated as an auxiliary signal only.
- `assistant` messages are excluded from topic clustering.
- LLM enrichment works on topics, not on every individual message.
- `merged_messages` mode merges consecutive user turns by time gap and count limit. Semantic shift detection is intentionally left for a later iteration.

