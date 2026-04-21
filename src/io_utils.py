"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_table(path: str | Path, file_format: str, encoding: str = "utf-8", sheet_name: str | int | None = 0) -> pd.DataFrame:
    source = Path(path)
    normalized_format = file_format.lower()
    if normalized_format == "csv":
        return pd.read_csv(source, encoding=encoding)
    if normalized_format == "parquet":
        return pd.read_parquet(source)
    if normalized_format in {"xlsx", "xls"}:
        return pd.read_excel(source, sheet_name=sheet_name)
    raise ValueError(f"Unsupported input format: {file_format}")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    destination = ensure_parent_dir(path)
    if destination.suffix == ".parquet":
        df.to_parquet(destination, index=False)
    elif destination.suffix == ".csv":
        df.to_csv(destination, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output format: {destination.suffix}")
    return destination


def read_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported dataframe input format: {source.suffix}")


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = ensure_parent_dir(path)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def write_markdown(content: str, path: str | Path) -> Path:
    destination = ensure_parent_dir(path)
    destination.write_text(content, encoding="utf-8")
    return destination


def save_numpy(array: np.ndarray, path: str | Path) -> Path:
    destination = ensure_parent_dir(path)
    np.save(destination, array)
    return destination


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(Path(path))

