"""Embedding providers and cache-aware embedding materialization."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import stable_text_hash
from src.config import AppConfig, EmbeddingsConfig
from src.io_utils import ensure_dir, load_numpy, save_numpy
from src.schemas import ValidationError


logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an array of embeddings with one row per text."""


class OpenAICompatibleEmbeddingClient(BaseEmbeddingProvider):
    """Calls an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Embedding request failed on attempt %s/%s: %s", attempt, self.config.max_retries, exc)
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"Embedding request failed after retries: {last_error}") from last_error

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_rows: list[np.ndarray] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": self.config.model, "input": batch}
            data = self._request(payload)
            embeddings = data.get("data", [])
            if len(embeddings) != len(batch):
                raise ValidationError("Embedding response size does not match the request batch")
            ordered = sorted(embeddings, key=lambda item: item.get("index", 0))
            batch_array = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
            all_rows.append(batch_array)
        if not all_rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(all_rows)


class HashingEmbeddingClient(BaseEmbeddingProvider):
    """Deterministic local embeddings used for tests and smoke runs."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = max(8, dimensions)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [self._embed_single(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)

    def _embed_single(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in str(text).lower().split():
            token_hash = stable_text_hash(token)
            bucket = int(token_hash[:8], 16) % self.dimensions
            sign = 1.0 if int(token_hash[-1], 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm


def build_embedding_provider(config: AppConfig, provider: BaseEmbeddingProvider | None = None) -> BaseEmbeddingProvider:
    if provider is not None:
        return provider
    if config.embeddings.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(config.embeddings)
    if config.embeddings.provider == "hashing":
        return HashingEmbeddingClient(config.embeddings.hash_dimensions)
    raise ValidationError(f"Unsupported embeddings.provider value: {config.embeddings.provider}")


def materialize_embeddings(
    units: pd.DataFrame,
    config: AppConfig,
    provider: BaseEmbeddingProvider | None = None,
) -> np.ndarray:
    if "text" not in units.columns:
        raise ValidationError("Analysis units must contain a `text` column before embeddings are computed")

    active_provider = build_embedding_provider(config, provider)
    cache_dir = ensure_dir(config.artifacts.embedding_cache_dir)
    ordered_rows: list[np.ndarray] = []
    missing_texts: list[str] = []
    missing_hashes: list[str] = []

    for text in units["text"].astype(str):
        text_hash = stable_text_hash(text)
        cache_path = cache_dir / f"{text_hash}.npy"
        if cache_path.exists():
            ordered_rows.append(load_numpy(cache_path))
            continue
        ordered_rows.append(None)  # type: ignore[arg-type]
        missing_texts.append(text)
        missing_hashes.append(text_hash)

    if missing_texts:
        logger.info("Embedding %s uncached units", len(missing_texts))
        generated = active_provider.embed_texts(missing_texts)
        if len(generated) != len(missing_texts):
            raise ValidationError("Embedding provider returned a mismatched number of vectors")
        for text_hash, vector in zip(missing_hashes, generated, strict=True):
            cache_path = Path(cache_dir) / f"{text_hash}.npy"
            save_numpy(np.asarray(vector, dtype=np.float32), cache_path)

    for index, row in enumerate(ordered_rows):
        if row is None:
            text_hash = stable_text_hash(str(units.iloc[index]["text"]))
            cache_path = Path(cache_dir) / f"{text_hash}.npy"
            ordered_rows[index] = load_numpy(cache_path)

    array = np.vstack(ordered_rows).astype(np.float32) if ordered_rows else np.zeros((0, 0), dtype=np.float32)
    save_numpy(array, config.artifacts.embeddings_path)
    logger.info("Saved embeddings to %s", config.artifacts.embeddings_path)
    return array

