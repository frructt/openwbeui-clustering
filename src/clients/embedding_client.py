"""Embedding providers and cache-aware embedding materialization."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import stable_text_hash
from src.config import AppConfig, EmbeddingsConfig
from src.io_utils import ensure_dir, load_numpy, save_numpy
from src.schemas import ValidationError


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingRequestError(RuntimeError):
    """Structured embedding request failure."""

    message: str
    status_code: int | None = None
    response_body: str = ""

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status_code={self.status_code})"


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an array of embeddings with one row per text."""


class OpenAICompatibleEmbeddingClient(BaseEmbeddingProvider):
    """Calls an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config

    @staticmethod
    def _read_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            payload = exc.read()
        except Exception:
            return ""
        if not payload:
            return ""
        return payload.decode("utf-8", errors="replace")

    def _is_retryable_status(self, status_code: int | None) -> bool:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

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
            except urllib.error.HTTPError as exc:
                response_body = self._read_error_body(exc)
                error = EmbeddingRequestError(
                    message=str(exc),
                    status_code=exc.code,
                    response_body=response_body,
                )
                last_error = error
                if response_body:
                    logger.warning(
                        "Embedding request failed on attempt %s/%s: %s | response=%s",
                        attempt,
                        self.config.max_retries,
                        error,
                        response_body[:500],
                    )
                else:
                    logger.warning(
                        "Embedding request failed on attempt %s/%s: %s",
                        attempt,
                        self.config.max_retries,
                        error,
                    )
                if not self._is_retryable_status(exc.code) or attempt == self.config.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Embedding request failed on attempt %s/%s: %s", attempt, self.config.max_retries, exc)
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise EmbeddingRequestError(f"Embedding request failed after retries: {last_error}") from last_error

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        payload = {"model": self.config.model, "input": batch}
        data = self._request(payload)
        embeddings = data.get("data", [])
        if len(embeddings) != len(batch):
            raise ValidationError("Embedding response size does not match the request batch")
        ordered = sorted(embeddings, key=lambda item: item.get("index", 0))
        return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)

    def _embed_batch_with_fallback(self, batch: list[str], batch_start: int, total_texts: int) -> np.ndarray:
        try:
            return self._embed_batch(batch)
        except EmbeddingRequestError as exc:
            if exc.status_code == 400 and len(batch) > 1:
                midpoint = len(batch) // 2
                logger.warning(
                    "Embedding batch %s-%s/%s failed with 400; splitting it into chunks of %s and %s items",
                    batch_start + 1,
                    batch_start + len(batch),
                    total_texts,
                    midpoint,
                    len(batch) - midpoint,
                )
                left = self._embed_batch_with_fallback(batch[:midpoint], batch_start, total_texts)
                right = self._embed_batch_with_fallback(batch[midpoint:], batch_start + midpoint, total_texts)
                return np.vstack([left, right])
            if exc.status_code == 400 and len(batch) == 1:
                preview = batch[0][:240].replace("\n", " ")
                raise ValidationError(
                    "Embedding failed for a single text. "
                    f"chars={len(batch[0])}, preview={preview!r}, "
                    f"server_response={exc.response_body[:1000] or str(exc)}"
                ) from exc
            raise

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_rows: list[np.ndarray] = []
        batch_size = max(1, self.config.batch_size)
        total_texts = len(texts)
        for start in range(0, total_texts, batch_size):
            batch = texts[start : start + batch_size]
            batch_array = self._embed_batch_with_fallback(batch, start, total_texts)
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


def resolve_embedding_texts(units: pd.DataFrame) -> pd.Series:
    if "modeling_text" in units.columns:
        return units["modeling_text"].fillna("").astype(str)
    if "text" in units.columns:
        return units["text"].fillna("").astype(str)
    raise ValidationError("Analysis units must contain either `modeling_text` or `text` before embeddings are computed")


def prepare_text_for_embedding(text: str, config: EmbeddingsConfig) -> str:
    prepared = str(text)
    if config.strip_null_bytes:
        prepared = prepared.replace("\x00", " ")
    if config.max_text_chars is not None and len(prepared) > config.max_text_chars:
        prepared = prepared[: config.max_text_chars]
    return prepared


def build_embedding_cache_key(text: str, config: EmbeddingsConfig) -> str:
    prepared = prepare_text_for_embedding(text, config)
    key_payload = "::".join(
        [
            config.provider,
            config.model,
            str(config.max_text_chars),
            str(config.strip_null_bytes),
            prepared,
        ]
    )
    return stable_text_hash(key_payload)


def materialize_embeddings(
    units: pd.DataFrame,
    config: AppConfig,
    provider: BaseEmbeddingProvider | None = None,
) -> np.ndarray:
    active_provider = build_embedding_provider(config, provider)
    cache_dir = ensure_dir(config.artifacts.embedding_cache_dir)
    ordered_rows: list[np.ndarray] = []
    missing_texts: list[str] = []
    missing_hashes: list[str] = []
    truncated_count = 0
    null_sanitized_count = 0
    embedding_texts = resolve_embedding_texts(units)

    for text in embedding_texts:
        prepared_text = prepare_text_for_embedding(text, config.embeddings)
        if prepared_text != text:
            if "\x00" in text:
                null_sanitized_count += 1
            if config.embeddings.max_text_chars is not None and len(text) > config.embeddings.max_text_chars:
                truncated_count += 1
        text_hash = build_embedding_cache_key(text, config.embeddings)
        cache_path = cache_dir / f"{text_hash}.npy"
        if cache_path.exists():
            ordered_rows.append(load_numpy(cache_path))
            continue
        ordered_rows.append(None)  # type: ignore[arg-type]
        missing_texts.append(prepared_text)
        missing_hashes.append(text_hash)

    if null_sanitized_count:
        logger.info("Sanitized null bytes in %s texts before embeddings", null_sanitized_count)
    if truncated_count:
        logger.info(
            "Truncated %s texts to max_text_chars=%s before embeddings",
            truncated_count,
            config.embeddings.max_text_chars,
        )

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
            text_hash = build_embedding_cache_key(str(embedding_texts.iloc[index]), config.embeddings)
            cache_path = Path(cache_dir) / f"{text_hash}.npy"
            ordered_rows[index] = load_numpy(cache_path)

    array = np.vstack(ordered_rows).astype(np.float32) if ordered_rows else np.zeros((0, 0), dtype=np.float32)
    save_numpy(array, config.artifacts.embeddings_path)
    logger.info("Saved embeddings to %s", config.artifacts.embeddings_path)
    return array
