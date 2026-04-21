"""Topic enrichment providers and caching helpers."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from src.config import AppConfig, LLMConfig
from src.schemas import ValidationError


logger = logging.getLogger(__name__)


RESPONSE_KEYS = (
    "topic_title",
    "topic_description",
    "suspected_jtbd",
    "suspected_pain_points",
    "suspected_business_value",
    "suspected_product_action",
    "confidence_note",
)


class BaseLLMProvider(ABC):
    """Abstract topic enrichment provider."""

    @abstractmethod
    def enrich_topic(self, topic_payload: dict[str, object]) -> dict[str, object]:
        """Return structured topic enrichment."""


def _clean_json_payload(text: str) -> dict[str, object]:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`")
        if "\n" in normalized:
            normalized = normalized.split("\n", 1)[1]
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1:
        raise ValidationError("LLM response does not contain a JSON object")
    payload = json.loads(normalized[start : end + 1])
    return {key: payload.get(key, "") for key in RESPONSE_KEYS}


class OpenAICompatibleLLMClient(BaseLLMProvider):
    """Calls an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
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
                logger.warning("LLM request failed on attempt %s/%s: %s", attempt, self.config.max_retries, exc)
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    def enrich_topic(self, topic_payload: dict[str, object]) -> dict[str, object]:
        system_prompt = (
            "You summarize enterprise AI usage topics. "
            "Return only JSON with the exact keys: "
            + ", ".join(RESPONSE_KEYS)
            + ". Keep wording short, business-like, and hypothesis-based. "
            "Do not invent facts beyond the provided examples."
        )
        user_prompt = json.dumps(topic_payload, ensure_ascii=False, indent=2)
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = self._request(payload)
        content = response["choices"][0]["message"]["content"]
        return _clean_json_payload(content)


class MockLLMClient(BaseLLMProvider):
    """Deterministic local topic enrichment for tests and smoke runs."""

    def enrich_topic(self, topic_payload: dict[str, object]) -> dict[str, object]:
        keywords = topic_payload.get("topic_keywords", "")
        examples = topic_payload.get("examples", [])
        title = str(keywords).split(",")[0].strip() or f"Topic {topic_payload.get('topic_id')}"
        return {
            "topic_title": title.title(),
            "topic_description": f"Hypothesis-based summary for topic {topic_payload.get('topic_id')}.",
            "suspected_jtbd": "Users want practical help on the topic shown by the examples.",
            "suspected_pain_points": "Likely friction around accuracy, workflow fit, or missing product affordances.",
            "suspected_business_value": f"Potentially important because the topic appears in {len(examples)} representative examples.",
            "suspected_product_action": "Review prompts, routing, and product affordances for this scenario.",
            "confidence_note": "Low to medium confidence; generated from synthetic or limited examples.",
        }


def build_llm_provider(config: AppConfig, provider: BaseLLMProvider | None = None) -> BaseLLMProvider:
    if provider is not None:
        return provider
    if config.llm.provider == "openai_compatible":
        return OpenAICompatibleLLMClient(config.llm)
    if config.llm.provider == "mock":
        return MockLLMClient()
    raise ValidationError(f"Unsupported llm.provider value: {config.llm.provider}")

