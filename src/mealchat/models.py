"""LLM client factories.

Two deliberately different models through one OpenAI-compatible API:
  - text:   the conversation model — fast, tool calling, runs the agent loop
  - vision: a separate multimodal model — only sees photos

Defaults (set in config.py) are public OpenAI models; both are configurable
via env (see .env.example) so a reviewer can point them at any provider
without touching code.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mealchat.config import Settings


def _extra_body(reasoning: str) -> dict | None:
    """Only send reasoning_effort when configured — many chat-completions
    models (including plain OpenAI ones) reject unknown body params."""
    return {"reasoning_effort": reasoning} if reasoning else None


def text_llm(s: Settings, temperature: float = 0.2) -> BaseChatModel:
    return ChatOpenAI(
        model=s.text_model,
        base_url=s.api_base,
        api_key=s.api_key,
        temperature=temperature,
        timeout=90,
        max_retries=1,
        extra_body=_extra_body(s.text_reasoning),
    )


def vision_llm(s: Settings, temperature: float = 0.1) -> BaseChatModel:
    # No retries: a slow vision model must hit the time budget and fall back
    # to the uncertain/ask path, not hold the turn hostage for minutes.
    return ChatOpenAI(
        model=s.vision_model,
        base_url=s.vision_api_base,
        api_key=s.vision_api_key,
        temperature=temperature,
        timeout=s.vision_timeout,
        max_retries=0,
        extra_body=_extra_body(s.vision_reasoning),
    )


class FoodEstimator(Protocol):
    """Estimates per-unit macros for foods we don't have in the table."""

    def __call__(self, foods: list[str]) -> dict[str, dict]: ...


_ESTIMATE_PROMPT = """You are a nutrition lookup service. For each food below, give the nutrition
of ONE typical serving / piece / cup (the natural everyday unit, the way a person in India would eat it).

Foods: {foods}

Respond with ONLY a JSON object, no markdown, no prose:
{{"results": [{{"food": "<name>", "unit": "<the unit your numbers are for>", "grams_per_unit": <typical grams (or ml) in ONE unit, or null if not applicable>, "calories": <number>, "protein": <grams>, "carbs": <grams>, "fat": <grams>}}]}}"""


def make_estimator(llm: BaseChatModel) -> FoodEstimator:
    """LLM-backed estimator used by the nutrition layer for unknown foods."""

    def estimate(foods: list[str]) -> dict[str, dict]:
        prompt = _ESTIMATE_PROMPT.format(foods=json.dumps(foods))
        raw = llm.invoke(prompt).content
        text = raw if isinstance(raw, str) else "".join(part.text or "" for part in raw)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return {str(r["food"]).lower().strip(): r for r in data.get("results", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    return estimate
