"""LLM client factories.

Two deliberately different models through one OpenAI-compatible gateway:
  - text:   deepseek-v4p1-flash — fast, tool calling, runs the agent loop
  - vision: glm-5.3-flash      — multimodal, only sees photos

Both are free-tier models on the local gateway. Both are configurable via env
(see .env.example) so a reviewer can point them at any OpenAI-compatible
provider without touching code.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mealchat.config import Settings


def text_llm(s: Settings, temperature: float = 0.2) -> BaseChatModel:
    return ChatOpenAI(
        model=s.text_model,
        base_url=s.api_base,
        api_key=s.api_key,
        temperature=temperature,
        timeout=90,
        max_retries=1,
        extra_body={"reasoning_effort": s.text_reasoning} if s.text_reasoning else None,
    )


def vision_llm(s: Settings, temperature: float = 0.1) -> BaseChatModel:
    return ChatOpenAI(
        model=s.vision_model,
        base_url=s.vision_api_base,
        api_key=s.vision_api_key,
        temperature=temperature,
        timeout=120,
        max_retries=1,
        extra_body={"reasoning_effort": s.vision_reasoning} if s.vision_reasoning else None,
    )


class FoodEstimator(Protocol):
    """Estimates per-unit macros for foods we don't have in the table."""

    def __call__(self, foods: list[str]) -> dict[str, dict]: ...


_ESTIMATE_PROMPT = """You are a nutrition lookup service. For each food below, give the nutrition
of ONE typical serving / piece / cup (the natural everyday unit, the way a person in India would eat it).

Foods: {foods}

Respond with ONLY a JSON object, no markdown, no prose:
{{"results": [{{"food": "<name>", "unit": "<the unit your numbers are for>", "calories": <number>, "protein": <grams>, "carbs": <grams>, "fat": <grams>}}]}}"""


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
