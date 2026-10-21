"""Configuration loaded from environment / .env file.

One OpenAI-compatible gateway serves both models by default; the vision model
can optionally point at a different base/key (it just has to be a different
model from the text one).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader — no dependency needed. Never overrides real env vars."""
    p = path or Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    api_base: str
    api_key: str
    text_model: str
    text_reasoning: str
    vision_model: str
    vision_reasoning: str
    vision_api_base: str
    vision_api_key: str
    vision_timeout: float
    timezone: str
    db_path: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def load_settings() -> Settings:
    _load_dotenv()
    # Defaults are plain public OpenAI models so a fresh clone runs with just an
    # OpenAI key. Any OpenAI-compatible gateway (Fireworks, Together, vLLM, …)
    # works by overriding these in .env.
    api_base = os.environ.get("MEALCHAT_API_BASE", "https://api.openai.com/v1")
    return Settings(
        api_base=api_base,
        api_key=os.environ.get("MEALCHAT_API_KEY", ""),
        text_model=os.environ.get("MEALCHAT_TEXT_MODEL", "gpt-4o-mini"),
        # reasoning_effort is gateway-specific (e.g. some chat-completions
        # models reject it) — only sent when explicitly set.
        text_reasoning=os.environ.get("MEALCHAT_TEXT_REASONING", ""),
        vision_model=os.environ.get("MEALCHAT_VISION_MODEL", "gpt-4o"),
        vision_reasoning=os.environ.get("MEALCHAT_VISION_REASONING", ""),
        vision_api_base=os.environ.get("MEALCHAT_VISION_API_BASE", "") or api_base,
        vision_api_key=os.environ.get("MEALCHAT_VISION_API_KEY", "")
        or os.environ.get("MEALCHAT_API_KEY", ""),
        # hard time budget for the vision call: on timeout we degrade to the
        # uncertain/ask path instead of holding the turn (or the whole agent)
        vision_timeout=float(os.environ.get("MEALCHAT_VISION_TIMEOUT", "20")),
        timezone=os.environ.get("MEALCHAT_TIMEZONE", "Asia/Kolkata"),
        db_path=os.environ.get("MEALCHAT_DB", "data/mealchat.db"),
    )
