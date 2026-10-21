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
    timezone: str
    db_path: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def load_settings() -> Settings:
    _load_dotenv()
    return Settings(
        api_base=os.environ.get("MEALCHAT_API_BASE", "https://code.in2peta.com/v1"),
        api_key=os.environ.get("MEALCHAT_API_KEY", ""),
        text_model=os.environ.get("MEALCHAT_TEXT_MODEL", "deepseek-v4p1-flash"),
        text_reasoning=os.environ.get("MEALCHAT_TEXT_REASONING", "low"),
        vision_model=os.environ.get("MEALCHAT_VISION_MODEL", "glm-5.3-flash"),
        vision_reasoning=os.environ.get("MEALCHAT_VISION_REASONING", "low"),
        vision_api_base=os.environ.get("MEALCHAT_VISION_API_BASE", "")
        or os.environ.get("MEALCHAT_API_BASE", "https://code.in2peta.com/v1"),
        vision_api_key=os.environ.get("MEALCHAT_VISION_API_KEY", "")
        or os.environ.get("MEALCHAT_API_KEY", ""),
        timezone=os.environ.get("MEALCHAT_TIMEZONE", "Asia/Kolkata"),
        db_path=os.environ.get("MEALCHAT_DB", "data/mealchat.db"),
    )
