"""Config defaults must stay a standard public provider — the app has to run
from a clean clone with a plain OpenAI key, with no private gateway wired in
by default. reasoning_effort must be opt-in (most models reject it)."""

from __future__ import annotations

import os

from mealchat.config import load_settings
from mealchat.models import _extra_body


def test_defaults_are_standard_public_models(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env present
    for k in list(os.environ):
        if k.startswith("MEALCHAT_"):
            monkeypatch.delenv(k)
    s = load_settings()
    assert s.api_base == "https://api.openai.com/v1"
    assert s.text_model == "gpt-4o-mini"
    assert s.vision_model == "gpt-4o"
    assert s.text_model != s.vision_model, "text and vision must be different models"
    assert s.text_reasoning == "" and s.vision_reasoning == ""


def test_extra_body_only_sent_when_configured():
    assert _extra_body("") is None
    assert _extra_body("low") == {"reasoning_effort": "low"}
