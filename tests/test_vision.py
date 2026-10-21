"""Vision-path fail-safe tests — no LLM, no network.

Any failure before or during the vision call (missing file, non-image file,
network error, unparseable output) must surface as uncertain=True, never as
an exception out of analyze_photo.
"""

from __future__ import annotations

import asyncio

from mealchat.vision import analyze_photo


def _run(llm, path):
    return asyncio.run(analyze_photo(llm, path))


def test_missing_file_is_uncertain_not_exception():
    r = _run(None, "fixtures/photos/does_not_exist.jpg")
    assert r.uncertain is True
    assert r.question
    assert "couldn't" in r.description.lower() or "failed" in r.description.lower()


def test_non_image_file_is_uncertain_not_exception():
    r = _run(None, "README.md")
    assert r.uncertain is True
    assert r.question
    assert r.items == []


def test_none_llm_is_uncertain_not_exception():
    # llm=None only fails once invoked; with a valid image that's exactly what happens
    r = _run(None, "fixtures/photos/Chapati.jpg")
    assert r.uncertain is True
    assert r.question
