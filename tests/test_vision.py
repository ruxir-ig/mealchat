"""Vision-path fail-safe tests — no LLM, no network.

Any failure before or during the vision call (missing file, non-image file,
network error, unparseable output) must surface as uncertain=True, never as
an exception out of analyze_photo.
"""

from __future__ import annotations

import asyncio

from mealchat.vision import _build_prompt, analyze_photo


def _run(llm, path):
    return asyncio.run(analyze_photo(llm, path))


def test_prompt_without_caption_has_no_caption_rules():
    p = _build_prompt(None)
    assert "caption" not in p.lower()
    assert "WHOLE visible portion" in p


def test_prompt_with_caption_forbids_applying_fractions():
    p = _build_prompt("half of this was my brother's")
    assert "half of this was my brother's" in p
    assert "WHOLE visible portion" in p
    # the double-scaling guard must be explicit
    assert "Do NOT apply any sharing or fraction" in p
    assert "double-scale" in p


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
