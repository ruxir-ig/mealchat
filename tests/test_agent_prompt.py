"""System-prompt shape tests: the prompt must be cache-friendly.

The date/weekday/timezone and the user's memory block differ per user and
day; the static policy text must be a byte-identical prefix so providers can
reuse it (prompt caching). Everything dynamic belongs AFTER the static part.
"""

from __future__ import annotations

from mealchat.agent import build_system_prompt


def test_dynamic_parts_come_after_the_identical_static_prefix():
    a = build_system_prompt(
        memory_block="- preference (diet) [m1]: vegetarian",
        today="2026-10-21",
        weekday="Wednesday",
        tz_name="Asia/Kolkata",
    )
    b = build_system_prompt(
        memory_block="",
        today="2026-12-25",
        weekday="Friday",
        tz_name="Europe/Berlin",
    )
    prefix_a, prefix_b = a.split("## Now")[0], b.split("## Now")[0]
    assert prefix_a == prefix_b  # static policy identical for every user/day
    assert "## Style" in prefix_a  # ALL policy sections are inside the prefix
    # the varying bits live at the end, after "## Now"
    assert "2026-10-21" in a.split("## Now", 1)[1]
    assert "vegetarian" in a.split("## Now", 1)[1]
    assert "2026-12-25" in b.split("## Now", 1)[1]


def test_empty_memory_block_has_fallback_text_at_the_tail():
    p = build_system_prompt("", "2026-10-21", "Wednesday", "Asia/Kolkata")
    assert "nothing saved yet" in p.split("## Now", 1)[1]
    assert p.rstrip().endswith(("come up)", "come up)"))
