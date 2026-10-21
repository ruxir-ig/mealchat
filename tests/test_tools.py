"""Tool-level tests — no network, no LLM (estimator=None or a fake)."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pytest

from mealchat import db, memory, tools
from mealchat.tools import build_tools

TZ = ZoneInfo("Asia/Kolkata")


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.init_db(c)
    yield c
    c.close()


@pytest.fixture()
def toolkit(conn):
    effects: dict = {}
    ts = build_tools(conn, TZ, None, "u1", "text", effects)
    return {t.name: t for t in ts}, effects


def test_log_meal_and_totals(toolkit, conn):
    ts, effects = toolkit
    out = ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 2}], "meal_type": "lunch", "description": "2 rotis"}
    )
    assert "Logged meal #1" in out
    assert "240 kcal" in out
    assert effects["logged"] == [1]
    t = db.day_totals(conn, "u1", db.local_today(TZ))
    assert t.calories == pytest.approx(240)


def test_get_meals_food_filter(toolkit):
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}]})
    ts["log_meal"].invoke({"items": [{"food": "chai", "quantity": 1}]})
    out = ts["get_meals"].invoke({"food_like": "roti"})
    assert "roti" in out and "chai" not in out
    assert "meal #2" not in out  # most recent roti meal is #1


def test_edit_meal_replaces_not_adds(toolkit, conn):
    ts, effects = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}]})
    out = ts["edit_meal"].invoke({"meal_id": 1, "items": [{"food": "roti", "quantity": 3}]})
    assert "Updated meal #1" in out
    assert "360 kcal" in out
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(360)


def test_delete_meal(toolkit, conn):
    ts, effects = toolkit
    ts["log_meal"].invoke({"items": [{"food": "chai", "quantity": 1}]})
    out = ts["delete_meal"].invoke({"meal_id": 1})
    assert "Deleted" in out
    assert db.get_meals(conn, "u1") == []
    assert effects["deleted"] == [1]


def test_edit_meal_wrong_user_rejected(toolkit, conn):
    ts, _ = toolkit
    db.add_meal(conn, "u2", "2026-10-21", "x", [db.MealItem("roti", 1, "roti", 120, 3, 18, 1.5)])
    out = ts["edit_meal"].invoke({"meal_id": 1, "items": [{"food": "roti", "quantity": 3}]})
    assert "Error" in out


def test_day_totals_with_targets(toolkit, conn):
    ts, _ = toolkit
    ts["save_memory"].invoke(
        {"kind": "target", "content": "2000 kcal daily", "key": "calories", "value": 2000, "unit": "kcal"}
    )
    ts["log_meal"].invoke({"items": [{"food": "paratha", "quantity": 2}]})
    out = ts["day_totals"].invoke({})
    assert "600 kcal" in out
    assert "target 2000" in out


def test_nutrition_lookup_builtin(toolkit):
    ts, _ = toolkit
    out = ts["nutrition_lookup"].invoke({"food": "samosa"})
    assert "180 kcal" in out


def test_save_memory_renders_in_block(toolkit, conn):
    ts, _ = toolkit
    ts["save_memory"].invoke({"kind": "preference", "content": "vegetarian"})
    ts["save_memory"].invoke({"kind": "usual", "content": "2 paratha and 1 cup chai for breakfast"})
    ts["save_memory"].invoke(
        {"kind": "target", "content": "140g protein daily", "key": "protein", "value": 140, "unit": "g"}
    )
    block = memory.memory_block(conn, "u1")
    assert "vegetarian" in block
    assert "usual" in block
    assert "target protein: 140g" in block


def test_bad_date_returns_error(toolkit):
    ts, _ = toolkit
    out = ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 1}], "date": "tomorrow"})
    assert "Error" in out
