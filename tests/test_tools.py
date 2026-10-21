"""Tool-level tests — no network, no LLM (estimator=None or a fake)."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pytest

from mealchat import db, memory
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
    ts, _effects = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}]})
    ts["log_meal"].invoke({"items": [{"food": "chai", "quantity": 1}]})
    out = ts["get_meals"].invoke({"food_like": "roti"})
    assert "roti" in out and "chai" not in out
    assert "meal #2" not in out  # most recent roti meal is #1


def test_edit_meal_spot_change_keeps_unmentioned_items(toolkit, conn):
    """The 'actually 3 rotis not 2' case: the chai must survive the edit."""
    ts, effects = toolkit
    ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 2}, {"food": "chai", "quantity": 1}]}
    )
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(330)

    out = ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "roti", "quantity": 3}]})
    assert "Updated meal #1" in out
    items = db.get_meal(conn, 1).items
    assert [(i.food, i.quantity) for i in items] == [("roti", 3.0), ("chai", 1.0)]
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(450)
    assert effects["edited"] == [1]


def test_edit_meal_removes_single_item(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 2}, {"food": "chai", "quantity": 1}]}
    )
    ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "chai", "remove": True}]})
    items = db.get_meal(conn, 1).items
    assert [i.food for i in items] == ["roti"]
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(240)


def test_edit_meal_adds_new_food(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}]})
    ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "dal", "quantity": 1}]})
    items = db.get_meal(conn, 1).items
    assert sorted(i.food for i in items) == ["dal", "roti"]
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(420)


def test_edit_meal_portion_scales_every_item(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 2}, {"food": "chai", "quantity": 1}]}
    )
    ts["edit_meal"].invoke({"meal_id": 1, "changes": [], "portion": 0.5})
    items = {i.food: i.quantity for i in db.get_meal(conn, 1).items}
    assert items["roti"] == pytest.approx(1.0)  # 2 x 0.5
    assert items["chai"] == pytest.approx(0.5)
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(165)


def test_delete_meal(toolkit, conn):
    ts, effects = toolkit
    ts["log_meal"].invoke({"items": [{"food": "chai", "quantity": 1}]})
    out = ts["delete_meal"].invoke({"meal_id": 1})
    assert "Deleted" in out
    assert db.get_meals(conn, "u1") == []
    assert effects["deleted"] == [1]


def test_edit_meal_guard_blocks_runaway_change(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}]})
    out = ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "roti", "quantity": 50}]})
    assert "Error" in out and "confirm" in out
    # meal untouched
    assert db.get_meal(conn, 1).items[0].quantity == pytest.approx(2)


def test_edit_meal_wrong_user_rejected(toolkit, conn):
    ts, _ = toolkit
    db.add_meal(conn, "u2", "2026-10-21", "x", [db.MealItem("roti", 1, "roti", 120, 3, 18, 1.5)])
    out = ts["edit_meal"].invoke(
        {"meal_id": 1, "changes": [{"food": "roti", "quantity": 3}]}
    )
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
    assert "1400 kcal left" in out  # remaining = target - eaten (regression: sign was flipped)


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


def test_forget_memory_removes_without_touching_others(toolkit, conn):
    ts, effects = toolkit
    ts["save_memory"].invoke({"kind": "preference", "content": "vegetarian"})
    ts["save_memory"].invoke({"kind": "preference", "content": "no eggs"})
    out = ts["forget_memory"].invoke({"search": "vegetarian"})
    assert "Forgot" in out and "vegetarian" in out and "no eggs" not in out
    contents = [m.content for m in db.get_memories(conn, "u1")]
    assert "vegetarian" not in contents and "no eggs" in contents
    assert "vegetarian" in effects.get("forgotten", [])

    # target keys are forgettable too
    ts["save_memory"].invoke(
        {"kind": "target", "content": "2000 kcal daily", "key": "calories", "value": 2000}
    )
    out = ts["forget_memory"].invoke({"search": "calories"})  # matches by key, not content
    assert "2000" in out
    assert all(m.key != "calories" for m in db.get_memories(conn, "u1"))


def test_meal_type_inferred_from_clock(toolkit, conn):
    """No meal_type given -> filled from the local hour, not left blank."""
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 1}]})
    meals = db.get_meals(conn, "u1")
    assert meals[0].meal_type in ("breakfast", "lunch", "dinner", "snack")


def test_meal_type_bands():
    from mealchat.tools import _meal_type_for_hour

    assert _meal_type_for_hour(8) == "breakfast"
    assert _meal_type_for_hour(13) == "lunch"
    assert _meal_type_for_hour(17) == "snack"
    assert _meal_type_for_hour(20) == "dinner"
    assert _meal_type_for_hour(1) == "snack"  # late night
    assert _meal_type_for_hour(23) == "snack"
    assert _meal_type_for_hour(5) == "breakfast"
    assert _meal_type_for_hour(11) == "lunch"
    assert _meal_type_for_hour(19) == "dinner"


def test_explicit_meal_type_not_overridden(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 1}], "meal_type": "dinner"}
    )
    assert db.get_meals(conn, "u1")[0].meal_type == "dinner"


def test_bad_date_returns_error(toolkit):
    ts, _ = toolkit
    out = ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 1}], "date": "tomorrow"})
    assert "Error" in out


# --- sanity-guard tests (review round 2, task A1) ---------------------------


def test_guard_blocks_runaway_single_item(toolkit, conn):
    """200 units of paneer curry = 64000 kcal must not be logged."""
    ts, effects = toolkit
    out = ts["log_meal"].invoke({"items": [{"food": "paneer curry", "quantity": 200}]})
    assert "Error" in out and "confirm" in out
    assert db.get_meals(conn, "u1") == []
    assert "logged" not in effects


def test_guard_blocks_runaway_meal_total(toolkit, conn):
    """Each item under the cap but the meal over it: still blocked."""
    ts, _ = toolkit
    out = ts["log_meal"].invoke(
        {
            "items": [
                {"food": "biryani", "quantity": 3},   # 2400
                {"food": "dal makhani", "quantity": 7},  # 1960 -> 4360 total
            ]
        }
    )
    assert "Error" in out and "4000" in out
    assert db.get_meals(conn, "u1") == []


def test_guard_allows_large_but_plausible_meal(toolkit, conn):
    ts, _ = toolkit
    out = ts["log_meal"].invoke(
        {
            "items": [
                {"food": "biryani", "quantity": 1.5},  # 1200
                {"food": "chicken curry", "quantity": 2},  # 700
            ]
        }
    )
    assert "Logged meal" in out
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(1900)
