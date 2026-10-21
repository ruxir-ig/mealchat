"""Tests for the persistence layer — the data-correctness backbone.

The invariant that matters most: day_totals is always recomputed from live
rows, so edits/corrections/deletions can never leave stale sums behind.
"""

from __future__ import annotations

import pytest

from mealchat import db


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.init_db(c)
    yield c
    c.close()


def roti_item(qty: float) -> db.MealItem:
    return db.MealItem("roti", qty, "roti", 120 * qty, 3 * qty, 18 * qty, 1.5 * qty)


def test_add_and_get_meal(conn):
    mid = db.add_meal(
        conn, "u1", "2026-10-21", "2 rotis and chai", [roti_item(2)], meal_type="lunch"
    )
    meals = db.get_meals(conn, "u1", date="2026-10-21")
    assert len(meals) == 1
    assert meals[0].id == mid
    assert meals[0].items[0].food == "roti"
    assert meals[0].items[0].quantity == 2


def test_totals_aggregate_from_rows(conn):
    db.add_meal(conn, "u1", "2026-10-21", "2 rotis", [roti_item(2)])
    db.add_meal(conn, "u1", "2026-10-21", "1 roti", [roti_item(1)])
    t = db.day_totals(conn, "u1", "2026-10-21")
    assert t.calories == pytest.approx(360)
    assert t.protein == pytest.approx(9)
    assert t.meals == 2


def test_edit_meal_fixes_totals_without_double_counting(conn):
    """The 'actually 3 rotis not 2' case, at the data layer."""
    mid = db.add_meal(conn, "u1", "2026-10-21", "2 rotis", [roti_item(2)])
    assert db.day_totals(conn, "u1", "2026-10-21").calories == pytest.approx(240)

    ok = db.edit_meal(conn, mid, items=[roti_item(3)])
    assert ok
    t = db.day_totals(conn, "u1", "2026-10-21")
    assert t.calories == pytest.approx(360)  # replaced, not added
    assert t.meals == 1
    items = db.get_meal(conn, mid).items
    assert len(items) == 1 and items[0].quantity == 3


def test_delete_meal_excluded_from_totals(conn):
    mid = db.add_meal(conn, "u1", "2026-10-21", "2 rotis", [roti_item(2)])
    db.add_meal(conn, "u1", "2026-10-21", "chai", [db.MealItem("chai", 1, "cup", 90, 2, 12, 3)])
    assert db.day_totals(conn, "u1", "2026-10-21").calories == pytest.approx(330)

    assert db.delete_meal(conn, mid)
    t = db.day_totals(conn, "u1", "2026-10-21")
    assert t.calories == pytest.approx(90)
    assert t.meals == 1
    assert db.get_meals(conn, "u1") == [] or all(m.id != mid for m in db.get_meals(conn, "u1"))


def test_user_isolation(conn):
    db.add_meal(conn, "u1", "2026-10-21", "2 rotis", [roti_item(2)])
    db.add_meal(conn, "u2", "2026-10-21", "1 roti", [roti_item(1)])
    assert db.day_totals(conn, "u1", "2026-10-21").calories == pytest.approx(240)
    assert db.day_totals(conn, "u2", "2026-10-21").calories == pytest.approx(120)


def test_food_lookup_and_upsert(conn):
    db.upsert_food(
        conn,
        db.Food("paratha", "paratha", "paratha", 300, 6, 40, 12, "builtin"),
    )
    f = db.get_food(conn, "Paratha")
    assert f is not None and f.calories == 300
    # llm estimate replaces builtin value on conflict
    db.upsert_food(
        conn,
        db.Food("paratha", "paratha", "paratha", 310, 6, 41, 12, "llm"),
    )
    assert db.get_food(conn, "paratha").calories == 310


def test_memory_target_upsert_replaces_old(conn):
    db.save_memory(conn, "u1", "target", "140g protein daily", key="protein", num_value=140, unit="g")
    db.save_memory(conn, "u1", "target", "160g protein daily", key="protein", num_value=160, unit="g")
    active = [m for m in db.get_memories(conn, "u1") if m.kind == "target"]
    assert len(active) == 1
    assert active[0].num_value == 160


def test_memory_preference_dedupes(conn):
    db.save_memory(conn, "u1", "preference", "vegetarian")
    db.save_memory(conn, "u1", "preference", "vegetarian")
    prefs = [m for m in db.get_memories(conn, "u1") if m.kind == "preference"]
    assert len(prefs) == 1


def test_recent_messages_order_and_limit(conn):
    for i in range(10):
        db.add_message(conn, "u1", "user", f"m{i}")
    msgs = db.recent_messages(conn, "u1", limit=4)
    assert [m["content"] for m in msgs] == ["m6", "m7", "m8", "m9"]


def test_recent_messages_time_window(conn):
    from datetime import datetime, timedelta

    db.add_message(conn, "u1", "user", "fresh")
    # a message from 4 hours ago, inserted with an old timestamp directly
    old = (datetime.now().astimezone() - timedelta(hours=4)).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, 'user', 'stale', ?)",
            ("u1", old),
        )
    cutoff = (datetime.now().astimezone() - timedelta(seconds=3 * 3600)).isoformat(timespec="seconds")
    assert [m["content"] for m in db.recent_messages(conn, "u1", since=cutoff)] == ["fresh"]
    # without a window both are returned, ordered by turn sequence (id)
    assert [m["content"] for m in db.recent_messages(conn, "u1")] == ["fresh", "stale"]
