"""Tool-level tests — no network, no LLM (estimator=None or a fake)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from mealchat import db, memory, nutrition
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


def test_grams_embedded_number_multiplies_quantity(conn):
    """{paneer, 2, '100 g'} is 2 x 100 g = 200 g — the embedded number used to
    swallow the quantity (logged 1 x 100 g = 265 instead of 530)."""
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("paneer", quantity=2, unit="100 g")]
    )
    assert items[0].quantity == pytest.approx(2)
    assert items[0].calories == pytest.approx(530, abs=2)

    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("chicken breast", quantity=2, unit="150 g")]
    )
    assert items[0].quantity == pytest.approx(3)  # 300 g = 3 x 100 g
    assert items[0].calories == pytest.approx(495, abs=2)


def test_grams_use_any_mass_basis_not_just_100g(conn):
    """Soy chunks are per '50 g dry': 10 g is 0.2 units (34 kcal), not 10
    servings (1700 kcal)."""
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("soy chunks", quantity=10, unit="g")]
    )
    assert items[0].quantity == pytest.approx(0.2)
    assert items[0].calories == pytest.approx(34, abs=1)


def test_unconvertible_grams_are_refused_not_guessed(toolkit, conn):
    """Almonds are per '10 nuts' — there's no gram basis, so 30 g must not
    become 30 x 70 kcal (2100, under the sanity cap). Nothing is logged."""
    ts, _ = toolkit
    out = ts["log_meal"].invoke({"items": [{"food": "almonds", "quantity": 30, "unit": "g"}]})
    assert out.startswith("Error:") and "10 nuts" in out
    assert db.get_meals(conn, "u1") == []


# --- edit_meal must not touch unmentioned items (urgent correction) --------


def test_edit_meal_spot_change_keeps_other_items(toolkit, conn):
    """The classic 'actually 3 rotis not 2' case: the chai must survive."""
    ts, effects = toolkit
    ts["log_meal"].invoke(
        {"items": [{"food": "roti", "quantity": 2}, {"food": "chai", "quantity": 1}]}
    )
    out = ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "roti", "quantity": 3}]})
    assert "Updated meal #1" in out
    items = {i.food: i.quantity for i in db.get_meal(conn, 1).items}
    assert items == {"roti": 3.0, "chai": 1.0}
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(450)
    assert effects["edited"] == [1]


def test_edit_meal_leaves_unmentioned_items_byte_identical(toolkit, conn):
    """Reviewer repro: chicken 200 g + 2 rotis; editing the rotis used to
    re-resolve the chicken and silently halve it (x2 -> x1)."""
    ts, _ = toolkit
    ts["log_meal"].invoke(
        {
            "items": [
                {"food": "chicken breast", "quantity": 200, "unit": "g"},
                {"food": "roti", "quantity": 2},
            ]
        }
    )
    before = next(i for i in db.get_meal(conn, 1).items if "chicken" in i.food)
    assert before.quantity == pytest.approx(2)  # 200 g = 2 x 100 g
    assert before.calories == pytest.approx(330, abs=1)

    ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "roti", "quantity": 3}]})

    after = next(i for i in db.get_meal(conn, 1).items if "chicken" in i.food)
    assert (after.food, after.quantity, after.unit) == (before.food, before.quantity, before.unit)
    assert (after.calories, after.protein, after.carbs, after.fat, after.notes) == (
        before.calories,
        before.protein,
        before.carbs,
        before.fat,
        before.notes,
    )  # byte-identical — never re-resolved
    roti = next(i for i in db.get_meal(conn, 1).items if i.food == "roti")
    assert roti.quantity == pytest.approx(3)
    assert roti.calories == pytest.approx(360)
    t = db.day_totals(conn, "u1", db.local_today(TZ))
    assert t.calories == pytest.approx(330 + 360, abs=2)


def test_edit_meal_quantity_change_scales_stored_macros(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "roti", "quantity": 2}, {"food": "chai", "quantity": 1}]})
    ts["edit_meal"].invoke({"meal_id": 1, "changes": [{"food": "roti", "quantity": 3}]})
    roti = next(i for i in db.get_meal(conn, 1).items if i.food == "roti")
    assert roti.calories == pytest.approx(240 * 1.5)  # stored 240 scaled by 3/2
    assert roti.protein == pytest.approx(3 * 2 * 1.5)  # 6g -> 9g
    assert roti.carbs == pytest.approx(18 * 2 * 1.5)


def test_edit_meal_unit_change_reresolves_that_item(toolkit, conn):
    ts, _ = toolkit
    ts["log_meal"].invoke({"items": [{"food": "dal", "quantity": 1}]})  # 1 bowl = 180
    ts["edit_meal"].invoke(
        {"meal_id": 1, "changes": [{"food": "dal", "quantity": 600, "unit": "g"}]}
    )
    dal = db.get_meal(conn, 1).items[0]
    assert dal.quantity == pytest.approx(2)  # 600 g = 2 bowls
    assert dal.calories == pytest.approx(360, abs=1)
    assert dal.unit == "bowl"  # re-resolved into the food's canonical unit


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


def test_nutrition_lookup_caches_grams_per_unit(conn):
    """A lookup-then-log must convert grams exactly like a direct log: the
    lookup's cache row used to drop grams_per_unit, so 10 g of a 150 g
    serving logged as 10 servings (2000 kcal) instead of ~13 kcal."""

    def fake_estimator(foods):
        return {
            f.lower(): {"food": f, "unit": "serving", "calories": 200, "protein": 5,
                        "carbs": 30, "fat": 6, "grams_per_unit": 150}
            for f in foods
        }

    ts = {t.name: t for t in build_tools(conn, TZ, fake_estimator, "u1", "text", {})}
    ts["nutrition_lookup"].invoke({"food": "ragi mudde"})
    assert db.get_food(conn, "ragi mudde").grams_per_unit == 150
    ts["log_meal"].invoke({"items": [{"food": "ragi mudde", "quantity": 10, "unit": "g"}]})
    assert db.day_totals(conn, "u1", db.local_today(TZ)).calories == pytest.approx(13.3, abs=0.1)


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


def test_memory_block_shows_ids(toolkit, conn):
    ts, _ = toolkit
    ts["save_memory"].invoke({"kind": "preference", "content": "vegetarian", "key": "diet"})
    block = memory.memory_block(conn, "u1")
    mid = next(m.id for m in db.get_memories(conn, "u1"))
    assert f"[m{mid}]" in block


def test_forget_memory_by_id_removes_only_that_one(toolkit, conn):
    ts, effects = toolkit
    veg_id = ts["save_memory"].invoke(
        {"kind": "preference", "content": "vegetarian", "key": "diet"}
    )
    # capture the id from the DB (the tool returns a string)
    veg_id = next(m.id for m in db.get_memories(conn, "u1") if m.key == "diet")
    ts["save_memory"].invoke({"kind": "preference", "content": "no eggs"})
    out = ts["forget_memory"].invoke({"memory_id": veg_id})
    assert "Forgot" in out and "vegetarian" in out and "no eggs" not in out
    contents = [m.content for m in db.get_memories(conn, "u1")]
    assert all("vegetarian" not in c for c in contents) and "no eggs" in contents
    assert "vegetarian" in effects.get("forgotten", [])


def test_forget_memory_rejects_other_users(toolkit, conn):
    ts, _ = toolkit
    db.save_memory(conn, "u2", "preference", "vegetarian", key="diet")
    other_id = next(m.id for m in db.get_memories(conn, "u2"))
    out = ts["forget_memory"].invoke({"memory_id": other_id})
    assert "Error" in out
    assert db.get_memory(conn, "u2", other_id) is not None  # untouched


def test_forget_twice_reports_already_forgotten(toolkit, conn):
    ts, _ = toolkit
    ts["save_memory"].invoke({"kind": "preference", "content": "vegetarian", "key": "diet"})
    mid = next(m.id for m in db.get_memories(conn, "u1"))
    ts["forget_memory"].invoke({"memory_id": mid})
    out = ts["forget_memory"].invoke({"memory_id": mid})
    assert "already forgotten" in out


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


def test_keyed_preference_upserts_for_any_kind(toolkit, conn):
    """key='diet' upserts like target keys do — re-saving replaces, not piles."""
    ts, _ = toolkit
    ts["save_memory"].invoke({"kind": "preference", "content": "vegetarian", "key": "diet"})
    ts["save_memory"].invoke({"kind": "preference", "content": "pescatarian now", "key": "diet"})
    prefs = [m for m in db.get_memories(conn, "u1") if m.kind == "preference" and m.key == "diet"]
    assert len(prefs) == 1
    assert prefs[0].content == "pescatarian now"


def test_usual_saved_with_structured_items_roundtrip(toolkit, conn):
    ts, _ = toolkit
    ts["save_memory"].invoke(
        {
            "kind": "usual",
            "content": "2 parathas and chai",
            "key": "breakfast",
            "items": [
                {"food": "paratha", "quantity": 2},
                {"food": "chai", "quantity": 1},
            ],
        }
    )
    usual = next(m for m in db.get_memories(conn, "u1") if m.kind == "usual")
    items = db.parse_memory_data(usual)
    assert items == [
        {"food": "paratha", "quantity": 2.0, "unit": ""},
        {"food": "chai", "quantity": 1.0, "unit": ""},
    ]
    block = memory.memory_block(conn, "u1")
    assert "usual (breakfast)" in block
    assert "items: paratha x2, chai x1" in block


def test_usual_items_render_with_units(toolkit, conn):
    """The unit is part of the usual: 'rice x1 [100 g]' (~83 kcal) replayed
    without its unit becomes 1 cup of rice (200 kcal)."""
    ts, _ = toolkit
    ts["save_memory"].invoke(
        {
            "kind": "usual",
            "content": "rice for breakfast",
            "key": "breakfast",
            "items": [{"food": "rice", "quantity": 1, "unit": "100 g"}],
        }
    )
    assert "items: rice x1 [100 g]" in memory.memory_block(conn, "u1")


def test_repeat_meals_copies_exactly(toolkit, conn):
    ts, effects = toolkit
    old_today = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
    db.add_meal(
        conn, "u1", old_today, "2 rotis and dal",
        [
            db.MealItem("roti", 2, "medium roti", 240, 6, 36, 3),
            db.MealItem("dal", 1, "bowl", 180, 9, 25, 4, notes="homestyle"),
        ],
        meal_type="lunch",
    )
    out = ts["repeat_meals"].invoke({"meal_ids": [1], "date": "today"})
    assert "Copied 1 meal(s)" in out
    today = db.local_today(TZ)
    meals = db.get_meals(conn, "u1", date=today)
    assert len(meals) == 1
    orig = db.get_meal(conn, 1).items
    assert [(i.food, i.quantity, i.calories, i.notes) for i in meals[0].items] == [
        (i.food, i.quantity, i.calories, i.notes) for i in orig
    ]  # exact copy, no re-estimation
    assert effects.get("repeated") == [meals[0].id]


def test_repeat_meals_rejects_other_users(toolkit, conn):
    ts, _ = toolkit
    db.add_meal(conn, "u2", "2026-10-21", "x", [db.MealItem("roti", 1, "roti", 120, 3, 18, 1.5)])
    out = ts["repeat_meals"].invoke({"meal_ids": [1]})
    assert "Error" in out
    assert db.get_meals(conn, "u1") == []


def test_memories_data_column_migrates(tmp_path):
    c = db.connect(tmp_path / "old.db")
    # simulate a DB created before the `data` column existed
    c.executescript(
        """
        CREATE TABLE memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
          kind TEXT NOT NULL, key TEXT, content TEXT NOT NULL,
          num_value REAL, unit TEXT, active INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """
    )
    c.execute(
        "INSERT INTO memories (user_id, kind, content, created_at, updated_at) VALUES ('u', 'note', 'x', 't', 't')"
    )
    c.commit()
    db.init_db(c)  # runs migrations
    cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
    assert "data" in cols
    assert db.get_memories(c, "u")[0].content == "x"  # old rows intact
    c.close()


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
