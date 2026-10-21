"""Nutrition resolution tests — table hits, unit conversion, LLM-estimate path."""

from __future__ import annotations

import pytest

from mealchat import db, nutrition


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.init_db(c)
    yield c
    c.close()


def fake_estimator(foods):
    return {
        f.lower(): {"food": f, "unit": "bowl", "calories": 300, "protein": 10, "carbs": 40, "fat": 10}
        for f in foods
    }


def failing_estimator(foods):
    return {}


def test_builtin_with_alias_and_plural(conn):
    for name in ("roti", "rotis", "chapati", "chapatis"):
        f = nutrition.lookup_food(conn, name)
        assert f is not None and f.key == "roti", name
    assert nutrition.lookup_food(conn, "curd").key == "dahi"


def test_quantity_scaling(conn):
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("paratha", quantity=2), nutrition.FoodItem("chai")]
    )
    assert items[0].calories == pytest.approx(600)
    assert items[1].calories == pytest.approx(90)


def test_fraction_quantity_biryani_box(conn):
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("biryani", quantity=0.67, unit="box", notes="two thirds of the box")]
    )
    assert items[0].calories == pytest.approx(800 * 0.67, abs=1)
    assert items[0].quantity == 0.67


def test_grams_to_100g_conversion(conn):
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("chicken breast", quantity=150, unit="g")]
    )
    assert items[0].calories == pytest.approx(165 * 1.5, abs=1)
    assert items[0].protein == pytest.approx(31 * 1.5, abs=1)


# --- unit-mismatch regression tests (review round 2, task A1) ---------------


def test_grams_of_bowl_food_converts_via_typical_unit(conn):
    """paneer curry 200 g used to become 200 bowls = 64000 kcal."""
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("paneer curry", quantity=200, unit="g")]
    )
    assert items[0].calories == pytest.approx(320 * 200 / 300, abs=15)  # ~213
    assert items[0].calories < 300


def test_ml_of_drink_converts_via_typical_unit(conn):
    """milk 250 ml used to become 250 cups = 37500 kcal."""
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("milk", quantity=250, unit="ml")]
    )
    assert items[0].calories == pytest.approx(150 * 250 / 240, abs=10)  # ~156
    assert items[0].calories < 200


def test_grams_of_unknown_food_uses_per_100g_fallback(conn):
    """mystery stew 300 g with no estimator used to become 300 servings = 75000 kcal."""
    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("mystery stew", quantity=300, unit="g")],
        estimator=lambda foods: {},
    )
    assert items[0].unit == "100 g"
    assert items[0].quantity == pytest.approx(3.0)
    assert items[0].calories == pytest.approx(180 * 3, abs=1)  # ~540
    assert items[0].calories < 700


def test_estimator_grams_per_unit_used_for_conversion(conn):
    def est(foods):
        return {
            "truffle pasta": {
                "food": "truffle pasta",
                "unit": "bowl",
                "grams_per_unit": 350,
                "calories": 500,
                "protein": 18,
                "carbs": 60,
                "fat": 18,
            }
        }

    items = nutrition.resolve_items(
        conn, [nutrition.FoodItem("truffle pasta", quantity=350, unit="g")], est
    )
    assert items[0].quantity == pytest.approx(1.0)
    assert items[0].calories == pytest.approx(500, abs=1)
    # and the grams_per_unit is cached with the food
    cached = db.get_food(conn, "truffle pasta")
    assert cached is not None and cached.grams_per_unit == 350


def test_unknown_food_estimated_and_cached(conn):
    items = nutrition.resolve_items(conn, [nutrition.FoodItem("rajma chawal bowl", 1)], fake_estimator)
    assert items[0].calories == pytest.approx(300)
    assert items[0].unit == "bowl"
    # cached now — resolves without an estimator
    cached = db.get_food(conn, "rajma chawal bowl")
    assert cached is not None and cached.source == "llm"
    again = nutrition.resolve_items(conn, [nutrition.FoodItem("rajma chawal bowl", 2)])
    assert again[0].calories == pytest.approx(600)


def test_estimator_failure_falls_back(conn):
    items = nutrition.resolve_items(conn, [nutrition.FoodItem("mystery curry", 1)], failing_estimator)
    assert items[0].calories == pytest.approx(250)
    assert "fallback" in (items[0].notes or "")


def test_lookup_food_returns_builtin_without_db_row(conn):
    f = nutrition.lookup_food(conn, "egg")
    assert f.source == "builtin" and f.calories == pytest.approx(72)
