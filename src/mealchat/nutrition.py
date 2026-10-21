"""Nutrition resolution: builtin table -> DB cache -> one batched LLM estimate.

Tradeoff (documented in the README): nutrition data is NOT the interesting
part of this task, so we use a small hardcoded Indian-first table for the
foods people actually log, and let the text model estimate everything else,
caching every estimate in the DB so each food is only ever estimated once.
Numbers are "close enough for a calorie tracker", not dietitian-grade.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from mealchat import db
from mealchat.models import FoodEstimator

# name -> (unit, kcal, protein_g, carbs_g, fat_g) per ONE unit
_BUILTIN: dict[str, tuple[str, float, float, float, float]] = {
    # Indian staples
    "roti": ("medium roti", 120, 3, 18, 1.5),
    "paratha": ("medium paratha", 300, 6, 40, 12),
    "aloo paratha": ("medium paratha", 320, 6.5, 42, 13),
    "chai": ("cup with sugar", 90, 2, 12, 3),
    "dal": ("bowl", 180, 9, 25, 4),
    "dal makhani": ("bowl", 280, 9, 25, 16),
    "rajma": ("bowl", 230, 11, 35, 5),
    "chole": ("bowl", 270, 11, 40, 7),
    "rice": ("cup cooked", 200, 4, 44, 0.5),
    "biryani": ("takeaway box", 800, 24, 95, 32),
    "veg biryani": ("takeaway box", 700, 16, 95, 26),
    "paneer": ("100 g", 265, 18, 2, 20),
    "paneer curry": ("bowl", 320, 16, 10, 24),
    "dahi": ("cup", 150, 8, 12, 8),
    "idli": ("idli", 40, 1.5, 8, 0.2),
    "dosa": ("plain dosa", 130, 3, 20, 4.5),
    "masala dosa": ("dosa", 250, 6, 35, 9),
    "sambar": ("bowl", 150, 7, 22, 4),
    "upma": ("plate", 270, 6, 40, 9),
    "poha": ("plate", 270, 5, 45, 8),
    "khichdi": ("bowl", 250, 9, 45, 5),
    "thepla": ("thepla", 200, 4, 28, 8),
    "pav": ("pav", 110, 3, 20, 1.5),
    "bhaji": ("bowl", 250, 6, 35, 9),
    "samosa": ("samosa", 180, 4, 20, 9),
    "maggi": ("pack", 400, 8, 55, 15),
    "sabzi": ("bowl", 150, 4, 15, 8),
    "mutton curry": ("bowl", 400, 25, 8, 30),
    "chicken curry": ("bowl", 350, 28, 8, 22),
    "fried chicken": ("piece", 200, 15, 8, 13),
    "lassi": ("glass, sweet", 300, 7, 50, 8),
    "gulab jamun": ("piece", 175, 3, 25, 8),
    "jalebi": ("piece", 150, 1, 25, 6),
    # Protein & western basics
    "egg": ("large egg", 72, 6.3, 0.4, 5),
    "chicken breast": ("100 g cooked", 165, 31, 0, 3.6),
    "fish": ("100 g cooked", 140, 25, 0, 4),
    "prawns": ("100 g", 99, 24, 0, 0.3),
    "tofu": ("100 g", 120, 12, 3, 7),
    "soy chunks": ("50 g dry", 170, 26, 15, 5),
    "sprouts": ("bowl", 150, 8, 25, 2),
    "protein shake": ("scoop with water", 120, 24, 3, 1.5),
    "greek yogurt": ("cup", 100, 17, 6, 0.7),
    "milk": ("cup", 150, 8, 12, 8),
    "bread": ("slice", 80, 3, 14, 1),
    "peanut butter": ("tbsp", 95, 4, 3, 8),
    "oatmeal": ("cup cooked", 150, 5, 27, 3),
    "banana": ("medium", 105, 1.3, 27, 0.4),
    "apple": ("medium", 95, 0.5, 25, 0.3),
    "mango": ("medium", 200, 2.8, 50, 1.3),
    "almonds": ("10 nuts", 70, 2.5, 3, 6),
    "pizza": ("slice", 285, 12, 36, 10),
    "burger": ("burger", 420, 15, 45, 18),
    "fries": ("medium", 380, 4, 48, 18),
    "pasta": ("plate", 350, 13, 60, 12),
    "sandwich": ("sandwich", 250, 8, 35, 9),
    "salad": ("bowl", 120, 3, 12, 5),
    "coffee": ("cup black", 5, 0.3, 0, 0),
    "juice": ("glass", 110, 1, 26, 0.3),
    "soft drink": ("can", 140, 0, 39, 0),
    "chocolate": ("bar", 235, 3, 26, 13),
    "biscuits": ("2 biscuits", 60, 1, 9, 2.5),
    "chips": ("small pack", 150, 2, 15, 9),
    "ice cream": ("scoop", 140, 2.3, 16, 7),
    "beer": ("bottle", 150, 1.5, 12, 0),
    "water": ("glass", 0, 0, 0, 0),
    "butter": ("tbsp", 100, 0.1, 0.1, 11),
}

_ALIASES: dict[str, str] = {
    "rotis": "roti",
    "chapati": "roti",
    "chapatis": "roti",
    "phulka": "roti",
    "phulkas": "roti",
    "parathas": "paratha",
    "parantha": "paratha",
    "tea": "chai",
    "masala chai": "chai",
    "chai with sugar": "chai",
    "sweet chai": "chai",
    "lentils": "dal",
    "curd": "dahi",
    "yogurt": "dahi",
    "curd rice": "rice",
    "white rice": "rice",
    "basmati rice": "rice",
    "biryani rice": "biryani",
    "biryani box": "biryani",
    "idlis": "idli",
    "dosas": "dosa",
    "eggs": "egg",
    "omelette": "egg",
    "boiled egg": "egg",
    "chicken": "chicken breast",
    "grilled chicken": "chicken breast",
    "whey": "protein shake",
    "protein": "protein shake",
    "coke": "soft drink",
    "coke zero": "soft drink",
    "instant noodles": "maggi",
    "noodles": "maggi",
    "veg biryani box": "veg biryani",
    "toast": "bread",
}

# Rough estimate when both the table and the LLM fail us
_FALLBACK = ("serving", 250, 8, 30, 10)


def normalize(name: str) -> str:
    n = " ".join(name.lower().strip().split())
    n = _ALIASES.get(n, n)
    if n not in _BUILTIN and n.endswith("s") and n[:-1] in _BUILTIN:
        n = n[:-1]
    if n not in _BUILTIN and " with " in n:  # "chai with sugar" -> "chai"
        base = n.split(" with ")[0].strip()
        if base in _BUILTIN or base in _ALIASES:
            n = _ALIASES.get(base, base)
    return n


@dataclass
class FoodItem:
    """What the agent wants to log, before nutrition is resolved."""

    food: str
    quantity: float = 1.0
    unit: str | None = None
    notes: str | None = None


def lookup_food(conn: sqlite3.Connection, name: str) -> db.Food | None:
    """Builtin table first, then the DB cache (which may hold LLM estimates)."""
    key = normalize(name)
    if key in _BUILTIN:
        unit, kcal, p, c, f = _BUILTIN[key]
        return db.Food(key, key, unit, kcal, p, c, f, "builtin")
    return db.get_food(conn, key)


def _grams(unit: str | None, quantity: float) -> float | None:
    """'150 g' -> 150.0, bare 'g'/'grams' -> the item's quantity. None if not grams."""
    if not unit:
        return None
    u = unit.lower().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", u)
    if m:
        return float(m.group(1))
    if u in ("g", "gram", "grams"):
        return quantity
    return None


def _effective_quantity(item: FoodItem, food: db.Food) -> tuple[float, str, str | None]:
    """Best-effort unit reconciliation. Returns (qty, unit, note)."""
    qty, unit, note = item.quantity, item.unit or food.unit, item.notes
    g = _grams(item.unit, item.quantity)
    if g is not None and food.unit.startswith("100 g"):
        qty = g / 100.0
        unit = food.unit
        note = note or f"{item.unit}"
    return qty, unit, note


def resolve_items(
    conn: sqlite3.Connection,
    items: list[FoodItem],
    estimator: FoodEstimator | None = None,
) -> list[db.MealItem]:
    """Resolve nutrition for each item; unknown foods batch-estimated once."""
    resolved: list[db.MealItem] = []
    missing: list[FoodItem] = []

    for item in items:
        food = lookup_food(conn, item.food)
        if food is None:
            missing.append(item)
        else:
            qty, unit, note = _effective_quantity(item, food)
            resolved.append(_scale(item, food, qty, unit, note))

    if missing:
        estimates = estimator([m.food for m in missing]) if estimator else {}
        for item in missing:
            est = estimates.get(normalize(item.food)) or estimates.get(
                item.food.lower().strip()
            )
            if est:
                food = db.Food(
                    key=normalize(item.food),
                    name=item.food.strip(),
                    unit=str(est.get("unit", "serving")),
                    calories=float(est.get("calories", _FALLBACK[1])),
                    protein=float(est.get("protein", _FALLBACK[2])),
                    carbs=float(est.get("carbs", _FALLBACK[3])),
                    fat=float(est.get("fat", _FALLBACK[4])),
                    source="llm",
                )
                db.upsert_food(conn, food)  # cache: estimated only once
                qty, unit, note = _effective_quantity(item, food)
                resolved.append(_scale(item, food, qty, unit, note))
            else:
                unit, kcal, p, c, f = _FALLBACK
                resolved.append(
                    db.MealItem(
                        food=item.food.strip(),
                        quantity=item.quantity,
                        unit=item.unit or unit,
                        calories=kcal * item.quantity,
                        protein=p * item.quantity,
                        carbs=c * item.quantity,
                        fat=f * item.quantity,
                        notes=(item.notes + " (rough fallback estimate)") if item.notes else "rough fallback estimate",
                    )
                )
    return resolved


def _scale(item: FoodItem, food: db.Food, qty: float, unit: str, note: str | None) -> db.MealItem:
    return db.MealItem(
        food=food.name if food.key in _BUILTIN else item.food.strip(),
        quantity=round(qty, 2),
        unit=unit,
        calories=round(food.calories * qty, 1),
        protein=round(food.protein * qty, 1),
        carbs=round(food.carbs * qty, 1),
        fat=round(food.fat * qty, 1),
        notes=note,
    )
