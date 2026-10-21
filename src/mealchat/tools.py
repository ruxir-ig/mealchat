"""Agent tools.

Tool surface (7 tools, deliberately split this way — see README):
  write path:   log_meal, edit_meal, delete_meal, save_memory
  read path:    get_meals, day_totals, nutrition_lookup

log_meal resolves nutrition internally (table -> cache -> one batched LLM
estimate) so the common case is ONE tool call, not a lookup-then-log dance.
Corrections go through edit_meal (replace items), never a second log_meal —
that is what prevents double-counting. Tools are built per turn via a factory
closure that binds user_id/date/source so every tool stays user-isolated.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import BaseModel

from mealchat import db, memory, nutrition
from mealchat.models import FoodEstimator

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


class LogItem(BaseModel):
    food: str
    quantity: float = 1.0
    unit: str | None = None
    notes: str | None = None


def _resolve_date(raw: str, tz: ZoneInfo) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "today"):
        return datetime.now(tz).strftime("%Y-%m-%d")
    if s == "yesterday":
        return (datetime.now(tz) - timedelta(days=1)).strftime("%Y-%m-%d")
    if _DATE_RE.match(s):
        return s
    raise ValueError(f"bad date {raw!r} — use 'today', 'yesterday' or YYYY-MM-DD")


def _fmt_item(it: db.MealItem) -> str:
    s = f"{it.quantity:g} {it.unit} {it.food} ({it.calories:g} kcal"
    if it.notes:
        s += f"; {it.notes}"
    return s + ")"


def _fmt_meal(m: db.Meal) -> str:
    items = " + ".join(_fmt_item(i) for i in m.items)
    kcal = sum(i.calories for i in m.items)
    head = f"meal #{m.id} {m.date} {m.meal_type or 'snack'}"
    if m.source == "image":
        head += " (from photo)"
    return f"{head}: {items} = {kcal:g} kcal"


def _totals_line(conn: sqlite3.Connection, user_id: str, date: str, tz: ZoneInfo) -> str:
    t = db.day_totals(conn, user_id, date)
    line = (
        f"Day {date}: {t.calories:g} kcal, protein {t.protein:g}g, "
        f"carbs {t.carbs:g}g, fat {t.fat:g}g across {t.meals} meal(s)."
    )
    tgts = memory.targets(conn, user_id)
    if "calories" in tgts:
        line += f" (calorie target {tgts['calories'].num_value:g}, {t.calories - tgts['calories'].num_value:g} kcal left)"
    if "protein" in tgts:
        line += f" (protein target {tgts['protein'].num_value:g}g, {tgts['protein'].num_value - t.protein:g}g left)"
    return line


def build_tools(
    conn: sqlite3.Connection,
    tz: ZoneInfo,
    estimator: FoodEstimator | None,
    user_id: str,
    source: str,
    effects: dict,
) -> list:
    """Build the per-turn tool set bound to one user and input source."""

    @tool
    def log_meal(
        items: list[LogItem],
        meal_type: str = "",
        date: str = "today",
        description: str = "",
    ) -> str:
        """Log ONE meal (breakfast/lunch/etc). Use for new meals only — for
        corrections to an already-logged meal use edit_meal, never this.

        items: one entry per food. quantity in the food's natural unit
        (2 rotis -> food "roti", quantity 2; two-thirds of a takeaway box of
        biryani -> food "biryani", quantity 0.67, unit "box"). Fractions of a
        shared dish go in quantity. Nutrition is resolved automatically — do
        not call nutrition_lookup first for a normal log.
        meal_type: breakfast/lunch/dinner/snack, or "" if unknown.
        date: 'today', 'yesterday' or YYYY-MM-DD.
        description: short human phrase for the meal log, e.g. '2 parathas and chai'.
        """
        try:
            d = _resolve_date(date, tz)
        except ValueError as e:
            return f"Error: {e}"
        mtype = meal_type.strip().lower() if meal_type else ""
        if mtype not in _MEAL_TYPES:
            mtype = ""
        resolved = nutrition.resolve_items(
            conn, [nutrition.FoodItem(i.food, i.quantity, i.unit, i.notes) for i in items], estimator
        )
        if not resolved:
            return "Error: no items to log."
        desc = description.strip() or " + ".join(i.food for i in resolved)
        mid = db.add_meal(conn, user_id, d, desc, resolved, meal_type=mtype, source=source)
        effects.setdefault("logged", []).append(mid)
        out = [f"Logged meal #{mid} ({d}, {mtype or 'snack'}):"]
        for r in resolved:
            out.append(f"- {_fmt_item(r)}")
        out.append(_totals_line(conn, user_id, d, tz))
        return "\n".join(out)

    @tool
    def get_meals(
        date: str = "",
        food_like: str = "",
        limit: int = 5,
    ) -> str:
        """Retrieve the user's past meals, most recent first.

        date: '' = any day, 'today', 'yesterday' or YYYY-MM-DD.
        food_like: case-insensitive food substring, e.g. 'roti' — use this to
        find the meal a correction refers to.
        limit: max meals to return (default 5).
        """
        try:
            d = _resolve_date(date, tz) if date.strip() else None
        except ValueError as e:
            return f"Error: {e}"
        meals = db.get_meals(
            conn, user_id, date=d, food_like=food_like.strip() or None, limit=max(1, min(limit, 20))
        )
        if not meals:
            return "No meals found." + (f" (filtered date={d}, food~{food_like!r})" if d or food_like else "")
        return "\n".join(_fmt_meal(m) for m in meals)

    @tool
    def edit_meal(
        meal_id: int,
        items: list[LogItem],
        meal_type: str = "",
        description: str = "",
    ) -> str:
        """Edit an already-logged meal by REPLACING its items — the right tool
        for corrections like 'actually that was 3 rotis not 2'. Nutrition is
        re-resolved for the new items and day totals update automatically;
        nothing gets double-counted because items are replaced, not added.

        meal_id: from get_meals. items: the CORRECTED full item list for the
        whole meal (not just the changed food). meal_type/description optional.
        """
        meal = db.get_meal(conn, meal_id)
        if meal is None or meal.user_id != user_id:
            return f"Error: meal #{meal_id} not found for this user."
        resolved = nutrition.resolve_items(
            conn, [nutrition.FoodItem(i.food, i.quantity, i.unit, i.notes) for i in items], estimator
        )
        mtype = meal_type.strip().lower() if meal_type and meal_type.strip().lower() in _MEAL_TYPES else meal.meal_type
        desc = description.strip() or meal.description
        db.edit_meal(conn, meal_id, items=resolved, meal_type=mtype, description=desc)
        effects.setdefault("edited", []).append(meal_id)
        out = [f"Updated meal #{meal_id} ({meal.date}):"]
        for r in resolved:
            out.append(f"- {_fmt_item(r)}")
        out.append(_totals_line(conn, user_id, meal.date, tz))
        return "\n".join(out)

    @tool
    def delete_meal(meal_id: int) -> str:
        """Delete a logged meal (soft delete). Use when the user says they
        didn't eat something, or wants a mistaken log removed."""
        meal = db.get_meal(conn, meal_id)
        if meal is None or meal.user_id != user_id:
            return f"Error: meal #{meal_id} not found for this user."
        db.delete_meal(conn, meal_id)
        effects.setdefault("deleted", []).append(meal_id)
        return f"Deleted meal #{meal_id} ({meal.description}).\n" + _totals_line(
            conn, user_id, meal.date, tz
        )

    @tool
    def day_totals(date: str = "today") -> str:
        """Return calorie + macro totals for a day, with the meal-by-meal
        breakdown and any saved targets. date: 'today', 'yesterday' or
        YYYY-MM-DD."""
        try:
            d = _resolve_date(date, tz)
        except ValueError as e:
            return f"Error: {e}"
        meals = db.get_meals(conn, user_id, date=d, limit=20)
        out = [_totals_line(conn, user_id, d, tz)]
        if meals:
            out.append("Meals:")
            out.extend("  " + _fmt_meal(m) for m in reversed(meals))
        else:
            out.append("Nothing logged yet.")
        return "\n".join(out)

    @tool
    def nutrition_lookup(food: str) -> str:
        """Per-unit nutrition for ONE food (e.g. 'how many calories in a
        samosa?'). Unknown foods get a one-time model estimate that is then
        cached."""
        f = nutrition.lookup_food(conn, food)
        if f is None and estimator:
            est = estimator([food])
            row = est.get(nutrition.normalize(food)) or est.get(food.lower().strip())
            if row:
                f = db.Food(
                    key=nutrition.normalize(food),
                    name=food.strip(),
                    unit=str(row.get("unit", "serving")),
                    calories=float(row.get("calories", 0)),
                    protein=float(row.get("protein", 0)),
                    carbs=float(row.get("carbs", 0)),
                    fat=float(row.get("fat", 0)),
                    source="llm",
                )
                db.upsert_food(conn, f)
        if f is None:
            return f"No data for {food!r}."
        src = "builtin table" if f.source == "builtin" else "model estimate"
        return (
            f"1 {f.unit} of {f.name}: {f.calories:g} kcal, "
            f"protein {f.protein:g}g, carbs {f.carbs:g}g, fat {f.fat:g}g ({src})."
        )

    @tool
    def save_memory(
        kind: Literal["preference", "target", "usual", "note"],
        content: str,
        key: str = "",
        value: float = 0,
        unit: str = "",
    ) -> str:
        """Save a durable fact about the user that must survive across
        sessions. Use sparingly — only facts worth remembering, never one-off
        meal details or small talk.

        kind: 'preference' (e.g. 'vegetarian'), 'target' (a daily nutrition
        goal), 'usual' (a meal they eat regularly, itemized like '2 paratha
        and 1 cup chai'), 'note' (anything else durable).
        For targets also set key ('calories'|'protein'|'carbs'|'fat'),
        value (number) and unit ('kcal'|'g'). Saving again with the same kind+key
        updates the fact instead of duplicating it.
        """
        content = content.strip()
        if not content:
            return "Error: empty memory content."
        k = key.strip().lower() if key else None
        if kind == "target":
            if k not in ("calories", "protein", "carbs", "fat") or not value:
                return "Error: targets need key in calories/protein/carbs/fat and a numeric value."
            db.save_memory(
                conn, user_id, kind, content, key=k, num_value=float(value), unit=unit or ("kcal" if k == "calories" else "g")
            )
        else:
            db.save_memory(conn, user_id, kind, content)
        effects.setdefault("memories", []).append(f"{kind}:{content[:60]}")
        return f"Remembered ({kind}): {content}"

    return [log_meal, get_meals, edit_meal, delete_meal, day_totals, nutrition_lookup, save_memory]
