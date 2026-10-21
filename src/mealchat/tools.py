"""Agent tools.

Tool surface (9 tools, deliberately split this way — see README):
  write path:   log_meal, edit_meal, delete_meal, repeat_meals, save_memory, forget_memory
  read path:    get_meals, day_totals, nutrition_lookup

log_meal resolves nutrition internally (table -> cache -> one batched LLM
estimate) so the common case is ONE tool call, not a lookup-then-log dance.
Corrections go through edit_meal (spot changes), never a second log_meal —
that is what prevents double-counting. repeat_meals copies items exactly.
Tools are built per turn via a factory closure that binds user_id/date/source
so every tool stays user-isolated.
"""

from __future__ import annotations

import json
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


def _meal_type_for_hour(hour: int) -> str:
    """Infer the meal type from the local hour when the user didn't say."""
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 19:
        return "snack"
    if 19 <= hour < 23:
        return "dinner"
    return "snack"  # late night / early morning grazing

# Sanity caps: a single logged item or whole meal above these is almost
# certainly a mis-parsed quantity — make the agent confirm instead of logging.
MAX_ITEM_KCAL = 2500
MAX_MEAL_KCAL = 4000


class LogItem(BaseModel):
    food: str
    quantity: float = 1.0
    unit: str | None = None
    notes: str | None = None


class MealChange(BaseModel):
    """One spot change to an existing meal item, matched by food name."""

    food: str
    quantity: float | None = None
    unit: str | None = None
    remove: bool = False


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
    note = f"; {it.notes}" if it.notes else ""
    return f"{it.food} x{it.quantity:g} [{it.unit}] {it.calories:g} kcal{note}"


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
        line += f" (calorie target {tgts['calories'].num_value:g}, {tgts['calories'].num_value - t.calories:g} kcal left)"
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
        portion: float = 1.0,
    ) -> str:
        """Log ONE meal (breakfast/lunch/etc). Use for new meals only — for
        corrections to an already-logged meal use edit_meal, never this.

        items: one entry per food, quantities for the WHOLE meal as eaten/seen
        (2 rotis -> food "roti", quantity 2). Nutrition is resolved automatically
        — do not call nutrition_lookup first for a normal log.
        meal_type: breakfast/lunch/dinner/snack, or "" if unknown.
        date: 'today', 'yesterday' or YYYY-MM-DD.
        description: short human phrase for the meal log, e.g. '2 parathas and chai'.
        portion: what fraction of this whole meal the user actually ate —
        0.5 when 'half of this was my brother's', 1.0 for a full meal. The tool
        scales every item by this number, so pass FULL quantities in items and
        put the share here; do NOT pre-multiply quantities yourself.
        """
        try:
            d = _resolve_date(date, tz)
        except ValueError as e:
            return f"Error: {e}"
        mtype = meal_type.strip().lower() if meal_type else ""
        if mtype not in _MEAL_TYPES:
            mtype = ""
        if not mtype:
            mtype = _meal_type_for_hour(datetime.now(tz).hour)  # from the clock
        try:
            share = float(portion)
        except (TypeError, ValueError):
            share = 1.0
        if share <= 0:
            share = 1.0
        resolved = nutrition.resolve_items(
            conn,
            [
                nutrition.FoodItem(i.food, i.quantity * share, i.unit, i.notes)
                for i in items
            ],
            estimator,
        )
        if not resolved:
            return "Error: no items to log."
        # Sanity guard: a bad unit or a runaway quantity should stop here and
        # get confirmed by the user — never poison the day's totals.
        item_kcal = [r.calories for r in resolved]
        if max(item_kcal) > MAX_ITEM_KCAL or sum(item_kcal) > MAX_MEAL_KCAL:
            worst = max(item_kcal)
            return (
                f"Error: this log looks too large to be real (largest item {worst:g} kcal,"
                f" meal total {sum(item_kcal):g} kcal; caps are {MAX_ITEM_KCAL} per item /"
                f" {MAX_MEAL_KCAL} per meal). Nothing was logged. Ask the user to confirm"
                " what and how much they actually ate, then log the confirmed amounts."
            )
        desc = description.strip() or " + ".join(i.food for i in resolved)
        if share != 1.0 and "portion" not in desc:
            desc = f"{desc} ({share:g} portion)"
        mid = db.add_meal(conn, user_id, d, desc, resolved, meal_type=mtype, source=source)
        effects.setdefault("logged", []).append(mid)
        effects.setdefault("portions", []).append(share)
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
    def repeat_meals(meal_ids: list[int], date: str = "today") -> str:
        """Copy past meals to a date with their items EXACTLY as logged — no
        re-estimation, no nutrition drift. The tool for 'same as yesterday':
        get_meals(date='yesterday') first, then ONE repeat_meals call with all
        those meal ids.

        meal_ids: from get_meals. date: 'today', 'yesterday' or YYYY-MM-DD.
        """
        try:
            d = _resolve_date(date, tz)
        except ValueError as e:
            return f"Error: {e}"
        if not meal_ids:
            return "Error: no meal ids to repeat."
        meals = []
        for mid in meal_ids:
            meal = db.get_meal(conn, mid)
            if meal is None or meal.user_id != user_id:
                return f"Error: meal #{mid} not found for this user."
            meals.append(meal)
        copied = []
        for meal in meals:
            new_id = db.add_meal(
                conn, user_id, d, meal.description, meal.items,
                meal_type=meal.meal_type, source=source,
            )
            copied.append(new_id)
            effects.setdefault("repeated", []).append(new_id)
        out = [f"Copied {len(copied)} meal(s) to {d}:"]
        for meal, new_id in zip(meals, copied):
            out.append(f"- meal #{new_id} = {meal.description} ({_fmt_meal(meal)})")
        out.append(_totals_line(conn, user_id, d, tz))
        return "\n".join(out)

    @tool
    def edit_meal(
        meal_id: int,
        changes: list[MealChange],
        meal_type: str = "",
        description: str = "",
        portion: float = 1.0,
    ) -> str:
        """Edit an already-logged meal with SPOT changes — the right tool for
        corrections like 'actually that was 3 rotis not 2' or 'skip the chai'.
        Only the foods you mention are touched; every other item is KEPT
        unchanged, so nothing gets lost or double-counted.

        meal_id: from get_meals.
        changes: one entry per food to change, matched to existing items by
        food name. {"food": "roti", "quantity": 3} updates just the rotis;
        {"food": "chai", "remove": true} removes the chai; a food that is not
        in the meal yet is ADDED. Give quantity in the food's natural unit.
        meal_type/description optional overrides.
        portion: like on log_meal — what fraction of the resulting meal the
        user ate (e.g. 0.5). Pass item quantities for the WHOLE meal and the
        share here; it scales every item.
        """
        meal = db.get_meal(conn, meal_id)
        if meal is None or meal.user_id != user_id:
            return f"Error: meal #{meal_id} not found for this user."

        def _same_food(ch: MealChange, it: db.MealItem) -> bool:
            return nutrition.normalize(ch.food) == nutrition.normalize(it.food)

        # Spot changes on STORED rows — unchanged items are never re-resolved,
        # so their quantity/unit/macros stay exactly as logged.
        pending: list[MealChange] = list(changes)
        final: list[db.MealItem] = []
        requeue: list[nutrition.FoodItem] = []  # unit changes / additions -> resolve

        for it in meal.items:
            remove_idx = next(
                (k for k, ch in enumerate(pending) if ch.remove and _same_food(ch, it)), None
            )
            match_idx = next(
                (k for k, ch in enumerate(pending) if not ch.remove and _same_food(ch, it)), None
            )
            if remove_idx is not None:
                pending.pop(remove_idx)
                continue
            if match_idx is None:
                final.append(it)  # untouched — kept byte-identical
                continue
            ch = pending.pop(match_idx)
            new_qty = ch.quantity if ch.quantity is not None else it.quantity
            new_unit = (ch.unit or "").strip()
            if new_unit and new_unit != it.unit:
                # basis change (e.g. grams instead of bowls) -> re-resolve this item
                requeue.append(nutrition.FoodItem(it.food, new_qty, new_unit, it.notes))
            elif ch.quantity is None or new_qty == it.quantity:
                final.append(it)
            elif it.quantity:
                # same unit, new amount: scale the STORED macros by the ratio
                factor = new_qty / it.quantity
                final.append(
                    db.MealItem(
                        food=it.food,
                        quantity=new_qty,
                        unit=it.unit,
                        calories=it.calories * factor,
                        protein=it.protein * factor,
                        carbs=it.carbs * factor,
                        fat=it.fat * factor,
                        notes=it.notes,
                    )
                )
            else:
                requeue.append(nutrition.FoodItem(it.food, new_qty, it.unit, it.notes))

        # unmatched non-remove changes are additions
        for ch in pending:
            if not ch.remove:
                requeue.append(nutrition.FoodItem(ch.food, ch.quantity or 1.0, ch.unit, None))

        resolved_extra = nutrition.resolve_items(conn, requeue, estimator) if requeue else []

        try:
            share = float(portion)
        except (TypeError, ValueError):
            share = 1.0
        if share <= 0:
            share = 1.0
        all_items = final + resolved_extra
        if share != 1.0:  # portion scales the stored macros of every item
            all_items = [
                db.MealItem(
                    food=i.food,
                    quantity=i.quantity * share,
                    unit=i.unit,
                    calories=i.calories * share,
                    protein=i.protein * share,
                    carbs=i.carbs * share,
                    fat=i.fat * share,
                    notes=i.notes,
                )
                for i in all_items
            ]
        if not all_items:
            return "Error: the edit would leave the meal empty."
        item_kcal = [r.calories for r in all_items]
        if max(item_kcal) > MAX_ITEM_KCAL or sum(item_kcal) > MAX_MEAL_KCAL:
            return (
                f"Error: the edited meal looks too large to be real (largest item"
                f" {max(item_kcal):g} kcal, total {sum(item_kcal):g} kcal; caps are"
                f" {MAX_ITEM_KCAL} per item / {MAX_MEAL_KCAL} per meal). Nothing was"
                " changed. Ask the user to confirm what and how much they ate."
            )
        mtype = meal_type.strip().lower() if meal_type and meal_type.strip().lower() in _MEAL_TYPES else meal.meal_type
        desc = description.strip() or meal.description
        db.edit_meal(conn, meal_id, items=all_items, meal_type=mtype, description=desc)
        effects.setdefault("edited", []).append(meal_id)
        effects.setdefault("portions", []).append(share)
        out = [f"Updated meal #{meal_id} ({meal.date}):"]
        for r in all_items:
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
        items: list[LogItem] | None = None,
    ) -> str:
        """Save a durable fact about the user that must survive across
        sessions. Use sparingly — only facts worth remembering, never one-off
        meal details or small talk.

        kind: 'preference', 'target', 'usual', 'note'.
        key: an upsert key so saving again REPLACES the previous version of
        the same fact instead of duplicating it — use 'diet' for diet
        preferences and the MEAL TYPE (breakfast/lunch/dinner/snack) for
        usuals; targets use calories|protein|carbs|fat.
        content: one short human-readable sentence for the fact.
        items: for usuals, the meal's foods (food/quantity/unit) — stored
        structured, so 'my usual' logs exactly these items later.
        For targets also set value (number) and unit ('kcal'|'g').
        """
        content = content.strip()
        if not content:
            return "Error: empty memory content."
        k = key.strip().lower() if key else None
        if kind == "target":
            if k not in ("calories", "protein", "carbs", "fat") or not value:
                return "Error: targets need key in calories/protein/carbs/fat and a numeric value."
            db.save_memory(
                conn, user_id, kind, content, key=k,
                num_value=float(value), unit=unit or ("kcal" if k == "calories" else "g"),
            )
        else:
            data = None
            if items:
                data = json.dumps(
                    [
                        {"food": i.food, "quantity": i.quantity, "unit": i.unit or ""}
                        for i in items
                    ]
                )
            db.save_memory(conn, user_id, kind, content, key=k, data=data)
        effects.setdefault("memories", []).append(f"{kind}:{content[:60]}")
        return f"Remembered ({kind}" + (f", {k}" if k else "") + f"): {content}"

    @tool
    def forget_memory(memory_id: int) -> str:
        """Remove one saved memory about the user by its id. The memory block
        in your context shows ids like [m12] — pass that number (12). Use when
        the user corrects or retracts a durable fact ('not vegetarian
        anymore', 'new protein target'), then save_memory the replacement.

        memory_id: the number from the [mN] tag shown in the memory block.
        """
        mem = db.get_memory(conn, user_id, memory_id)
        if mem is None:
            gone = db.get_memory(conn, user_id, memory_id, active_only=False)
            if gone is not None:
                return f"Error: memory #{memory_id} is already forgotten."
            return f"Error: no memory #{memory_id} for this user."
        db.forget_memory(conn, memory_id)
        effects.setdefault("forgotten", []).append(mem.content)
        return f"Forgot #{memory_id} ({mem.kind}): {mem.content}"

    return [
        log_meal,
        get_meals,
        repeat_meals,
        edit_meal,
        delete_meal,
        day_totals,
        nutrition_lookup,
        save_memory,
        forget_memory,
    ]
