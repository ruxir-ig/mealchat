"""Eval cases — the full test conversation set from the task, with a defined
notion of "correct" per case (in prose + DB assertions).

Three of these are the ones the task says separate good submissions from
average ones (marked SEPARATOR): corrections must not double-count, "same as
yesterday"/"my usual" must come from memory, photo+caption must be one meal.

Each case runs against a FRESH user in a throwaway eval DB. check() returns
a list of failures — empty list = pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mealchat import db

THALI = "fixtures/photos/Indian_thali.jpg"  # multi-bowl thali (used for the caption case)
PLATE = "fixtures/photos/Chapati.jpg"  # clean single plate of rotis
NOT_FOOD = "fixtures/photos/not_food.jpg"  # laptop keyboard


@dataclass
class EvalCase:
    name: str
    correct_is: str  # what "correct" means, in one sentence
    turns: list[tuple[str, str | None]]  # (text, image_path or None)
    check: Callable  # (conn, user_id, replies) -> list[str] failures
    seed: Callable | None = None  # optional DB seeding before the turns
    separator: bool = False


def _meals(conn, user_id):
    return db.get_meals(conn, user_id, limit=50)


def _items(conn, user_id):
    out = []
    for m in _meals(conn, user_id):
        out.extend(m.items)
    return out


def _totals(conn, user_id):
    return db.day_totals(conn, user_id, db.local_today(__import__("zoneinfo").ZoneInfo("Asia/Kolkata")))


CASES: list[EvalCase] = []


# --------------------------------------------------------------------- cases


def check_simple(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"expected 1 meal, got {len(meals)}")
        return fails
    if {i.food for i in meals[0].items} != {"paratha", "chai"}:
        fails.append(f"items wrong: {[i.food for i in meals[0].items]}")
    t = _totals(conn, user_id)
    if t.calories < 650 or t.calories > 700:
        fails.append(f"expected ~690 kcal (2x300 + 90), got {t.calories}")
    return fails


CASES.append(
    EvalCase(
        name="simple_breakfast_log",
        correct_is="'had 2 parathas and chai for breakfast' -> exactly one meal, paratha+chai, ~690 kcal",
        turns=[("had 2 parathas and chai for breakfast", None)],
        check=check_simple,
    )
)


def check_correction(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"SEPARATOR FAIL: expected 1 meal after correction, got {len(meals)} (double-count?)")
        return fails
    roti = [i for i in meals[0].items if "roti" in i.food.lower()]
    if not roti or roti[0].quantity != 3:
        fails.append(f"expected 3 rotis after correction, got {[(i.food, i.quantity) for i in meals[0].items]}")
    t = _totals(conn, user_id)
    # 3 x 120 = 360 (+ chai 90 if logged) — must NOT be 240+360
    if t.calories not in (360, 450):
        fails.append(f"expected 360 kcal (or 450 with chai), got {t.calories} (double-count?)")
    if "edited" not in replies[-1].effects:
        fails.append("correction did not go through edit_meal (effects missing 'edited')")
    return fails


CASES.append(
    EvalCase(
        name="correction_no_double_count",
        correct_is="'actually that was 3 rotis not 2' -> the SAME meal edited to 3 rotis; totals equal 3 rotis, never 2+3",
        turns=[
            ("had 2 rotis and chai for lunch", None),
            ("actually that was 3 rotis not 2", None),
        ],
        check=check_correction,
        separator=True,
    )
)


def check_fraction(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"expected 1 meal, got {len(meals)}")
        return fails
    bir = [i for i in meals[0].items if "biryani" in i.food.lower()]
    if not bir:
        fails.append(f"no biryani item: {[i.food for i in meals[0].items]}")
    elif abs(bir[0].quantity - 0.67) > 0.06 and abs(bir[0].quantity - 2 / 3) > 0.06:
        fails.append(f"biryani quantity should be ~0.67 box, got {bir[0].quantity}")
    t = _totals(conn, user_id)
    if t.calories > 800:
        fails.append(f"two-thirds of a box (~540 kcal) exceeded 800: {t.calories}")
    return fails


CASES.append(
    EvalCase(
        name="fraction_of_box",
        correct_is="'leftover biryani, maybe two thirds of the box' -> one meal, biryani quantity ~0.67 box (~540 kcal)",
        turns=[("leftover biryani, maybe two thirds of the box", None)],
        check=check_fraction,
    )
)


def check_grazing(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if meals:
        fails.append(f"should NOT have logged unnamed grazing, got {[m.description for m in meals]}")
    if "?" not in replies[-1].reply:
        fails.append(f"should ask a follow-up question, replied: {replies[-1].reply[:120]!r}")
    return fails


CASES.append(
    EvalCase(
        name="grazing_asks_not_guesses",
        correct_is="'skipped lunch but grazed all afternoon' -> nothing logged, ONE question asking what they grazed on",
        turns=[("skipped lunch but grazed all afternoon", None)],
        check=check_grazing,
    )
)


def seed_same_as_yesterday(conn, user_id):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    yday = (datetime.now(tz) - timedelta(days=1)).strftime("%Y-%m-%d")
    db.add_meal(
        conn, user_id, yday, "2 rotis and dal",
        [db.MealItem("roti", 2, "medium roti", 240, 6, 36, 3),
         db.MealItem("dal", 1, "bowl", 180, 9, 25, 4)],
        meal_type="lunch",
    )
    db.add_meal(
        conn, user_id, yday, "1 cup chai",
        [db.MealItem("chai", 1, "cup with sugar", 90, 2, 12, 3)],
        meal_type="snack",
    )


def check_same_as_yesterday(conn, user_id, replies):
    fails = []
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    today = datetime.now(tz).strftime("%Y-%m-%d")
    yday = (datetime.now(tz) - timedelta(days=1)).strftime("%Y-%m-%d")
    today_meals = db.get_meals(conn, user_id, date=today, limit=50)
    if len(today_meals) != 2:
        fails.append(f"SEPARATOR FAIL: expected yesterday's 2 meals copied to today, got {len(today_meals)}")
        return fails
    t_today = db.day_totals(conn, user_id, today)
    t_yday = db.day_totals(conn, user_id, yday)
    if t_today.calories < t_yday.calories - 5 or t_today.calories > t_yday.calories + 5:
        fails.append(f"today {t_today.calories} != yesterday {t_yday.calories} kcal")
    foods_today = sorted(i.food for m in today_meals for i in m.items)
    if foods_today != sorted(["roti", "dal", "chai"]):
        fails.append(f"copied foods wrong: {foods_today}")
    return fails


CASES.append(
    EvalCase(
        name="same_as_yesterday",
        correct_is="'same as yesterday' -> yesterday's meals copied to today exactly (totals match yesterday)",
        turns=[("same as yesterday", None)],
        check=check_same_as_yesterday,
        seed=seed_same_as_yesterday,
        separator=True,
    )
)


def seed_usual_memory(conn, user_id):
    db.save_memory(conn, user_id, "usual", "breakfast: 2 paratha and 1 cup chai")
    db.save_memory(conn, user_id, "target", "140g protein daily", key="protein", num_value=140, unit="g")


def check_my_usual(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"SEPARATOR FAIL: 'my usual' should log exactly 1 meal, got {len(meals)}")
        return fails
    foods = {i.food.lower() for i in meals[0].items}
    if not {"paratha", "chai"} <= foods:
        fails.append(f"usual memory says paratha+chai, logged: {foods}")
    return fails


CASES.append(
    EvalCase(
        name="my_usual_from_memory",
        correct_is="with a saved usual, 'my usual' logs that exact meal — memory, not parsing",
        turns=[("my usual", None)],
        check=check_my_usual,
        seed=seed_usual_memory,
        separator=True,
    )
)


def seed_usual_pattern(conn, user_id):
    """Two days of the same breakfast — a real pattern, not n=1."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    for days_ago in (2, 1):
        d = (datetime.now(tz) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        db.add_meal(
            conn, user_id, d, "2 parathas and chai",
            [
                db.MealItem("paratha", 2, "medium paratha", 600, 12, 80, 24),
                db.MealItem("chai", 1, "cup with sugar", 90, 2, 12, 3),
            ],
            meal_type="breakfast",
        )


def check_usual_learned(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if not meals:
        return ["no meal logged for 'my usual'"]
    today_items = {i.food.lower() for i in meals[0].items}
    if not {"paratha", "chai"} <= today_items:
        fails.append(f"'my usual' should repeat the 2-day breakfast pattern (paratha+chai), got {today_items}")
    mems = db.get_memories(conn, user_id)
    if not any(m.kind == "usual" for m in mems):
        fails.append("expected a 'usual' memory saved once the pattern was used")
    return fails


CASES.append(
    EvalCase(
        name="my_usual_learned_and_saved",
        correct_is="after the same breakfast on 2 straight days, 'my usual' logs that pattern from history AND saves it as a usual memory",
        turns=[("my usual", None)],
        check=check_usual_learned,
        seed=seed_usual_pattern,
    )
)


def check_totals_query(conn, user_id, replies):
    fails = []
    t = _totals(conn, user_id)
    if t.calories < 680 or t.calories > 700:
        fails.append(f"expected ~690 kcal in DB, got {t.calories}")
    if "690" not in replies[-1].reply:
        fails.append(f"reply should include the day's kcal number: {replies[-1].reply[:120]!r}")
    return fails


CASES.append(
    EvalCase(
        name="how_am_i_doing",
        correct_is="after breakfast, 'how am I doing on calories?' reports ~690 kcal from day_totals",
        turns=[
            ("had 2 parathas and chai for breakfast", None),
            ("how am I doing on calories?", None),
        ],
        check=check_totals_query,
    )
)


def check_protein_target(conn, user_id, replies):
    fails = []
    tgts = db.get_memories(conn, user_id)
    prot = [m for m in tgts if m.kind == "target" and m.key == "protein"]
    if not prot:
        fails.append("protein target not saved as memory")
        return fails
    if not prot[0].num_value or abs(prot[0].num_value - 140) > 1:
        fails.append(f"protein target should be 140g, got {prot[0].num_value}")
    if not any("140" in r.reply for r in replies[1:]):
        fails.append(f"protein answer should reference the 140g target: {replies[-1].reply[:120]!r}")
    return fails


CASES.append(
    EvalCase(
        name="protein_target_memory",
        correct_is="'i'm targeting 140g of protein' saved as a target memory, and later protein answers compare against it",
        turns=[
            ("i'm targeting 140g of protein a day", None),
            ("how much protein have I had today?", None),
        ],
        check=check_protein_target,
    )
)


def check_vegetarian(conn, user_id, replies):
    fails = []
    prefs = [m for m in db.get_memories(conn, user_id) if m.kind == "preference" and "vegetarian" in m.content.lower()]
    if not prefs:
        fails.append("vegetarian preference not saved")
    meals = _meals(conn, user_id)
    if meals:
        fails.append(f"meat logged without confirming the vegetarian conflict: {[m.description for m in meals]}")
    if "vegetarian" not in replies[-1].reply.lower():
        fails.append(f"reply should flag the vegetarian conflict: {replies[-1].reply[:120]!r}")
    return fails


CASES.append(
    EvalCase(
        name="vegetarian_conflict_flagged",
        correct_is="'i'm vegetarian' is saved; a following chicken log is FLAGGED and not silently logged",
        turns=[
            ("i'm vegetarian btw", None),
            ("i had chicken curry for dinner", None),
        ],
        check=check_vegetarian,
    )
)


def check_photo_logs(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"a plate photo should log exactly 1 meal, got {len(meals)}")
        return fails
    if meals[0].source != "image":
        fails.append(f"meal source should be 'image', got {meals[0].source!r}")
    items = meals[0].items
    if not items or not any("roti" in i.food.lower() or "chapati" in i.food.lower() for i in items):
        fails.append(f"expected a roti/chapati item, got {[i.food for i in items]}")
    kcal = sum(i.calories for i in items)
    if kcal < 120:  # at least one real roti's worth
        fails.append(f"expected >=120 kcal from a plate of rotis, got {kcal}")
    return fails


CASES.append(
    EvalCase(
        name="photo_of_plate_logs",
        correct_is="a photo of one person's plate (stack of rotis) is logged as ONE image meal",
        turns=[("", PLATE)],
        check=check_photo_logs,
    )
)


def _fuzzy_vision_match(logged_food: str, vis_items: list[dict]):
    """Best-matching vision item by token overlap, or None."""
    best, best_score = None, 0.0
    a = {t for t in logged_food.lower().split() if len(t) > 2}
    for vi in vis_items:
        b = {t for t in str(vi.get("food", "")).lower().split() if len(t) > 2}
        if not a or not b:
            continue
        score = len(a & b) / len(a | b)
        if score > best_score:
            best, best_score = vi, score
    return best if best_score > 0 else None


def check_photo_caption_one_meal(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"SEPARATOR FAIL: photo + caption must be ONE meal, got {len(meals)}")
        return fails
    items = meals[0].items
    if len(items) < 2:
        fails.append(f"thali should yield >=2 items, got {[i.food for i in items]}")
    vis = replies[-1].vision
    if not vis or not vis.items:
        fails.append("no vision items recorded — cannot verify halving")
        return fails
    for li in items:
        vm = _fuzzy_vision_match(li.food, vis.items)
        if vm is None:
            continue  # item the vision didn't name distinctly; skip
        expect = float(vm.get("quantity", 1)) * 0.5
        if li.quantity > expect + 0.06:
            fails.append(
                f"'half was brother's' not applied: {li.food} x{li.quantity:g} "
                f"(vision saw {vm.get('food')} x{float(vm.get('quantity', 1)):g} -> expected ~{expect:g})"
            )
        elif li.quantity < expect - 0.35:
            fails.append(
                f"{li.food} x{li.quantity:g} looks double-halved (vision {vm.get('food')} "
                f"x{float(vm.get('quantity', 1)):g} -> expected ~{expect:g})"
            )
    return fails


CASES.append(
    EvalCase(
        name="photo_plus_caption_one_meal",
        correct_is="photo + 'half of this was my brother's' -> ONE meal, every item halved — both models resolve to one log",
        turns=[("half of this was my brother's", THALI)],
        check=check_photo_caption_one_meal,
        separator=True,
    )
)


def check_photo_asks(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if meals:
        fails.append(f"non-food photo must NOT be logged, got {[m.description for m in meals]}")
    r = replies[-1].reply.lower()
    if not any(k in r for k in ("?", "send", "retake", "another", "tell me")):
        fails.append(f"agent should ask for a real photo / clarification, replied: {replies[-1].reply[:120]!r}")
    return fails


CASES.append(
    EvalCase(
        name="vision_unsure_asks",
        correct_is="a photo the vision model can't identify -> nothing logged, user asked instead of guessed",
        turns=[("", NOT_FOOD)],
        check=check_photo_asks,
    )
)
