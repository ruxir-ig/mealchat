"""Eval cases — the full test conversation set from the task, with a defined
notion of "correct" per case (in prose + DB assertions).

Three of these are the ones the task says separate good submissions from
average ones (marked SEPARATOR): corrections must not double-count, "same as
yesterday"/"my usual" must come from memory, photo+caption must be one meal.

Each case runs against a FRESH user in a throwaway eval DB. check() returns
a list of failures — empty list = pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from mealchat import db
from mealchat.config import load_settings

# same timezone the agent uses, whatever it's configured to
TZ = load_settings().tz

THALI = "fixtures/photos/Indian_thali.jpg"  # multi-bowl thali (used for the caption case)
PLATE = "fixtures/photos/Chapati.jpg"  # clean single plate of rotis
NOT_FOOD = "fixtures/photos/not_food.jpg"  # laptop keyboard

# eval-runner marker: start a NEW MealChatAgent on the same DB with the user's
# chat history deleted — anything that survives is memory, not history
NEW_SESSION = "__new_session__"


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
    return db.day_totals(conn, user_id, db.local_today(TZ))


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
    by_food = {i.food: i for i in meals[0].items}
    roti = by_food.get("roti")
    if roti is None or roti.quantity != 3:
        fails.append(f"expected 3 rotis after correction, got {[(i.food, i.quantity) for i in meals[0].items]}")
    chai = by_food.get("chai")
    if chai is None or chai.quantity != 1:
        fails.append("the chai must SURVIVE the correction (spot-change semantics, not full-list replace)")
    t = _totals(conn, user_id)
    if t.calories != 450:  # 3 x 120 + 90 — not 360 (chai dropped), not 780 (double-count)
        fails.append(f"expected exactly 450 kcal (3 rotis + chai), got {t.calories}")
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

    tz = TZ
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

    tz = TZ
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
    db.save_memory(
        conn, user_id, "usual", "2 paratha and 1 cup chai", key="breakfast",
        data=json.dumps(
            [
                {"food": "paratha", "quantity": 2, "unit": "medium paratha"},
                {"food": "chai", "quantity": 1, "unit": "cup with sugar"},
            ]
        ),
    )
    db.save_memory(conn, user_id, "target", "140g protein daily", key="protein", num_value=140, unit="g")


def check_my_usual(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"SEPARATOR FAIL: 'my usual' should log exactly 1 meal, got {len(meals)}")
        return fails
    items = {i.food.lower(): i.quantity for i in meals[0].items}
    if items.get("paratha") != 2 or items.get("chai") != 1:
        fails.append(
            f"structured usual says paratha x2 + chai x1; logged exactly: {items}"
        )
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

    tz = TZ
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
    # only TODAY's meals count — the seeded pattern days must not satisfy this
    today_meals = db.get_meals(conn, user_id, date=db.local_today(TZ), limit=50)
    if len(today_meals) != 1:
        return [f"'my usual' should log exactly 1 meal today, got {len(today_meals)}"]
    items = {i.food.lower(): i.quantity for i in today_meals[0].items}
    if items != {"paratha": 2, "chai": 1}:
        fails.append(f"'my usual' should repeat the 2-day breakfast (paratha x2 + chai x1), got {items}")
    usuals = [m for m in db.get_memories(conn, user_id) if m.kind == "usual"]
    saved = {str(i.get("food", "")).lower() for m in usuals for i in db.parse_memory_data(m)}
    if not {"paratha", "chai"} <= saved:
        fails.append(f"expected a STRUCTURED usual (items paratha + chai) saved, got {[m.content for m in usuals]}")
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
    r = replies[-1].reply.lower()
    flagged = "vegetarian" in r or " veg" in r  # "you were veg" counts as flagging
    if not flagged:
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


def _share_error(logged_qty: float, vision_item: dict, share: float = 0.5) -> str | None:
    """Symmetric check that a logged quantity is `share` of what the vision
    model saw: off by more than 15% either way fails — so a double-halved
    item (0.25 of the whole) fails just like an unhalved one."""
    whole = float(vision_item.get("quantity", 1))
    expect = whole * share
    if abs(logged_qty - expect) > max(0.06, 0.15 * expect):
        return (
            f"x{logged_qty:g} should be ~{expect:g} ({share:g} of vision's"
            f" {vision_item.get('food')} x{whole:g})"
        )
    return None


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
        err = _share_error(li.quantity, vm)
        if err:
            fails.append(f"'half was brother's' wrong for {li.food}: {err}")
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


def check_photo_then_caption(conn, user_id, replies):
    """Photo first, caption as a SEPARATE follow-up message. The invariant that
    matters: exactly ONE halved meal at the end, with no second log. Two valid
    paths: (a) photo logged -> caption EDITS that meal; (b) photo asked (vision
    flagged a shared spread) -> caption logs ONE meal at half. Both end the same."""
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"photo + later caption must end as ONE meal, got {len(meals)}")
        return fails
    first, second = replies[0], replies[1]
    first_logged = "logged" in first.effects
    second_logged = "logged" in second.effects
    if first_logged and second_logged:
        fails.append("double log: caption turn logged a SECOND meal instead of editing")
    elif first_logged and "edited" not in second.effects:
        fails.append(f"photo meal was logged, so the caption must EDIT it, effects: {second.effects}")
    elif not first_logged and not second_logged:
        fails.append(f"photo turn asked instead of logging, so the caption must log ONE meal, effects: {second.effects}")
    if first_logged and second_logged:
        return fails  # quantities already wrong; skip the halving noise
    vis = first.vision
    if not vis or not vis.items:
        fails.append("no vision items recorded for the photo turn")
        return fails
    for li in meals[0].items:
        vm = _fuzzy_vision_match(li.food, vis.items)
        if vm is None:
            continue
        err = _share_error(li.quantity, vm)
        if err:
            fails.append(f"{li.food} {err}")
    return fails


CASES.append(
    EvalCase(
        name="photo_then_separate_caption",
        correct_is="photo, then 'half of this was my brother's' as a separate message -> exactly ONE halved meal; if the photo logged, the caption edits it — never a second log",
        turns=[("", PLATE), ("half of this was my brother's", None)],
        check=check_photo_then_caption,
        separator=True,
    )
)


def check_delete_item(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"removing one item must keep the meal, got {len(meals)} meals")
        return fails
    items = {i.food: i.quantity for i in meals[0].items}
    if "chai" in items:
        fails.append(f"the chai should be removed, items: {items}")
    if items.get("roti") != 2:
        fails.append(f"the rotis must survive the removal, items: {items}")
    t = _totals(conn, user_id)
    if t.calories != 240:  # 2 rotis (240) + chai (90) = 330, minus the chai = 240
        fails.append(f"totals should drop by exactly 90 (the chai) to 240, got {t.calories}")
    if "edited" not in replies[-1].effects:
        fails.append(f"single-item removal should go through edit_meal, effects: {replies[-1].effects}")
    return fails


CASES.append(
    EvalCase(
        name="delete_item_from_meal",
        correct_is="'i didn't actually have the chai' removes ONLY the chai: rotis kept, totals drop by 90, same meal edited",
        turns=[
            ("had 2 rotis and chai for lunch", None),
            ("i didn't actually have the chai", None),
        ],
        check=check_delete_item,
    )
)


def check_grazing_follow_up(conn, user_id, replies):
    fails = []
    if "?" not in replies[0].reply:
        fails.append(f"turn 1 should ask what the grazing was: {replies[0].reply[:100]!r}")
    meals = _meals(conn, user_id)
    if not meals:
        fails.append("after naming the foods, the grazing should be logged")
        return fails
    foods = " ".join(i.food.lower() for m in meals for i in m.items)
    if "chips" not in foods or "samosa" not in foods:
        fails.append(f"expected chips + samosa logged, got: {foods!r}")
    return fails


CASES.append(
    EvalCase(
        name="grazing_follow_up_answer",
        correct_is="grazing asks on turn 1; naming the foods ('some chips and a samosa') logs them on turn 2",
        turns=[
            ("skipped lunch but grazed all afternoon", None),
            ("some chips and a samosa", None),
        ],
        check=check_grazing_follow_up,
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


# --------------------------------------------------------------------------
# Memory evals (the area the spec cares about most)
# --------------------------------------------------------------------------


def check_no_memory_for_one_off(conn, user_id, replies):
    fails = []
    mems = db.get_memories(conn, user_id)
    if mems:
        fails.append(
            "one-off meal chatter must NOT be saved as memory, got: "
            f"{[(m.kind, m.content) for m in mems]}"
        )
    return fails


CASES.append(
    EvalCase(
        name="memory_negative_selectivity",
        correct_is="'had biryani at rahul's party last night, it was amazing' logs the meal but saves NOTHING as memory — selectivity means refusing to remember one-offs",
        turns=[("had biryani at rahul's party last night, it was amazing", None)],
        check=check_no_memory_for_one_off,
    )
)


def seed_vegetarian_preference(conn, user_id):
    db.save_memory(conn, user_id, "preference", "vegetarian", key="diet")


def check_retraction(conn, user_id, replies):
    fails = []
    active = db.get_memories(conn, user_id)
    if any(m.content.strip().lower() == "vegetarian" for m in active):
        fails.append("the retracted 'vegetarian' preference is still active after the update")
    return fails


CASES.append(
    EvalCase(
        name="memory_retraction",
        correct_is="with 'vegetarian' saved, 'i'm not vegetarian anymore, i eat chicken now' deactivates the old preference (replaced via forget_memory, not piled onto)",
        turns=[("i'm not vegetarian anymore, i eat chicken now", None)],
        seed=seed_vegetarian_preference,
        check=check_retraction,
    )
)


def check_cross_session_vegetarian(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if meals:
        fails.append(
            "chicken after a saved vegetarian preference must be confirmed first, "
            f"silently logged: {[m.description for m in meals]}"
        )
    if "vegetarian" not in replies[-1].reply.lower():
        fails.append(f"reply should flag the vegetarian conflict: {replies[-1].reply[:120]!r}")
    prefs = [
        m
        for m in db.get_memories(conn, user_id)
        if m.kind == "preference" and "vegetarian" in m.content.lower()
    ]
    if not prefs:
        fails.append("the vegetarian preference must persist across the session boundary")
    # the fact can only have come from memory: history was wiped at NEW_SESSION
    veg_in_history = any(
        m["role"] == "user" and "vegetarian" in m["content"].lower()
        for m in db.recent_messages(conn, user_id, limit=20)
    )
    if veg_in_history:
        fails.append("history should not carry the fact across NEW_SESSION — memory must")
    return fails


CASES.append(
    EvalCase(
        name="memory_cross_session_conflict",
        correct_is="vegetarian saved in session 1; in a FRESH session (history wiped) 'had chicken curry' is flagged, not silently logged — the fact traveled via memory only",
        turns=[
            ("i'm vegetarian btw", None),
            (NEW_SESSION, None),
            ("had chicken curry for dinner", None),
        ],
        check=check_cross_session_vegetarian,
    )
)


def seed_structured_usual(conn, user_id):
    db.save_memory(
        conn, user_id, "usual", "2 paratha and 1 cup chai", key="breakfast",
        data=json.dumps(
            [
                {"food": "paratha", "quantity": 2, "unit": "medium paratha"},
                {"food": "chai", "quantity": 1, "unit": "cup with sugar"},
            ]
        ),
    )


def check_usual_cross_session(conn, user_id, replies):
    fails = []
    meals = _meals(conn, user_id)
    if len(meals) != 1:
        fails.append(f"'my usual' in a fresh session should log exactly 1 meal, got {len(meals)}")
        return fails
    items = {i.food.lower(): i.quantity for i in meals[0].items}
    if items.get("paratha") != 2 or items.get("chai") != 1:
        fails.append(
            f"the structured usual (paratha x2 + chai x1) must be logged exactly, got {items}"
        )
    return fails


CASES.append(
    EvalCase(
        name="memory_usual_cross_session",
        correct_is="a structured usual saved in one session is logged EXACTLY (paratha x2, chai x1) from memory in a fresh session",
        turns=[(NEW_SESSION, None), ("my usual", None)],
        seed=seed_structured_usual,
        check=check_usual_cross_session,
    )
)
