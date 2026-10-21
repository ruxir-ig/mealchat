"""Latency bench — p50/p95 of agent.handle() for text and photo paths.

    uv run python scripts/bench.py          # full run (~6-10 min on the tested config)
    uv run python scripts/bench.py text      # only scenarios matching 'text'

Scenarios (fresh user + seeded DB state per run so each measured turn is
comparable):
  text_log        'had 2 rotis and dal for lunch'        — the common path
  text_query      'how am I doing today?'                — read path
  text_correction 'actually that was 3 rotis not 2'      — edit path
  photo_log       [chapati plate photo, no caption]      — vision + agent
  photo_caption   [same photo] + 'this was my lunch'     — vision + caption

Numbers are wall-clock around agent.handle() — the full core, including
vision preprocessing and the graph loop, excluding any UI/HTTP.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from mealchat import db
from mealchat.agent import MealChatAgent
from mealchat.config import load_settings

PHOTO = "fixtures/photos/Chapati.jpg"

# a food that is NOT in the builtin table — its cached estimate row is wiped
# before every bench run, so each run pays for the full estimator round trip
UNCACHED_FOOD_KEY = "truffle pasta with mushrooms"
UNCACHED_FOOD_TURN = f"had a bowl of {UNCACHED_FOOD_KEY} for lunch"


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs) + 0.999999) - 1)]


def seed_meal(conn, user_id, tz):
    today = db.local_today(tz)
    db.add_meal(
        conn, user_id, today, "2 rotis",
        [db.MealItem("roti", 2, "medium roti", 240, 6, 36, 3)],
        meal_type="lunch",
    )


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    settings = load_settings()
    bench_db = Path("data/bench.db")
    if bench_db.exists():
        bench_db.unlink()
    settings = replace(settings, db_path=str(bench_db))
    agent = MealChatAgent(settings)

    scenarios = [
        ("text_log", "had 2 rotis and dal for lunch", None, False, 20),
        ("text_query", "how am I doing today?", None, True, 20),
        ("text_correction", "actually that was 3 rotis not 2", None, True, 20),
        ("text_log_unknown", UNCACHED_FOOD_TURN, None, False, 20),
        ("photo_log", "", PHOTO, False, 10),
        ("photo_caption", "this was my lunch", PHOTO, False, 10),
    ]

    all_text: list[float] = []
    all_photo: list[float] = []
    print(f"{'scenario':<18} {'n':>3} {'p50':>7} {'p95':>7} {'mean':>7}  vision/agent split (photo)")
    for name, text, photo, seed, n in scenarios:
        if which and which not in name:
            continue
        lats: list[float] = []
        vis_lats: list[float] = []
        agent_lats: list[float] = []
        tool_calls = 0
        for i in range(n):
            uid = f"bench_{name}_{i}"
            if seed:
                seed_meal(agent.conn, uid, settings.tz)
            if name == "text_log_unknown":
                # wipe the cached estimate so every run pays for the estimator
                with agent.conn:
                    agent.conn.execute(
                        "DELETE FROM foods WHERE key = ?", (UNCACHED_FOOD_KEY,)
                    )
            r = agent.handle(uid, text, image_path=photo)
            lats.append(r.latency_ms / 1000)
            if photo:
                vis_lats.append(r.vision_ms / 1000)
                agent_lats.append(r.agent_ms / 1000)
            tool_calls += r.tool_calls
            (all_photo if photo else all_text).append(r.latency_ms / 1000)
        split = (
            f"{pct(vis_lats, .5):.1f}/{pct(agent_lats, .5):.1f}s"
            if photo
            else f"tools/turn {tool_calls / n:.1f}"
        )
        print(
            f"{name:<18} {n:>3} {pct(lats, .50):>6.1f}s {pct(lats, .95):>6.1f}s "
            f"{sum(lats) / n:>6.1f}s  {split}"
        )

    print("\npooled (all text runs vs all photo runs):")
    for label, xs in (("text", all_text), ("photo", all_photo)):
        if xs:
            print(
                f"  {label:<6} n={len(xs):<3} p50={pct(xs, .50):.1f}s p95={pct(xs, .95):.1f}s "
                f"mean={sum(xs) / len(xs):.1f}s"
            )


if __name__ == "__main__":
    main()
