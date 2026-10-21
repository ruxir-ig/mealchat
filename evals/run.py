"""Eval runner: fresh eval DB, fresh user per case, DB-level assertions.

    uv run python -m evals.run            # all cases
    uv run python -m evals.run correction photo

Prints one PASS/FAIL line per case plus the failure reasons, then a summary.
Replies and latencies are printed too — the eval doubles as a conversation
demo of the full test-conversation set from the task.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

from mealchat import db
from mealchat.agent import MealChatAgent
from mealchat.config import load_settings

from evals.cases import CASES


def run(selected: list[str] | None = None) -> int:
    settings = load_settings()
    # dedicated throwaway DB for evals — never touches real data
    eval_db = Path("data/evals.db")
    if eval_db.exists():
        eval_db.unlink()
    settings = replace(settings, db_path=str(eval_db))

    agent = MealChatAgent(settings)
    passed, failed = [], []
    for case in CASES:
        if selected and not any(s in case.name for s in selected):
            continue
        user_id = f"eval_{case.name}"
        if case.seed:
            case.seed(agent.conn, user_id)
        replies = []
        t0 = time.perf_counter()
        for text, image in case.turns:
            r = agent.handle(user_id, text, image_path=image)
            replies.append(r)
        wall = time.perf_counter() - t0
        fails = case.check(agent.conn, user_id, replies)
        status = "PASS" if not fails else "FAIL"
        tag = " [separator]" if case.separator else ""
        print(f"{status}: {case.name}{tag} ({wall:.1f}s)")
        for text, _ in case.turns:
            print(f"    > {text if text else '[photo]' }")
        for r in replies:
            print(f"    < {r.reply}")
            if fails:
                print(f"      latency {r.latency_ms:.0f}ms, effects {r.effects}")
        for f in fails:
            print(f"    ! {f}")
        (passed if not fails else failed).append(case.name)
    print(f"\n{len(passed)} passed, {len(failed)} failed / {len(passed) + len(failed)} total")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or None))
