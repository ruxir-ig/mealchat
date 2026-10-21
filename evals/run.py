"""Eval runner: fresh eval DB, fresh user per case/run, DB-level assertions.

    uv run python -m evals.run                # all cases, once
    uv run python -m evals.run correction photo  # only matching cases
    uv run python -m evals.run --repeat 3    # 3 runs per case, pass rates

Prints one PASS/FAIL line per case (per run when --repeat), then a summary.
With --repeat it also reports a soft style metric across all replies — the
share of replies with 45 words or fewer, and with at most one question mark —
as signal, not pass/fail.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from evals.cases import CASES, NEW_SESSION
from mealchat.agent import MealChatAgent
from mealchat.config import load_settings


def _print_turns(case):
    for text, _ in case.turns:
        label = "[new session]" if text == NEW_SESSION else (text or "[photo]")
        print(f"    > {label}")


def run(selected: list[str] | None = None, repeat: int = 1) -> int:
    settings = load_settings()
    # dedicated throwaway DB for evals — never touches real data
    eval_db = Path("data/evals.db")
    if eval_db.exists():
        eval_db.unlink()
    settings = replace(settings, db_path=str(eval_db))

    agent = MealChatAgent(settings)
    all_replies = []
    case_rates: list[tuple[str, int, int]] = []
    for case in CASES:
        if selected and not any(s in case.name for s in selected):
            continue
        passes, first_fail = 0, None
        for run_idx in range(repeat):
            suffix = "" if repeat == 1 else f"_r{run_idx}"
            user_id = f"eval_{case.name}{suffix}"
            if case.seed:
                case.seed(agent.conn, user_id)
            replies = []
            t0 = time.perf_counter()
            for text, image in case.turns:
                if text == NEW_SESSION:
                    # fresh agent on the SAME DB, history wiped — only memory
                    # can carry facts across this boundary. Adopts the
                    # existing event loop (a second loop in-process would
                    # break the HTTP library's pooled connections; a real
                    # restart is a new process, where that can't happen).
                    agent = MealChatAgent(settings, loop=agent._loop)
                    with agent.conn:
                        agent.conn.execute(
                            "DELETE FROM messages WHERE user_id = ?", (user_id,)
                        )
                    continue
                r = agent.handle(user_id, text, image_path=image)
                replies.append(r)
            wall = time.perf_counter() - t0
            fails = case.check(agent.conn, user_id, replies)
            all_replies.extend(replies)
            if not fails:
                passes += 1
                if repeat == 1:
                    print(f"PASS: {case.name} ({wall:.1f}s)")
                    _print_turns(case)
                    for r in replies:
                        print(f"    < {r.reply}")
            else:
                first_fail = first_fail or (fails, replies, wall)
                if repeat == 1:
                    print(f"FAIL: {case.name} ({wall:.1f}s)")
                    _print_turns(case)
                    for r in replies:
                        print(f"    < {r.reply}")
                        print(f"      latency {r.latency_ms:.0f}ms, effects {r.effects}")
                else:
                    print(f"  run {run_idx + 1}: FAIL ({wall:.1f}s)")
            if repeat > 1 and not fails:
                print(f"  run {run_idx + 1}: PASS ({wall:.1f}s)")
        tag = " [separator]" if case.separator else ""
        if repeat == 1:
            if passes == 1:
                case_rates.append((case.name, 1, 1))
            else:
                case_rates.append((case.name, 0, 1))
        else:
            mark = "PASS" if passes == repeat else ("PARTIAL" if passes else "FAIL")
            print(f"{mark} {passes}/{repeat}: {case.name}{tag}")
            case_rates.append((case.name, passes, repeat))
            if first_fail:
                fails, replies, _ = first_fail
                _print_turns(case)
                for r in replies:
                    print(f"    < {r.reply}")
                for f in fails:
                    print(f"    ! {f}")

    total_runs = sum(t for _, _, t in case_rates)
    total_passes = sum(p for _, p, _ in case_rates)
    print(f"\n{total_passes}/{total_runs} case runs passed across {len(case_rates)} case(s)")
    for name, p, t in case_rates:
        if p != t:
            print(f"  flaky/failed: {name} {p}/{t}")

    if all_replies:
        n = len(all_replies)
        short = sum(1 for r in all_replies if len(r.reply.split()) <= 45)
        q = sum(1 for r in all_replies if r.reply.count("?") <= 1)
        print(
            f"style (soft metric): {short}/{n} replies ≤45 words ({100 * short // n}%), "
            f"{q}/{n} with ≤1 '?' ({100 * q // n}%)"
        )
    return 0 if total_passes == total_runs else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="evals")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case (pass rates)")
    ap.add_argument("filters", nargs="*", help="substring case filters")
    args = ap.parse_args()
    sys.exit(run(args.filters or None, max(1, args.repeat)))
