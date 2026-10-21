"""Terminal chat with the agent.

    uv run mealchat                          # REPL (cli-user)
    uv run mealchat --user ash "had 2 rotis" # one-shot
    uv run mealchat --image plate.jpg lunch  # one-shot with a photo

In the REPL, prefix a line with @path to send a photo:
    @fixtures/photos/Chapati.jpg this was lunch
"""

from __future__ import annotations

import argparse
import re
import sys

from mealchat.agent import MealChatAgent


def _parse_photo_line(line: str) -> tuple[str, str | None]:
    m = re.match(r"@(\S+)\s*(.*)", line)
    if m:
        return m.group(2), m.group(1)
    return line, None


def main() -> None:
    ap = argparse.ArgumentParser(prog="mealchat", description="chat with the meal-logging agent")
    ap.add_argument("--user", default="cli-user", help="user id (memory + meals are per-user)")
    ap.add_argument("--image", help="send this image with the one-shot message")
    ap.add_argument("message", nargs="*", help="one-shot message; omit for the REPL")
    args = ap.parse_args()

    agent = MealChatAgent()

    if args.message or args.image:
        r = agent.handle(args.user, " ".join(args.message), image_path=args.image)
        print(r.reply)
        print(f"[{r.latency_ms / 1000:.1f}s]", file=sys.stderr)
        return

    print("mealchat — text what you ate. '@photo.jpg caption' sends a photo. Ctrl-D to quit.")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            break
        if line.lower() in ("quit", "exit"):
            break
        if not line:
            continue
        text, image = _parse_photo_line(line)
        r = agent.handle(args.user, text, image_path=image)
        print(f"mealchat> {r.reply}   [{r.latency_ms / 1000:.1f}s]")


if __name__ == "__main__":
    main()
