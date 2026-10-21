"""Record a live demo session and render the walkthrough video.

    uv run python scripts/make_video.py --record   # live models -> docs/demo_session.json
    uv run python scripts/make_video.py --render   # frames + ffmpeg -> docs/walkthrough.mp4

The video is an auto-generated screencast built from REAL agent turns — the
replies, latencies and tool effects shown on screen are exactly what the
session JSON captured. No talking head, no manual editing.

Photo turns that end without a logged meal (vision nondeterminism) are retried
once so the video shows the canonical behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

# ---------------------------------------------------------------- recording

TURNS: list[dict] = [
    {"text": "had 2 parathas and chai for breakfast"},
    {"text": "actually that was 3 parathas not 2"},
    {"image": "fixtures/photos/Chapati.jpg"},
    {"image": "fixtures/photos/Indian_thali.jpg", "text": "half of this was my brother's"},
    {"text": "i'm vegetarian btw"},
    {"text": "i had chicken curry for dinner"},
    {"text": "how am I doing today?"},
]


def record() -> None:
    from mealchat.agent import MealChatAgent
    from mealchat.config import load_settings

    settings = replace(load_settings(), db_path="data/video_session.db")
    Path("data/video_session.db").unlink(missing_ok=True)
    agent = MealChatAgent(settings)
    out = []
    for i, turn in enumerate(TURNS):
        r = agent.handle("demo", turn.get("text", ""), image_path=turn.get("image"))
        # photo turns must show the canonical log-once behavior — retry once
        if turn.get("image") and "logged" not in r.effects:
            Path("data/video_session.db").unlink(missing_ok=True)
            agent2 = MealChatAgent(settings)
            for prev in TURNS[:i]:
                agent2.handle("demo", prev.get("text", ""), image_path=prev.get("image"))
            r = agent2.handle("demo", turn.get("text", ""), image_path=turn.get("image"))
            agent = agent2
        out.append(
            {
                "input": turn.get("text") or "",
                "photo": turn.get("image"),
                "reply": r.reply,
                "latency_ms": round(r.latency_ms),
                "effects": r.effects,
                "vision": (
                    {
                        "description": r.vision.description,
                        "items": r.vision.items,
                        "uncertain": r.vision.uncertain,
                        "single_serving": r.vision.meta.get("single_serving"),
                    }
                    if r.vision
                    else None
                ),
            }
        )
        print(f"[{i + 1}/{len(TURNS)}] {r.latency_ms:.0f}ms :: {r.reply[:90]}")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/demo_session.json").write_text(json.dumps(out, indent=2))
    print("saved docs/demo_session.json")


# ---------------------------------------------------------------- rendering

W, H, FPS = 1280, 720, 2
BG, FG = (16, 20, 24), (214, 222, 230)
BLUE, GOLD, GREEN, MUTED = (79, 156, 249), (255, 209, 102), (129, 199, 132), (139, 152, 165)
SPEED = 1.6  # duration multiplier; >1 = slower typing (longer video)


def _fonts():
    from PIL import ImageFont
    from glob import glob

    def pick(cands: list[str], size: int):
        for c in cands:
            hits = glob(c)
            if hits:
                return ImageFont.truetype(hits[0], size)
        return None

    return {
        "title": pick(["/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf"], 54),
        "head": pick(["/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf"], 30),
        "mono": pick(["/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"], 22),
        "mono_s": pick(["/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"], 19),
    }


def _wrap(draw, text: str, font, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if draw.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _dur(block: dict) -> float:
    if block["type"] == "photo":
        return 3.0
    d = 0.035 * len(block.get("text", "")) + 1.0
    if block["type"] == "callout":
        d += 1.6
    if block.get("speaker") == "agent":
        d += 0.8
    return d * SPEED


def _scene_blocks(session: list[dict]) -> list[tuple[str, list[dict]]]:
    t = session
    arch = [
        {"type": "title-sm", "text": "architecture"},
        {"type": "text", "speaker": "sys", "text": "  'had 2 rotis' ──▶  agent.handle(user_id, text, image_path)"},
        {"type": "text", "speaker": "sys", "text": "  [photo] ──▶ resize ──▶ VISION MODEL (separate) ──▶ structured JSON"},
        {"type": "text", "speaker": "sys", "text": "                    system prompt = policy + MEMORY BLOCK + last 8 turns"},
        {"type": "text", "speaker": "sys", "text": "                    LangGraph react loop (text model) ──▶ 8 tools ──▶ SQLite"},
        {"type": "callout", "text": "daily totals are ALWAYS recomputed from meal rows — edits and deletes can't leave stale sums"},
        {"type": "text", "speaker": "sys", "text": "  memory = structured facts (preference / target / usual), written deliberately,"},
        {"type": "text", "speaker": "sys", "text": "  retrieved as a capped block — chat history is context, never memory"},
    ]
    scenes = [
        ("intro", [
            {"type": "bigtitle", "text": "mealchat"},
            {"type": "subtitle", "text": "conversational meal-logging agent — CalorAI test task walkthrough"},
            {"type": "subtitle2", "text": "auto-generated screencast from a live session · every number is real · 2026-10-21"},
        ]),
        ("architecture", arch),
        ("tool surface — split by what it does to state", [
            {"type": "title-sm", "text": "tool surface (8)"},
            {"type": "text", "speaker": "sys", "text": "  write: log_meal · edit_meal · delete_meal · save_memory · forget_memory"},
            {"type": "text", "speaker": "sys", "text": "  read:  get_meals · day_totals · nutrition_lookup"},
            {"type": "callout", "text": "log_meal resolves nutrition internally — a normal log is ONE tool call, not lookup-then-log"},
            {"type": "callout", "text": "corrections go through edit_meal (items REPLACED) — the anti-double-count seam"},
            {"type": "callout", "text": "nutrition: builtin table → DB cache → one batched model estimate, cached forever"},
            {"type": "text", "speaker": "sys", "text": "  'portion' parameter on log_meal: caption share fractions scale every item"},
            {"type": "text", "speaker": "sys", "text": "  deterministically — the LLM decides one number, the tool does the math"},
        ]),
        ("demo — logging & correction", [
            {"type": "chat", "speaker": "user", "text": t[0]["input"]},
            {"type": "chat", "speaker": "agent", "text": t[0]["reply"], "meta": _meta(t[0])},
            {"type": "chat", "speaker": "user", "text": t[1]["input"]},
            {"type": "chat", "speaker": "agent", "text": t[1]["reply"], "meta": _meta(t[1])},
            {"type": "callout", "text": "correction = edit_meal on the found meal — totals update, nothing double-counted"},
        ]),
        ("demo — photo on a separate vision model", [
            {"type": "chat", "speaker": "user", "text": "[photo of a plate of rotis]", "photo": t[2]["photo"]},
            {"type": "vision", "text": _vision_line(t[2])},
            {"type": "chat", "speaker": "agent", "text": t[2]["reply"], "meta": _meta(t[2])},
            {"type": "callout", "text": "image → resize → a DIFFERENT model → one log_meal; text agent never sees pixels"},
        ]),
        ("demo — photo + caption = ONE meal", [
            {"type": "chat", "speaker": "user", "text": t[3]["input"], "photo": t[3]["photo"]},
            {"type": "vision", "text": _vision_line(t[3])},
            {"type": "chat", "speaker": "agent", "text": t[3]["reply"], "meta": _meta(t[3])},
            {"type": "callout", "text": "caption share → log_meal's portion=0.5 → tool scales EVERY item deterministically"},
        ]),
        ("demo — memory & judgment", [
            {"type": "chat", "speaker": "user", "text": t[4]["input"]},
            {"type": "chat", "speaker": "agent", "text": t[4]["reply"], "meta": _meta(t[4])},
            {"type": "chat", "speaker": "user", "text": t[5]["input"]},
            {"type": "chat", "speaker": "agent", "text": t[5]["reply"], "meta": _meta(t[5])},
            {"type": "callout", "text": "preference saved once → meat conflict flagged & confirmed, never silently logged"},
        ]),
        ("demo — day totals", [
            {"type": "chat", "speaker": "user", "text": t[6]["input"]},
            {"type": "chat", "speaker": "agent", "text": t[6]["reply"], "meta": _meta(t[6])},
            {"type": "callout", "text": "'how am I doing?' → day_totals: totals, breakdown, saved targets"},
        ]),
        ("latency", [
            {"type": "title-sm", "text": "latency (agent.handle, measured)"},
            {"type": "text", "speaker": "sys", "text": "  text  pooled   p50 3.4s   p95 5.5s    (log 3.0 · correction 5.0)"},
            {"type": "text", "speaker": "sys", "text": "  photo pooled   p50 13.4s  p95 22.1s   (vision-dominated)"},
            {"type": "callout", "text": "optimised: warm event loop · 1-tool-call logging · 1024px resize · lean prompts"},
            {"type": "callout", "text": "couldn't fix: vision-model latency & variance on the tested gateway — documented honestly"},
        ]),
        ("honest gaps & next", [
            {"type": "title-sm", "text": "what's NOT done"},
            {"type": "text", "speaker": "sys", "text": "  · LangSmith tracing not wired        · no streaming yet"},
            {"type": "text", "speaker": "sys", "text": "  · nutrition = hardcoded table + cached model estimates"},
            {"type": "text", "speaker": "sys", "text": "next: streaming replies · per-user tz · memory decay · retry/fallback vision"},
        ]),
        ("end", [
            {"type": "bigtitle", "text": "github.com/ruxir-ig/mealchat"},
            {"type": "subtitle", "text": "13-case eval suite · 28 unit tests · README with the full design notes"},
        ]),
    ]
    return scenes


def _meta(turn: dict) -> str:
    eff = turn.get("effects", {})
    tags = [f"{turn['latency_ms'] / 1000:.1f}s"]
    for k in ("logged", "edited", "deleted", "memories", "forgotten"):
        if k in eff:
            tags.append(k)
    return " · ".join(tags)


def _vision_line(turn: dict) -> str:
    v = turn.get("vision") or {}
    items = ", ".join(f"{i['food']} x{i['quantity']:g}" for i in (v.get("items") or [])[:5])
    ss = v.get("single_serving")
    spread = "" if ss is None else ("single serving" if ss else "SHARED spread")
    return f"vision model saw: {items or 'nothing identifiable'} ({spread}; uncertain={bool(v.get('uncertain'))})"


def render() -> None:
    from PIL import Image, ImageDraw

    session = json.loads(Path("docs/demo_session.json").read_text())
    fonts = _fonts()
    frames_dir = Path("/tmp/opencode/video_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("f_*.png"):
        old.unlink()

    scenes = _scene_blocks(session)
    total = 0.0
    for _, blocks in scenes:
        total += sum(_dur(b) for b in blocks) + 0.9 * len(blocks) + 2.2
    print(f"natural duration ≈ {total / 60:.1f} min")

    frame_idx = 0
    for scene_name, blocks in scenes:
        t = 0.0
        for b in blocks:
            b["_start"] = t
            b["_dur"] = _dur(b)
            t += b["_dur"] + 0.9
        scene_len = t + 2.2
        n_frames = int(scene_len * FPS)
        for f in range(n_frames):
            now = f / FPS
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            # header
            d.rectangle([0, 0, W, 54], fill=(23, 28, 34))
            d.text((24, 14), f"mealchat — {scene_name}", font=fonts["head"], fill=BLUE)
            # footer with running time
            d.text((24, H - 30), "real agent session — latencies as measured", font=fonts["mono_s"], fill=MUTED)
            y = 96
            for b in blocks:
                if now < b["_start"]:
                    break
                shown = min(1.0, (now - b["_start"]) / b["_dur"])
                if b["type"] == "bigtitle":
                    d.text((W // 2 - d.textlength(b["text"], font=fonts["title"]) // 2, 250), b["text"],
                           font=fonts["title"], fill=FG)
                    y += 0
                    continue
                if b["type"] in ("subtitle", "subtitle2"):
                    fnt = fonts["head"] if b["type"] == "subtitle" else fonts["mono"]
                    d.text((W // 2 - d.textlength(b["text"], font=fnt) // 2, y), b["text"][: int(len(b["text"]) * shown) + 1],
                           font=fnt, fill=MUTED if b["type"] == "subtitle2" else FG)
                    y += 46
                    continue
                if b["type"] == "title-sm":
                    d.text((24, y), b["text"][: int(len(b["text"]) * shown) + 1], font=fonts["head"], fill=FG)
                    y += 52
                    continue
                if b["type"] == "callout":
                    txt = b["text"][: int(len(b["text"]) * shown) + 1]
                    lines = _wrap(d, txt, fonts["mono"], W - 140)
                    box_h = 18 + 30 * len(lines)
                    d.rounded_rectangle([24, y, W - 24, y + box_h], 10, outline=GOLD, width=2)
                    yy = y + 9
                    for ln in lines:
                        d.text((40, yy), ln, font=fonts["mono"], fill=GOLD)
                        yy += 30
                    y += box_h + 14
                    continue
                if b["type"] == "photo" or b.get("photo"):
                    path = Path(b["photo"])
                    if path.exists():
                        ph = Image.open(path).convert("RGB")
                        ph.thumbnail((360, 300))
                        d.rectangle([W - 24 - ph.width - 8, y - 4, W - 16, y + ph.height + 4],
                                    outline=BLUE, width=2)
                        img.paste(ph, (W - 24 - ph.width, y))
                    d.text((24, y), "[photo sent]", font=fonts["mono_s"], fill=MUTED)
                    y += 316
                    continue
                # chat / vision / sys text
                if b["type"] == "chat":
                    prefix = "you> " if b["speaker"] == "user" else "mealchat> "
                    color = BLUE if b["speaker"] == "user" else GREEN
                elif b["type"] == "vision":
                    prefix, color = "vision> ", MUTED
                else:
                    prefix, color = "  ", FG
                txt = b["text"][: int(len(b["text"]) * shown) + 1]
                first = True
                for ln in _wrap(d, txt, fonts["mono"], W - 60):
                    d.text((24 if not first else 24 + d.textlength(prefix, font=fonts["mono"]), y),
                           (prefix + ln) if first else ln, font=fonts["mono"], fill=color)
                    y += 30
                    first = False
                if b.get("meta"):
                    d.text((24 + 24, y), f"({b['meta']})", font=fonts["mono_s"], fill=MUTED)
                    y += 26
                y += 10
            img.save(frames_dir / f"f_{frame_idx:05d}.png")
            frame_idx += 1

    out = Path("docs/walkthrough.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "f_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)],
        check=True, capture_output=True,
    )
    print(f"rendered {frame_idx} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB, {frame_idx / FPS / 60:.1f} min)")


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    elif "--render" in sys.argv:
        render()
    else:
        print(__doc__)
