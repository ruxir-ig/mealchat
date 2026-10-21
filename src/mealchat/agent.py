"""Agent core — interface-agnostic.

    agent = MealChatAgent()
    result = agent.handle(user_id="ruchir", text="had 2 parathas and chai")

The same core serves the CLI, the web UI, the evals and the latency bench.
Control flow per turn:
  1. (photo path) resize + one structured call to the SEPARATE vision model
  2. build a per-turn tool set bound to (user_id, source, effects)
  3. dynamic system prompt: base + durable memory block + today's date
  4. LangGraph react loop with a strict correction/references policy
  5. persist the turn (short history window), return reply + effects + timings
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from mealchat import db, memory, models, tools, vision
from mealchat.config import Settings, load_settings

_SYSTEM_TEMPLATE = """You are CalorAI's meal-logging assistant chatting over WhatsApp. Users text what they ate ("had 2 parathas and chai for breakfast"), send photos of their plates, make corrections, and ask about their day. You log meals, keep the day's calorie and macro totals accurate, and remember what matters about the user. You are a texting buddy with a nutrition database — warm, quick, never a form.

## Logging rules
- log_meal: one entry per food, quantity in the food's natural unit (2 rotis -> food "roti", quantity 2; two thirds of a biryani box -> food "biryani", quantity 0.67, unit "box", notes "two-thirds of a takeaway box"). Nutrition resolves automatically — do not call nutrition_lookup before a normal log.
- After logging, confirm in ONE short line: what you logged, the meal's kcal, and today's total. State assumptions briefly in parentheses, e.g. "(assumed medium parathas)".
- Do not log skipped meals. If the user grazed or snacked without naming foods, ask ONE short question about what they had.

## Corrections — never double-log
- "actually that was 3 rotis not 2" EDITS an existing meal: get_meals(date="today", food_like="roti") to find it, then edit_meal(meal_id, changes=[{"food": "roti", "quantity": 3}]). Changes are SPOT changes — only the foods you mention change; every other item in the meal (like the chai) is KEPT unchanged. NEVER log a correction as a new meal.
- Removing one item of a meal ("i didn't actually have the chai"): edit_meal with {"food": "chai", "remove": true}. Removing a whole meal ("i didn't eat that at all"): get_meals, then delete_meal.
- A follow-up about a just-logged photo meal ("half of this was my brother's" after the photo) also EDITS that meal — edit_meal(meal_id, changes=[], portion=0.5), never a second log.
- edit_meal's portion works like log_meal's: pass whole-meal item quantities, the share goes in portion.
- If several meals could match a correction, ask which one — briefly.

## References — resolve from data or memory, not guesswork
- "same as yesterday": get_meals(date="yesterday"), then ONE repeat_meals call with all those meal ids — it copies the items exactly, no re-estimation.
- "my usual": if a usual for that meal is in your memory above, log exactly its listed items (shown as 'items: ...' — pass those foods and quantities to log_meal). Otherwise look at get_meals(limit=10): if the same meal repeats across recent days, log/repeat it and save_memory(kind="usual", key=<meal type>, content="...", items=[the foods]) so next time is one step. If there's no repeating meal yet, ask once what their usual is, then save it.

## Photos
- The [Photo analysis] block in the user's message comes from a separate vision model. Its quantities cover the WHOLE visible portion.
- A photo with any caption is ONE meal — ONE log_meal call containing ALL items. The caption beats the vision guess, especially for portions. A share fraction in the caption ("half of this was my brother's", "ate a third of it") goes into log_meal's `portion` argument (0.5, 0.33...) — pass the vision items at their FULL quantities and let the tool scale every one; never halve only some items yourself.
- If the analysis is NOT marked UNCERTAIN, log it right away — map vague items to a sensible generic food ("dal", "mixed veg curry") instead of asking about each bowl. State assumptions in one parenthetical.
- One exception: if a MAIN dish (not a side, garnish or drink) is marked low-confidence, do NOT guess it silently — either log a clearly-generic version and flag it in your reply ("that bowl I guessed as fish curry — tell me if I got it wrong"), or ask one question about it. Low-confidence sides/drinks are fine to log generically.
- ASK once, instead of logging, only when: the analysis is marked UNCERTAIN (never guess then), or it says the photo is a SHARED spread and the caption gives no clue what share was the user's (ask what portion was theirs).
- If your memory says the user is vegetarian and a photo or text shows meat, flag it kindly — don't silently log.

## Ask vs log (where the line is)
- LOG NOW, stating assumptions: named foods with known or standard amounts; clear photos; "same as yesterday"; "my usual"; small unknowns (estimate, note it).
- ASK ONE short question only when: the food is unknown ("grazed all afternoon" — on what?); the portion is unknowable AND calorie-heavy; the vision analysis is uncertain; a correction is ambiguous; or a log contradicts a saved preference (confirm, never refuse).
- Never ask twice for the same missing info.

## Memory
- Save durable facts with save_memory when the user states them: preferences ("i'm vegetarian" -> kind preference, key "diet"), daily targets ("aiming for 140g protein" -> kind target, key protein, value 140, unit g), usuals (kind usual, key = the meal type e.g. breakfast, content = one short sentence, items = the meal's foods — they are stored structured so 'my usual' logs exactly those). Never save one-off meal details or small talk — keep memory lean.
- When the user CORRECTS or retracts a durable fact ("not vegetarian anymore", "new protein target"), forget_memory the old one first, then save the replacement. The memory block shows ids like [m12] — pass that number. Outdated facts must not linger or contradict.
- Use what you remember: targets in day summaries, the vegetarian flag on meat logs, usuals for "my usual".

## Style
- Reply like a text message: under ~35 words, friendly, no bullet lists unless asked, kcal. Round numbers.
- "how am I doing today?" / "how much protein have I had?" -> day_totals; answer quick and specific, compare against targets when they exist."""


def build_system_prompt(memory_block: str, today: str, weekday: str, tz_name: str) -> str:
    """Static policy first, dynamic bits LAST.

    The date, weekday, timezone and the user's memory block differ between
    users and days; keeping them at the end means the long policy text above
    (plus the tool schemas bound before it) is a byte-identical prefix for
    every request — the shape prompt caching needs. Whether a cache actually
    hits is up to the provider.
    """
    dynamic = (
        "## Now\n"
        f"Today is {today} ({weekday}), timezone {tz_name}.\n\n"
        "## What you remember about the user (durable facts — not chat history)\n"
        + (memory_block or "(nothing saved yet — pick up preferences and targets as they come up)")
    )
    return _SYSTEM_TEMPLATE.rstrip() + "\n\n" + dynamic


@dataclass
class AgentResult:
    reply: str
    vision: vision.VisionResult | None = None
    effects: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    vision_ms: float = 0.0
    agent_ms: float = 0.0
    tool_calls: int = 0

    @property
    def photo_path(self) -> bool:
        return self.vision is not None


# Chat history is short-lived context: cap both how many turns and how old
# they can be. Anything durable belongs in memories, not here.
HISTORY_MAX_MESSAGES = 8
HISTORY_WINDOW_SECONDS = 3 * 60 * 60


class MealChatAgent:
    """The agent owns ONE persistent event loop (background thread).

    Why: the LLM clients' httpx connection pools bind to the loop they first
    run on. Re-creating a loop per call (asyncio.run) dead-hands the pooled
    TLS transports ("Event loop is closed") and pays a fresh TLS handshake per
    turn. A single long-lived loop keeps pools warm and turns fast. All
    surfaces (CLI, FastAPI, evals, bench) go through the sync handle() — which
    is also exactly the interface the task spec asks for.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.settings = settings or load_settings()
        self.conn = db.connect(self.settings.db_path)
        db.init_db(self.conn)
        self.text_llm = models.text_llm(self.settings)
        self.vision_llm = models.vision_llm(self.settings)
        # estimator = same text model, temperature 0 — nutrition numbers shouldn't be creative
        self.estimator = models.make_estimator(models.text_llm(self.settings, temperature=0.0))
        # An existing loop can be adopted (used by the eval runner to simulate
        # fresh sessions in-process: the HTTP library pools connections per
        # event loop, so a second loop in one process breaks pooled clients —
        # a real restart is a fresh process where that can't happen).
        if loop is not None:
            self._loop = loop
            self._thread = None
        else:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="mealchat-agent-loop"
            )
            self._thread.start()

    # ------------------------------------------------------------------ core

    def handle(self, user_id: str, text: str, image_path: str | None = None) -> AgentResult:
        """The stable core interface: agent.handle(user_id, text, image_path=None)."""
        fut = asyncio.run_coroutine_threadsafe(
            self._handle(user_id, text, image_path), self._loop
        )
        return fut.result(timeout=180)

    async def _handle(
        self, user_id: str, text: str, image_path: str | None = None
    ) -> AgentResult:
        t0 = time.perf_counter()
        result = AgentResult(reply="")
        vis = None
        if image_path:
            vis = await vision.analyze_photo(
                self.vision_llm, image_path, caption=(text.strip() or None)
            )
            result.vision = vis

        today = datetime.now(self.settings.tz).strftime("%Y-%m-%d")
        source = "image" if image_path else "text"
        effects: dict = {}
        tool_set = tools.build_tools(
            self.conn, self.settings.tz, self.estimator, user_id, source, effects
        )

        system = self._system_prompt(user_id, today)
        history = self._history(user_id)
        human_content = (text.strip() or "(photo, no caption)") if image_path else text
        if vis is not None:
            human_content = f"{human_content}\n\n{vis.prompt_block()}"
        input_msgs = [SystemMessage(system), *history, HumanMessage(human_content)]

        t_agent0 = time.perf_counter()
        graph = create_react_agent(self.text_llm, tool_set)
        state = await graph.ainvoke(
            {"messages": input_msgs}, config={"recursion_limit": 12}
        )
        result.agent_ms = (time.perf_counter() - t_agent0) * 1000
        result.vision_ms = (
            (t_agent0 - t0) * 1000 if image_path else 0.0
        )

        msgs = state.get("messages", [])
        result.tool_calls = sum(1 for m in msgs if isinstance(m, ToolMessage))
        reply = ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                content = m.content
                reply = content if isinstance(content, str) else "".join(
                    getattr(p, "text", "") or "" for p in content
                )
                if reply.strip():
                    break
        result.reply = (reply or "Done — anything else?").strip()
        result.effects = effects
        result.latency_ms = (time.perf_counter() - t0) * 1000

        db.add_message(self.conn, user_id, "user", human_content[:500])
        db.add_message(self.conn, user_id, "assistant", result.reply[:500])
        return result

    # ------------------------------------------------------------- internals

    def _system_prompt(self, user_id: str, today: str) -> str:
        block = memory.memory_block(self.conn, user_id)
        return build_system_prompt(
            memory_block=block,
            today=today,
            weekday=datetime.now(self.settings.tz).strftime("%A"),
            tz_name=self.settings.timezone,
        )

    def _history(self, user_id: str) -> list:
        """Turns from the last 3 hours ONLY (capped count) — conversational
        context, explicitly not memory. Old turns drop out on purpose."""
        cutoff = (
            datetime.now().astimezone() - timedelta(seconds=HISTORY_WINDOW_SECONDS)
        ).isoformat(timespec="seconds")
        out = []
        for m in db.recent_messages(
            self.conn, user_id, limit=HISTORY_MAX_MESSAGES, since=cutoff
        ):
            if m["role"] == "user":
                out.append(HumanMessage(m["content"]))
            elif m["role"] == "assistant":
                out.append(AIMessage(m["content"]))
        return out
