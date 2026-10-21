# mealchat 🥗

A conversational meal-logging agent. You text what you ate — *"had 2 parathas and chai for breakfast"*, *"leftover biryani, maybe two thirds of the box"*, a photo of your plate with *"half of this was my brother's"* — and it logs it, keeps running daily calorie/macro totals that survive edits and corrections, and remembers the things worth remembering about you across sessions.

Built as the CalorAI AI-Engineer test task. Python, LangGraph, SQLite, two deliberately different models for text and vision.

**Walkthrough video (6.4 min): [docs/walkthrough.mp4](docs/walkthrough.mp4)** — an auto-generated screencast rendered from a real live session; every reply, latency and tool effect on screen comes verbatim from [docs/demo_session.json](docs/demo_session.json). It covers the architecture, the tool split, memory write/read, a photo case, the photo+caption case, a correction, and the latency discussion.

```
                    ┌────────────────────────────────────────────┐
   "had 2 rotis" ──▶│ agent.handle(user_id, text, image_path)   │
   [photo of plate]─▶│                                            │
                    │  photo? ── resize ──▶ VISION MODEL (separate)│
                    │                        └── structured JSON │
                    │  system prompt = policy + MEMORY BLOCK      │
                    │                + last 8 turns (context)     │
                    │        LangGraph react loop (text model)      │
                    │   tools ──▶ SQLite (meals, foods, memories) │
                    └────────────────────────────────────────────┘
```

## Status & honest gaps

All six core features are implemented and covered by a 13-case eval suite (all passing on the final state) plus 28 unit tests. What is **not** done:

- **LangSmith tracing** — not wired (no key). The agent emits per-turn latency / vision / tool-call effects in `AgentResult`, which is where the numbers below come from.
- **Streaming** — not implemented; `handle()` returns a complete turn.
- Vision-model latency on the tested gateway is the weak point — see the latency section for what I tried and what I'd do with budget.
- Nutrition data is a hardcoded Indian-first table + cached model estimates — by design (the task says nutrition accuracy is not the point; the tradeoff is documented below).
- The commit history is genuine but compressed — several chunks were finished inside the same working window, so some commits share timestamps; it is not a minute-by-minute narrative.

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), an API key for any OpenAI-compatible provider. **Defaults are plain public OpenAI models**, so a fresh clone runs with just an OpenAI key; every value is overridable in `.env`.

```bash
git clone https://github.com/ruxir-ig/mealchat && cd mealchat
uv sync                       # deps + .venv from the lockfile
cp .env.example .env          # then edit MEALCHAT_API_KEY (and models if you want)
```

| var | default | what it is |
|---|---|---|
| `MEALCHAT_API_BASE` | `https://api.openai.com/v1` | any OpenAI-compatible base URL |
| `MEALCHAT_API_KEY` | — | your key |
| `MEALCHAT_TEXT_MODEL` | `gpt-4o-mini` | conversation/tool-calling model (text-only is fine) |
| `MEALCHAT_VISION_MODEL` | `gpt-4o` | must accept image input, must be a **different** model |
| `MEALCHAT_TEXT_REASONING` / `MEALCHAT_VISION_REASONING` | *(empty)* | optional `reasoning_effort`; only sent when set — most chat-completions models reject the param, so leave blank unless yours supports it |
| `MEALCHAT_VISION_API_BASE` / `MEALCHAT_VISION_API_KEY` | same gateway | optional separate route for vision |
| `MEALCHAT_TIMEZONE` | `Asia/Kolkata` | local date for "today" boundaries |
| `MEALCHAT_DB` | `data/mealchat.db` | SQLite file |

Run it:

```bash
uv run mealchat                                     # terminal chat (REPL)
uv run mealchat --image fixtures/photos/Chapati.jpg this was lunch
uv run uvicorn mealchat.server:app                 # web chat at http://127.0.0.1:8000
uv run pytest                                       # 28 unit tests (no network)
uv run python -m evals.run                          # 13-case eval suite (live models)
uv run python scripts/bench.py                      # latency p50/p95 (live models)
uv run python scripts/make_video.py --record && uv run python scripts/make_video.py --render
```

The core interface is interface-agnostic — the CLI, web UI, evals, bench and video all call exactly:

```python
result = agent.handle(user_id, text, image_path=None)   # -> AgentResult
# result.reply, result.vision, result.effects, result.latency_ms / vision_ms / agent_ms
```

## Model choices and why

There are two configurations:

- **Defaults (what a clean clone runs): `gpt-4o-mini` for text, `gpt-4o` for vision.** Public, stable model names on the standard OpenAI API, so the repo has zero private dependencies. gpt-4o-mini is cheap and fast with dependable tool-calling; gpt-4o is a stronger image reasoner for the vision path — and they are different models, as the task requires.
- **Tested configuration (what the evals/latency numbers below actually ran on): `deepseek-v4p1-flash` for text, `glm-5.3-flash` for vision**, behind an OpenAI-compatible gateway I have access to. Rationale on the axes the spec asks about:
  - **Cost** — flash-class models cost a small fraction of frontier-tier pricing per turn, and this is a high-volume, low-stakes domain (a wrong calorie estimate is recoverable; a $0.05 turn is not).
  - **Latency** — measured 1.2–2.5s round trips on the text path. I verified structured `tool_calls` output and a live multimodal call on each model *before* building on them.
  - **Quality** — tool-calling was the hard requirement for the text model (verified clean: correct JSON arguments, no hallucinated tools); for vision, food identification on the fixture plates was accurate with well-calibrated confidence flags.

The handoff between them is the same regardless of provider: photo → resize → one structured vision call → its JSON (foods, quantities, confidence, single-serving flag, uncertainty) is injected as a delimited block into the text agent's turn. **If the vision model returns `uncertain: true`** (non-food, unidentifiable) **or the call/parse fails, the agent is instructed — and prompted with the vision model's own suggested question — to ask the user instead of guessing.** The eval `vision_unsure_asks` covers this with a keyboard photo: nothing is logged, the agent asks for a real photo. A low-confidence *main* dish is never silently guessed either — it's logged generically and flagged in the reply, or questioned.

## How memory works

This is the part I spent the most design time on, so the rules are explicit:

- **What I store:** structured, deduplicated *facts* in a `memories` table — `kind ∈ {preference, target, usual, note}` (+ `key`/`num_value`/`unit` for numeric targets like `protein=140g`). Nothing else. Targets upsert by `(user, kind, key)`; repeated facts dedupe instead of piling up.
- **When it's written:** only when the agent decides a fact is durable, via the `save_memory` tool — "i'm vegetarian" → preference; "i'm targeting 140g of protein" → target; a "my usual" that had to be inferred from a repeating pattern gets saved so it's one step next time. One-off meal details and small talk never get stored.
- **How facts get removed:** `forget_memory` — when the user corrects or retracts a durable fact ("not vegetarian anymore", "new protein target"), the agent forgets the old one before saving the replacement, so contradicting facts can't pile up.
- **How it's retrieved:** one SQLite query per turn, rendered into a capped block in the system prompt — preferences and targets first, then usuals, then newest notes, max 10 lines, a few hundred tokens. No embeddings needed at this scale (dozens of rows, not thousands); if memory grew, the same table would take a keyword/vector index without changing the contract.
- **What memory is *not*:** the last 8 chat turns are also injected (a separate `messages` table, capped, for conversational coherence), but that's *context*, not memory — it neither survives growth nor carries durable facts. The proof it works: a fresh session with a seeded `usual` memory and zero history resolves "my usual" correctly (eval `my_usual_from_memory`), and "same as yesterday" resolves from the `meals` table, not from anything said in chat (eval `same_as_yesterday`).
- **Why this satisfies "selective":** every byte in the prompt block was individually judged worth keeping, is structured, deduplicated, updateable and removable — versus "store every message and hope".

## Tool design — why this split

Eight tools, split by what they do to state, not by data type:

| tool | path | why it exists |
|---|---|---|
| `log_meal` | write | the one happy-path call: items + `portion` fraction; resolves nutrition internally |
| `edit_meal` | write | corrections REPLACE items — this is what makes "3 rotis not 2" impossible to double-count |
| `delete_meal` | write | soft delete, keeps audit trail, drops out of totals |
| `save_memory` | write | durable facts only (see above) |
| `forget_memory` | write | retract outdated facts so they don't linger and contradict |
| `get_meals` | read | past meals, filterable by date/food — the correction-finder and "same as yesterday" resolver |
| `day_totals` | read | the "how am I doing?" answer: totals + breakdown + targets + remaining |
| `nutrition_lookup` | read | one-off questions ("calories in a samosa?"), table hit or one-time estimate then cached |

The judgment calls: `log_meal` resolves nutrition *inside* the tool (builtin table → DB cache → one batched LLM estimate for the whole meal, cached forever) so a normal log is **one** tool call, not a lookup-then-log dance. `edit_meal` is separate from `log_meal` because corrections are a different *act* — the system prompt says corrections must find the meal (`get_meals(food_like=...)`) and edit it, never re-log. And `portion` is a parameter (not LLM arithmetic) because "half of this was my brother's" must scale every item deterministically — the model decides one number, the tool does the multiplication.

Totals correctness is a data-layer property, not a prompt property: `day_totals` always aggregates from live rows, never incrementally maintained — so edits and deletions can't leave stale sums (unit-tested in `test_db.py`, including the sign of "kcal left" vs a saved target).

## When the agent asks vs just logs — where I drew the line

**Logs immediately, stating assumptions in a parenthetical:** named foods with known/standard amounts ("2 parathas" → medium parathas, 690 kcal); fractions of standard containers ("two thirds of the box" → 0.67 × takeaway box ≈ 536 kcal); confident photo analyses (generic names for unclear dishes, not interrogation); "same as yesterday"; "my usual".

**Asks exactly one short question when:** the food is unknown ("grazed all afternoon" — on *what*?); the vision analysis is `uncertain` (never guess); the photo clearly shows a shared/family spread with no caption clue about the user's share (asks what portion was theirs); a correction is ambiguous (several candidate meals); or a log contradicts a saved preference (chicken after "i'm vegetarian" → confirms, never refuses, never silently logs).

The principle: over-asking kills the texting feel, under-asking poisons the data. Small unknowns get estimates + a stated assumption; identity/portion unknowns that would materially change the numbers get one question. Never two questions in one reply.

## Evals

`uv run python -m evals.run` — 13 cases, fresh user + throwaway DB each, with a per-case definition of "correct" in `evals/cases.py` and DB-level assertions (not just reply-string matching). The three separator cases from the task:

- **`correction_no_double_count`** — "actually that was 3 rotis not 2" → same meal edited to 3 rotis, day total 360 (or 450 with the chai), effects show `edit_meal`, never a second meal. ✅
- **`same_as_yesterday` / `my_usual_from_memory`** — resolved from the meals table / the memory block, not from parsing or chat history. ✅
- **`photo_plus_caption_one_meal`** — thali photo + "half of this was my brother's" → exactly ONE meal, every item halved (verified against the vision model's own whole-portion quantities, so double-halving is caught too). ✅

Current run: **13/13 passing** (live models; conversation-level nondeterminism means photo cases occasionally wobble on *which* dishes the vision model reports — the assertions are robust to that).

## Latency

Measured by `scripts/bench.py` around `agent.handle()` — the full core (vision preprocessing + graph loop + tools), excluding UI/HTTP. Single user, warm connections, tested configuration, 2026-10-21.

| scenario | n | p50 | p95 | mean | notes |
|---|---|---|---|---|---|
| text: log a meal | 10 | **3.0s** | 4.1s | 3.2s | 1 tool call |
| text: day totals query | 10 | **3.4s** | 4.7s | 3.4s | 1 tool call |
| text: correction | 10 | **5.0s** | 5.6s | 5.0s | 2 tool calls (find + edit) |
| photo: log plate | 6 | **9.4s** | 22.1s | 13.5s | vision 5.8s + agent 4.0s (p50 splits) |
| photo: + caption | 6 | **13.4s** | 17.7s | 13.5s | vision 9.6s + agent 3.9s |
| **pooled text** | 30 | **3.4s** | **5.5s** | 3.8s | |
| **pooled photo** | 12 | **13.4s** | **22.1s** | 13.5s | |

**What I did to get there:**

- **One persistent event loop** owned by the agent (sync `handle()` schedules onto it) — the LLM clients' httpx connection pools stay warm across turns instead of paying a fresh TLS handshake per message, and it dodges the "Event loop is closed" trap of `asyncio.run`-per-call. The vision call itself runs `ainvoke` on that loop, so one slow photo never blocks other users.
- **One-tool-call logging** — `log_meal` resolves nutrition internally from the builtin table → DB cache → a single *batched* estimate call for unknown foods (cached forever, so "same as yesterday" is pure table hits).
- **Image resize before the vision call** — EXIF-oriented, longest side 1024px, JPEG q85 (~50KB vs multi-MB phone photos): less upload, fewer vision tokens. I A/B-tested 768px (no reliable gain — reasoning dominates, and it risked truncated JSON) so 1024 stayed.
- **`reasoning_effort: low` where the provider supports it** (opt-in env var; the tested config uses it) — measurably less reasoning volume with no quality loss on this task.
- **Lean prompts** — system prompt + memory block + 8-turn history + one photo block ≈ 1.2K tokens per turn.
- **Correct tool ordering** — a simple log is 2 LLM round trips (decide+tool, then confirm); a correction is 3 (find, edit, confirm).

**What I couldn't fix (and why):** on the tested configuration the photo path is vision-model-bound — 4–10s on a simple plate, up to ~20s on a busy thali, with occasional long stalls (timeout 120s, one retry). With budget this is the first swap: a faster vision endpoint, plus an async queue + streaming replies so the user sees "analyzing your photo…" immediately. On the text path, the second round trip (the confirmation message) could be templated away to save ~1.5s — I chose reply quality over the shave.

## Assumptions and trade-offs

- **Nutrition data:** hardcoded per-unit table (Indian-first, ~65 foods, values are "tracker-grade", not dietitian-grade) + LLM estimates for everything else, cached. Chosen over a nutrition API for zero-latency table hits and offline determinism; the trade-off is coverage, which the estimator path covers.
- **Timezone:** "today" boundaries use `MEALCHAT_TIMEZONE` (default Asia/Kolkata — CalorAI's users). One global tz, not per-user tz — would be a per-user setting in production.
- **Skipped meals aren't logged** — a zero row adds nothing; the agent asks what the grazing was instead.
- **Soft deletes only** — keeps the audit trail so a mistaken correction is recoverable by hand.
- **History is 8 turns**, stored trimmed to 500 chars — enough for conversational coherence, small enough to never dominate the prompt.
- **Single agent instance, one event loop, thread-lock around SQLite** — tools execute on LangGraph's executor threads. Fine for a single-process demo; multi-worker deployment would want a connection pool per worker.
- **Vision uncertainty policy:** `uncertain=true` OR unparseable JSON OR a thrown network error all degrade to "ask the user" — the photo path fails safe, never silent.
- **The web UI gives each browser its own user id** (localStorage UUID), so visitors don't share meals/memory; there's no auth — by task spec.

## Time breakdown

Roughly **one hour** of focused build time, AI-accelerated:

| what | time |
|---|---|
| model selection — live probes of tool-calling and image input before writing code | ~10 min |
| persistence + nutrition layers with 16 unit tests | ~12 min |
| agent core, 7→8 tools, vision pipeline, event-loop fix | ~15 min |
| judgment tuning + eval suite driven to 13/13 (portion param, same-food-name copies, ask-vs-log line) | ~12 min |
| CLI + web UI + latency bench | ~10 min |
| walkthrough video + README | ~10 min |

## What I'd fix or build next

1. **Streaming replies** — the single biggest felt-latency win; the graph already emits per-token events, the UI just doesn't consume them.
2. **Per-user timezone + meal-time inference** from local clock ("late lunch" → meal_type).
3. **Memory expiry/decay** — `note`-kind memories older than N days drop out of the block.
4. **Better correction UX** — edit conflicts ("2 rotis at lunch AND dinner — which?") could offer the last-meal default with a one-tap confirm.
5. **LangSmith tracing** behind an env flag (the tool-call effects in `AgentResult` already give the local skeleton).
6. **A real nutrition source** behind the same `resolve_items` seam (USDA/FatSecret), keeping the builtin table as the fast path.
7. **Retry/fallback on the vision path** — a second VL provider, raced after 8s, first answer wins.

## Notes on AI tool usage

- The entire build was done with an AI coding agent (OpenCode/T3, GLM-5.3 model) driving the small-base/test/add/repeat loop — including this README's first draft, the eval assertions, and the fix for the double-halving edge case its own eval caught. Human (me) set the architecture decisions, the ask-vs-log product line, the memory design, and reviewed every commit.
- The app's own models were chosen by capability probes (a live tool-calling check for the text model, a live image call for the vision model) and cost/latency fit, before any code was written.
- AI-assisted ≠ untested: every layer has unit tests, and the 13 eval cases run against live models, not mocks.
- The walkthrough video was also agent-generated (scripts/make_video.py) from a real session — stated plainly rather than passed off as a human recording.

## Credits

Test food photos are from Wikimedia Commons: `Chapati.jpg`, `Biryani.jpg`, `Indian_thali.jpg`, plus a keyboard photo for the not-food case — committed under `fixtures/photos/` for reproducible evals.
