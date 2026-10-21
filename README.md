# mealchat 🥗

A conversational meal-logging agent. You text what you ate — *"had 2 parathas and chai for breakfast"*, *"leftover biryani, maybe two thirds of the box"*, a photo of your plate with *"half of this was my brother's"* — and it logs it, keeps running daily calorie/macro totals that survive edits and corrections, and remembers the things worth remembering about you across sessions.

Built as the CalorAI AI-Engineer test task. Python, LangGraph, SQLite, two deliberately different models for text and vision.

**Walkthrough video: [docs/walkthrough.mp4](docs/walkthrough.mp4)** — an auto-generated screencast rendered from a real live session; every reply, latency and tool effect on screen comes verbatim from [docs/demo_session.json](docs/demo_session.json). It covers the architecture, the tool split, memory write/read, a photo case, the photo+caption case, a correction, and the latency discussion.

```
                 "had 2 rotis" ──┐
                 [photo of plate]─┤
                                   ▼
                agent.handle(user_id, text, image_path)
                                   │
              photo? ── resize ──▶ VISION MODEL (separate)
                                   │         └─ structured JSON
                                   ▼
          system prompt = static policy + MEMORY BLOCK + last 3h of turns
                                   ▼
                LangGraph react loop (text model, 9 tools)
                                   ▼
              SQLite (meals, items, foods, memories, messages)
```

## Status & honest gaps

All six core features are implemented: 20-case eval suite (three repeats per case, 59/60 passing — details below), 63 unit tests, measured p50/p95 latency. What is **not** done:

- **LangSmith tracing** — wired but off: no key exists in this environment, so **no public trace link is included**. With `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` set in `.env`, one user turn becomes one trace with the vision call and the agent graph nested under it, tagged with `user_id` and the path (text/photo). With the vars unset (the default), tracing is fully off — verified by running the evals in that mode.
- **Streaming** — not implemented, including a CLI `--stream` flag I chose to skip: doing it properly needs a second token-streaming path next to the single `handle()` core that the evals and benchmark measure, and the task allows skipping when it complicates the core. `handle()` returns a complete turn.
- Vision-model latency on the tested gateway is the weak point — see the latency section.
- Nutrition data is a hardcoded Indian-first table + cached model estimates — by design (the task says nutrition accuracy is not the point; the tradeoff is documented below).
- The tested vision model sometimes reports a low-confidence dish its own description doesn't mention (e.g. a "mixed curry" on a plate of plain rotis — visible in the recorded demo). Mitigations: low-confidence MAIN dishes must be flagged in the reply or questioned (never silently logged), plus the 2500/4000 kcal sanity caps. Known failure mode, not fixed.
- The commit history is genuine but compressed — several chunks were finished inside the same working window, so some commits share timestamps; it is not a minute-by-minute narrative.

## Where I disagree with the spec

The task's latency framing — "a user texting 'had 2 rotis' expects a reply at messaging speed" — is achievable, and the text path hits it (p50 3.0s for a known-foods log). But **"adding image handling makes this harder, which is the point"** can't be fully "won" in the way the sentence implies: with a vision model in the loop, a photo reply at messaging speed (sub-3s) is not reachable on any provider I could test — the vision call alone costs 4–10s on a simple plate. The right product answer isn't a faster vision model; it's **an instant acknowledgement ("looking at your plate…") followed by the analyzed log**, so the perceived latency is messaging-speed even when the real path isn't. That needs an async/queued reply channel (WhatsApp allows it), which is outside this task's CLI/web shape — so I bounded the damage instead: a hard 20s vision budget that degrades to asking the user, never a silent hang.

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
| `MEALCHAT_TEXT_REASONING` / `MEALCHAT_VISION_REASONING` | *(empty)* | optional `reasoning_effort`; only sent when set — most chat-completions models reject the param |
| `MEALCHAT_VISION_API_BASE` / `MEALCHAT_VISION_API_KEY` | same gateway | optional separate route for vision |
| `MEALCHAT_VISION_TIMEOUT` | `20` | hard seconds-budget for one vision call; on timeout the agent asks the user instead of waiting |
| `MEALCHAT_TIMEZONE` | `Asia/Kolkata` | local date for "today" boundaries and clock-based meal types |
| `MEALCHAT_DB` | `data/mealchat.db` | SQLite file |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | *(unset)* | optional tracing — see above |

Run it:

```bash
uv run mealchat                                     # terminal chat (REPL)
uv run mealchat --image fixtures/photos/Chapati.jpg this was lunch
uv run uvicorn mealchat.server:app                 # web chat at http://127.0.0.1:8000
uv run pytest                                       # 63 unit tests (no network)
uv run python -m evals.run                          # 20-case eval suite (live models)
uv run python -m evals.run --repeat 3               # pass rates per case + style metric
uv run python scripts/bench.py                      # latency p50/p95 (live models)
uv run python scripts/make_video.py --record && uv run python scripts/make_video.py --render
```

The core interface is interface-agnostic — the CLI, web UI, evals, benchmark and video all call exactly:

```python
result = agent.handle(user_id, text, image_path=None)   # -> AgentResult
# result.reply, result.vision, result.effects, result.latency_ms / vision_ms / agent_ms
```

## Model choices and why

There are two configurations:

- **Defaults (what a clean clone runs): `gpt-4o-mini` for text, `gpt-4o` for vision.** Public, stable model names on the standard OpenAI API, so the repo has zero private dependencies. gpt-4o-mini is cheap and fast with dependable tool-calling; gpt-4o is a stronger image reasoner for the vision path — and they are different models, as the task requires.
- **Tested configuration (what every eval and latency number below actually ran on): `deepseek-v4p1-flash` for text, `glm-5.3-flash` for vision**, behind an OpenAI-compatible gateway I have access to. Rationale on the axes the spec asks about:
  - **Cost** — flash-class models cost a small fraction of frontier-tier pricing per turn, and this is a high-volume, low-stakes domain (a wrong calorie estimate is recoverable; a $0.05 turn is not).
  - **Latency** — measured 1.2–2.5s round trips on the text path. I verified structured `tool_calls` output and a live multimodal call on each model *before* building on them.
  - **Quality** — tool-calling was the hard requirement for the text model (verified clean: correct JSON arguments, no hallucinated tools); for vision, food identification on the fixture plates is accurate with useful confidence flags — though not perfectly calibrated (see the known failure mode above).

The handoff between them is the same regardless of provider: photo → resize → one structured vision call (`response_format=json_object`, verified accepted by the tested gateway with a live call) → its JSON (foods, quantities, confidence, single-serving flag, uncertainty) is injected as a delimited block into the text agent's turn. **If the vision model returns `uncertain: true`** (non-food, unidentifiable) **or the call/parse fails, the agent is instructed — and prompted with the vision model's own suggested question — to ask the user instead of guessing.** The eval `vision_unsure_asks` covers this with a keyboard photo: nothing is logged, the agent asks for a real photo. A low-confidence *main* dish is never silently guessed either — it's logged generically and flagged in the reply, or questioned. The caption reaches the vision model too, but only as an identification hint: the prompt forbids applying sharing/fractions there (the share is applied once, later, by the agent — double-scaling is a silent calorie bug the eval `photo_plus_caption_one_meal` checks for).

## How memory works

This is the part I spent the most design time on, so the rules are explicit:

- **What I store:** structured, deduplicated *facts* in a `memories` table — `kind ∈ {preference, target, usual, note}`, each with an optional **key** (diet preferences are keyed `diet`, usuals by meal type, targets by `calories|protein|carbs|fat`), a `data` JSON payload (a usual stores its items as `[{food, quantity, unit}]`), and a numeric value/unit for targets. Nothing else. Keyed facts upsert — saving again **replaces** the previous version instead of piling up.
- **When it's written:** only when the agent decides a fact is durable, via `save_memory` — "i'm vegetarian" → preference keyed `diet`; "i'm targeting 140g of protein" → target; a "my usual" inferred from a repeating pattern gets saved keyed by meal type **with its items**, so next time it's logged from structured data, not re-parsed free text. One-off meal details and small talk never get stored (the negative-selectivity eval checks this).
- **How facts get removed:** `forget_memory(memory_id)` — the memory block shows ids like `[m12]`, and when the user corrects or retracts a fact the agent forgets the old one by id before saving the replacement. Ids are ownership-checked; substring matching was deliberately rejected (it could catch the wrong fact).
- **How it's retrieved:** one SQLite query per turn, rendered into a capped block in the system prompt — preferences first, then targets, then usuals (with their items), max 10 lines, a few hundred tokens. No embeddings needed at this scale; if memory grew, the same table would take a keyword/vector index without changing the contract.
- **What memory is *not*:** the last 8 chat turns *from the last 3 hours* are also injected (a separate `messages` table, capped on both count and age, for conversational coherence) — that's *context*, not memory. The proof it works: four cross-session evals use a `NEW_SESSION` marker (a fresh `MealChatAgent` on the same DB with the user's chat history deleted), and the facts still travel: a vegetarian preference saved in session 1 flags a chicken log in session 2; a structured usual logs exactly (paratha x2 + chai) in a fresh session.
- **Why this satisfies "selective":** every byte in the prompt block was individually judged worth keeping, is structured, keyed, updateable and removable — versus "store every message and hope".

## Tool design — why this split

Nine tools, split by what they do to state, not by data type:

| tool | path | why it exists |
|---|---|---|
| `log_meal` | write | the one happy-path call: items + `portion` fraction; resolves nutrition internally |
| `edit_meal` | write | corrections are **spot changes** — only mentioned foods change, everything else is kept byte-identical |
| `delete_meal` | write | soft delete, keeps audit trail, drops out of totals |
| `repeat_meals` | write | copies past meals with items exactly as logged — "same as yesterday" with zero re-estimation |
| `save_memory` | write | durable facts only, keyed for upsert, usuals with structured items |
| `forget_memory` | write | retract outdated facts by id |
| `get_meals` | read | past meals, filterable by date/food — the correction-finder |
| `day_totals` | read | the "how am I doing?" answer: totals + breakdown + targets + remaining |
| `nutrition_lookup` | read | one-off questions ("calories in a samosa?"), table hit or one-time estimate then cached |

The judgment calls: `log_meal` resolves nutrition *inside* the tool (builtin table → DB cache → one batched LLM estimate for the whole meal, cached forever) so a normal log is **one** tool call, not a lookup-then-log dance. `edit_meal` is separate from `log_meal` because corrections are a different *act* — and it never re-resolves unmentioned items, so a "3 rotis not 2" fix can't quietly change the chicken row. `portion` is a parameter (not LLM arithmetic) because "half of this was my brother's" must scale every item deterministically — the model decides one number, the tool does the multiplication. Gram/ml quantities are converted through the food's grams-per-unit (never multiplied raw), and a sanity guard refuses any log with an item over 2500 kcal or a meal over 4000 — the agent asks instead of poisoning the day.

Totals correctness is a data-layer property, not a prompt property: `day_totals` always aggregates from live rows, never incrementally maintained — so edits and deletions can't leave stale sums (unit-tested in `test_db.py`, including the sign of "kcal left" vs a saved target).

## When the agent asks vs just logs — where I drew the line

**Logs immediately, stating assumptions in a parenthetical:** named foods with known/standard amounts ("2 parathas" → medium parathas, 690 kcal); fractions of standard containers ("two thirds of the box" → 0.67 × takeaway box ≈ 536 kcal); confident photo analyses (generic names for unclear dishes, not interrogation); "same as yesterday" (one `repeat_meals` call); "my usual".

**Asks exactly one short question when:** the food is unknown ("grazed all afternoon" — on *what*?); the vision analysis is `uncertain` (never guess); the photo clearly shows a shared/family spread with no caption clue about the user's share (asks what portion was theirs); a correction is ambiguous (several candidate meals); or a log contradicts a saved preference (chicken after "i'm vegetarian" → confirms, never refuses, never silently logs).

When `log_meal` gets no meal type, it infers one from the local clock (5–11 breakfast, 11–16 lunch, 16–19 snack, 19–23 dinner) in the configured timezone; an explicit type is never overridden.

The principle: over-asking kills the texting feel, under-asking poisons the data. Small unknowns get estimates + a stated assumption; identity/portion unknowns that would materially change the numbers get one question. Never two questions in one reply.

## Evals

`uv run python -m evals.run` — 20 cases, fresh user + throwaway DB each, with a per-case definition of "correct" in `evals/cases.py` and DB-level assertions (not just reply-string matching). The three separator cases from the task, plus memory cases that use a `NEW_SESSION` marker (fresh agent, wiped history — only memory can carry facts):

- **`correction_no_double_count`** — "actually that was 3 rotis not 2" → the same meal edited via spot change, chai kept, exactly 450 kcal. ✅
- **`same_as_yesterday` / `my_usual_from_memory`** — from the meals table / the memory block, not from parsing or chat history. ✅
- **`photo_plus_caption_one_meal`** and **`photo_then_separate_caption`** — photo + "half was my brother's", together or as a follow-up message: exactly ONE halved meal, no double-scaling, no second log. ✅
- **Memory selectivity, retraction, and cross-session persistence** — one-off chatter saves nothing; "not vegetarian anymore" deactivates the old preference; facts and structured usuals survive a `NEW_SESSION`. ✅
- **`delete_item_from_meal`** ("i didn't actually have the chai"), **`grazing_follow_up_answer`** (asks, then logs the named foods), **`vision_unsure_asks`** (keyboard photo → asks, never guesses). ✅

**Final run, `--repeat 3` (60 case-runs on the final code, tested configuration): 59/60.** The single miss was `vegetarian_conflict_flagged` on one run where the agent flagged the conflict correctly ("Thought you were veg 🌱 — want me to log it anyway?") but the check demanded the literal word "vegetarian"; the matcher now accepts the abbreviation and that case is 3/3 standalone. Soft style metric across all 84 replies: **100% ≤45 words, 95% with ≤1 question mark.**

*(Like the latency numbers, eval numbers come from the tested configuration; the public defaults (gpt-4o-mini / gpt-4o) have not been evaluated — no OpenAI key exists in this environment.)*

## Latency

Measured by `scripts/bench.py` around `agent.handle()` — the full core (vision preprocessing + graph loop + tools), excluding UI/HTTP. Single user, warm connections, **final code, tested configuration**, 2026-10-21. *(The public defaults — gpt-4o-mini / gpt-4o — have **not** been benchmarked: no OpenAI key exists in this environment. Every number below comes from the tested config.)*

| scenario | n | p50 | p95 | mean | notes |
|---|---|---|---|---|---|
| text: log a meal | 20 | **3.0s** | 3.8s | 3.2s | 1 tool call, all foods known |
| text: day totals query | 20 | **2.9s** | 4.3s | 3.1s | 1 tool call |
| text: correction | 20 | **4.6s** | 5.5s | 4.7s | 2 tool calls (find + edit) |
| text: log unknown food | 20 | **8.9s** | 13.0s | 9.6s | estimator round trip paid every run (cache row wiped) |
| photo: log plate | 10 | **12.0s** | 22.2s | 13.6s | vision 6.7s + agent 5.2s (p50 splits) |
| photo: + caption | 10 | **10.1s** | 15.3s | 10.8s | vision 5.7s + agent 4.2s |
| **pooled text** | 80 | **3.9s** | **10.6s** | 5.1s | |
| **pooled photo** | 20 | **10.4s** | **18.7s** | 12.2s | |

The known-foods log path is the messaging-speed story (3.0s p50). The honest tails: a first-time unknown food costs an extra estimator round trip (~3× the simple log), and the photo path is vision-model-bound with a hard 20s time budget before it degrades to asking the user.

**What I did to get there:**

- **One persistent event loop** owned by the agent (sync `handle()` schedules onto it) — the LLM clients' httpx connection pools stay warm across turns, and it dodges the "Event loop is closed" trap of `asyncio.run`-per-call. The vision call runs `ainvoke` on that loop, so one slow photo never blocks other users.
- **One-tool-call logging** — `log_meal` resolves nutrition internally; `repeat_meals` copies without re-estimating, so "same as yesterday" is pure row copies.
- **Cache-friendly prompt shape** — the static policy text and tool schemas form a byte-identical prefix for every user and day; the date/timezone/memory block sit at the very end (whether a provider's prompt cache actually hits is up to the provider).
- **Image resize before the vision call** — EXIF-oriented, longest side 1024px, JPEG q85 (~50KB vs multi-MB phone photos). A/B-tested 768px (no reliable gain — reasoning dominates) so 1024 stayed.
- **`reasoning_effort: low` where the provider supports it** (opt-in env var; the tested config uses it) — measurably less reasoning volume with no quality loss on this task.
- **Lean prompts, measured not guessed:** ~3.5K tokens per text turn — system prompt ≈1.5K + 9 tool schemas ≈1.9K + capped history ≈0.1K (chars/4 approximation; photo turns add the ~120-token vision block). An earlier README claimed "≈1.2K per turn", which was roughly the system prompt alone — corrected.
- **Correct tool ordering** — a simple log is 2 LLM round trips (decide+tool, then confirm); a correction is 3 (find, edit, confirm).

**What I couldn't fix (and why):** on the tested configuration the photo path is vision-model-bound — p50 10.4s, p95 18.7s. I bounded the damage instead: a hard `MEALCHAT_VISION_TIMEOUT` (20s default, zero retries) after which the turn degrades to the ask-the-user path rather than stalling. With budget this is the first swap: a faster vision endpoint, plus an async queue + streaming replies so the user sees "looking at your plate…" immediately. On the text path, the second round trip (the confirmation message) could be templated away to save ~1.5s — I chose reply quality over the shave.

## Assumptions and trade-offs

- **Nutrition data:** hardcoded per-unit table (Indian-first, ~65 foods, values are "tracker-grade", not dietitian-grade) + LLM estimates for everything else, cached with grams-per-unit. Chosen over a nutrition API for zero-latency table hits and offline determinism; the trade-off is coverage, which the estimator path covers.
- **Timezone:** "today" boundaries and clock-inferred meal types use `MEALCHAT_TIMEZONE` (default Asia/Kolkata — CalorAI's users). One global tz, not per-user tz — would be a per-user setting in production.
- **Skipped meals aren't logged** — a zero row adds nothing; the agent asks what the grazing was instead.
- **Soft deletes only** — keeps the audit trail so a mistaken correction is recoverable by hand.
- **History is 8 turns from the last 3 hours**, stored trimmed to 500 chars — conversational coherence without a growing prompt.
- **Single agent instance, one event loop, thread-lock around SQLite** — tools execute on LangGraph's executor threads. Fine for a single-process demo; multi-worker deployment would want a connection pool per worker.
- **Vision uncertainty policy:** `uncertain=true` OR unparseable JSON OR a thrown network error OR a timeout all degrade to "ask the user" — the photo path fails safe, never silent.
- **The web UI gives each browser its own user id** (localStorage UUID), so visitors don't share meals/memory; there's no auth — by task spec.

## Time breakdown

Real durations from the git timestamps:

| when | what | time |
|---|---|---|
| 19:17–20:12 | initial build — model probing, persistence + nutrition layers, agent core, evals to green, CLI/web UI, bench, video, README | ~55 min |
| 22:11–22:24 | review round 1 — public OpenAI defaults, calorie-sign bug, async vision, forget tool, web isolation, README honesty | ~13 min |
| 22:36–23:35 | review round 2 (this one) — unit conversions + sanity guard, fail-safe vision, spot-change edit semantics, keyed/structured memory + repeat_meals, 3h history window, memory evals, delete/grazing evals + --repeat, prompt caching, vision timeout, caption-to-vision, LangSmith wiring, realistic benchmark, video render fixes | ~59 min |

## What I'd fix or build next

1. **Streaming replies** — the single biggest felt-latency win; the graph already emits per-token events, the UI just doesn't consume them.
2. **Instant photo ack** — queue the vision work, reply "looking at your plate…" immediately, follow with the log (see "Where I disagree with the spec").
3. **Per-user timezone** — one global `MEALCHAT_TIMEZONE` today; meal-type inference from the local clock is already in (8885fd8).
4. **Memory expiry/decay** — `note`-kind memories older than N days drop out of the block.
5. **Better correction UX** — edit conflicts ("2 rotis at lunch AND dinner — which?") could offer the last-meal default with a one-tap confirm.
6. **A real nutrition source** behind the same `resolve_items` seam (USDA/FatSecret), keeping the builtin table as the fast path.
7. **Raced vision fallback** — a second VL provider, raced after 8s, first answer wins.

## Notes on AI tool usage

- The entire build was done with an AI coding agent (OpenCode/T3, GLM-5.3 model) driving the small-base/test/add/repeat loop — including this README's first draft, the eval assertions, and the fixes for edge cases its own evals caught (the photo double-halving, the edit_meal re-resolution bug). Human (me) set the architecture decisions, the ask-vs-log product line, the memory design, and reviewed every commit.
- The app's own models were chosen by capability probes (a live tool-calling check for the text model, a live image call for the vision model) and cost/latency fit, before any code was written.
- AI-assisted ≠ untested: every layer has unit tests, and the 20 eval cases run against live models, not mocks.
- The walkthrough video was also agent-generated (scripts/make_video.py) from a real session — stated plainly rather than passed off as a human recording.

## Credits

Test food photos are from Wikimedia Commons: `Chapati.jpg`, `Biryani.jpg`, `Indian_thali.jpg`, plus a keyboard photo for the not-food case — committed under `fixtures/photos/` for reproducible evals.
