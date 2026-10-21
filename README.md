# mealchat

A conversational meal-logging agent — you text what you ate (or send a photo of
your plate), it logs it, keeps running daily calorie/macro totals, and remembers
what matters about you across sessions.

Built as a test task for CalorAI. **This README is being filled in as the build
progresses — see the commit history for the working progression.**

## Status

- [x] Scaffold (uv, pyproject, CI-friendly layout)
- [ ] Persistent SQLite schema (meals, items, foods, memories)
- [ ] Nutrition lookup + estimation with cache
- [ ] Selective memory (store / retrieve / inject)
- [ ] Agent core with tool calling (LangGraph)
- [ ] Vision path on a separate model
- [ ] Corrections without double-counting
- [ ] Web chat UI
- [ ] Evals + latency benchmarks
- [ ] Final README sections

## Quick start (placeholder — final instructions at the end)

```bash
uv sync
cp .env.example .env  # fill in your key
uv run mealchat
```
