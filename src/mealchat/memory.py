"""Selective memory: rendering durable facts into a small prompt block.

The design (spelled out in the README):
- memories are DEDUPED, STRUCTURED FACTS (preference / target / usual / note),
  never raw chat history;
- they're written explicitly by the agent via the save_memory tool, only when
  a fact is worth keeping across sessions;
- retrieval = one SQLite query per turn, rendered into a capped block
  (preferences first, then targets, then usuals, then the newest notes),
  a few hundred tokens at most. Chat history is capped separately and is
  context, not memory.
"""

from __future__ import annotations

import sqlite3

from mealchat import db

_KIND_ORDER = {"preference": 0, "target": 1, "usual": 2, "note": 3}
_MAX_LINES = 10


def targets(conn: sqlite3.Connection, user_id: str) -> dict[str, db.Memory]:
    """Active target memories keyed by target name (calories/protein/carbs/fat)."""
    out = {}
    for m in db.get_memories(conn, user_id):
        if m.kind == "target" and m.key:
            out[m.key] = m
    return out


def memory_block(conn: sqlite3.Connection, user_id: str) -> str:
    """Compact 'what you remember about the user' block for the system prompt."""
    mems = db.get_memories(conn, user_id)
    if not mems:
        return ""
    # newest first from the DB; sort by kind importance, then recency
    mems.sort(key=lambda m: (_KIND_ORDER.get(m.kind, 9), -m.id))
    lines: list[str] = []
    for m in mems:
        if m.kind == "target" and m.key:
            lines.append(f"- target {m.key}: {m.num_value:g}{m.unit or ''} per day [m{m.id}]")
            continue
        label = f"- {m.kind} ({m.key})" if m.key else f"- {m.kind}"
        suffix = _items_suffix(m)
        lines.append(f"{label} [m{m.id}]: {m.content}{suffix}")
        if len(lines) >= _MAX_LINES:
            break
    return "\n".join(lines)


def _items_suffix(m: db.Memory) -> str:
    """Structured item list for usuals saved with items (JSON in `data`)."""
    items = db.parse_memory_data(m)
    if not items:
        return ""
    parts = []
    for it in items:
        if isinstance(it, dict) and it.get("food"):
            try:
                parts.append(f"{it['food']} x{float(it.get('quantity', 1)):g}")
            except (TypeError, ValueError):
                parts.append(str(it.get("food")))
    return f" — items: {', '.join(parts)}" if parts else ""
