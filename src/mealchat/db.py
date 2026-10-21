"""SQLite persistence layer.

Design note on correctness: daily totals are ALWAYS aggregated from the
current meal/meal_items rows — they are never incrementally maintained. That is
what makes edits, corrections and deletions safe: any mutation lands as new row
state and the next read recomputes the truth.

Schema:
  meals       — one logged meal (user, local date, type, source, description)
  meal_items  — food rows with resolved nutrition per item
  foods       — nutrition cache: builtin table + LLM estimates, keyed by food
  memories    — durable selective memory facts (preference/target/usual/note)
  messages    — recent chat turns (short-lived context, NOT memory)
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Tools execute inside langgraph's thread executor, so the connection is hit
# from worker threads. check_same_thread=False + this lock serialize access;
# every public op here is a sub-millisecond SQLite call, so no latency cost.
_LOCK = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)

    return wrapper

SCHEMA = """
CREATE TABLE IF NOT EXISTS meals (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id     TEXT NOT NULL,
  date        TEXT NOT NULL,                -- local date, YYYY-MM-DD
  meal_type   TEXT,                         -- breakfast/lunch/dinner/snack
  source      TEXT NOT NULL DEFAULT 'text', -- text | image
  description TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, date);

CREATE TABLE IF NOT EXISTS meal_items (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  meal_id  INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
  food     TEXT NOT NULL,
  quantity REAL NOT NULL DEFAULT 1,
  unit     TEXT NOT NULL DEFAULT 'serving',
  calories REAL NOT NULL,
  protein  REAL NOT NULL,
  carbs    REAL NOT NULL,
  fat      REAL NOT NULL,
  notes    TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_meal ON meal_items(meal_id);

CREATE TABLE IF NOT EXISTS foods (
  key            TEXT PRIMARY KEY,              -- normalized lowercase food name
  name           TEXT NOT NULL,
  unit           TEXT NOT NULL,                 -- what quantity=1 means
  calories       REAL NOT NULL,
  protein        REAL NOT NULL,
  carbs          REAL NOT NULL,
  fat            REAL NOT NULL,
  grams_per_unit REAL,                         -- typical grams (or ml) in ONE unit
  source         TEXT NOT NULL,                 -- builtin | llm
  created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    TEXT NOT NULL,
  kind       TEXT NOT NULL,                 -- preference | target | usual | note
  key        TEXT,                         -- upsert key; e.g. 'diet', 'breakfast', 'protein'
  content    TEXT NOT NULL,
  data       TEXT,                         -- JSON payload, e.g. structured usual items
  num_value  REAL,
  unit       TEXT,
  active     INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, active);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL,                 -- user | assistant
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id, id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@_locked
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations for DBs created by older versions."""
    _add_column_if_missing(conn, "foods", "grams_per_unit", "REAL")
    _add_column_if_missing(conn, "memories", "data", "TEXT")


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def local_today(tz: ZoneInfo) -> str:
    return datetime.now(tz).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Meals
# --------------------------------------------------------------------------


@dataclass
class MealItem:
    food: str
    quantity: float
    unit: str
    calories: float
    protein: float
    carbs: float
    fat: float
    notes: str | None = None


@dataclass
class Meal:
    id: int
    user_id: str
    date: str
    meal_type: str | None
    source: str
    description: str
    created_at: str
    updated_at: str
    items: list[MealItem]


@_locked
def add_meal(
    conn: sqlite3.Connection,
    user_id: str,
    date: str,
    description: str,
    items: list[MealItem],
    meal_type: str | None = None,
    source: str = "text",
) -> int:
    now = _now()
    with conn:
        cur = conn.execute(
            "INSERT INTO meals (user_id, date, meal_type, source, description, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, date, meal_type, source, description, now, now),
        )
        meal_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO meal_items (meal_id, food, quantity, unit, calories, protein, carbs, fat, notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    meal_id,
                    it.food,
                    it.quantity,
                    it.unit,
                    it.calories,
                    it.protein,
                    it.carbs,
                    it.fat,
                    it.notes,
                )
                for it in items
            ],
        )
    return meal_id


def _row_to_meal(conn: sqlite3.Connection, row: sqlite3.Row) -> Meal:
    item_rows = conn.execute(
        "SELECT * FROM meal_items WHERE meal_id = ? ORDER BY id", (row["id"],)
    ).fetchall()
    return Meal(
        id=row["id"],
        user_id=row["user_id"],
        date=row["date"],
        meal_type=row["meal_type"],
        source=row["source"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        items=[
            MealItem(
                food=ir["food"],
                quantity=ir["quantity"],
                unit=ir["unit"],
                calories=ir["calories"],
                protein=ir["protein"],
                carbs=ir["carbs"],
                fat=ir["fat"],
                notes=ir["notes"],
            )
            for ir in item_rows
        ],
    )


@_locked
def get_meals(
    conn: sqlite3.Connection,
    user_id: str,
    date: str | None = None,
    food_like: str | None = None,
    limit: int = 10,
) -> list[Meal]:
    """Most recent first. Optional filters: exact local date, food substring."""
    q = "SELECT * FROM meals WHERE user_id = ? AND deleted_at IS NULL"
    params: list = [user_id]
    if date is not None:
        q += " AND date = ?"
        params.append(date)
    if food_like is not None:
        # join to items for food match, keep distinct meals
        q += " AND id IN (SELECT meal_id FROM meal_items WHERE food LIKE ?)"
        params.append(f"%{food_like}%")
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    return [_row_to_meal(conn, r) for r in rows]


@_locked
def get_meal(conn: sqlite3.Connection, meal_id: int) -> Meal | None:
    row = conn.execute(
        "SELECT * FROM meals WHERE id = ? AND deleted_at IS NULL", (meal_id,)
    ).fetchone()
    return _row_to_meal(conn, row) if row else None


@_locked
def edit_meal(
    conn: sqlite3.Connection,
    meal_id: int,
    items: list[MealItem] | None = None,
    meal_type: str | None = None,
    description: str | None = None,
) -> bool:
    """Replace items (nutrition recomputed by caller) and/or metadata."""
    with conn:
        if items is not None:
            conn.execute("DELETE FROM meal_items WHERE meal_id = ?", (meal_id,))
            conn.executemany(
                "INSERT INTO meal_items (meal_id, food, quantity, unit, calories, protein, carbs, fat, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        meal_id,
                        it.food,
                        it.quantity,
                        it.unit,
                        it.calories,
                        it.protein,
                        it.carbs,
                        it.fat,
                        it.notes,
                    )
                    for it in items
                ],
            )
        sets, params = ["updated_at = ?"], [_now()]
        if meal_type is not None:
            sets.append("meal_type = ?")
            params.append(meal_type)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        params.append(meal_id)
        cur = conn.execute(f"UPDATE meals SET {', '.join(sets)} WHERE id = ?", params)
        return cur.rowcount > 0


@_locked
def delete_meal(conn: sqlite3.Connection, meal_id: int) -> bool:
    """Soft delete — keeps an audit trail, excluded from all reads/totals."""
    with conn:
        cur = conn.execute(
            "UPDATE meals SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
            (_now(), _now(), meal_id),
        )
        return cur.rowcount > 0


# --------------------------------------------------------------------------
# Daily totals (always aggregated from live rows)
# --------------------------------------------------------------------------


@dataclass
class DayTotals:
    date: str
    calories: float
    protein: float
    carbs: float
    fat: float
    meals: int


@_locked
def day_totals(conn: sqlite3.Connection, user_id: str, date: str) -> DayTotals:
    row = conn.execute(
        "SELECT COUNT(DISTINCT m.id) AS meal_count,"
        " COALESCE(SUM(i.calories), 0) AS calories,"
        " COALESCE(SUM(i.protein), 0) AS protein,"
        " COALESCE(SUM(i.carbs), 0) AS carbs,"
        " COALESCE(SUM(i.fat), 0) AS fat"
        " FROM meals m JOIN meal_items i ON i.meal_id = m.id"
        " WHERE m.user_id = ? AND m.date = ? AND m.deleted_at IS NULL",
        (user_id, date),
    ).fetchone()
    return DayTotals(
        date=date,
        calories=round(row["calories"], 1),
        protein=round(row["protein"], 1),
        carbs=round(row["carbs"], 1),
        fat=round(row["fat"], 1),
        meals=row["meal_count"],
    )


# --------------------------------------------------------------------------
# Food nutrition cache
# --------------------------------------------------------------------------


@dataclass
class Food:
    key: str
    name: str
    unit: str
    calories: float
    protein: float
    carbs: float
    fat: float
    source: str
    grams_per_unit: float | None = None


def normalize_food(name: str) -> str:
    return " ".join(name.lower().strip().split())


@_locked
def upsert_food(conn: sqlite3.Connection, food: Food) -> None:
    with conn:
        conn.execute(
            "INSERT INTO foods (key, name, unit, calories, protein, carbs, fat, grams_per_unit, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET name=excluded.name, unit=excluded.unit,"
            " calories=excluded.calories, protein=excluded.protein, carbs=excluded.carbs,"
            " fat=excluded.fat, grams_per_unit=excluded.grams_per_unit, source=excluded.source",
            (
                food.key,
                food.name,
                food.unit,
                food.calories,
                food.protein,
                food.carbs,
                food.fat,
                food.grams_per_unit,
                food.source,
                _now(),
            ),
        )


@_locked
def get_food(conn: sqlite3.Connection, key: str) -> Food | None:
    row = conn.execute("SELECT * FROM foods WHERE key = ?", (normalize_food(key),)).fetchone()
    if not row:
        return None
    return Food(
        key=row["key"],
        name=row["name"],
        unit=row["unit"],
        calories=row["calories"],
        protein=row["protein"],
        carbs=row["carbs"],
        fat=row["fat"],
        source=row["source"],
        grams_per_unit=row["grams_per_unit"],
    )


# --------------------------------------------------------------------------
# Memories (durable selective facts — see README "How memory works")
# --------------------------------------------------------------------------


@dataclass
class Memory:
    id: int
    user_id: str
    kind: str
    key: str | None
    content: str
    data: str | None = None
    num_value: float | None = None
    unit: str | None = None
    created_at: str = ""
    updated_at: str = ""


def parse_memory_data(m: Memory) -> list[dict]:
    """Decode a memory's JSON data payload (e.g. structured usual items)."""
    if not m.data:
        return []
    try:
        parsed = json.loads(m.data)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


@_locked
def save_memory(
    conn: sqlite3.Connection,
    user_id: str,
    kind: str,
    content: str,
    key: str | None = None,
    data: str | None = None,
    num_value: float | None = None,
    unit: str | None = None,
) -> int:
    """Upsert. With a key: one active row per (user, kind, key) — saving again
    replaces the previous version for ANY kind. Without a key: exact-content
    dedupe. `data` is an optional JSON payload (structured usual items)."""
    now = _now()
    with conn:
        if key:
            conn.execute(
                "UPDATE memories SET active = 0, updated_at = ?"
                " WHERE user_id = ? AND kind = ? AND key = ?",
                (now, user_id, kind, key),
            )
        else:
            dup = conn.execute(
                "SELECT id FROM memories WHERE user_id = ? AND kind = ? AND content = ? AND active = 1",
                (user_id, kind, content),
            ).fetchone()
            if dup:
                return int(dup["id"])
        cur = conn.execute(
            "INSERT INTO memories (user_id, kind, key, content, data, num_value, unit, active, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (user_id, kind, key, content, data, num_value, unit, now, now),
        )
        return int(cur.lastrowid)


def _row_to_memory(r: sqlite3.Row) -> Memory:
    return Memory(
        id=r["id"],
        user_id=r["user_id"],
        kind=r["kind"],
        key=r["key"],
        content=r["content"],
        data=r["data"],
        num_value=r["num_value"],
        unit=r["unit"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


@_locked
def get_memory(
    conn: sqlite3.Connection,
    user_id: str,
    memory_id: int,
    active_only: bool = True,
) -> Memory | None:
    """One memory by id, ownership-scoped. active_only=False also sees forgotten rows."""
    q = "SELECT * FROM memories WHERE user_id = ? AND id = ?"
    if active_only:
        q += " AND active = 1"
    row = conn.execute(q, (user_id, memory_id)).fetchone()
    return _row_to_memory(row) if row else None


@_locked
def get_memories(conn: sqlite3.Connection, user_id: str, active_only: bool = True) -> list[Memory]:
    q = "SELECT * FROM memories WHERE user_id = ?"
    if active_only:
        q += " AND active = 1"
    q += " ORDER BY id DESC"
    rows = conn.execute(q, (user_id,)).fetchall()
    return [_row_to_memory(r) for r in rows]


@_locked
def forget_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    with conn:
        cur = conn.execute(
            "UPDATE memories SET active = 0, updated_at = ? WHERE id = ?", (_now(), memory_id)
        )
        return cur.rowcount > 0


# --------------------------------------------------------------------------
# Chat history (short-lived conversational context — explicitly NOT memory)
# --------------------------------------------------------------------------


@_locked
def add_message(conn: sqlite3.Connection, user_id: str, role: str, content: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, _now()),
        )


@_locked
def recent_messages(conn: sqlite3.Connection, user_id: str, limit: int = 8) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
