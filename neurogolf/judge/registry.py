"""Transactional SQLite champion registry with immutable history records."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChampionRecord:
    task_num: int
    task_id: str
    candidate_sha256: str
    onnx_path: str
    builder_path: str | None
    explanation_path: str | None
    judge_version: str
    generator_revision: str
    official_utils_sha256: str
    public_passed: int
    public_total: int
    regression_passed: int
    regression_total: int
    fuzz_passed: int
    fuzz_total: int
    sealed_passed: int
    sealed_total: int
    memory_bytes: int
    parameter_elements: int
    objective: int
    points: float
    created_at: str
    parent_candidate: str | None
    status: str
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA = """
CREATE TABLE IF NOT EXISTS champions (
  task_num INTEGER NOT NULL,
  task_id TEXT NOT NULL,
  candidate_sha256 TEXT NOT NULL,
  onnx_path TEXT NOT NULL,
  builder_path TEXT,
  explanation_path TEXT,
  judge_version TEXT NOT NULL,
  generator_revision TEXT NOT NULL,
  official_utils_sha256 TEXT NOT NULL,
  public_passed INTEGER NOT NULL,
  public_total INTEGER NOT NULL,
  regression_passed INTEGER NOT NULL,
  regression_total INTEGER NOT NULL,
  fuzz_passed INTEGER NOT NULL,
  fuzz_total INTEGER NOT NULL,
  sealed_passed INTEGER NOT NULL,
  sealed_total INTEGER NOT NULL,
  memory_bytes INTEGER NOT NULL,
  parameter_elements INTEGER NOT NULL,
  objective INTEGER NOT NULL,
  points REAL NOT NULL,
  created_at TEXT NOT NULL,
  parent_candidate TEXT,
  status TEXT NOT NULL CHECK(status IN ('champion', 'historical', 'rolled_back')),
  notes TEXT,
  PRIMARY KEY(task_num, candidate_sha256)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_current_champion
ON champions(task_num) WHERE status = 'champion';
CREATE TABLE IF NOT EXISTS promotion_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_num INTEGER NOT NULL,
  candidate_sha256 TEXT NOT NULL,
  previous_sha256 TEXT,
  action TEXT NOT NULL CHECK(action IN ('promote', 'rollback')),
  created_at TEXT NOT NULL,
  notes TEXT
);
"""


class ChampionRegistry:
    def __init__(self, champion_root: Path):
        self.root = champion_root
        self.path = champion_root / "registry.sqlite"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "history").mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row | None) -> ChampionRecord | None:
        return ChampionRecord(**dict(row)) if row is not None else None

    def current(self, task_num: int) -> ChampionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM champions WHERE task_num=? AND status='champion'", (task_num,)
            ).fetchone()
        return self._record(row)

    def get(self, task_num: int, sha256: str) -> ChampionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM champions WHERE task_num=? AND candidate_sha256=?",
                (task_num, sha256),
            ).fetchone()
        return self._record(row)

    def list(self, *, current_only: bool = True) -> list[ChampionRecord]:
        query = "SELECT * FROM champions"
        if current_only:
            query += " WHERE status='champion'"
        query += " ORDER BY task_num, created_at"
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def history(self, task_num: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM promotion_history WHERE task_num=? ORDER BY id", (task_num,)
            ).fetchall()
        return [dict(row) for row in rows]
