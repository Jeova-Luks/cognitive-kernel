"""Component [7] EPISODE LOG (cru).

SQLite table with the columns from spec §6.6, plus the returned answer, the
reader's plan and the confidence of the final answer. Rows with success = 1
are what the episodic memory bank is built from; with cross-checking on,
success means the program's answer was confirmed by an independent derivation.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    problem_text  TEXT    NOT NULL,
    problem_type  TEXT,
    plan          TEXT,
    final_program TEXT,
    answer        TEXT,
    confidence    TEXT,
    num_attempts  INTEGER NOT NULL,
    success       INTEGER NOT NULL,
    error_trace   TEXT,
    embedding     TEXT
)
"""


class EpisodeLog:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One kernel may be driven from a worker thread (never two at once).
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def log(self, problem_text: str, problem_type: str, final_program: str | None,
            answer: object, num_attempts: int, success: bool, error_trace: str | None,
            embedding: list[float] | None = None, plan: str | None = None,
            confidence: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes (timestamp, problem_text, problem_type, plan, final_program,"
            " answer, confidence, num_attempts, success, error_trace, embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), problem_text, problem_type, plan, final_program,
             None if answer is None else json.dumps(answer), confidence, num_attempts, int(success),
             error_trace, None if embedding is None else json.dumps(embedding)),
        )
        self.conn.commit()
        return cur.lastrowid

    def successes(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM episodes WHERE success = 1 ORDER BY id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
            out.append(d)
        return out

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
