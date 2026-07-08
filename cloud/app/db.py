from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "rcp-cloud-data"


def data_dir() -> Path:
    return Path(os.environ.get("RCP_CLOUD_DATA_DIR", str(DEFAULT_DATA_DIR)))


def db_path() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "jobs.sqlite3"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            output_dir TEXT,
            error_message TEXT,
            page_count INTEGER DEFAULT 0,
            fixture_total INTEGER DEFAULT 0,
            fan_total INTEGER DEFAULT 0,
            warning_count INTEGER DEFAULT 0,
            summary_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """
    )
    conn.commit()
    conn.close()