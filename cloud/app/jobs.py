from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from cloud.app.db import connect, data_dir, init_db

MAX_BYTES = 50 * 1024 * 1024
JOB_STATUSES = ("queued", "processing", "completed", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    p = data_dir() / "jobs" / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_job(filename: str, file_bytes: bytes) -> dict:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF uploads are accepted in v1.")
    if len(file_bytes) > MAX_BYTES:
        raise ValueError(f"File exceeds {MAX_BYTES // (1024*1024)} MB limit.")

    job_id = secrets.token_urlsafe(12)
    token = secrets.token_urlsafe(24)
    job_path = _job_dir(job_id)
    upload_path = job_path / "upload.pdf"
    upload_path.write_bytes(file_bytes)

    conn = connect()
    conn.execute(
        """
        INSERT INTO jobs (id, access_token, status, created_at, updated_at,
                          original_filename, upload_path, output_dir)
        VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            token,
            _now(),
            _now(),
            filename,
            str(upload_path),
            str(job_path / "output"),
        ),
    )
    conn.commit()
    conn.close()
    return {"job_id": job_id, "access_token": token, "status": "queued"}


def get_job(job_id: str, token: str | None) -> dict | None:
    conn = connect()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    if token is None or token != row["access_token"]:
        return None
    return dict(row)


def _update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn = connect()
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
    conn.commit()
    conn.close()


def claim_next_queued_job() -> dict | None:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
    ).fetchone()
    if row is None:
        conn.close()
        return None
    job_id = row["id"]
    conn.execute(
        "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ? AND status = 'queued'",
        (_now(), job_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row and row["status"] == "processing":
        return dict(row)
    return None


def _summarize_from_json(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    summary = data.get("summary") or {}
    pages = data.get("pages") or []
    fixture_total = int(summary.get("fixture_total", 0) or 0)
    fan_total = int(summary.get("fans", 0) or 0)
    warnings = int(summary.get("zero_schedule_codes", 0) or 0)
    warnings += int(summary.get("unscheduled_codes", 0) or 0)
    warnings += int(summary.get("low_confidence_occurrences", 0) or 0)
    return {
        "page_count": int(summary.get("pages", len(pages)) or 0),
        "fixture_total": fixture_total,
        "fan_total": fan_total,
        "warning_count": warnings,
    }


def process_job(job: dict) -> None:
    job_id = job["id"]
    upload = Path(job["upload_path"])
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_base = out_dir / "report.md"

    try:
        from click.testing import CliRunner
        from rcp_detector.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["ocr-count", str(upload), "--output", str(report_base), "--dpi", "300"],
        )
        if result.exit_code != 0:
            raise RuntimeError((result.output or result.exception or "ocr-count failed")[:4000])

        json_path = report_base.with_suffix(".json")
        summary = _summarize_from_json(json_path)
        if not summary:
            summary = {"page_count": 0, "fixture_total": 0, "fan_total": 0, "warning_count": 0}

        _update_job(
            job_id,
            status="completed",
            page_count=summary["page_count"],
            fixture_total=summary["fixture_total"],
            fan_total=summary["fan_total"],
            warning_count=summary["warning_count"],
            summary_json=json.dumps(summary),
            error_message=None,
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error_message=str(exc)[:2000])


def artifact_path(job: dict, name: str) -> Path | None:
    out = Path(job["output_dir"])
    mapping = {
        "report.md": out / "report.md",
        "report.html": out / "report.html",
        "report.json": out / "report.json",
        "report.csv": out / "report.csv",
        "report.marked.pdf": out / "report.marked.pdf",
    }
    path = mapping.get(name)
    if path and path.exists():
        return path
    return None