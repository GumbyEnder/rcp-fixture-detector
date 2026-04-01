from __future__ import annotations

import csv
import html as html_lib
import json
from datetime import datetime, timezone
from pathlib import Path


def write_json(all_results: dict[str, list[dict]], output_path: str | Path, class_names: dict[int, str]) -> None:
    """Write results as pretty JSON.

    The output preserves the existing nested structure and includes a compact
    class-name lookup alongside the raw detections.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "class_names": {int(k): v for k, v in class_names.items()},
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))


def write_csv(all_results: dict[str, list[dict]], output_path: str | Path, class_names: dict[int, str]) -> None:
    """Write results as a flat CSV for spreadsheet review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "confidence", "bbox_abs", "bbox_norm", "quantity"])
        for image_name, dets in all_results.items():
            for det in dets:
                class_id = int(det.get("class_id", -1))
                writer.writerow([
                    image_name,
                    class_id,
                    class_names.get(class_id, f"class_{class_id}"),
                    det.get("confidence", ""),
                    det.get("bbox_abs", ""),
                    det.get("bbox_norm", ""),
                    det.get("quantity", ""),
                ])


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _html_escape(value) -> str:
    return html_lib.escape("" if value is None else str(value), quote=True)


def _format_counts(counts: dict) -> str:
    if not counts:
        return "—"
    parts = [f"{_html_escape(code)}: {int(count)}" for code, count in sorted(counts.items())]
    return ", ".join(parts)


def _format_schedule(schedule: dict) -> str:
    if not schedule:
        return "<tr><td colspan=\"2\">—</td></tr>"
    rows = []
    for code, desc in sorted(schedule.items()):
        rows.append(f"<tr><td>{_html_escape(code)}</td><td>{_html_escape(desc)}</td></tr>")
    return "".join(rows)


def _summary_metrics(results: list) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "pages": len(results),
        "elapsed_s": 0.0,
        "fixture_types": 0,
        "fixture_total": 0,
        "raw_ocr_texts": 0,
        "raw_occurrences": 0,
        "deduped_occurrences": 0,
        "tiles": 0,
        "blank_tiles_skipped": 0,
        "schedule_entries": 0,
        "zero_schedule_codes": 0,
        "unscheduled_codes": 0,
        "low_confidence_occurrences": 0,
        "fans": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "min_confidence": None,
    }

    for result in results:
        metrics = _get(result, "metrics", {}) or {}
        fixture_counts = _get(result, "fixture_counts", {}) or {}
        occurrences = _get(result, "occurrences", []) or []
        schedule_entries = _get(result, "schedule_entries", {}) or {}

        summary["elapsed_s"] += float(metrics.get("elapsed_s", metrics.get("page_elapsed_s", 0.0)) or 0.0)
        summary["fixture_types"] += len(fixture_counts)
        summary["fixture_total"] += sum(int(v) for v in fixture_counts.values())
        summary["raw_ocr_texts"] += int(metrics.get("raw_ocr_texts", metrics.get("ocr_lines", 0)) or 0)
        summary["raw_occurrences"] += int(metrics.get("raw_occurrences", metrics.get("occurrences", 0)) or 0)
        summary["deduped_occurrences"] += int(metrics.get("deduped_occurrences", len(occurrences)) or 0)
        summary["tiles"] += int(metrics.get("tile_count", 0) or 0)
        summary["blank_tiles_skipped"] += int(metrics.get("blank_tiles_skipped", 0) or 0)
        summary["schedule_entries"] += int(metrics.get("schedule_entries", len(schedule_entries)) or 0)
        summary["zero_schedule_codes"] += int(metrics.get("zero_schedule_codes", 0) or 0)
        summary["unscheduled_codes"] += int(metrics.get("unscheduled_codes", 0) or 0)
        summary["low_confidence_occurrences"] += int(metrics.get("low_confidence_occurrences", 0) or 0)
        summary["fans"] += int(metrics.get("fans", _get(result, "fan_count", 0)) or 0)

        confidences = [float(_get(occ, "confidence", 0.0) or 0.0) for occ in occurrences]
        summary["confidence_sum"] += sum(confidences)
        summary["confidence_count"] += len(confidences)
        if confidences:
            page_min = min(confidences)
            if summary["min_confidence"] is None:
                summary["min_confidence"] = page_min
            else:
                summary["min_confidence"] = min(float(summary["min_confidence"]), page_min)

    summary["elapsed_s"] = round(float(summary["elapsed_s"]), 3)
    if summary["confidence_count"]:
        summary["avg_confidence"] = round(float(summary["confidence_sum"]) / float(summary["confidence_count"]), 3)
        summary["min_confidence"] = round(float(summary["min_confidence"]), 3) if summary["min_confidence"] is not None else 0.0
    else:
        summary["avg_confidence"] = 0.0
        summary["min_confidence"] = 0.0
    return summary


def write_html_report(results: list, output_path: str | Path, markdown_path: str | Path | None = None) -> None:
    """Write a standalone technical HTML report for OCR fixture counts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _summary_metrics(results)
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    markdown_name = Path(markdown_path).name if markdown_path else None

    def metric_card(label: str, value, detail: str = "") -> str:
        detail_html = f'<div class="card-detail">{_html_escape(detail)}</div>' if detail else ""
        return (
            f'<div class="card"><div class="card-label">{_html_escape(label)}</div>'
            f'<div class="card-value">{_html_escape(value)}</div>{detail_html}</div>'
        )

    page_rows = []
    for result in results:
        metrics = _get(result, "metrics", {}) or {}
        fixture_counts = _get(result, "fixture_counts", {}) or {}
        page_rows.append(
            "<tr>"
            f"<td>{_html_escape(_get(result, 'page_name', ''))}</td>"
            f"<td>{_html_escape(metrics.get('page_kind', 'n/a'))}</td>"
            f"<td>{_html_escape(sum(int(v) for v in fixture_counts.values()))}</td>"
            f"<td>{_html_escape(len(fixture_counts))}</td>"
            f"<td>{_html_escape(metrics.get('raw_occurrences', metrics.get('occurrences', 0)))}</td>"
            f"<td>{_html_escape(metrics.get('deduped_occurrences', len(_get(result, 'occurrences', []))))}</td>"
            f"<td>{_html_escape(metrics.get('schedule_entries', len(_get(result, 'schedule_entries', {}) or {})))}</td>"
            f"<td>{_html_escape(metrics.get('fans', _get(result, 'fan_count', 0)))}</td>"
            f"<td>{_html_escape(metrics.get('elapsed_s', metrics.get('page_elapsed_s', '')))}</td>"
            "</tr>"
        )

    per_page_sections = []
    for result in results:
        metrics = _get(result, "metrics", {}) or {}
        fixture_counts = _get(result, "fixture_counts", {}) or {}
        schedule_entries = _get(result, "schedule_entries", {}) or {}
        occurrences = _get(result, "occurrences", []) or []
        raw_texts = _get(result, "all_ocr_texts", []) or []
        page_name = _get(result, "page_name", "")

        metrics_rows = []
        for key, label in (
            ("page_kind", "Page kind"),
            ("page_elapsed_s", "Page elapsed (s)"),
            ("elapsed_s", "Elapsed (s)"),
            ("raw_ocr_texts", "OCR texts"),
            ("raw_occurrences", "Raw occurrences"),
            ("deduped_occurrences", "Deduped occurrences"),
            ("tile_count", "Tiles"),
            ("blank_tiles_skipped", "Blank tiles skipped"),
            ("schedule_entries", "Schedule entries"),
            ("zero_schedule_codes", "Schedule zero-counts"),
            ("unscheduled_codes", "Unscheduled codes"),
            ("low_confidence_occurrences", "Low-confidence hits"),
            ("avg_confidence", "Avg confidence"),
            ("min_confidence", "Min confidence"),
            ("fans", "Fans"),
        ):
            value = metrics.get(key, "")
            if value == "" and key == "fans":
                value = _get(result, "fan_count", 0)
            metrics_rows.append(f"<tr><th>{_html_escape(label)}</th><td>{_html_escape(value)}</td></tr>")

        occurrence_rows = []
        for occ in occurrences:
            occurrence_rows.append(
                "<tr>"
                f"<td>{_html_escape(_get(occ, 'code', ''))}</td>"
                f"<td>{_html_escape(_get(occ, 'quantity', ''))}</td>"
                f"<td>{_html_escape(_get(occ, 'source', ''))}</td>"
                f"<td>{_html_escape(round(float(_get(occ, 'confidence', 0.0) or 0.0), 3))}</td>"
                f"<td>{_html_escape(_get(occ, 'bbox', _get(occ, 'bbox_abs', '')))}</td>"
                f"<td>{_html_escape(_get(occ, 'raw_text', ''))}</td>"
                "</tr>"
            )

        raw_text_block = ""
        if raw_texts:
            raw_text_block = "<details><summary>Raw OCR text samples</summary><ul>" + "".join(
                f"<li>{_html_escape(text)}</li>" for text in raw_texts[:50]
            ) + ("</ul></details>" if len(raw_texts) <= 50 else f"</ul><p>Showing first 50 of {len(raw_texts)} texts.</p></details>")

        per_page_sections.append(
            f"""
            <details class="page-card" {'open' if len(per_page_sections) == 0 else ''}>
              <summary>
                <strong>{_html_escape(page_name)}</strong>
                <span>{_html_escape(metrics.get('page_kind', 'n/a'))}</span>
                <span>{_html_escape(sum(int(v) for v in fixture_counts.values()))} fixtures</span>
                <span>{_html_escape(len(occurrences))} occurrences</span>
              </summary>
              <div class="page-grid">
                <section>
                  <h3>Metrics</h3>
                  <table class="metric-table">
                    <tbody>
                      {''.join(metrics_rows)}
                    </tbody>
                  </table>
                </section>
                <section>
                  <h3>Fixture counts</h3>
                  <p>{_format_counts(fixture_counts)}</p>
                </section>
                <section>
                  <h3>Schedule entries</h3>
                  <table class="data-table">
                    <thead><tr><th>Code</th><th>Description</th></tr></thead>
                    <tbody>{_format_schedule(schedule_entries)}</tbody>
                  </table>
                </section>
                <section>
                  <h3>Occurrences</h3>
                  <table class="data-table">
                    <thead><tr><th>Code</th><th>Qty</th><th>Source</th><th>Conf</th><th>BBox</th><th>Text</th></tr></thead>
                    <tbody>{''.join(occurrence_rows) if occurrence_rows else '<tr><td colspan="6">No occurrences</td></tr>'}</tbody>
                  </table>
                </section>
                <section>
                  <h3>Raw OCR samples</h3>
                  {raw_text_block or '<p>No OCR text samples captured.</p>'}
                </section>
              </div>
            </details>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCR Fixture Count Report</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d8dce6;
      --text: #1d2433;
      --muted: #5f6b85;
      --accent: #2b6cb0;
      --accent-soft: #edf4ff;
      --shadow: 0 8px 30px rgba(29, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 28px 32px 20px; background: linear-gradient(180deg, #ffffff, #f6f7fb); border-bottom: 1px solid var(--border); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .subtitle {{ color: var(--muted); margin: 0; }}
    .top-links a {{ color: var(--accent); text-decoration: none; margin-right: 14px; }}
    .wrap {{ padding: 24px 32px 40px; max-width: 1400px; margin: 0 auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }}
    .card-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }}
    .card-value {{ font-size: 24px; font-weight: 700; line-height: 1.1; }}
    .card-detail {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px; box-shadow: var(--shadow); margin-bottom: 20px; }}
    .section h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .data-table, .metric-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .data-table th, .data-table td, .metric-table th, .metric-table td {{ border-bottom: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }}
    .metric-table th {{ width: 220px; color: var(--muted); font-weight: 600; }}
    .page-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; margin-bottom: 14px; box-shadow: var(--shadow); }}
    .page-card > summary {{ cursor: pointer; list-style: none; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; font-size: 16px; }}
    .page-card > summary::-webkit-details-marker {{ display: none; }}
    .page-card summary span {{ color: var(--muted); font-size: 13px; }}
    .page-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-top: 14px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    details summary {{ color: var(--accent); }}
    ul {{ margin: 8px 0 0 20px; }}
    .note {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>OCR Fixture Count Report</h1>
    <p class="subtitle">Generated {_html_escape(generated_at)}</p>
    <p class="subtitle">Technical review output for fixture counting, schedule reconciliation, and fan separation.</p>
    <p class="top-links">
      {f'<a href="{_html_escape(markdown_name)}">Markdown summary</a>' if markdown_name else ''}
      <a href="#pages">Page details</a>
    </p>
  </header>
  <div class="wrap">
    <section class="cards">
      {metric_card('Pages', summary['pages'])}
      {metric_card('Fixture total', summary['fixture_total'])}
      {metric_card('Fixture types', summary['fixture_types'])}
      {metric_card('Elapsed', f"{summary['elapsed_s']} s")}
      {metric_card('Raw hits', summary['raw_occurrences'])}
      {metric_card('Deduped hits', summary['deduped_occurrences'])}
      {metric_card('Schedule entries', summary['schedule_entries'])}
      {metric_card('Fans', summary['fans'])}
      {metric_card('Avg confidence', summary['avg_confidence'])}
      {metric_card('Min confidence', summary['min_confidence'])}
    </section>

    <section class="section" id="overview">
      <h2>Run overview</h2>
      <table class="data-table">
        <thead>
          <tr>
            <th>Page</th><th>Kind</th><th>Fixture total</th><th>Types</th><th>Raw hits</th><th>Deduped</th><th>Schedule</th><th>Fans</th><th>Elapsed (s)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(page_rows) if page_rows else '<tr><td colspan="9">No pages processed.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section class="section" id="pages">
      <h2>Page details</h2>
      <p class="note">Use each page card to review counts, schedule links, occurrence details, and raw OCR samples.</p>
      {''.join(per_page_sections) if per_page_sections else '<p>No page details available.</p>'}
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html)
