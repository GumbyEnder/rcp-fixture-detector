from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, '__dict__') and not isinstance(value, type):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith('_')}
    return value


def _bbox_values(occ: Any) -> list[float]:
    bbox = _get(occ, 'bbox', _get(occ, 'bbox_abs', _get(occ, 'bbox_norm', [])))
    if isinstance(bbox, str):
        return []
    if not bbox:
        return []
    if len(bbox) != 4:
        return []
    try:
        return [float(v) for v in bbox]
    except Exception:
        return []


def _normalize_occurrence(occ: Any, page_name: str, page_index: int, page_kind: str, page_total: int, fan_count: int, qc_flags: list[str]) -> dict[str, Any]:
    bbox = _bbox_values(occ)
    confidence = float(_get(occ, 'confidence', 0.0) or 0.0)
    return {
        'page_index': page_index,
        'page_name': page_name,
        'page_kind': page_kind,
        'code': _get(occ, 'code', ''),
        'quantity': int(_get(occ, 'quantity', 1) or 1),
        'source': _get(occ, 'source', ''),
        'confidence': round(confidence, 3),
        'bbox': bbox,
        'raw_text': _get(occ, 'raw_text', ''),
        'fan_count': fan_count,
        'qc_flags': list(qc_flags),
        'page_total': page_total,
    }


def _result_metrics(result: Any) -> dict[str, Any]:
    metrics = _get(result, 'metrics', {}) or {}
    return _plain(metrics)


def _page_summary(result: Any, page_index: int) -> dict[str, Any]:
    fixture_counts = _get(result, 'fixture_counts', {}) or {}
    schedule_entries = _get(result, 'schedule_entries', {}) or {}
    occurrences = _get(result, 'occurrences', []) or []
    fan_count = int(_get(result, 'fan_count', _get(result, 'metrics', {}).get('fans', 0)) or 0)
    metrics = _result_metrics(result)
    page_name = _get(result, 'page_name', f'page_{page_index}')
    page_kind = str(metrics.get('page_kind', 'n/a'))
    page_total = sum(int(v) for v in fixture_counts.values())
    qc_flags = _qc_flags(result)

    return {
        'page_index': page_index,
        'page_name': page_name,
        'page_kind': page_kind,
        'fixture_total': page_total,
        'fixture_types': len(fixture_counts),
        'fixture_counts': {str(k): int(v) for k, v in fixture_counts.items()},
        'schedule_entries': {str(k): str(v) for k, v in schedule_entries.items()},
        'fan_count': fan_count,
        'metrics': metrics,
        'qc_flags': qc_flags,
        'occurrence_count': len(occurrences),
        'reconciliation': {
            'schedule_codes': int(metrics.get('schedule_entries', len(schedule_entries)) or 0),
            'zero_schedule_codes': int(metrics.get('zero_schedule_codes', 0) or 0),
            'unscheduled_codes': int(metrics.get('unscheduled_codes', 0) or 0),
            'low_confidence_occurrences': int(metrics.get('low_confidence_occurrences', 0) or 0),
            'avg_confidence': float(metrics.get('avg_confidence', 0.0) or 0.0),
            'min_confidence': float(metrics.get('min_confidence', 0.0) or 0.0),
        },
    }


def _qc_flags(result: Any) -> list[str]:
    metrics = _result_metrics(result)
    fixture_counts = _get(result, 'fixture_counts', {}) or {}
    schedule_entries = _get(result, 'schedule_entries', {}) or {}
    fan_count = int(_get(result, 'fan_count', metrics.get('fans', 0)) or 0)
    page_total = sum(int(v) for v in fixture_counts.values())

    flags: list[str] = []
    if int(metrics.get('zero_schedule_codes', 0) or 0):
        flags.append(f"{int(metrics['zero_schedule_codes'])} schedule code(s) not found in counts")
    if int(metrics.get('unscheduled_codes', 0) or 0):
        flags.append(f"{int(metrics['unscheduled_codes'])} counted code(s) absent from schedule")
    if int(metrics.get('low_confidence_occurrences', 0) or 0):
        flags.append(f"{int(metrics['low_confidence_occurrences'])} low-confidence OCR hit(s)")
    if page_total == 0:
        if schedule_entries:
            flags.append('Schedule present, but no qualifying fixture counts remained')
        else:
            flags.append('No qualifying fixture counts found')
    if fan_count:
        flags.append(f"{fan_count} fan(s) kept separate from fixture totals")
    return flags


def _summary_metrics(results: list) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'pages': len(results),
        'elapsed_s': 0.0,
        'fixture_types': 0,
        'fixture_total': 0,
        'raw_ocr_texts': 0,
        'raw_occurrences': 0,
        'deduped_occurrences': 0,
        'tiles': 0,
        'blank_tiles_skipped': 0,
        'schedule_entries': 0,
        'zero_schedule_codes': 0,
        'unscheduled_codes': 0,
        'low_confidence_occurrences': 0,
        'fans': 0,
        'confidence_sum': 0.0,
        'confidence_count': 0,
        'min_confidence': None,
    }

    for result in results:
        metrics = _result_metrics(result)
        fixture_counts = _get(result, 'fixture_counts', {}) or {}
        occurrences = _get(result, 'occurrences', []) or []
        schedule_entries = _get(result, 'schedule_entries', {}) or {}

        summary['elapsed_s'] += float(metrics.get('elapsed_s', metrics.get('page_elapsed_s', 0.0)) or 0.0)
        summary['fixture_types'] += len(fixture_counts)
        summary['fixture_total'] += sum(int(v) for v in fixture_counts.values())
        summary['raw_ocr_texts'] += int(metrics.get('raw_ocr_texts', metrics.get('ocr_lines', 0)) or 0)
        summary['raw_occurrences'] += int(metrics.get('raw_occurrences', metrics.get('occurrences', 0)) or 0)
        summary['deduped_occurrences'] += int(metrics.get('deduped_occurrences', len(occurrences)) or 0)
        summary['tiles'] += int(metrics.get('tile_count', 0) or 0)
        summary['blank_tiles_skipped'] += int(metrics.get('blank_tiles_skipped', 0) or 0)
        summary['schedule_entries'] += int(metrics.get('schedule_entries', len(schedule_entries)) or 0)
        summary['zero_schedule_codes'] += int(metrics.get('zero_schedule_codes', 0) or 0)
        summary['unscheduled_codes'] += int(metrics.get('unscheduled_codes', 0) or 0)
        summary['low_confidence_occurrences'] += int(metrics.get('low_confidence_occurrences', 0) or 0)
        summary['fans'] += int(metrics.get('fans', _get(result, 'fan_count', 0)) or 0)

        confidences = [float(_get(occ, 'confidence', 0.0) or 0.0) for occ in occurrences]
        summary['confidence_sum'] += sum(confidences)
        summary['confidence_count'] += len(confidences)
        if confidences:
            page_min = min(confidences)
            if summary['min_confidence'] is None:
                summary['min_confidence'] = page_min
            else:
                summary['min_confidence'] = min(float(summary['min_confidence']), page_min)

    summary['elapsed_s'] = round(float(summary['elapsed_s']), 3)
    if summary['confidence_count']:
        summary['avg_confidence'] = round(float(summary['confidence_sum']) / float(summary['confidence_count']), 3)
        summary['min_confidence'] = round(float(summary['min_confidence']), 3) if summary['min_confidence'] is not None else 0.0
    else:
        summary['avg_confidence'] = 0.0
        summary['min_confidence'] = 0.0
    return summary


def build_report_bundle(results: list) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    qc_findings: list[dict[str, Any]] = []

    for page_index, result in enumerate(results, start=1):
        page = _page_summary(result, page_index)
        pages.append(page)
        if page['qc_flags']:
            qc_findings.append({
                'page_index': page_index,
                'page_name': page['page_name'],
                'page_kind': page['page_kind'],
                'fixture_total': page['fixture_total'],
                'schedule_entries': len(page['schedule_entries']),
                'flags': page['qc_flags'],
                'reconciliation': page['reconciliation'],
            })

        for occ in _get(result, 'occurrences', []) or []:
            detections.append(_normalize_occurrence(
                occ,
                page_name=page['page_name'],
                page_index=page_index,
                page_kind=page['page_kind'],
                page_total=page['fixture_total'],
                fan_count=page['fan_count'],
                qc_flags=page['qc_flags'],
            ))

    return {
        'schema_version': 'rcp-fixture-report-v1',
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'summary': _summary_metrics(results),
        'qc_findings': qc_findings,
        'pages': pages,
        'detections': detections,
    }


def write_normalized_json(results: list, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report_bundle(results), indent=2, default=_plain))


def _write_page_summary_csv(results: list, output_path: Path) -> None:
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'page_index', 'page_name', 'page_kind', 'fixture_total', 'fixture_types', 'fan_count',
            'raw_occurrences', 'deduped_occurrences', 'schedule_entries', 'zero_schedule_codes',
            'unscheduled_codes', 'low_confidence_occurrences', 'avg_confidence', 'min_confidence',
            'elapsed_s', 'qc_flags',
        ])
        for page in build_report_bundle(results)['pages']:
            rec = page['reconciliation']
            metrics = page['metrics'] or {}
            writer.writerow([
                page['page_index'],
                page['page_name'],
                page['page_kind'],
                page['fixture_total'],
                page['fixture_types'],
                page['fan_count'],
                metrics.get('raw_occurrences', metrics.get('occurrences', 0)),
                metrics.get('deduped_occurrences', page['occurrence_count']),
                rec.get('schedule_codes', page['reconciliation']['schedule_codes']),
                rec.get('zero_schedule_codes', 0),
                rec.get('unscheduled_codes', 0),
                rec.get('low_confidence_occurrences', 0),
                rec.get('avg_confidence', 0.0),
                rec.get('min_confidence', 0.0),
                metrics.get('elapsed_s', metrics.get('page_elapsed_s', '')),
                ' | '.join(page['qc_flags']),
            ])


def write_normalized_csv(results: list, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_report_bundle(results)

    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'page_index', 'page_name', 'page_kind', 'code', 'quantity', 'source', 'confidence',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'raw_text', 'fan_count', 'fixture_total',
            'schedule_entries', 'zero_schedule_codes', 'unscheduled_codes', 'low_confidence_occurrences',
            'qc_flags',
        ])
        for row in bundle['detections']:
            bbox = row.get('bbox') or []
            while len(bbox) < 4:
                bbox = list(bbox) + ['']
            page = bundle['pages'][row['page_index'] - 1]
            rec = page['reconciliation']
            writer.writerow([
                row['page_index'], row['page_name'], row['page_kind'], row['code'], row['quantity'],
                row['source'], row['confidence'], bbox[0], bbox[1], bbox[2], bbox[3], row['raw_text'],
                row['fan_count'], row['page_total'], rec.get('schedule_codes', 0), rec.get('zero_schedule_codes', 0),
                rec.get('unscheduled_codes', 0), rec.get('low_confidence_occurrences', 0), ' | '.join(row['qc_flags']),
            ])

    _write_page_summary_csv(results, output_path.with_name(f"{output_path.stem}.pages.csv"))


def write_marked_pdf(source_pdf: str | Path, results: list, output_path: str | Path, dpi: int = 300) -> None:
    """Create a marked-up PDF by overlaying OCR detections on the source pages.

    The annotations are drawn using the render DPI used for OCR, so the pixel
    coordinates can be mapped back to PDF points.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - dependency not installed in some shells
        raise RuntimeError('PyMuPDF (fitz) is required for marked PDF output') from exc

    source_pdf = Path(source_pdf)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(source_pdf)
    scale = 72.0 / float(dpi)
    palette = {
        'plan': (0.18, 0.45, 0.78),
        'schedule': (0.33, 0.58, 0.28),
        'fan': (0.82, 0.42, 0.12),
        'default': (0.75, 0.18, 0.22),
    }

    for page_index, (page, result) in enumerate(zip(doc, results), start=1):
        page_name = _get(result, 'page_name', f'page_{page_index}')
        occurrences = _get(result, 'occurrences', []) or []
        metrics = _result_metrics(result)
        page_kind = str(metrics.get('page_kind', 'plan'))
        page_total = sum(int(v) for v in (_get(result, 'fixture_counts', {}) or {}).values())
        fan_count = int(_get(result, 'fan_count', metrics.get('fans', 0)) or 0)

        # Small page label near the top-left edge so viewers know which sheet they are on.
        header_color = palette.get(page_kind, palette['default'])
        page.insert_text(
            (18, 18),
            f"{page_name} • {page_kind} • {page_total} fixture(s) • {fan_count} fan(s)",
            fontsize=8,
            color=header_color,
        )

        for occ in occurrences:
            bbox = _bbox_values(occ)
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [coord * scale for coord in bbox]
            rect = fitz.Rect(x1, y1, x2, y2)
            source = str(_get(occ, 'source', 'plan') or 'plan')
            color = palette['fan'] if source == 'fan' else palette.get(source, header_color)
            code = str(_get(occ, 'code', ''))
            quantity = int(_get(occ, 'quantity', 1) or 1)
            confidence = float(_get(occ, 'confidence', 0.0) or 0.0)
            label = f"{code} ×{quantity} ({confidence:.2f})"

            page.draw_rect(rect, color=color, width=1.2)
            label_left = rect.x0
            label_top = max(0, rect.y0 - 11)
            label_bottom = max(label_top + 9, rect.y0 - 1)
            label_right = label_left + max(85, len(label) * 4.6)
            label_rect = fitz.Rect(label_left, label_top, label_right, label_bottom)
            page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.6)
            page.insert_textbox(label_rect, label, fontsize=7, color=color, align=0)

    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
