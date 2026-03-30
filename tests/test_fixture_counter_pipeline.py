from pathlib import Path


from rcp_detector.ocr.fixture_counter import FixtureCountResult, count_fixtures_from_pdf


def _fake_ocr_result(text: str):
    return [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], text, 0.99],
        ]
    ]


def test_count_fixtures_from_pdf_skips_detail_and_irrelevant_pages(monkeypatch, tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for name in ["plan.png", "schedule.png", "detail.png", "cover.png"]:
        (pages_dir / name).write_bytes(b"fake")

    ocr_map = {
        "plan.png": _fake_ocr_result("RCP page with L-1 callout"),
        "schedule.png": _fake_ocr_result("FIXTURE SCHEDULE L-1 Recessed LED"),
        "detail.png": _fake_ocr_result("Typical detail section at scale 1/2"),
        "cover.png": _fake_ocr_result("Cover sheet"),
    }

    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter._ocr_page_result",
        lambda image_path, lang="en": ocr_map[Path(image_path).name],
    )
    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter._classify_page_ocr",
        lambda result: (
            "schedule" if "FIXTURE SCHEDULE" in result[0][0][1].upper()
            else "detail" if "DETAIL" in result[0][0][1].upper()
            else "irrelevant" if "COVER" in result[0][0][1].upper()
            else "plan",
            {"reason": "test"},
        ),
    )
    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter._extract_schedule_from_page",
        lambda image_path, lang="en": {"L-1": "Recessed LED"},
    )
    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter.count_fixtures_ocr",
        lambda image_path, lang="en": FixtureCountResult(
            page_name=Path(image_path).stem,
            fixture_counts={"L-1": 1},
            schedule_entries={},
            occurrences=[],
            fan_count=0,
            metrics={"elapsed_s": 0.1, "ocr_lines": 1, "occurrences": 1, "schedule_entries": 0},
        ),
    )

    results = count_fixtures_from_pdf(
        "dummy.pdf",
        pages_dir=pages_dir,
        dpi=300,
        lang="en",
        use_tiling=False,
        detect_fans=False,
    )

    assert len(results) == 1
    result = results[0]
    assert result.page_name == "plan"
    assert result.metrics["page_kind"] == "plan"
    assert result.schedule_entries == {"L-1": "Recessed LED"}
    assert result.fixture_counts["L-1"] == 1


import cv2
import numpy as np


def test_count_fixtures_from_pdf_skips_blank_pages_without_ocr(monkeypatch, tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    blank_img = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(pages_dir / "blank.png"), blank_img)
    (pages_dir / "plan.png").write_bytes(b"fake")

    calls = {"ocr": 0}

    def fake_ocr(image_path, lang="en"):
        calls["ocr"] += 1
        name = Path(image_path).name
        if name == "blank.png":
            raise AssertionError("blank page should have been skipped before OCR")
        return _fake_ocr_result("RCP page with L-1 callout")

    monkeypatch.setattr("rcp_detector.ocr.fixture_counter._ocr_page_result", fake_ocr)
    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter._classify_page_ocr",
        lambda result: ("plan", {"reason": "test"}),
    )
    monkeypatch.setattr(
        "rcp_detector.ocr.fixture_counter.count_fixtures_ocr",
        lambda image_path, lang="en": FixtureCountResult(
            page_name=Path(image_path).stem,
            fixture_counts={"L-1": 1},
            schedule_entries={},
            occurrences=[],
            fan_count=0,
            metrics={"elapsed_s": 0.1, "ocr_lines": 1, "occurrences": 1, "schedule_entries": 0},
        ),
    )

    results = count_fixtures_from_pdf(
        "dummy.pdf",
        pages_dir=pages_dir,
        dpi=300,
        lang="en",
        use_tiling=False,
        detect_fans=False,
    )

    assert len(results) == 1
    assert calls["ocr"] == 1
