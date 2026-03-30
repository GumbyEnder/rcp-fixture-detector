from rcp_detector.ocr.fixture_counter import _classify_page_text, _parse_schedule_entries


def test_parse_schedule_entries_extracts_codes_and_descriptions():
    lines = [
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "L-1A Recessed LED downlight", 0.99],
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "CF-12 Ceiling fan, 52in", 0.88],
    ]

    schedule = _parse_schedule_entries(lines)

    assert schedule == {
        "L-1A": "Recessed LED downlight",
        "CF-12": "Ceiling fan, 52in",
    }


def test_parse_schedule_entries_ignores_invalid_lines():
    lines = [
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "A-1 Sheet index", 0.99],
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "random text only", 0.88],
    ]

    schedule = _parse_schedule_entries(lines)

    assert schedule == {}


def test_classify_page_text_detects_schedule_pages():
    page_kind, meta = _classify_page_text("Fixture Schedule L-1A Recessed LED downlight")

    assert page_kind == "schedule"
    assert meta["reason"] == "schedule_keyword"


def test_classify_page_text_detects_detail_pages():
    page_kind, meta = _classify_page_text("Typical detail section at scale 1/2")

    assert page_kind == "detail"
    assert meta["reason"] == "detail_keyword"


def test_classify_page_text_marks_sparse_pages_irrelevant():
    page_kind, meta = _classify_page_text("Cover sheet")

    assert page_kind == "irrelevant"
    assert meta["reason"] == "irrelevant_keyword"


def test_classify_page_text_defaults_plan_for_fixture_drawings():
    page_kind, meta = _classify_page_text("RCP page with L-1, L-2, and CF-1 callouts")

    assert page_kind == "plan"
    assert meta["reason"] == "default"
