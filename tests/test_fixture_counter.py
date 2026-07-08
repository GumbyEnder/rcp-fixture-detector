from rcp_detector.ocr.fixture_counter import (
    FixtureCountResult,
    FixtureOccurrence,
    _calculate_reconciliation,
    _classify_page_text,
    _parse_schedule_entries,
    _tile_origins,
)


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


def test_calculate_reconciliation_flags_schedule_gaps_and_low_confidence():
    result = FixtureCountResult(
        page_name="page-1",
        fixture_counts={"L-1": 2, "L-2": 0, "L-3": 1, "CEILING_FAN": 1},
        schedule_entries={"L-1": "Recessed LED", "L-2": "Pendant"},
        occurrences=[
            FixtureOccurrence(code="L-1", quantity=2, bbox=[0, 0, 10, 10], raw_text="L-1", confidence=0.91),
            FixtureOccurrence(code="L-3", quantity=1, bbox=[0, 0, 10, 10], raw_text="L-3", confidence=0.42),
        ],
        fan_count=1,
    )

    metrics = _calculate_reconciliation(result)

    assert metrics["schedule_codes"] == 2
    assert metrics["zero_schedule_codes"] == 1
    assert metrics["unscheduled_codes"] == 1
    assert metrics["low_confidence_occurrences"] == 1
    assert metrics["avg_confidence"] == 0.665
    assert metrics["min_confidence"] == 0.42



def test_tile_origins_include_trailing_edge_when_step_misses_tail():
    origins = _tile_origins(length=2000, patch_size=640, step=320)

    assert origins[0] == 0
    assert origins[-1] == 1360
    assert 1280 in origins


def test_tile_origins_handle_small_or_invalid_dimensions():
    assert _tile_origins(length=500, patch_size=640, step=320) == [0]
    assert _tile_origins(length=0, patch_size=640, step=320) == [0]
    assert _tile_origins(length=1000, patch_size=0, step=320) == [0]
