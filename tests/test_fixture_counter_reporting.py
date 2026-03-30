from rcp_detector.ocr.fixture_counter import FixtureCountResult, FixtureOccurrence, format_results_markdown


def test_format_results_markdown_includes_run_summary_and_page_metrics():
    results = [
        FixtureCountResult(
            page_name="page-1",
            fixture_counts={"L-1": 3, "CF-1": 1, "L-3": 1},
            schedule_entries={"L-1": "Recessed LED", "CF-1": "Ceiling fan"},
            occurrences=[
                FixtureOccurrence(code="L-1", quantity=3, bbox=[0, 0, 10, 10], raw_text="L-1", confidence=0.98),
                FixtureOccurrence(code="CF-1", quantity=1, bbox=[10, 10, 20, 20], raw_text="CF-1", confidence=0.72),
                FixtureOccurrence(code="L-3", quantity=1, bbox=[20, 20, 30, 30], raw_text="L-3", confidence=0.48),
            ],
            fan_count=1,
            metrics={
                "elapsed_s": 1.25,
                "raw_ocr_texts": 18,
                "raw_occurrences": 4,
                "deduped_occurrences": 2,
                "tile_count": 8,
                "blank_tiles_skipped": 3,
                "schedule_entries": 2,
                "zero_schedule_codes": 1,
                "unscheduled_codes": 1,
                "low_confidence_occurrences": 1,
                "avg_confidence": 0.85,
                "min_confidence": 0.72,
                "fans": 1,
                "fan_detection_method": "template",
            },
        ),
        FixtureCountResult(
            page_name="page-2",
            fixture_counts={"L-2": 4},
            schedule_entries={"L-2": "Pendant"},
            occurrences=[
                FixtureOccurrence(code="L-2", quantity=4, bbox=[0, 0, 10, 10], raw_text="L-2", confidence=0.66),
            ],
            fan_count=0,
            metrics={
                "elapsed_s": 0.75,
                "raw_ocr_texts": 10,
                "raw_occurrences": 1,
                "deduped_occurrences": 1,
                "tile_count": 4,
                "blank_tiles_skipped": 1,
                "schedule_entries": 1,
                "zero_schedule_codes": 0,
                "unscheduled_codes": 1,
                "low_confidence_occurrences": 0,
                "avg_confidence": 0.66,
                "min_confidence": 0.66,
                "fans": 0,
                "fan_detection_method": "disabled",
            },
        ),
    ]

    md = format_results_markdown(results)

    assert "## Run Summary" in md
    assert "| Pages | 2 |" in md
    assert "| Total elapsed | 2.0 s |" in md
    assert "| Fixture total | 9 |" in md
    assert "| Schedule zero-counts | 1 |" in md
    assert "| Low-confidence hits | 1 |" in md
    assert "### page-1" in md
    assert "| L-1 | 3 | Recessed LED |" in md
    assert "| CF-1 | 1 |" in md
    assert "**Reconciliation:**" in md
    assert "| Unscheduled codes | 1 |" in md
    assert "Fan Detection Method: template" in md
    assert "### page-2" in md
    assert "**Page Total: 4 fixtures**" in md
