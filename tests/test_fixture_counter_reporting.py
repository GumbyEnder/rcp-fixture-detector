from rcp_detector.ocr.fixture_counter import FixtureCountResult, format_results_markdown


def test_format_results_markdown_includes_run_summary_and_page_metrics():
    results = [
        FixtureCountResult(
            page_name="page-1",
            fixture_counts={"L-1": 3, "CF-1": 1},
            schedule_entries={"L-1": "Recessed LED", "CF-1": "Ceiling fan"},
            occurrences=[],
            fan_count=1,
            metrics={
                "elapsed_s": 1.25,
                "raw_ocr_texts": 18,
                "raw_occurrences": 4,
                "deduped_occurrences": 2,
                "tile_count": 8,
                "blank_tiles_skipped": 3,
                "schedule_entries": 2,
                "fans": 1,
            },
        ),
        FixtureCountResult(
            page_name="page-2",
            fixture_counts={"L-2": 4},
            schedule_entries={"L-2": "Pendant"},
            occurrences=[],
            fan_count=0,
            metrics={
                "elapsed_s": 0.75,
                "raw_ocr_texts": 10,
                "raw_occurrences": 1,
                "deduped_occurrences": 1,
                "tile_count": 4,
                "blank_tiles_skipped": 1,
                "schedule_entries": 1,
                "fans": 0,
            },
        ),
    ]

    md = format_results_markdown(results)

    assert "## Run Summary" in md
    assert "| Pages | 2 |" in md
    assert "| Total elapsed | 2.0 s |" in md
    assert "| Fixture total | 8 |" in md
    assert "### page-1" in md
    assert "| L-1 | 3 | Recessed LED |" in md
    assert "| CF-1 | 1 |" in md
    assert "### page-2" in md
    assert "**Page Total: 4 fixtures**" in md
