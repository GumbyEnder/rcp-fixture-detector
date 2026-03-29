
from rcp_detector.ocr.fixture_counter import _parse_schedule_entries


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
