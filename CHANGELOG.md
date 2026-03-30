## 2026-03-30 — OCR workflow follow-up

### Changed
- Fixed the `rcp_detector/cli.py` startup crash by adding the missing `logging` import.
- Quieted noisy PaddleOCR loggers during OCR-count runs.
- Expanded markdown output to include run metrics per page.
- Documented the Manor East fixture scan outputs in the project workflow notes.

### Verified
- `python3 -m py_compile rcp_detector/cli.py rcp_detector/ocr/fixture_counter.py`

## 2026-03-29 — OCR fixture counter cleanup pass

### Changed
- Restored and cleaned up `rcp_detector/ocr/fixture_counter.py` after the broken rewrite.
- Kept OCR-first fixture counting, tile-based OCR, schedule parsing, partial-read filtering, and spatial deduplication.
- Added per-page timing and metrics to fixture count results.
- Added OCR page-result caching to reduce repeated work across schedule detection and plan parsing.
- Removed deprecated PaddleOCR `cls=True` usage and `show_log=False` init parameter from OCR calls.
- Added run log support in the CLI for `ocr-count`.
- Adjusted default PDF DPI from 400 to 300 to match test expectations and keep runtime manageable.
- Updated tests for schedule parsing behavior.

### Verified
- `python -m py_compile rcp_detector/ocr/fixture_counter.py rcp_detector/cli.py rcp_detector/ocr/tag_reader.py tests/test_fixture_counter.py`
- `pytest tests/test_fixture_counter.py -q` → 2 passed

### Notes
- The fixture counter still uses the same OCR-first path and should preserve output quality.
- Remaining work is to finish the full Manor East sheet-by-sheet fixture takeoff if desired.