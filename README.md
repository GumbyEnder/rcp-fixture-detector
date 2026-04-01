# RCP Fixture Detector

Detect and count ceiling-mounted fixtures from Reflected Ceiling Plan (RCP) PDFs.

This repository started as an OCR-first fixture counter and has grown into a broader toolkit for plan parsing, report generation, dataset prep, and model training. The current codebase supports tile-based OCR extraction, spatial deduplication, page classification, schedule reconciliation, fan separation, and markdown/JSON reporting.

## What this project already covers

- OCR-based fixture counting from plan text and tags
- Tile-based page scanning with spatial deduplication
- Page classification to skip blank, irrelevant, or non-target sheets
- Cached OCR / blank-page checks to reduce runtime
- Fan detection kept separate from fixture totals
- Schedule reconciliation and confidence summaries
- Markdown and JSON result output, plus visual previews
- Dataset utilities for annotation prep, splitting, and label stats
- YOLOv8 training scaffolding for the ML path
- A limited cloud UI plan for a future upload/process workflow

## Completed work to date

The active roadmap in `TODO.md` is complete. The major shipped improvements include:

1. OCR reporting and metrics
   - Added run summaries and clearer per-page metrics
   - Improved default report readability
   - Quieted noisy OCR logs during normal runs

2. Sheet classification before heavy OCR
   - Classifies pages as plan, schedule, detail, or irrelevant
   - Skips non-target pages in multi-page runs
   - Reuses cached OCR text for classification and extraction

3. Runtime reduction and caching
   - Added blank-page prechecks and page OCR caching
   - Reduced repeated work across classification and parsing

4. Count confidence and schedule reconciliation
   - Flags disagreements between schedule totals and OCR totals
   - Surfaces low-confidence pages and unscheduled codes
   - Adds reconciliation sections to the markdown report

5. Fan detection and symbol separation
   - Keeps fans separate from light / fixture totals
   - Records the fan detection method in the report

6. Limited cloud UI planning
   - Documented a small anonymous-job v1 upload flow
   - Defined storage, queueing, retention, and result-page boundaries

## CLI

The package exposes the `rcp-detect` command.

### Full pipeline

```bash
rcp-detect analyze <PDF_PATH> --output output/ --format json
```

### OCR-based counting

```bash
rcp-detect ocr-count <PDF_PATH> --output results.md
```

### PDF prep / tiling

```bash
rcp-detect prepare <PDF_PATH> --output output/
```

### Dataset helpers

```bash
rcp-detect stats <DATASET_PATH>
rcp-detect train <DATASET_PATH>
```

## Setup

A Python 3.9+ environment is required.

Typical local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

If you want to use the existing pinned environment from this repo, the checked-in `.venv311` can also be used.

## Verification

```bash
pytest -q
```

## Current notes

- OCR-first counting is the most mature path today.
- The ML / YOLO path is present, but the trained model and target drawings determine how useful it is.
- The sample PDFs in this repo are good for pipeline validation, but not all of them are ideal RCP fixtures.
- The cloud UI work is intentionally documented only as a plan for now.

## Supporting docs

- `CHANGELOG.md` — shipped changes and verification notes
- `TODO.md` — completed roadmap items and work order
- `SETUP_STATUS.md` — environment and compatibility notes
- `OCR_TEST_RESULTS.md` — OCR test observations and sample outputs
- `LIMITED_CLOUD_UI_PLAN.md` — future upload/process UI outline

## Project goal

Make fixture counting for reflected ceiling plans faster, more consistent, and easier to verify than manual takeoff.
