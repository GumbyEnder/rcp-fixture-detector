
This file is the active tracked todo list for the OCR fixture-counting project.

Rule of work:
- Do one item at a time.
- After each item, run a QC/QA cycle against known drawings or fixtures.
- Do not start the next item until the current item has a pass/fail verdict.
- If QC/QA fails, fix the issue and rerun the same cycle before moving on.

## 0. OCR reporting and metrics pass
Status: pending

Improve report output so each run shows useful diagnostics.

Target outcomes:
- Per-page runtime
- Raw OCR hit counts
- Deduped fixture counts
- Tile counts / skipped blanks
- Fan detections separated from fixture totals
- Client-friendly default output and a debug mode

QC/QA gate:
- Run on a known sheet and verify the output is readable, stable, and useful.

## 1. Sheet classification before heavy OCR
Status: pending

Add lightweight pre-processing to identify page type before expensive OCR.

Target outcomes:
- Classify pages as primary fixture-count sheets, schedules/details, or irrelevant pages
- Skip or down-rank non-target pages
- Keep false positives/negatives visible in logs or metrics

QC/QA gate:
- Test on a mixed sheet set and confirm the right pages are scanned heavily.

## 2. Runtime reduction and caching
Status: pending

Reduce end-to-end OCR time without lowering quality.

Target outcomes:
- Cache page renders and OCR results more aggressively
- Skip blank or mostly blank tiles earlier
- Avoid duplicate work across schedule detection and plan parsing
- Reuse intermediate outputs when a page is scanned multiple times

QC/QA gate:
- Benchmark before/after on the same drawings and compare counts.

## 3. Count confidence and schedule reconciliation
Status: pending

Cross-check OCR counts against fixture schedules and surface mismatches.

Target outcomes:
- Flag disagreements between schedule totals and OCR totals
- Distinguish inferred counts from directly read counts
- Surface low-confidence pages for review

QC/QA gate:
- Validate against sheets with known schedule totals and confirm mismatch reporting is helpful.

## 4. Fan detection and symbol separation
Status: pending

Improve handling of ceiling fans and other non-light symbols.

Target outcomes:
- Detect fan symbols explicitly
- Keep fan callouts separate from fixture counts
- Avoid mixing fan symbols into light totals

QC/QA gate:
- Test on sheets containing fans and fixtures together and confirm the separation logic matches human review.

## 5. Limited cloud UI planning
Status: pending

Plan a small cloud-facing UI that lets users submit drawings and receive results.

Questions to resolve before implementation:
- Accepted file types and max file sizes
- Single drawing vs batch upload
- Anonymous jobs vs authenticated users
- Storage, retention, and deletion policy
- Job queueing and status tracking
- What the first user-facing results page should show
- Which details stay internal vs user-visible

QC/QA gate:
- Review the proposed flow with a mock submission before building anything.

## Working order
1. Finish OCR reporting and metrics.
2. Add sheet classification.
3. Reduce runtime and tighten caching.
4. Add confidence and reconciliation checks.
5. Improve fan handling.
6. Do the limited cloud UI plan last.

## Notes
- Keep the roadmap in sync with changelog entries after each completed item.
- If a task needs a broader design decision, pause and document the decision before implementation.
