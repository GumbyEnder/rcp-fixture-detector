
Goal: let users submit their own drawings to a cloud-deployed version of the OCR fixture counter, while keeping the first version deliberately small and safe.

## v1 scope

1. Upload a drawing set
2. Queue OCR processing
3. Show job status
4. Show a concise result page
5. Allow download of the full markdown/JSON report

This is intentionally not a full multi-tenant product suite yet. No project collaboration, no annotations, and no public sharing in v1.

## Accepted inputs

Preferred first release:
- PDF only
- Optional ZIP of PDFs if batch upload is needed later

Out of scope for first release:
- DWG/DXF native support
- Image sequences without PDF packaging
- Active CAD editing

Suggested limits:
- PDF max size: 50 MB per file
- Batch cap: 10 files per submission
- Total batch size cap: 200 MB

## Submission model

Two reasonable options:

1. Anonymous job token
- User submits a drawing
- System returns a job ID + secret access token
- User can later revisit status/results with that token

2. Account-backed jobs
- User signs in
- Jobs are associated with a user account
- Easier long-term sharing/history, but heavier upfront scope

Recommended v1 choice:
- Anonymous job token
- Add accounts later only if there is a real need

## Processing flow

1. User uploads PDF(s)
2. API validates file type, size, and count
3. Files are stored in object storage
4. A job record is created with status = queued
5. Worker picks up the job
6. Worker renders pages, classifies pages, runs OCR, produces report artifacts
7. Status updates through queued -> processing -> completed/failed
8. User sees a results page and can download artifacts

## Storage and retention

Suggested storage layout:
- Raw uploads in object storage
- Rendered page images in ephemeral processing storage
- Final artifacts in durable storage

Retention policy for v1:
- Raw uploads: 7 days
- Generated outputs: 14 days
- Logs/debug bundles: 7 days unless explicitly pinned

## User-facing result page

Keep the first page simple:
- Job status
- Submission timestamp
- Page count processed
- Total fixture count summary
- Fan count summary
- Confidence / reconciliation warnings
- Download links for markdown and JSON

Do not show raw OCR text by default.

## Internal-only information

Keep these hidden from the normal result page:
- Raw OCR text dumps
- Page-level debug images
- Internal thresholds
- Confidence traces
- Worker logs
- Security/debug metadata

Provide an explicit debug mode for trusted users/admins only.

## Queueing and execution

Recommended shape:
- HTTP API accepts upload and creates job record
- Background worker pulls queued jobs
- Worker can run on a container or serverless job runner
- Job status stored in a database row
- UI polls job status or subscribes to updates

## MVP architecture recommendation

- Frontend: small Next.js app
- API: upload + job-status endpoints
- Storage: object storage for PDFs and outputs
- Queue: database-backed queue or managed job queue
- Worker: Python OCR worker running the existing pipeline
- Auth: anonymous job token for v1

## Mock submission review

Before building, walk through this mock flow:

1. Upload one PDF
2. Confirm the upload is accepted and validated
3. Confirm the job enters queued state
4. Confirm the worker processes it
5. Confirm the result page shows a concise summary
6. Confirm the user can download the final report
7. Confirm raw debug data is not exposed to normal users

## Design guardrails

- Keep the first version intentionally limited
- Prefer one clear submission path over many options
- Do not expose internal OCR complexity in the public UI
- Make results trustworthy before making the UI expansive
