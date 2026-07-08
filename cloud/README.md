# RCP Fixture Detector — Cloud MVP (v1)

Anonymous PDF upload → queued job → OCR worker → status + result pages.

## Run locally

```bash
cd ~/rcp-fixture-detector
pip install -e ".[dev]"
pip install -r cloud/requirements.txt

export RCP_CLOUD_DATA_DIR=~/rcp-cloud-data
python -m cloud.worker   # terminal 1
uvicorn cloud.app.main:app --reload --port 8090   # terminal 2
```

Open http://127.0.0.1:8090/

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/jobs` | multipart `file` (PDF), returns `job_id` + `access_token` |
| GET | `/api/jobs/{job_id}` | JSON status (`?token=`) |
| GET | `/jobs/{job_id}` | HTML status page (`?token=`) |
| GET | `/jobs/{job_id}/result` | HTML summary + downloads (`?token=`) |
| GET | `/api/jobs/{job_id}/download/{artifact}` | `report.md`, `report.html`, `report.json`, `report.csv` |

## Limits (v1)

- PDF only, max 50 MB
- Local filesystem storage under `RCP_CLOUD_DATA_DIR`
- SQLite job queue

See `LIMITED_CLOUD_UI_PLAN.md` for product guardrails.