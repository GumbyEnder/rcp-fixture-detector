from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from cloud.app.db import init_db
from cloud.app import jobs as job_service

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

app = FastAPI(title="RCP Fixture Detector Cloud", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


@app.post("/api/jobs")
async def api_create_job(file: UploadFile = File(...)):
    content = await file.read()
    try:
        created = job_service.create_job(file.filename or "upload.pdf", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(created)


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str, token: str = Query(...)):
    job = job_service.get_job(job_id, token)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "page_count": job["page_count"],
        "fixture_total": job["fixture_total"],
        "fan_total": job["fan_total"],
        "warning_count": job["warning_count"],
        "error_message": job["error_message"],
    }


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status_page(request: Request, job_id: str, token: str = Query(...)):
    job = job_service.get_job(job_id, token)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TEMPLATES.TemplateResponse(
        "status.html",
        {"request": request, "job": job, "token": token},
    )


@app.get("/jobs/{job_id}/result", response_class=HTMLResponse)
def job_result_page(request: Request, job_id: str, token: str = Query(...)):
    job = job_service.get_job(job_id, token)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")
    return TEMPLATES.TemplateResponse(
        "result.html",
        {"request": request, "job": job, "token": token},
    )


@app.get("/api/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str, token: str = Query(...)):
    job = job_service.get_job(job_id, token)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job_service.artifact_path(job, artifact)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)