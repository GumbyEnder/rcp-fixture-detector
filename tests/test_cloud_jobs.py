import pytest

from cloud.app.db import init_db
from cloud.app import jobs as job_service


@pytest.fixture(autouse=True)
def _tmp_cloud_data(tmp_path, monkeypatch):
    monkeypatch.setenv("RCP_CLOUD_DATA_DIR", str(tmp_path))
    init_db()


def test_create_job_rejects_non_pdf():
    with pytest.raises(ValueError, match="PDF"):
        job_service.create_job("drawing.dwg", b"123")


def test_create_job_returns_token():
    created = job_service.create_job("plan.pdf", b"%PDF-1.4\n%fake\n")
    assert created["status"] == "queued"
    job = job_service.get_job(created["job_id"], created["access_token"])
    assert job is not None
    assert job["original_filename"] == "plan.pdf"