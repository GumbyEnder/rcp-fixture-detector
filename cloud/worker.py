"""Poll SQLite queue and run OCR jobs."""

from __future__ import annotations

import logging
import time

from cloud.app.db import init_db
from cloud.app.jobs import claim_next_queued_job, process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rcp_cloud_worker")


def main() -> None:
    init_db()
    logger.info("Worker started; polling for queued jobs")
    while True:
        job = claim_next_queued_job()
        if job is None:
            time.sleep(2)
            continue
        logger.info("Processing job %s (%s)", job["id"], job["original_filename"])
        process_job(job)
        logger.info("Job %s finished with status update", job["id"])


if __name__ == "__main__":
    main()