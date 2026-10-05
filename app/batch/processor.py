"""
app/batch/processor.py
-----------------------
Asynchronous batch invoice processing.

Flow:
  1. Caller uploads a .zip file containing multiple PDF invoices.
  2. A unique `job_id` (UUID) is returned immediately (non-blocking).
  3. FastAPI BackgroundTask runs `process_batch_job()` in the background:
       - Extracts each PDF from the zip.
       - Runs the full extraction + PO matching pipeline on each.
       - Stores results and partial failures in an in-memory job registry.
  4. Caller polls `GET /batch/{job_id}` to check progress and retrieve results.

Limitations (acceptable for v1):
  - Job state is in-memory: results are lost on server restart.
  - For production, replace job registry with Redis or a database table.
"""

import asyncio
import io
import logging
import time
import uuid
import zipfile
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from app.extractor import ExtractionResult, extract_invoice
from app.rag.matcher import match_invoice_to_po

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job state models
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FileResult(BaseModel):
    filename: str
    status: str  # "success" | "failed"
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class BatchJob(BaseModel):
    job_id: str
    status: JobStatus
    total_files: int
    processed_files: int = 0
    created_at: str
    completed_at: Optional[str] = None
    results: list[FileResult] = []


# ---------------------------------------------------------------------------
# In-memory job registry (replace with Redis for production)
# ---------------------------------------------------------------------------
_JOB_REGISTRY: dict[str, BatchJob] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_job(total_files: int) -> BatchJob:
    """Register and return a new batch job."""
    job = BatchJob(
        job_id=str(uuid.uuid4()),
        status=JobStatus.PENDING,
        total_files=total_files,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _JOB_REGISTRY[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[BatchJob]:
    """Retrieve a job by ID. Returns None if not found."""
    return _JOB_REGISTRY.get(job_id)


async def process_batch_job(job_id: str, zip_bytes: bytes) -> None:
    """
    Background task: extract each PDF from the zip archive sequentially.
    Updates the job registry as each file is processed.
    """
    job = _JOB_REGISTRY[job_id]
    job.status = JobStatus.RUNNING
    logger.info("Batch job %s started (%d files).", job_id, job.total_files)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            valid_exts = (".pdf", ".jpg", ".jpeg", ".png")
            file_names = [
                n for n in zf.namelist() 
                if n.lower().endswith(valid_exts) and not n.startswith("__MACOSX")
            ]

            for file_name in file_names:
                file_result = await _process_single_file(zf, file_name)
                job.results.append(file_result)
                job.processed_files += 1
                logger.info(
                    "Batch job %s: %d/%d — %s [%s]",
                    job_id, job.processed_files, job.total_files,
                    file_name, file_result.status,
                )

        job.status = JobStatus.COMPLETED

    except Exception as exc:
        logger.error("Batch job %s crashed: %s", job_id, exc)
        job.status = JobStatus.FAILED

    finally:
        job.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info("Batch job %s finished with status: %s", job_id, job.status)


async def _process_single_file(zf: zipfile.ZipFile, file_name: str) -> FileResult:
    """Extract and match a single PDF or Image from within a ZipFile object."""
    from app.vision import process_file_to_base64
    
    try:
        file_bytes = zf.read(file_name)
        base64_image = process_file_to_base64(file_bytes, filename=file_name)
        result: ExtractionResult = await extract_invoice(base64_image)
        po_match = await match_invoice_to_po(result.invoice)

        return FileResult(
            filename=file_name,
            status="success",
            data={
                "invoice": result.invoice.model_dump(by_alias=True, mode="json"),
                "po_match": po_match.model_dump(),
                "metrics": {
                    "input_tokens": result.metrics.input_tokens,
                    "output_tokens": result.metrics.output_tokens,
                    "cost_usd": result.metrics.cost_usd,
                    "latency_seconds": result.metrics.latency_seconds,
                    "retries": result.metrics.retries,
                },
            },
        )

    except Exception as exc:
        return FileResult(filename=file_name, status="failed", error=str(exc))
