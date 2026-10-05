"""
main.py — FastAPI application with all 5 upgraded capabilities:

  POST /extract-invoice    — Single invoice extraction + PO match
  POST /batch/upload       — Upload a .zip of PDFs for async batch processing
  GET  /batch/{job_id}     — Poll batch job status and results
  GET  /metrics/summary    — Aggregate cost / token / latency dashboard
  GET  /metrics/recent     — Last 20 individual extraction records
  GET  /health             — Liveness probe
"""

import io
import logging
import zipfile
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.batch.processor import (
    BatchJob,
    create_job,
    get_job,
    process_batch_job,
)
from app.config import settings
from app.extractor import ExtractionError, ExtractionResult, extract_invoice
from app.observability.tracker import (
    ExtractionMetrics,
    get_recent,
    get_summary,
    init_db,
    record,
)
from app.rag.matcher import POMatchResult, match_invoice_to_po
from app.rag.po_store import seed_mock_pos
from app.schemas import InvoiceSchema
from app.vision import pdf_to_base64_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ExtractionResponse(BaseModel):
    """Full response: invoice data + PO match + observability metrics."""
    invoice: InvoiceSchema
    po_match: POMatchResult
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Invoice Extractor API…")
    await init_db()          # Create SQLite metrics table
    seed_mock_pos()          # Seed ChromaDB with mock POs (idempotent)
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down Invoice Extractor API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision-RAG Invoice Data Extractor (v2)",
    description=(
        "Production-grade invoice extraction: Vision AI + PO matching RAG + "
        "async batch processing + LLM cost observability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /extract-invoice — Single PDF extraction
# ---------------------------------------------------------------------------
@app.post(
    "/extract-invoice",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured data from a single PDF invoice",
    tags=["Extraction"],
)
async def extract_invoice_endpoint(
    file: UploadFile = File(..., description="PDF or Image invoice file."),
) -> ExtractionResponse:
    _validate_file(file)
    file_bytes = await _read_file(file)

    base64_image = _convert_file(file_bytes, file.filename, file.content_type)

    result: ExtractionResult = await _run_extraction(base64_image)

    # PO matching
    po_match = await match_invoice_to_po(result.invoice)
    result.metrics.po_matched = po_match.matched

    # Persist metrics
    await record(result.metrics)

    return ExtractionResponse(
        invoice=result.invoice,
        po_match=po_match,
        metrics={
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "cost_usd": result.metrics.cost_usd,
            "latency_seconds": result.metrics.latency_seconds,
            "retries": result.metrics.retries,
            "model": result.metrics.model,
            "provider": result.metrics.provider,
        },
    )


# ---------------------------------------------------------------------------
# POST /purchase-orders — Dynamic RAG Database Insertion
# ---------------------------------------------------------------------------
class POCreate(BaseModel):
    po_number: str
    vendor_name: str
    total_amount: float
    items: str

@app.post(
    "/purchase-orders",
    status_code=status.HTTP_201_CREATED,
    summary="Embed and store a new Purchase Order in ChromaDB",
    tags=["RAG"],
)
async def create_purchase_order(po: POCreate) -> dict:
    from app.rag.po_store import add_purchase_order
    add_purchase_order(po.po_number, po.vendor_name, po.total_amount, po.items)
    return {"status": "success", "message": f"PO {po.po_number} securely embedded and stored."}


# ---------------------------------------------------------------------------
# POST /approve-invoice — Human-In-The-Loop save endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/approve-invoice",
    status_code=status.HTTP_201_CREATED,
    summary="Save human-verified invoice data to the local database/disk",
    tags=["Extraction"],
)
async def approve_invoice(invoice: dict) -> dict:
    import os
    import json
    from datetime import datetime
    
    os.makedirs("data/approved_invoices", exist_ok=True)
    vendor = invoice.get("VendorName", "UnknownVendor").replace(" ", "_").replace("/", "")
    ts = int(datetime.now().timestamp())
    filename = f"data/approved_invoices/invoice_{vendor}_{ts}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(invoice, f, indent=2)
        
    return {"status": "success", "file_saved": filename}


# ---------------------------------------------------------------------------
# POST /batch/upload — Upload .zip for async batch processing
# ---------------------------------------------------------------------------
@app.post(
    "/batch/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a ZIP of PDF invoices for async batch extraction",
    tags=["Batch"],
)
async def batch_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description=".zip file containing PDF invoices."),
) -> dict:
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted for batch upload.",
        )

    zip_bytes = await file.read()

    # Count valid files in the archive
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            valid_exts = (".pdf", ".jpg", ".jpeg", ".png")
            valid_files = [
                n for n in zf.namelist() 
                if n.lower().endswith(valid_exts) and not n.startswith("__MACOSX")
            ]
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or corrupt ZIP file."
        )

    if not valid_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ZIP archive contains no valid PDF, JPG, or PNG files.",
        )

    if len(valid_files) > settings.batch_max_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files ({len(valid_files)}). Max allowed: {settings.batch_max_files}.",
        )

    job = create_job(total_files=len(valid_files))
    background_tasks.add_task(process_batch_job, job.job_id, zip_bytes)

    logger.info("Batch job %s queued with %d files.", job.job_id, len(valid_files))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_files": job.total_files,
        "message": f"Batch job accepted. Poll GET /batch/{job.job_id} for progress.",
    }


# ---------------------------------------------------------------------------
# GET /batch/{job_id} — Poll batch job status
# ---------------------------------------------------------------------------
@app.get(
    "/batch/{job_id}",
    response_model=BatchJob,
    summary="Get the status and results of a batch extraction job",
    tags=["Batch"],
)
async def get_batch_status(job_id: str) -> BatchJob:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job


# ---------------------------------------------------------------------------
# GET /metrics/summary — Aggregate observability dashboard
# ---------------------------------------------------------------------------
@app.get(
    "/metrics/summary",
    summary="Aggregate cost, token usage, latency and self-healing metrics",
    tags=["Observability"],
)
async def metrics_summary() -> dict:
    return await get_summary()


# ---------------------------------------------------------------------------
# GET /metrics/recent — Last 20 extraction records
# ---------------------------------------------------------------------------
@app.get(
    "/metrics/recent",
    summary="Last 20 individual extraction records",
    tags=["Observability"],
)
async def metrics_recent() -> list[dict]:
    return await get_recent(limit=20)


# ---------------------------------------------------------------------------
# GET /health — Liveness probe
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _validate_file(file: UploadFile) -> None:
    valid_types = (
        "application/pdf", 
        "image/jpeg", 
        "image/png", 
        "image/webp", 
        "application/octet-stream"
    )
    if file.content_type not in valid_types and not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Only PDF, JPG, and PNG files are accepted.",
        )


async def _read_file(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )
    logger.info("Received File: %s (%d bytes)", file.filename, len(file_bytes))
    return file_bytes


def _convert_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    from app.vision import process_file_to_base64
    try:
        return process_file_to_base64(file_bytes, filename, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"File processing failed: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Server config error: {exc}")


async def _run_extraction(base64_image: str) -> ExtractionResult:
    try:
        return await extract_invoice(base64_image)
    except ExtractionError as exc:
        detail: dict = {"error": str(exc)}
        if exc.last_validation_error:
            detail["validation_errors"] = exc.last_validation_error.errors()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
