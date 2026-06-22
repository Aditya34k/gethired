import uuid
import asyncio
import structlog

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query

from services.api.schemas.profile import IngestResponse
from services.ingestion.resume_parser import parse_resume
from services.ingestion.embedder import store_profile_embeddings
from services.ingestion.linkedin_fetcher import fetch_and_merge_linkedin

log = structlog.get_logger()
router = APIRouter()

# In-memory store for job status.
# Phase 2 will move this to Redis — for now a dict is fine.
job_status: dict = {}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_candidate(
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    linkedin_url: str | None = Query(default=None),
):
    """
    Accepts a resume PDF and optional LinkedIn URL.
    Returns a candidate_id immediately.
    Processing happens in the background.

    WHY BACKGROUND TASKS?
    Parsing + LLM call + embedding takes 5-10 seconds.
    We don't want the HTTP request to hang that long.
    Instead we return instantly and process async.
    """
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    candidate_id = str(uuid.uuid4())
    pdf_bytes = await resume.read()

    # Mark as processing immediately
    job_status[candidate_id] = "processing"

    # Queue the heavy work to run after response is sent
    background_tasks.add_task(
        run_ingestion_pipeline,
        candidate_id=candidate_id,
        pdf_bytes=pdf_bytes,
        linkedin_url=linkedin_url,
    )

    return IngestResponse(
        candidate_id=candidate_id,
        status="processing",
        message="Resume received. Processing in background."
    )


@router.get("/status/{candidate_id}")
async def get_status(candidate_id: str):
    """Check the processing status of a submitted resume."""
    status = job_status.get(candidate_id)
    if not status:
        raise HTTPException(status_code=404, detail="candidate_id not found")
    return {"candidate_id": candidate_id, "status": status}


async def run_ingestion_pipeline(
    candidate_id: str,
    pdf_bytes: bytes,
    linkedin_url: str | None,
):
    """
    The actual pipeline — runs in the background after response is sent.
    1. Parse resume PDF
    2. Merge LinkedIn data if URL provided
    3. Store embeddings in Qdrant
    """
    try:
        log.info("pipeline.start", candidate_id=candidate_id)

        # Step 1 — parse resume
        profile = parse_resume(pdf_bytes)

        # Step 2 — merge LinkedIn if provided
        if linkedin_url:
            profile = await fetch_and_merge_linkedin(profile, linkedin_url)

        # Step 3 — embed and store in Qdrant
        chunks = store_profile_embeddings(profile, candidate_id)

        job_status[candidate_id] = "complete"
        log.info("pipeline.complete", candidate_id=candidate_id, chunks=chunks)

    except Exception as e:
        job_status[candidate_id] = f"failed: {str(e)}"
        log.error("pipeline.failed", candidate_id=candidate_id, error=str(e))