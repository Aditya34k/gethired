import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import ingest, interview, prep
from services.api.middleware import APIKeyMiddleware
from services.api.config import settings

log = structlog.get_logger()

app = FastAPI(
    title="GetHired AI",
    description="AI-powered interview platform",
    version="0.1.0",
    docs_url="/docs",
)

# Auth middleware — runs before every request
app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1")
app.include_router(interview.router, prefix="/api/v1")
app.include_router(prep.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}