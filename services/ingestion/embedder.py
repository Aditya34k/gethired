import uuid
import structlog
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from services.api.config import settings
from services.api.schemas.profile import CandidateProfile

log = structlog.get_logger()

# Load the embedding model once at module level.
# This means it loads when the file is first imported, not on every call.
# "all-MiniLM-L6-v2" is small (80MB), fast, and good enough for our use case.
# It outputs 384-dimensional vectors.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = 384

# Collection names — constants so we never typo a string
COLLECTION_RESUMES = "resume_chunks"
COLLECTION_KNOWLEDGE = "knowledge_base"


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collections(client: QdrantClient) -> None:
    """
    Creates Qdrant collections if they don't exist yet.
    Safe to call every time — it checks before creating.
    """
    existing = {c.name for c in client.get_collections().collections}

    for name in [COLLECTION_RESUMES, COLLECTION_KNOWLEDGE]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    # COSINE measures the angle between vectors.
                    # Two texts with similar meaning will have vectors
                    # pointing in similar directions — high cosine similarity.
                ),
            )
            log.info("qdrant.collection_created", name=name)


def embed_text(text: str) -> list[float]:
    """
    Converts a string into a vector (list of floats).
    This is what gets stored in Qdrant and searched against.
    """
    vector = EMBEDDING_MODEL.encode(text)
    return vector.tolist()


def chunk_profile(profile: CandidateProfile) -> list[dict]:
    """
    Splits a profile into meaningful chunks for storage.

    WHY CHUNK?
    If you store the whole profile as one vector, you lose granularity.
    A question about Python skills should match the skills chunk strongly,
    not get diluted by experience and education text.

    Each chunk has:
    - text: what gets embedded (the semantic content)
    - payload: metadata stored alongside the vector (filterable later)
    """
    chunks = []

    # Chunk 1 — skills
    if profile.skills:
        skills_text = "Technical skills: " + ", ".join(profile.skills)
        chunks.append({
            "text": skills_text,
            "payload": {
                "chunk_type": "skills",
                "skills": profile.skills,
                "candidate_name": profile.full_name,
            }
        })

    # Chunk 2 — one chunk per experience entry
    for exp in profile.experience:
        exp_text = (
            f"{exp.title} at {exp.company} "
            f"({exp.start_year} - {exp.end_year or 'present'}): "
            f"{exp.description}"
        )
        chunks.append({
            "text": exp_text,
            "payload": {
                "chunk_type": "experience",
                "company": exp.company,
                "title": exp.title,
                "start_year": exp.start_year,
                "end_year": exp.end_year,
                "candidate_name": profile.full_name,
            }
        })

    # Chunk 3 — education summary
    if profile.education:
        edu_parts = [
            f"{e.degree} from {e.institution} ({e.year or 'unknown year'})"
            for e in profile.education
        ]
        edu_text = "Education: " + "; ".join(edu_parts)
        chunks.append({
            "text": edu_text,
            "payload": {
                "chunk_type": "education",
                "candidate_name": profile.full_name,
            }
        })

    # Chunk 4 — LinkedIn summary if available
    if profile.linkedin_data.get("summary"):
        chunks.append({
            "text": profile.linkedin_data["summary"],
            "payload": {
                "chunk_type": "linkedin_summary",
                "headline": profile.linkedin_data.get("headline", ""),
                "candidate_name": profile.full_name,
            }
        })

    return chunks


def store_profile_embeddings(
    profile: CandidateProfile,
    candidate_id: str
) -> int:
    """
    Main public function — embeds all profile chunks and stores in Qdrant.
    Returns the number of chunks stored.
    """
    log.info("embedder.start", candidate_id=candidate_id)

    client = get_qdrant_client()
    ensure_collections(client)

    chunks = chunk_profile(profile)
    points = []

    for chunk in chunks:
        vector = embed_text(chunk["text"])

        # PointStruct = one record in Qdrant
        # id: unique UUID for this point
        # vector: the float array from embedding
        # payload: JSON metadata — filterable and returnable in search
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                **chunk["payload"],
                "candidate_id": candidate_id,
                "source_text": chunk["text"],
            }
        )
        points.append(point)

    # Upsert all points in one batch — much faster than one at a time
    client.upsert(collection_name=COLLECTION_RESUMES, points=points)

    log.info(
        "embedder.complete",
        candidate_id=candidate_id,
        chunks_stored=len(points)
    )

    return len(points)