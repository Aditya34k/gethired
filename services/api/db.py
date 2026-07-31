import structlog
from supabase import Client, create_client

from services.api.config import settings

log = structlog.get_logger()

# Single shared Supabase client
# Created once at module level — reused for every request
def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


def save_session(
    session_id: str,
    candidate_id: str,
    mode: str,
    status: str,
    data: dict,
    job_description: str = "",
    gap_analysis: dict = {},
) -> None:
    """
    Saves or updates a session in Supabase.
    Uses upsert — inserts if new, updates if exists.

    WHY UPSERT?
    The first call for a session creates the row.
    Every subsequent call updates it.
    Upsert handles both cases in one operation.
    """
    client = get_supabase()

    client.table("sessions").upsert({
        "id": session_id,
        "candidate_id": candidate_id,
        "mode": mode,
        "status": status,
        "data": data,                    # full InterviewState as JSON
        "job_description": job_description,
        "gap_analysis": gap_analysis,
    }).execute()

    log.info("db.session_saved", session_id=session_id, status=status)


def load_session(session_id: str) -> dict | None:
    """
    Loads a session from Supabase by session_id.
    Returns None if not found.
    """
    client = get_supabase()

    result = client.table("sessions") \
        .select("*") \
        .eq("id", session_id) \
        .execute()

    if not result.data:
        log.warning("db.session_not_found", session_id=session_id)
        return None

    row = result.data[0]
    log.info("db.session_loaded", session_id=session_id, status=row["status"])
    return row


def delete_session(session_id: str) -> None:
    """Deletes a session — useful for cleanup and testing."""
    client = get_supabase()
    client.table("sessions").delete().eq("id", session_id).execute()
    log.info("db.session_deleted", session_id=session_id)