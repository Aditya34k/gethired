import json
import structlog
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from services.api.config import settings
from services.agents.state import InterviewState

log = structlog.get_logger()

# These are the valid values for each classification dimension.
# The LLM must pick from these — no free-form answers.
VALID_DOMAINS = [
    "software_engineering",
    "data_science",
    "finance",
    "marketing",
    "product_management",
    "general",
]

VALID_YOE_TIERS = ["junior", "mid", "senior"]

VALID_INTENTS = ["new_job", "career_switch", "promotion", "lateral_move"]


def fetch_profile_from_qdrant(candidate_id: str) -> dict:
    """
    Pulls all stored chunks for a candidate from Qdrant.
    Returns a dict with their skills, experience, and name.

    We filter by candidate_id in the payload — this is why we stored
    candidate_id as a payload field in the embedder.
    """
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Scroll retrieves all points matching a filter.
    # Unlike search (which finds similar vectors), scroll finds exact matches.
    results, _ = client.scroll(
        collection_name="resume_chunks",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="candidate_id",
                    match=MatchValue(value=candidate_id)
                )
            ]
        ),
        limit=50,           # max chunks to retrieve
        with_payload=True,  # include the metadata we stored
        with_vectors=False, # we don't need the vectors themselves here
    )

    if not results:
        log.warning("classifier.no_profile_found", candidate_id=candidate_id)
        return {}

    # Reassemble profile from chunks
    profile = {
        "candidate_name": "",
        "skills": [],
        "experience": [],
        "education": [],
    }

    for point in results:
        payload = point.payload
        chunk_type = payload.get("chunk_type", "")

        if not profile["candidate_name"] and payload.get("candidate_name"):
            profile["candidate_name"] = payload["candidate_name"]

        if chunk_type == "skills":
            profile["skills"] = payload.get("skills", [])

        elif chunk_type == "experience":
            profile["experience"].append({
                "company": payload.get("company", ""),
                "title": payload.get("title", ""),
                "start_year": payload.get("start_year", 0),
                "end_year": payload.get("end_year", None),
            })

        elif chunk_type == "education":
            profile["education"].append(payload.get("source_text", ""))

    log.info(
        "classifier.profile_fetched",
        candidate_id=candidate_id,
        skills=len(profile["skills"]),
        experience=len(profile["experience"]),
    )

    return profile


def classify_candidate(profile: dict) -> dict:
    """
    Sends the profile to an LLM and gets back domain, yoe_tier, intent.
    Returns a dict with those three fields.
    """
    # Calculate total YOE from experience entries
    total_yoe = 0
    for exp in profile.get("experience", []):
        start = exp.get("start_year", 0)
        end = exp.get("end_year") or 2025
        if start > 0:
            total_yoe += max(0, end - start)

    profile_summary = f"""
Candidate: {profile.get('candidate_name', 'Unknown')}
Skills: {', '.join(profile.get('skills', [])[:20])}
Experience: {json.dumps(profile.get('experience', []), indent=2)}
Education: {'; '.join(profile.get('education', []))}
Total YOE calculated: {total_yoe}
"""

    system_prompt = f"""You are a candidate classifier for an AI interview system.
Analyse the candidate profile and classify them.

Return ONLY a valid JSON object with exactly this structure:
{{
  "domain": "one of: {', '.join(VALID_DOMAINS)}",
  "yoe_tier": "one of: {', '.join(VALID_YOE_TIERS)}",
  "intent": "one of: {', '.join(VALID_INTENTS)}",
  "reasoning": "one sentence explaining your classification"
}}

Domain rules:
- software_engineering: developers, engineers, DevOps, backend, frontend
- data_science: ML engineers, data scientists, AI researchers, analysts
- finance: financial analysts, investment bankers, accountants
- marketing: marketers, growth, brand, content
- product_management: product managers, program managers
- general: if domain is unclear

YOE tier rules:
- junior: 0-2 years
- mid: 3-6 years
- senior: 7+ years

Intent rules:
- new_job: looking for a new role in same field
- career_switch: moving to a different domain
- promotion: seeking a higher level role
- lateral_move: same level, different company

Return ONLY the JSON, no explanation."""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this candidate:\n{profile_summary}"}
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    # Validate — if LLM returns something outside our valid values, use defaults
    if result.get("domain") not in VALID_DOMAINS:
        result["domain"] = "general"
    if result.get("yoe_tier") not in VALID_YOE_TIERS:
        result["yoe_tier"] = "mid"
    if result.get("intent") not in VALID_INTENTS:
        result["intent"] = "new_job"

    result["total_yoe"] = total_yoe

    log.info(
        "classifier.complete",
        domain=result["domain"],
        yoe_tier=result["yoe_tier"],
        intent=result["intent"],
        total_yoe=total_yoe,
    )

    return result


def run_classifier(state: InterviewState) -> InterviewState:
    """
    LangGraph node function — takes state, returns updated state.
    This is what LangGraph calls when it enters the classifier node.
    """
    log.info("classifier.node_start", candidate_id=state.candidate_id)

    # Fetch profile from Qdrant
    profile = fetch_profile_from_qdrant(state.candidate_id)

    if not profile:
        # If no profile found, use safe defaults so interview can still run
        return state.model_copy(update={
            "domain": "general",
            "yoe_tier": "mid",
            "intent": "new_job",
            "total_yoe": 0,
            "candidate_name": "Candidate",
            "next_action": "build_persona",
        })

    # Classify the candidate
    classification = classify_candidate(profile)

    # Return updated state with everything we learned
    return state.model_copy(update={
        "candidate_name": profile.get("candidate_name", "Candidate"),
        "candidate_skills": profile.get("skills", []),
        "candidate_experience": profile.get("experience", []),
        "domain": classification["domain"],
        "yoe_tier": classification["yoe_tier"],
        "intent": classification["intent"],
        "total_yoe": classification["total_yoe"],
        "next_action": "build_persona",
    })