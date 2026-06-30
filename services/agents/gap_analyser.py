import json
import structlog
from litellm import completion
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from services.api.config import settings

log = structlog.get_logger()

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_candidate_skills(candidate_id: str) -> list[str]:
    """
    Fetches the candidate's skills from Qdrant.
    Same pattern as classifier.py — filter by candidate_id,
    find the skills chunk, return the skills list.
    """
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    results, _ = client.scroll(
        collection_name="resume_chunks",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="candidate_id",
                    match=MatchValue(value=candidate_id)
                ),
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="skills")
                ),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        log.warning("gap_analyser.no_skills_found", candidate_id=candidate_id)
        return []

    skills = results[0].payload.get("skills", [])
    log.info("gap_analyser.skills_fetched", count=len(skills))
    return skills


def semantic_gap_score(
    candidate_skills: list[str],
    job_description: str,
) -> float:
    """
    Measures how well the candidate's skills match the job description
    using cosine similarity between embeddings.

    Returns a score from 0.0 (no match) to 1.0 (perfect match).

    WHY THIS WORKS:
    We embed the candidate's skills as one text block and the job
    description as another. If they're semantically similar — same
    domain, same technologies — the vectors will point in similar
    directions and cosine similarity will be high.
    """
    skills_text = "Skills: " + ", ".join(candidate_skills)

    skills_vector = EMBEDDING_MODEL.encode(skills_text)
    job_vector = EMBEDDING_MODEL.encode(job_description[:2000])

    # Cosine similarity — dot product of normalised vectors
    import numpy as np
    skills_norm = skills_vector / np.linalg.norm(skills_vector)
    job_norm = job_vector / np.linalg.norm(job_vector)
    similarity = float(np.dot(skills_norm, job_norm))

    log.info("gap_analyser.semantic_score", similarity=round(similarity, 3))
    return similarity


def analyse_gaps_with_llm(
    candidate_skills: list[str],
    job_description: str,
    domain: str,
    yoe_tier: str,
) -> dict:
    """
    Uses the LLM to reason about skill gaps between the candidate
    and the job description.

    Returns a structured dict with:
    - matching_skills: skills the candidate has that the job wants
    - missing_skills: skills the job wants that the candidate lacks
    - transferable_skills: candidate skills that partially cover gaps
    - priority_gaps: the 3 most important gaps to address first
    - overall_match_pct: rough percentage match
    """
    system_prompt = """You are a technical recruiter analysing a candidate's fit for a job.
Compare the candidate's skills against the job description.

Return ONLY a valid JSON object with exactly this structure:
{
  "matching_skills": ["skills the candidate has that the job requires"],
  "missing_skills": ["skills the job requires that the candidate lacks"],
  "transferable_skills": ["candidate skills that partially cover gaps"],
  "priority_gaps": ["top 3 most important gaps to address, ordered by importance"],
  "overall_match_pct": 65,
  "summary": "2 sentence assessment of fit"
}

Be specific — name actual technologies and skills, not vague categories.
Return ONLY the JSON, nothing else."""

    user_message = f"""Candidate skills: {', '.join(candidate_skills)}
Candidate level: {yoe_tier} in {domain}

Job description:
{job_description[:3000]}

Analyse the gap."""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    log.info(
        "gap_analyser.llm_complete",
        matching=len(result.get("matching_skills", [])),
        missing=len(result.get("missing_skills", [])),
        match_pct=result.get("overall_match_pct"),
    )

    return result


def run_gap_analysis(
    candidate_id: str,
    job_description: str,
    domain: str,
    yoe_tier: str,
) -> dict:
    """
    Main public function — runs the full gap analysis.
    Combines semantic scoring and LLM reasoning.
    """
    log.info("gap_analyser.start", candidate_id=candidate_id)

    # Get candidate skills from Qdrant
    skills = get_candidate_skills(candidate_id)

    if not skills:
        return {
            "error": "No skills found for candidate",
            "matching_skills": [],
            "missing_skills": [],
            "priority_gaps": [],
            "overall_match_pct": 0,
        }

    # Semantic similarity score
    similarity = semantic_gap_score(skills, job_description)

    # LLM gap analysis
    llm_result = analyse_gaps_with_llm(
        candidate_skills=skills,
        job_description=job_description,
        domain=domain,
        yoe_tier=yoe_tier,
    )

    # Combine both into final result
    result = {
        **llm_result,
        "semantic_similarity": round(similarity, 3),
        "candidate_skills": skills,
    }

    log.info(
        "gap_analyser.complete",
        match_pct=result.get("overall_match_pct"),
        priority_gaps=result.get("priority_gaps"),
    )

    return result