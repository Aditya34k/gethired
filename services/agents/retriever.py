import os
import structlog
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from services.api.config import settings

log = structlog.get_logger()

# Same model as the embedder — must match or similarity search breaks
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_knowledge_base_from_yaml(domain: str) -> list[dict]:
    """
    Reads questions from the YAML file for a given domain.
    Returns a list of question dicts.
    """
    # Build path to knowledge base file
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    yaml_path = os.path.join(base_dir, "knowledge_base", f"{domain}.yaml")

    # Fall back to general if domain file doesn't exist
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(base_dir, "knowledge_base", "software_engineering.yaml")
        log.warning("retriever.domain_file_missing", domain=domain, fallback="software_engineering")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("questions", [])


def seed_knowledge_base(domain: str) -> int:
    """
    Embeds all questions from a domain's YAML file and stores in Qdrant.
    Returns number of questions seeded.

    Call this once per domain at setup, or whenever you update the YAML.
    """
    questions = load_knowledge_base_from_yaml(domain)
    if not questions:
        log.warning("retriever.no_questions_found", domain=domain)
        return 0

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    from qdrant_client.models import PointStruct
    import uuid

    points = []
    for q in questions:
        # Embed the question text
        vector = EMBEDDING_MODEL.encode(q["question"]).tolist()

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "domain": domain,
                "difficulty": q.get("difficulty", "mid"),
                "topic": q.get("topic", "general"),
                "question": q["question"],
                "good_answer_criteria": q.get("good_answer_criteria", ""),
                "question_id": q.get("id", ""),
            }
        )
        points.append(point)

    client.upsert(collection_name="knowledge_base", points=points)

    log.info("retriever.seeded", domain=domain, count=len(points))
    return len(points)


def retrieve_questions(
    domain: str,
    yoe_tier: str,
    candidate_skills: list[str],
    n: int = 5,
) -> list[dict]:
    """
    Retrieves the most relevant questions for this candidate from Qdrant.

    HOW RAG WORKS HERE:
    1. We build a query string from the candidate's domain and skills
    2. We embed that query into a vector
    3. We search Qdrant for the most similar question vectors
    4. We filter by domain and difficulty to match the candidate's level
    5. We return the top N questions

    This means candidates with ML skills get ML questions,
    not generic software engineering questions.
    """
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Build a query that represents what we want to find
    # The more specific, the better the retrieval
    query = f"{domain} interview questions for {yoe_tier} candidate skills: {', '.join(candidate_skills[:10])}"
    query_vector = EMBEDDING_MODEL.encode(query).tolist()

    # Map yoe_tier to difficulty
    difficulty_map = {
        "junior": "junior",
        "mid": "mid",
        "senior": "senior",
    }
    difficulty = difficulty_map.get(yoe_tier, "mid")

    # Search with filter — only get questions matching domain AND difficulty
    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="domain",
                    match=MatchValue(value=domain)
                ),
                FieldCondition(
                    key="difficulty",
                    match=MatchValue(value=difficulty)
                ),
            ]
        ),
        limit=n,
        with_payload=True,
    )

    # If not enough domain-specific results, fall back without domain filter
    if len(results) < n:
        log.info("retriever.fallback_search", domain=domain, found=len(results))
        results = client.search(
            collection_name="knowledge_base",
            query_vector=query_vector,
            limit=n,
            with_payload=True,
        )

    questions = []
    for r in results:
        questions.append({
            "question": r.payload.get("question", ""),
            "topic": r.payload.get("topic", ""),
            "difficulty": r.payload.get("difficulty", ""),
            "good_answer_criteria": r.payload.get("good_answer_criteria", ""),
        })

    log.info("retriever.retrieved", domain=domain, count=len(questions))
    return questions