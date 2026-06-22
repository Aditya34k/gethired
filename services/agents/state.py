from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class InterviewState(BaseModel):
    """
    The complete state of one interview session.
    Every node in the LangGraph graph reads from and writes to this.

    WHY PYDANTIC HERE?
    Pydantic validates the state at every step. If a node accidentally
    sets score to a string instead of an int, it fails immediately
    with a clear error rather than silently corrupting the session.
    """

    # --- Identity ---
    session_id: str = ""
    candidate_id: str = ""
    mode: str = "commercial"      # "commercial" or "prep"

    # --- Candidate profile (loaded from Qdrant) ---
    candidate_name: str = ""
    candidate_skills: list[str] = []
    candidate_experience: list[dict] = []
    raw_profile_text: str = ""

    # --- Classification (set by classifier node) ---
    domain: str = ""              # e.g. "software_engineering", "finance"
    yoe_tier: str = ""            # "junior", "mid", "senior"
    intent: str = ""              # "new_job", "career_switch", "promotion"
    total_yoe: int = 0

    # --- Persona (set by persona builder node) ---
    system_prompt: str = ""       # the composed interviewer persona

    # --- Interview progress ---
    questions_asked: list[str] = []     # all questions asked so far
    answers_given: list[str] = []       # all answers given so far
    scores: list[int] = []             # score per answer (0-10)
    feedbacks: list[str] = []          # feedback per answer
    current_question: str = ""         # the current question being asked
    question_number: int = 0           # which question we're on
    total_questions: int = 5           # how many questions in total

    # --- Conversation history ---
    # This is the list of messages sent to the LLM each turn.
    # It grows with every question and answer.
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages: list[dict] = []

    # --- Final report (set by the last node) ---
    report: dict = {}

    # --- Graph control ---
    status: str = "not_started"   # "not_started" | "in_progress" | "complete"
    next_action: str = ""         # tells the graph which node to go to next

    class Config:
        arbitrary_types_allowed = True