import uuid
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agents.prep_graph import prep_graph
from services.agents.prep_graph import (
    prep_evaluate_node, prep_ask_question_node, prep_end_node
)
from services.agents.state import InterviewState
from services.agents.gap_analyser import run_gap_analysis
from services.agents.roadmap import generate_study_roadmap
from services.agents.classifier import run_classifier

log = structlog.get_logger()
router = APIRouter()

# In-memory session store for prep sessions
prep_sessions: dict = {}


# --- Request / Response schemas ---

class PrepStartRequest(BaseModel):
    candidate_id: str
    job_description: str = ""    # optional — if provided, gap analysis runs
    total_questions: int = 3


class PrepStartResponse(BaseModel):
    session_id: str
    question: str
    question_number: int
    total_questions: int
    domain: str
    yoe_tier: str
    candidate_name: str
    gap_analysis: dict = {}      # populated if job_description was provided


class PrepMessageRequest(BaseModel):
    session_id: str
    message: str


class PrepMessageResponse(BaseModel):
    status: str
    question: str = ""
    question_number: int = 0
    feedback: str = ""
    score: int = 0
    report: dict = {}
    roadmap: dict = {}           # populated when interview completes


# --- Endpoints ---

@router.post("/prep/start", response_model=PrepStartResponse)
async def start_prep(req: PrepStartRequest):
    """
    Starts a prep session.
    If job_description is provided, runs gap analysis first.
    Then starts the mock interview with coaching persona.
    """
    session_id = str(uuid.uuid4())
    log.info("prep.start", candidate_id=req.candidate_id, session_id=session_id)

    # Create initial state
    initial_state = InterviewState(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode="prep",
        total_questions=req.total_questions,
    )

    try:
        # Run the prep graph — stops at ask_question and returns Q1
        result = prep_graph.invoke(initial_state)
        state = InterviewState(**result)
    except Exception as e:
        log.error("prep.start_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prep start failed: {str(e)}")

    # Run gap analysis if job description was provided
    gap_analysis = {}
    if req.job_description.strip():
        try:
            gap_analysis = run_gap_analysis(
                candidate_id=req.candidate_id,
                job_description=req.job_description,
                domain=state.domain,
                yoe_tier=state.yoe_tier,
            )
            log.info("prep.gap_analysis_complete", match_pct=gap_analysis.get("overall_match_pct"))
        except Exception as e:
            log.warning("prep.gap_analysis_failed", error=str(e))

    # Save session — include job_description and gap_analysis for later
    session_data = state.model_dump()
    session_data["job_description"] = req.job_description
    session_data["gap_analysis"] = gap_analysis
    prep_sessions[session_id] = session_data

    return PrepStartResponse(
        session_id=session_id,
        question=state.current_question,
        question_number=state.question_number,
        total_questions=state.total_questions,
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        candidate_name=state.candidate_name,
        gap_analysis=gap_analysis,
    )


@router.post("/prep/message", response_model=PrepMessageResponse)
async def send_prep_message(req: PrepMessageRequest):
    """
    Sends an answer and advances the prep session.
    Same manual node invocation pattern as interview.py.
    On completion, generates the study roadmap.
    """
    session_data = prep_sessions.get(req.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Extract extra fields before building state
    job_description = session_data.pop("job_description", "")
    gap_analysis = session_data.pop("gap_analysis", {})

    state = InterviewState(**session_data)

    if state.status == "complete":
        raise HTTPException(status_code=400, detail="Session already complete")

    log.info(
        "prep.message",
        session_id=req.session_id,
        question_number=state.question_number,
    )

    # Add answer to state
    updated_state = state.model_copy(update={
        "answers_given": state.answers_given + [req.message],
        "messages": state.messages + [{"role": "user", "content": req.message}],
    })

    # Run evaluate node
    eval_result = prep_evaluate_node(updated_state)
    new_state = updated_state.model_copy(update=eval_result)

    # Route — end or next question
    if new_state.next_action == "end":
        end_result = prep_end_node(new_state)
        new_state = new_state.model_copy(update=end_result)
    else:
        ask_result = prep_ask_question_node(new_state)
        new_state = new_state.model_copy(update=ask_result)

    # Save session
    updated_data = new_state.model_dump()
    updated_data["job_description"] = job_description
    updated_data["gap_analysis"] = gap_analysis
    prep_sessions[req.session_id] = updated_data

    latest_score = new_state.scores[-1] if new_state.scores else 0
    latest_feedback = new_state.feedbacks[-1] if new_state.feedbacks else ""

    log.info(
        "prep.message_complete",
        score=latest_score,
        status=new_state.status,
    )

    if new_state.status == "complete":
        # Generate study roadmap on completion
        roadmap = {}
        try:
            priority_gaps = gap_analysis.get("priority_gaps", [])
            missing_skills = gap_analysis.get("missing_skills", [])

            # If no gap analysis was done, use weak areas from the interview
            if not priority_gaps:
                priority_gaps = [
                    f"Improve answers on {q[:50]}"
                    for q, s in zip(
                        new_state.questions_asked,
                        new_state.scores
                    ) if s < 6
                ]

            roadmap = generate_study_roadmap(
                candidate_name=new_state.candidate_name,
                domain=new_state.domain,
                yoe_tier=new_state.yoe_tier,
                priority_gaps=priority_gaps or ["Practice interview answers"],
                missing_skills=missing_skills or [],
                matching_skills=new_state.candidate_skills,
                timeframe_weeks=4,
            )
        except Exception as e:
            log.warning("prep.roadmap_failed", error=str(e))

        return PrepMessageResponse(
            status="complete",
            feedback=latest_feedback,
            score=latest_score,
            report=new_state.report,
            roadmap=roadmap,
        )

    return PrepMessageResponse(
        status="in_progress",
        question=new_state.current_question,
        question_number=new_state.question_number,
        feedback=latest_feedback,
        score=latest_score,
    )


@router.get("/prep/session/{session_id}")
async def get_prep_session(session_id: str):
    """Debug endpoint — returns full prep session state."""
    session = prep_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session