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
from services.api.db import save_session, load_session

log = structlog.get_logger()
router = APIRouter()


class PrepStartRequest(BaseModel):
    candidate_id: str
    job_description: str = ""
    total_questions: int = 3


class PrepStartResponse(BaseModel):
    session_id: str
    question: str
    question_number: int
    total_questions: int
    domain: str
    yoe_tier: str
    candidate_name: str
    gap_analysis: dict = {}


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
    roadmap: dict = {}


@router.post("/prep/start", response_model=PrepStartResponse)
async def start_prep(req: PrepStartRequest):
    session_id = str(uuid.uuid4())
    log.info("prep.start", candidate_id=req.candidate_id, session_id=session_id)

    initial_state = InterviewState(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode="prep",
        total_questions=req.total_questions,
    )

    try:
        result = prep_graph.invoke(initial_state)
        state = InterviewState(**result)
    except Exception as e:
        log.error("prep.start_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prep start failed: {str(e)}")

    gap_analysis = {}
    if req.job_description.strip():
        try:
            gap_analysis = run_gap_analysis(
                candidate_id=req.candidate_id,
                job_description=req.job_description,
                domain=state.domain,
                yoe_tier=state.yoe_tier,
            )
        except Exception as e:
            log.warning("prep.gap_analysis_failed", error=str(e))

    # Save to Supabase
    save_session(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode="prep",
        status=state.status,
        data=state.model_dump(),
        job_description=req.job_description,
        gap_analysis=gap_analysis,
    )

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
    # Load from Supabase
    row = load_session(req.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    state = InterviewState(**row["data"])
    job_description = row.get("job_description", "")
    gap_analysis = row.get("gap_analysis", {})

    if state.status == "complete":
        raise HTTPException(status_code=400, detail="Session already complete")

    log.info("prep.message", session_id=req.session_id, question_number=state.question_number)

    updated_state = state.model_copy(update={
        "answers_given": state.answers_given + [req.message],
        "messages": state.messages + [{"role": "user", "content": req.message}],
    })

    eval_result = prep_evaluate_node(updated_state)
    new_state = updated_state.model_copy(update=eval_result)

    if new_state.next_action == "end":
        end_result = prep_end_node(new_state)
        new_state = new_state.model_copy(update=end_result)
    else:
        ask_result = prep_ask_question_node(new_state)
        new_state = new_state.model_copy(update=ask_result)

    # Save updated session to Supabase
    save_session(
        session_id=req.session_id,
        candidate_id=state.candidate_id,
        mode="prep",
        status=new_state.status,
        data=new_state.model_dump(),
        job_description=job_description,
        gap_analysis=gap_analysis,
    )

    latest_score = new_state.scores[-1] if new_state.scores else 0
    latest_feedback = new_state.feedbacks[-1] if new_state.feedbacks else ""

    if new_state.status == "complete":
        roadmap = {}
        try:
            priority_gaps = gap_analysis.get("priority_gaps", [])
            missing_skills = gap_analysis.get("missing_skills", [])
            if not priority_gaps:
                priority_gaps = [
                    f"Improve answers on: {q[:50]}"
                    for q, s in zip(new_state.questions_asked, new_state.scores)
                    if s < 6
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
    row = load_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row