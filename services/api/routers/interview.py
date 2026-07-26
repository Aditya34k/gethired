import uuid
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agents.graph import interview_graph
from services.agents.graph import (
    evaluate_node, route_after_evaluate,
    ask_question_node, followup_node, end_node
)
from services.agents.state import InterviewState
from services.api.db import save_session, load_session

log = structlog.get_logger()
router = APIRouter()


class StartRequest(BaseModel):
    candidate_id: str
    mode: str = "commercial"
    total_questions: int = 3


class StartResponse(BaseModel):
    session_id: str
    question: str
    question_number: int
    total_questions: int
    domain: str
    yoe_tier: str
    candidate_name: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    status: str
    question: str = ""
    question_number: int = 0
    feedback: str = ""
    score: int = 0
    report: dict = {}


@router.post("/interview/start", response_model=StartResponse)
async def start_interview(req: StartRequest):
    session_id = str(uuid.uuid4())
    log.info("interview.start", candidate_id=req.candidate_id, session_id=session_id)

    initial_state = InterviewState(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode=req.mode,
        total_questions=req.total_questions,
    )

    try:
        result = interview_graph.invoke(initial_state)
        state = InterviewState(**result)
    except Exception as e:
        log.error("interview.start_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Interview start failed: {str(e)}")

    # Save to Supabase instead of in-memory dict
    save_session(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode=req.mode,
        status=state.status,
        data=state.model_dump(),
    )

    log.info(
        "interview.started",
        session_id=session_id,
        domain=state.domain,
        question=state.current_question[:50],
    )

    return StartResponse(
        session_id=session_id,
        question=state.current_question,
        question_number=state.question_number,
        total_questions=state.total_questions,
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        candidate_name=state.candidate_name,
    )


@router.post("/interview/message", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    # Load from Supabase instead of in-memory dict
    row = load_session(req.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    state = InterviewState(**row["data"])

    if state.status == "complete":
        raise HTTPException(status_code=400, detail="Interview already complete")

    log.info(
        "interview.message",
        session_id=req.session_id,
        question_number=state.question_number,
        answer_length=len(req.message),
    )

    updated_state = state.model_copy(update={
        "answers_given": state.answers_given + [req.message],
        "messages": state.messages + [{"role": "user", "content": req.message}],
    })

    eval_result = evaluate_node(updated_state)
    new_state = updated_state.model_copy(update=eval_result)

    next_node = route_after_evaluate(new_state)
    log.info("interview.routing", next_node=next_node)

    if next_node == "end":
        end_result = end_node(new_state)
        new_state = new_state.model_copy(update=end_result)
    elif next_node == "followup":
        followup_result = followup_node(new_state)
        new_state = new_state.model_copy(update=followup_result)
        ask_result = ask_question_node(new_state)
        new_state = new_state.model_copy(update=ask_result)
    else:
        ask_result = ask_question_node(new_state)
        new_state = new_state.model_copy(update=ask_result)

    # Save updated session to Supabase
    save_session(
        session_id=req.session_id,
        candidate_id=state.candidate_id,
        mode=state.mode,
        status=new_state.status,
        data=new_state.model_dump(),
    )

    latest_score = new_state.scores[-1] if new_state.scores else 0
    latest_feedback = new_state.feedbacks[-1] if new_state.feedbacks else ""

    log.info(
        "interview.message_complete",
        session_id=req.session_id,
        score=latest_score,
        status=new_state.status,
    )

    if new_state.status == "complete":
        return MessageResponse(
            status="complete",
            feedback=latest_feedback,
            score=latest_score,
            report=new_state.report,
        )

    return MessageResponse(
        status="in_progress",
        question=new_state.current_question,
        question_number=new_state.question_number,
        feedback=latest_feedback,
        score=latest_score,
    )


@router.get("/interview/session/{session_id}")
async def get_session(session_id: str):
    row = load_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row