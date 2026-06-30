import uuid
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agents.graph import interview_graph
from services.agents.state import InterviewState
from services.agents.graph import evaluate_node, route_after_evaluate, ask_question_node, followup_node, end_node

log = structlog.get_logger()
router = APIRouter()

# In-memory session store — holds all active interview states
# Key: session_id, Value: InterviewState dict
sessions: dict = {}


# --- Request / Response schemas ---

class StartRequest(BaseModel):
    candidate_id: str
    mode: str = "commercial"
    total_questions: int = 3    # default 3 for testing, 5 for production


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
    status: str                 # "in_progress" or "complete"
    question: str = ""          # next question (if in_progress)
    question_number: int = 0
    feedback: str = ""          # feedback on previous answer
    score: int = 0              # score for previous answer
    report: dict = {}           # final report (if complete)


# --- Endpoints ---

@router.post("/interview/start", response_model=StartResponse)
async def start_interview(req: StartRequest):
    """
    Starts a new interview session.
    Runs the graph from start → ask_question and returns Q1.
    """
    session_id = str(uuid.uuid4())

    log.info("interview.start", candidate_id=req.candidate_id, session_id=session_id)

    # Create initial state
    initial_state = InterviewState(
        session_id=session_id,
        candidate_id=req.candidate_id,
        mode=req.mode,
        total_questions=req.total_questions,
    )

    try:
        # Run the graph — it will stop at ask_question (which ends at END)
        # and return the state after Q1 is set
        result = interview_graph.invoke(initial_state)

        # LangGraph returns a dict — convert back to InterviewState
        state = InterviewState(**result)
        sessions[session_id] = state.model_dump()

    except Exception as e:
        log.error("interview.start_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Interview start failed: {str(e)}")

    # Save session state
    sessions[session_id] = state.model_dump()

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
    session_data = sessions.get(req.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    state = InterviewState(**session_data)

    if state.status == "complete":
        raise HTTPException(status_code=400, detail="Interview already complete")

    log.info(
        "interview.message",
        session_id=req.session_id,
        question_number=state.question_number,
        answer_length=len(req.message),
    )

    # Add candidate answer to state
    updated_state = state.model_copy(update={
        "answers_given": state.answers_given + [req.message],
        "messages": state.messages + [{"role": "user", "content": req.message}],
    })

    # Import node functions
    from services.agents.graph import (
        evaluate_node, route_after_evaluate,
        ask_question_node, followup_node, end_node
    )

    # Run evaluate
    eval_result = evaluate_node(updated_state)
    new_state = updated_state.model_copy(update=eval_result)

    # Route to next node
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

    # Save session
    sessions[req.session_id] = new_state.model_dump()

    # Get latest score and feedback
    latest_score = new_state.scores[-1] if new_state.scores else 0
    latest_feedback = new_state.feedbacks[-1] if new_state.feedbacks else ""

    log.info(
        "interview.message_complete",
        session_id=req.session_id,
        score=latest_score,
        status=new_state.status,
        next_question=new_state.current_question[:50] if new_state.current_question else "",
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
    """Debug endpoint — returns full session state."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session