import uuid
import structlog
from langgraph.graph import StateGraph, END
from litellm import completion

from services.api.config import settings
from services.agents.state import InterviewState
from services.agents.classifier import run_classifier
from services.agents.persona import run_persona_builder
from services.agents.retriever import retrieve_questions
from services.agents.evaluator import evaluate_answer, generate_report

log = structlog.get_logger()


def start_node(state: InterviewState) -> dict:
    print("DEBUG start_node candidate_id:", state.candidate_id)
    print("DEBUG start_node type:", type(state))
    """
    First node — runs once when interview begins.
    1. Classify the candidate (domain, YOE, intent)
    2. Build the interviewer persona
    3. Fetch all questions from RAG upfront
    """
    log.info("graph.start_node", candidate_id=state.candidate_id)

    # Step 1 — classify
    state = run_classifier(state)

    # Step 2 — build persona
    state = run_persona_builder(state)

    # Step 3 — fetch questions from RAG
    questions = retrieve_questions(
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        candidate_skills=state.candidate_skills,
        n=state.total_questions,
    )

    # Extract just the question strings and their criteria
    question_texts = [q["question"] for q in questions]
    question_criteria = [q["good_answer_criteria"] for q in questions]
    # Debug — add this right after question_criteria = [q["good_answer_criteria"] for q in questions]
    print("DEBUG criteria stored:", question_criteria)
    print("DEBUG questions stored:", question_texts)

    # If RAG didn't return enough, pad with a generic fallback
    while len(question_texts) < state.total_questions:
        question_texts.append(
            f"Tell me about a challenging {state.domain} project you worked on."
        )
        question_criteria.append("Specific example, clear role, measurable outcome")

    log.info(
        "graph.start_complete",
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        questions_fetched=len(question_texts),
    )

    # Return only the fields that changed — LangGraph merges this into state
    return {
        "domain": state.domain,
        "yoe_tier": state.yoe_tier,
        "intent": state.intent,
        "total_yoe": state.total_yoe,
        "candidate_name": state.candidate_name,
        "candidate_skills": state.candidate_skills,
        "candidate_experience": state.candidate_experience,
        "system_prompt": state.system_prompt,
        "questions_asked": question_texts,
        "question_criteria": question_criteria,
        "question_number": 0,
        "status": "in_progress",
        "next_action": "ask_question",
    }


def ask_question_node(state: InterviewState) -> dict:
    """
    Picks the next question from the pre-fetched list and presents it.
    Adds it to the conversation history so the LLM has full context.
    """
    q_num = state.question_number
    questions = state.questions_asked

    if q_num >= len(questions):
        # Shouldn't happen but safe fallback
        return {"next_action": "end", "status": "complete"}

    current_q = questions[q_num]

    log.info(
        "graph.ask_question",
        question_number=q_num + 1,
        total=state.total_questions,
    )

    # Add question to conversation history
    new_messages = state.messages + [
        {"role": "assistant", "content": current_q}
    ]

    return {
        "current_question": current_q,
        "messages": new_messages,
        "question_number": q_num + 1,
        "next_action": "wait_for_answer",
    }


def evaluate_node(state: InterviewState) -> dict:
    """
    Scores the candidate's latest answer.
    Sets next_action based on score and question count.
    """
    if not state.answers_given:
        return {"next_action": "ask_question"}

    latest_answer = state.answers_given[-1]
    current_q = state.current_question

    # question_number was already incremented by ask_question_node
    # so index is question_number - 1
    q_index = state.question_number - 1
    criteria_list = state.question_criteria
    criteria = criteria_list[q_index] if criteria_list and q_index < len(criteria_list) else ""

    log.info(
        "graph.evaluate_debug",
        q_index=q_index,
        criteria_found=bool(criteria),
        criteria_preview=criteria[:60] if criteria else "EMPTY",
    )
    print("DEBUG state.question_criteria:", state.question_criteria)
    print("DEBUG q_index:", q_index)
    print("DEBUG criteria:", criteria)

    result = evaluate_answer(
        question=current_q,
        answer=latest_answer,
        good_answer_criteria=criteria,
        domain=state.domain,
        yoe_tier=state.yoe_tier,
    )

    score = result["score"]
    feedback = result["feedback"]
    needs_followup = result["needs_followup"]

    new_scores = state.scores + [score]
    new_feedbacks = state.feedbacks + [feedback]

    new_messages = state.messages + [
        {"role": "assistant", "content": f"Feedback: {feedback}"}
    ]

    log.info(
        "graph.evaluate",
        question_number=state.question_number,
        score=score,
        needs_followup=needs_followup,
    )

    if state.question_number >= state.total_questions:
        next_action = "end"
    elif needs_followup:
        next_action = "followup"
    else:
        next_action = "ask_question"

    return {
        "scores": new_scores,
        "feedbacks": new_feedbacks,
        "messages": new_messages,
        "next_action": next_action,
    }


def followup_node(state: InterviewState) -> dict:
    """
    Generates a follow-up question when the evaluator flags needs_followup.
    Uses the LLM with the persona to generate a contextual follow-up.
    """
    log.info("graph.followup", question_number=state.question_number)

    # Ask the LLM to generate a follow-up based on the conversation so far
    messages = [
        {"role": "system", "content": state.system_prompt},
        *state.messages,
        {
            "role": "user",
            "content": (
                "The candidate's answer was somewhat vague. "
                "Generate ONE short follow-up question to get more specific detail. "
                "Ask only the question, nothing else."
            )
        }
    ]

    response = completion(
        model=settings.extraction_model,
        messages=messages,
        max_tokens=100,
    )

    followup_q = response.choices[0].message.content.strip()

    new_messages = state.messages + [
        {"role": "assistant", "content": followup_q}
    ]

    return {
        "current_question": followup_q,
        "messages": new_messages,
        "next_action": "wait_for_answer",
    }


def end_node(state: InterviewState) -> dict:
    """
    Final node — generates the complete interview report.
    """
    log.info(
        "graph.end",
        candidate=state.candidate_name,
        total_score=sum(state.scores),
    )

    report = generate_report(
        candidate_name=state.candidate_name,
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        questions=state.questions_asked[:len(state.scores)],
        answers=state.answers_given,
        scores=state.scores,
        feedbacks=state.feedbacks,
    )

    return {
        "report": report,
        "status": "complete",
        "next_action": "done",
    }


def route_after_evaluate(state: InterviewState) -> str:
    """
    Conditional edge function — called after evaluate_node.
    Returns the name of the next node to go to.
    This is the brain of the routing logic.
    """
    action = state.next_action

    if action == "end":
        return "end"
    elif action == "followup":
        return "followup"
    else:
        return "ask_question"


def build_interview_graph():
    """
    Assembles all nodes and edges into a runnable LangGraph graph.
    Call this once and reuse the compiled graph.
    """
    # InterviewState is our state schema
    graph = StateGraph(InterviewState)

    # Add all nodes — name maps to function
    graph.add_node("start", start_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("followup", followup_node)
    graph.add_node("end", end_node)

    # Set entry point — first node to run
    graph.set_entry_point("start")

    # Fixed edges — always go from A to B
    graph.add_edge("start", "ask_question")
    graph.add_edge("followup", "evaluate")

    # Conditional edge — after evaluate, call route_after_evaluate
    # to decide whether to go to ask_question, followup, or end
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "ask_question": "ask_question",
            "followup": "followup",
            "end": "end",
        }
    )

    # ask_question pauses and waits — it ends the graph turn here
    # The API will resume it when the candidate sends their answer
    graph.add_edge("ask_question", END)
    graph.add_edge("end", END)

    return graph.compile()


# Single compiled graph instance — import this in interview.py
interview_graph = build_interview_graph()