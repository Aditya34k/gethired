import structlog
from langgraph.graph import StateGraph, END
from litellm import completion

from services.api.config import settings
from services.agents.state import InterviewState
from services.agents.classifier import run_classifier
from services.agents.persona import build_prep_persona
from services.agents.retriever import retrieve_questions
from services.agents.evaluator import evaluate_answer

log = structlog.get_logger()


def prep_start_node(state: InterviewState) -> dict:
    """
    Same as start_node in graph.py, but uses the coaching persona
    instead of the strict interviewer persona.
    """
    log.info("prep_graph.start", candidate_id=state.candidate_id)

    state = run_classifier(state)

    # Use the coaching persona instead of build_persona
    system_prompt = build_prep_persona(
        domain=state.domain,
        yoe_tier=state.yoe_tier,
    )

    questions = retrieve_questions(
        domain=state.domain,
        yoe_tier=state.yoe_tier,
        candidate_skills=state.candidate_skills,
        n=state.total_questions,
    )

    question_texts = [q["question"] for q in questions]
    question_criteria = [q["good_answer_criteria"] for q in questions]

    while len(question_texts) < state.total_questions:
        question_texts.append(
            f"Tell me about a project that demonstrates your {state.domain} skills."
        )
        question_criteria.append("Specific example, clear approach, lessons learned")

    log.info("prep_graph.start_complete", domain=state.domain, questions=len(question_texts))

    return {
        "domain": state.domain,
        "yoe_tier": state.yoe_tier,
        "intent": state.intent,
        "candidate_name": state.candidate_name,
        "candidate_skills": state.candidate_skills,
        "system_prompt": system_prompt,
        "questions_asked": question_texts,
        "question_criteria": question_criteria,
        "question_number": 0,
        "status": "in_progress",
        "mode": "prep",
        "next_action": "ask_question",
    }


def prep_ask_question_node(state: InterviewState) -> dict:
    """Identical pattern to ask_question_node in graph.py."""
    q_num = state.question_number
    questions = state.questions_asked

    if q_num >= len(questions):
        return {"next_action": "end", "status": "complete"}

    current_q = questions[q_num]
    log.info("prep_graph.ask_question", question_number=q_num + 1)

    new_messages = state.messages + [{"role": "assistant", "content": current_q}]

    return {
        "current_question": current_q,
        "messages": new_messages,
        "question_number": q_num + 1,
        "next_action": "wait_for_answer",
    }


def prep_evaluate_node(state: InterviewState) -> dict:
    """
    Like evaluate_node, but generates COACHING feedback instead of
    just a score. This is the key difference from commercial mode.
    """
    if not state.answers_given:
        return {"next_action": "ask_question"}

    latest_answer = state.answers_given[-1]
    current_q = state.current_question

    q_index = state.question_number - 1
    criteria_list = state.question_criteria
    criteria = criteria_list[q_index] if criteria_list and q_index < len(criteria_list) else ""

    # Get the base score from the evaluator
    result = evaluate_answer(
        question=current_q,
        answer=latest_answer,
        good_answer_criteria=criteria,
        domain=state.domain,
        yoe_tier=state.yoe_tier,
    )

    score = result["score"]

    # Generate coaching feedback — different from commercial feedback
    coaching_prompt = f"""The candidate answered: "{latest_answer}"

Question: {current_q}
What a strong answer covers: {criteria}
Score given: {score}/10

Write 2-3 sentences of COACHING feedback. Explain what was good,
what was missing, and give one specific tip for next time.
Be encouraging but specific. Do not just say "good job" — name
exactly what to add or improve."""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": state.system_prompt},
            {"role": "user", "content": coaching_prompt},
        ],
        max_tokens=200,
    )

    coaching_feedback = response.choices[0].message.content.strip()

    new_scores = state.scores + [score]
    new_feedbacks = state.feedbacks + [coaching_feedback]
    new_messages = state.messages + [
        {"role": "assistant", "content": coaching_feedback}
    ]

    log.info("prep_graph.evaluate", question_number=state.question_number, score=score)

    if state.question_number >= state.total_questions:
        next_action = "end"
    else:
        next_action = "ask_question"

    return {
        "scores": new_scores,
        "feedbacks": new_feedbacks,
        "messages": new_messages,
        "next_action": next_action,
    }


def prep_end_node(state: InterviewState) -> dict:
    """
    Final node for prep mode. Unlike commercial mode's strict report,
    this generates an encouraging summary with the roadmap reference.
    """
    log.info("prep_graph.end", candidate=state.candidate_name)

    avg_score = sum(state.scores) / len(state.scores) if state.scores else 0

    report = {
        "candidate_name": state.candidate_name,
        "domain": state.domain,
        "average_score": round(avg_score, 1),
        "total_questions": len(state.scores),
        "feedbacks": state.feedbacks,
        "questions_practiced": state.questions_asked[:len(state.scores)],
        "encouragement": (
            f"Great work completing this practice session, {state.candidate_name}! "
            f"You answered {len(state.scores)} questions with an average score of "
            f"{round(avg_score, 1)}/10. Review the feedback above and keep practicing."
        ),
    }

    return {
        "report": report,
        "status": "complete",
        "next_action": "done",
    }


def build_prep_graph():
    """Assembles the prep mode graph — simpler than commercial, no followup branch."""
    graph = StateGraph(InterviewState)

    graph.add_node("start", prep_start_node)
    graph.add_node("ask_question", prep_ask_question_node)
    graph.add_node("evaluate", prep_evaluate_node)
    graph.add_node("end", prep_end_node)

    graph.set_entry_point("start")
    graph.add_edge("start", "ask_question")

    def route(state: InterviewState) -> str:
        return "end" if state.next_action == "end" else "ask_question"

    graph.add_conditional_edges(
        "evaluate",
        route,
        {"ask_question": "ask_question", "end": "end"}
    )

    graph.add_edge("ask_question", "evaluate")  # makes evaluate reachable
    graph.add_edge("end", END)

    return graph.compile()
# Single compiled graph instance — import this in prep.py
prep_graph = build_prep_graph()