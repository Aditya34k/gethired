import json
import structlog
from litellm import completion
from services.api.config import settings

log = structlog.get_logger()


def evaluate_answer(
    question: str,
    answer: str,
    good_answer_criteria: str,
    domain: str,
    yoe_tier: str,
) -> dict:
    """
    Scores a candidate's answer from 0-10.
    Returns score, feedback, and whether to ask a follow-up.

    WHY A SEPARATE EVALUATOR?
    Keeping evaluation separate from the interviewer agent means
    we can swap or improve the scoring logic without touching
    the conversation flow. It also means scores are consistent
    across all domains since they use the same rubric.
    """
    system_prompt = """You are an objective interview answer evaluator.
Score the candidate's answer and provide brief feedback.

Return ONLY a valid JSON object with exactly this structure:
{
  "score": 7,
  "feedback": "one or two sentences of specific feedback",
  "needs_followup": false,
  "followup_reason": "only if needs_followup is true, explain why"
}

Scoring rubric:
- 9-10: Exceptional. Specific, deep, demonstrates mastery, clear impact
- 7-8:  Strong. Good depth, mostly specific, minor gaps
- 5-6:  Adequate. Correct direction but lacks depth or specifics
- 3-4:  Weak. Vague, incorrect in places, or missing key points
- 0-2:  Poor. Wrong, completely vague, or no real answer given

Set needs_followup to true if:
- The answer is vague and a follow-up would get more useful signal
- The candidate mentioned something interesting worth exploring
- The score is between 4-6 and more depth would change the assessment

Return ONLY the JSON, nothing else."""

    user_message = f"""Question: {question}

Good answer should cover: {good_answer_criteria}

Candidate's answer: {answer}

Candidate level: {yoe_tier} in {domain}

Evaluate this answer."""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    # Validate score is in range
    result["score"] = max(0, min(10, int(result.get("score", 5))))
    result["feedback"] = result.get("feedback", "Thank you for your answer.")
    result["needs_followup"] = bool(result.get("needs_followup", False))

    log.info(
        "evaluator.scored",
        score=result["score"],
        needs_followup=result["needs_followup"],
    )

    return result


def generate_report(
    candidate_name: str,
    domain: str,
    yoe_tier: str,
    questions: list[str],
    answers: list[str],
    scores: list[int],
    feedbacks: list[str],
) -> dict:
    """
    Generates the final interview report after all questions are done.
    Called by the last node in the LangGraph graph.
    """
    total_score = sum(scores)
    max_score = len(scores) * 10
    percentage = (total_score / max_score * 100) if max_score > 0 else 0

    # Grade based on percentage
    if percentage >= 85:
        grade = "A"
        recommendation = "Strong hire — recommend proceeding to next round"
    elif percentage >= 70:
        grade = "B"
        recommendation = "Leaning hire — some gaps but overall solid"
    elif percentage >= 55:
        grade = "C"
        recommendation = "Borderline — recommend a second opinion"
    else:
        grade = "D"
        recommendation = "Not recommended at this time"

    # Build Q&A summary for the LLM
    qa_summary = ""
    for i, (q, a, s, f) in enumerate(zip(questions, answers, scores, feedbacks), 1):
        qa_summary += f"\nQ{i}: {q}\nAnswer: {a}\nScore: {s}/10\nFeedback: {f}\n"

    system_prompt = """You are writing a professional interview assessment report.
Based on the interview transcript, identify 2-3 specific strengths and 2-3 specific areas to improve.
Be specific — reference actual answers, not generic statements.

Return ONLY a valid JSON object:
{
  "strengths": ["specific strength 1", "specific strength 2"],
  "areas_to_improve": ["specific area 1", "specific area 2"],
  "summary": "2-3 sentence overall assessment"
}"""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Interview transcript for {candidate_name}:\n{qa_summary}"}
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    llm_analysis = json.loads(raw)

    report = {
        "candidate_name": candidate_name,
        "domain": domain,
        "yoe_tier": yoe_tier,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": round(percentage, 1),
        "grade": grade,
        "recommendation": recommendation,
        "strengths": llm_analysis.get("strengths", []),
        "areas_to_improve": llm_analysis.get("areas_to_improve", []),
        "summary": llm_analysis.get("summary", ""),
        "question_breakdown": [
            {
                "question": q,
                "score": s,
                "feedback": f,
            }
            for q, s, f in zip(questions, scores, feedbacks)
        ],
    }

    log.info(
        "evaluator.report_generated",
        candidate=candidate_name,
        grade=grade,
        score=f"{total_score}/{max_score}",
    )

    return report