import json
import structlog
from litellm import completion
from services.api.config import settings

log = structlog.get_logger()


def generate_study_roadmap(
    candidate_name: str,
    domain: str,
    yoe_tier: str,
    priority_gaps: list[str],
    missing_skills: list[str],
    matching_skills: list[str],
    timeframe_weeks: int = 4,
) -> dict:
    """
    Generates a structured study plan to close the candidate's skill gaps.

    WHY STRUCTURE BY WEEK?
    A flat list of "things to learn" is overwhelming. Breaking it into
    weekly phases with a clear focus makes it actionable — the user
    knows exactly what to do this week, not just someday.
    """
    system_prompt = f"""You are a career coach creating a personalised study roadmap.
The candidate has skill gaps for their target role. Create a week-by-week plan.

Return ONLY a valid JSON object with exactly this structure:
{{
  "weeks": [
    {{
      "week_number": 1,
      "focus": "short title for this week's focus",
      "topics": ["specific topic 1", "specific topic 2"],
      "resources": ["specific resource or type of resource to use"],
      "practice_task": "one concrete hands-on task to complete this week"
    }}
  ],
  "overall_strategy": "2-3 sentences explaining the learning order and why"
}}

Rules:
- Create exactly {timeframe_weeks} weeks
- Order topics from foundational to advanced
- Each week should build on the previous one
- practice_task must be a specific, completable project or exercise
- Be specific with resource types (e.g. "official PyTorch docs tutorial on X" not just "read documentation")
- Return ONLY the JSON, nothing else"""

    user_message = f"""Candidate: {candidate_name}
Domain: {domain}
Level: {yoe_tier}

Priority gaps to address: {', '.join(priority_gaps)}
All missing skills: {', '.join(missing_skills)}
Existing strong skills to build on: {', '.join(matching_skills[:10])}

Create a {timeframe_weeks}-week study roadmap."""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Same robust extraction pattern as gap_analyser
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif "{" in part:
                raw = part.strip()
                break

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("roadmap.json_parse_failed", raw_preview=raw[:100])
        result = {
            "weeks": [],
            "overall_strategy": "Could not generate detailed roadmap. Please retry.",
        }

    log.info(
        "roadmap.generated",
        candidate=candidate_name,
        weeks=len(result.get("weeks", [])),
    )

    return result