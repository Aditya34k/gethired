import json

import fitz  # PyMuPDF — imported as fitz for historical reasons
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion

from services.api.config import settings
from services.api.schemas.profile import CandidateProfile, Experience, Education

log = structlog.get_logger()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Opens a PDF from raw bytes and extracts all text page by page.

    WHY BYTES? The API will receive the file as bytes over HTTP.
    We never write it to disk — cleaner and faster.

    fitz.open() can open from bytes directly using stream= parameter.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page in doc:
        text = page.get_text()   # returns plain text string for that page
        pages.append(text)

    doc.close()

    # Join pages with a separator so the LLM understands page structure
    return "\n---PAGE BREAK---\n".join(pages)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8)
)
def extract_profile_with_llm(raw_text: str) -> dict:
    """
    Sends resume text to an LLM and gets back structured JSON.

    @retry means: if this fails, wait 1s and try again.
    If it fails again, wait 2s. Then 4s. Then give up.
    This handles temporary API errors gracefully.

    WHY TRUNCATE TO 6000 CHARS?
    LLMs have context limits. Most resumes are under 6000 characters
    of actual content. Truncating avoids hitting token limits.
    """
    system_prompt =system_prompt = """You are a resume parser. Extract structured data from the resume text.
Return ONLY a valid JSON object with exactly this structure — no explanation, no markdown:
{
  "full_name": "string",
  "email": "string or null",
  "skills": ["individual skill name", "another skill"],
  "experience": [
    {
      "company": "company name only, without the job title",
      "title": "job title only",
      "start_year": 2024,
      "end_year": 2024,
      "description": "brief summary of what they did"
    }
  ],
  "education": [
    {
      "institution": "college or university name",
      "degree": "degree name",
      "year": 2024
    }
  ]
}

IMPORTANT RULES:
- company should be just the company name e.g. "Authnull" not "Authnull (Technical Trainee-AI/ML)"
- title should be just the role e.g. "Technical Trainee - AI/ML"
- start_year and end_year must be integers (the year number only e.g. 2024) — never null
- if end_year is "present" or "current" use null
- if you cannot find a year, use 0 as the integer, never use null for start_year
- skills must be individual technology or skill names, not full sentences
- Return ONLY the JSON object, nothing else"""

    response = completion(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this resume:\n\n{raw_text[:6000]}"}
        ],
    )

    raw_json = response.choices[0].message.content.strip()

    # Sometimes LLMs wrap JSON in markdown code blocks like ```json ... ```
    # This strips that wrapping if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    return json.loads(raw_json)


def parse_resume(pdf_bytes: bytes) -> CandidateProfile:
    """
    Main public function — this is what the rest of the app calls.

    Takes raw PDF bytes, returns a validated CandidateProfile.
    If anything goes wrong, raises a clear exception.
    """
    log.info("resume_parser.start", size=len(pdf_bytes))

    # Step 1: PDF → raw text
    raw_text = extract_text_from_pdf(pdf_bytes)
    log.info("resume_parser.text_extracted", chars=len(raw_text))

    # Step 2: raw text → structured dict via LLM
    extracted = extract_profile_with_llm(raw_text)
    log.info("resume_parser.llm_done", fields=list(extracted.keys()))

    # Step 3: validate the dict against our schema
    # Pydantic raises ValidationError here if the shape is wrong
    profile = CandidateProfile(
        full_name=extracted.get("full_name", "Unknown"),
        email=extracted.get("email"),
        skills=extracted.get("skills", []),
        experience=[
            Experience(**exp)
            for exp in extracted.get("experience", [])
        ],
        education=[
            Education(**edu)
            for edu in extracted.get("education", [])
        ],
        raw_text=raw_text,
    )

    log.info(
        "resume_parser.complete",
        name=profile.full_name,
        skills=len(profile.skills),
        experience=len(profile.experience),
        yoe=profile.total_yoe,
    )

    return profile