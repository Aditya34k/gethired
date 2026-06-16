from pydantic import BaseModel, EmailStr, field_validator


class Experience(BaseModel):
    """One job entry from a resume."""
    company: str
    title: str
    start_year: int
    end_year: int | None = None   # None means "current job"
    description: str = ""


class Education(BaseModel):
    """One education entry from a resume."""
    institution: str
    degree: str = ""
    year: int | None = None


class CandidateProfile(BaseModel):
    """
    The single source of truth for a parsed candidate.
    Both resume_parser and linkedin_fetcher produce this shape.
    Every downstream agent reads this shape.
    """
    full_name: str
    email: str | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    raw_text: str = ""          # original resume text, kept for debugging
    linkedin_data: dict = {}    # raw linkedin fields merged in later

    @field_validator("skills", mode="before")
    @classmethod
    def clean_skills(cls, v):
        """
        Strip whitespace and remove empty strings from skills list.
        LLMs sometimes return ["Python", " ", ""] — this cleans that up.
        """
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return v

    @property
    def total_yoe(self) -> int:
        """
        Calculate total years of experience from all job entries.
        Used later by the router agent to pick junior/mid/senior tier.
        """
        total = 0
        for exp in self.experience:
            start = exp.start_year
            end = exp.end_year or 2025
            total += max(0, end - start)
        return total


class IngestResponse(BaseModel):
    """What the POST /ingest endpoint returns immediately."""
    candidate_id: str
    status: str
    message: str = ""