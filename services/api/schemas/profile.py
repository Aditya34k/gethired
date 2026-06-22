from pydantic import BaseModel, field_validator, model_validator


class Experience(BaseModel):
    company: str = "Unknown"
    title: str = "Unknown"
    start_year: int = 0
    end_year: int | None = None
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def replace_nulls(cls, values):
        if isinstance(values, dict):
            if not values.get("title"):
                values["title"] = "Unknown"
            if not values.get("company"):
                values["company"] = "Unknown"
            if not values.get("start_year"):
                values["start_year"] = 0
            if not values.get("description"):
                values["description"] = ""
        return values


class Education(BaseModel):
    institution: str = "Unknown"
    degree: str = ""
    year: int | None = None


class CandidateProfile(BaseModel):
    full_name: str
    email: str | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    raw_text: str = ""
    linkedin_data: dict = {}

    @field_validator("skills", mode="before")
    @classmethod
    def clean_skills(cls, v):
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return v

    @property
    def total_yoe(self) -> int:
        total = 0
        for exp in self.experience:
            start = exp.start_year
            end = exp.end_year or 2025
            if start > 0:
                total += max(0, end - start)
        return total


class IngestRequest(BaseModel):
    linkedin_url: str | None = None


class IngestResponse(BaseModel):
    candidate_id: str
    status: str
    message: str = ""