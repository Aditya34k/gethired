from services.api.schemas.profile import CandidateProfile, Experience


def test_candidate_profile_skills_cleaned():
    """Skills validator removes empty strings and whitespace."""
    profile = CandidateProfile(
        full_name="Test User",
        skills=["Python", " ", "", "FastAPI", "  "],
    )
    assert profile.skills == ["Python", "FastAPI"]


def test_candidate_profile_yoe_calculation():
    """Total YOE is correctly calculated from experience entries."""
    profile = CandidateProfile(
        full_name="Test User",
        experience=[
            Experience(company="A", title="Dev", start_year=2020, end_year=2022),
            Experience(company="B", title="Dev", start_year=2022, end_year=2024),
        ]
    )
    assert profile.total_yoe == 4


def test_experience_null_replacement():
    """Model validator replaces null title and start_year with defaults."""
    exp = Experience(**{
        "company": "TestCo",
        "title": None,
        "start_year": None,
        "description": "Some work"
    })
    assert exp.title == "Unknown"
    assert exp.start_year == 0


def test_candidate_profile_defaults():
    """CandidateProfile has sensible defaults for all optional fields."""
    profile = CandidateProfile(full_name="Jane Doe")
    assert profile.email is None
    assert profile.skills == []
    assert profile.experience == []
    assert profile.total_yoe == 0