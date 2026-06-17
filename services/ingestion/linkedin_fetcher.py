import asyncio
import sys
import os
import structlog

# Explicit path fix for Windows — ensures mcp_servers is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp_servers.linkedin.scraper import fetch_linkedin_profile
from services.api.schemas.profile import CandidateProfile

log = structlog.get_logger()


def merge_linkedin_into_profile(
    profile: CandidateProfile,
    linkedin_data: dict
) -> CandidateProfile:
    if not linkedin_data:
        return profile

    existing_lower = {s.lower() for s in profile.skills}
    new_skills = [
        s for s in linkedin_data.get("skills", [])
        if s.lower() not in existing_lower
    ]

    log.info(
        "linkedin_fetcher.merged",
        new_skills_added=len(new_skills),
        total_skills=len(profile.skills) + len(new_skills)
    )

    return profile.model_copy(update={
        "skills": profile.skills + new_skills,
        "linkedin_data": linkedin_data,
    })


async def fetch_and_merge_linkedin(
    profile: CandidateProfile,
    linkedin_url: str
) -> CandidateProfile:
    log.info("linkedin_fetcher.start", url=linkedin_url)

    try:
        linkedin_data = await fetch_linkedin_profile(linkedin_url)
        log.info("linkedin_fetcher.got_data", keys=list(linkedin_data.keys()))
    except Exception as e:
        log.warning("linkedin_fetcher.failed", error=str(e))
        return profile

    if not linkedin_data:
        log.warning("linkedin_fetcher.no_data", url=linkedin_url)
        return profile

    return merge_linkedin_into_profile(profile, linkedin_data)