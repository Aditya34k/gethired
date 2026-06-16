async def fetch_linkedin_profile(url: str) -> dict:
    """
    Mock LinkedIn profile fetcher.

    In dev: returns realistic fake data keyed to the URL.
    In prod: swap this function's internals with a real API call.
    Everything else in the system stays the same.

    WHY ASYNC?
    Real network calls should never block the event loop.
    Even though this mock is instant, we write it async now
    so the interface never needs to change when we go real.
    """
    # Extract username from URL for slightly personalised mock data
    # e.g. "https://linkedin.com/in/aditya-kumar" -> "aditya-kumar"
    username = url.rstrip("/").split("/")[-1]

    return {
        "source": "linkedin_mock",
        "url": url,
        "username": username,
        "headline": "AI/ML Engineer | LLMs | RAG | Deep Learning",
        "summary": (
            "Passionate ML engineer with hands-on experience building "
            "LLM-powered applications, RAG pipelines, and deep learning models. "
            "Currently exploring multi-agent systems and vector databases."
        ),
        "skills": [
            "Python", "LangChain", "RAG", "Deep Learning",
            "NLP", "Hugging Face", "Streamlit", "FastAPI",
            "Vector Databases", "Large Language Models"
        ],
        "experience": [
            {
                "company": "Freelance / Projects",
                "title": "ML Engineer",
                "duration": "2023 - Present",
            }
        ],
        "education": [
            {
                "institution": "University",
                "degree": "B.Tech",
                "year": 2024,
            }
        ]
    }