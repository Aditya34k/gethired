from services.agents.roadmap import generate_study_roadmap

roadmap = generate_study_roadmap(
    candidate_name="Aditya Kumar",
    domain="data_science",
    yoe_tier="junior",
    priority_gaps=[
        "Kubernetes (required for container orchestration)",
        "MLOps practices (critical for efficient model deployment and monitoring)",
        "Prompt engineering (necessary for effective LLM utilization)",
    ],
    missing_skills=[
        "Kubernetes", "MLOps practices", "Prompt engineering",
        "RAG pipelines knowledge", "FastAPI experience",
    ],
    matching_skills=["PyTorch", "TensorFlow", "LangChain", "RAG", "Vector DB"],
    timeframe_weeks=4,
)

print("=" * 50)
print("STUDY ROADMAP")
print("=" * 50)
print(f"\nStrategy: {roadmap.get('overall_strategy')}\n")

for week in roadmap.get("weeks", []):
    print(f"WEEK {week['week_number']}: {week['focus']}")
    print(f"  Topics: {', '.join(week['topics'])}")
    print(f"  Resources: {', '.join(week['resources'])}")
    print(f"  Practice: {week['practice_task']}")
    print()