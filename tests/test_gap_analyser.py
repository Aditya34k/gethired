from services.agents.gap_analyser import run_gap_analysis

# A realistic ML Engineer job description
job_description = """
We are looking for a Machine Learning Engineer to join our team.

Requirements:
- Strong Python programming skills
- Experience with PyTorch or TensorFlow
- Knowledge of MLOps practices including model monitoring and deployment
- Experience with cloud platforms (AWS, GCP or Azure)
- Familiarity with Docker and Kubernetes
- Experience with LLMs and prompt engineering
- Knowledge of RAG pipelines and vector databases
- Strong understanding of NLP fundamentals
- Experience with FastAPI or similar frameworks
- Ability to deploy models to production

Nice to have:
- Experience with LangChain or LlamaIndex
- Knowledge of fine-tuning LLMs
- Experience with Airflow or similar pipeline tools
"""

result = run_gap_analysis(
    candidate_id="7e1df2af-83e1-46e5-962f-e023c58f4d66",
    job_description=job_description,
    domain="data_science",
    yoe_tier="junior",
)

print("=" * 50)
print("GAP ANALYSIS RESULT")
print("=" * 50)
print(f"Overall match    : {result['overall_match_pct']}%")
print(f"Semantic score   : {result['semantic_similarity']}")
print()
print("Matching skills:")
for s in result.get("matching_skills", []):
    print(f"  ✓ {s}")
print()
print("Missing skills:")
for s in result.get("missing_skills", []):
    print(f"  ✗ {s}")
print()
print("Priority gaps (address these first):")
for i, g in enumerate(result.get("priority_gaps", []), 1):
    print(f"  {i}. {g}")
print()
print("Summary:")
print(f"  {result.get('summary')}")