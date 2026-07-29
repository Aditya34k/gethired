import requests
import time

BASE = "http://localhost:8000/api/v1"
HEADERS = {"X-API-Key": "gethired-secret-2026"}


JOB_DESCRIPTION = """
We are looking for a Machine Learning Engineer to join our team.

Requirements:
- Strong Python programming skills
- Experience with PyTorch or TensorFlow
- Knowledge of MLOps practices including model monitoring and deployment
- Experience with cloud platforms (AWS, GCP or Azure)
- Familiarity with Docker and Kubernetes
- Experience with LLMs and prompt engineering
- Knowledge of RAG pipelines and vector databases
- Experience with FastAPI or similar frameworks
- Ability to deploy models to production
"""

print("=" * 55)
print("GETHIRED AI — PREP SESSION")
print("=" * 55)

# Start prep session
start_resp = requests.post(f"{BASE}/prep/start", json={
    "candidate_id": "7e1df2af-83e1-46e5-962f-e023c58f4d66",
    "job_description": JOB_DESCRIPTION,
    "total_questions": 3,
} , headers=HEADERS)

if start_resp.status_code != 200:
    print("ERROR starting prep:", start_resp.text)
    exit()

data = start_resp.json()
session_id = data["session_id"]

print(f"\nCandidate  : {data['candidate_name']}")
print(f"Domain     : {data['domain']}")
print(f"YOE tier   : {data['yoe_tier']}")

gap = data.get("gap_analysis", {})
if gap:
    print(f"\nJob match  : {gap.get('overall_match_pct')}%")
    print(f"Top gap    : {gap.get('priority_gaps', [''])[0]}")

print("-" * 55)
print(f"\nCOACH: {data['question']}\n")

# Sample answers for automated test
sample_answers = [
    "A RAG pipeline works by first converting documents into vector embeddings and storing them in a vector database like Chroma or Qdrant. When a user asks a question, the query is also embedded and used to retrieve the most similar document chunks. These chunks are then passed as context to the LLM to generate a grounded answer. This prevents hallucination and allows the model to reference real data.",
    "I would monitor data drift by comparing the distribution of incoming features against the training distribution using statistical tests. I would also track model performance metrics like accuracy and latency in production. If drift exceeds a threshold I would trigger retraining. I have used basic logging with AWS CloudWatch for this.",
    "I would choose RAG when the knowledge base is frequently updated and fine-tuning would be too expensive. RAG is also better when you need citations or transparency. Fine-tuning is better when you need the model to adopt a specific tone or style that cannot be achieved through prompting alone.",
]

for i, answer in enumerate(sample_answers):
    print(f"YOU: {answer[:80]}...\n")

    msg_resp = requests.post(f"{BASE}/prep/message", json={
        "session_id": session_id,
        "message": answer,
    } , headers=HEADERS)

    if msg_resp.status_code != 200:
        print("ERROR:", msg_resp.text)
        break

    msg_data = msg_resp.json()

    print(f"SCORE    : {msg_data['score']}/10")
    print(f"COACHING : {msg_data['feedback']}")
    print("-" * 55)

    if msg_data["status"] == "complete":
        print("\n" + "=" * 55)
        print("PREP SESSION COMPLETE")
        print("=" * 55)

        report = msg_data.get("report", {})
        print(f"\nAverage score : {report.get('average_score')}/10")
        print(f"\n{report.get('encouragement')}")

        roadmap = msg_data.get("roadmap", {})
        if roadmap:
            print("\n" + "=" * 55)
            print("YOUR STUDY ROADMAP")
            print("=" * 55)
            print(f"\nStrategy: {roadmap.get('overall_strategy')}\n")
            for week in roadmap.get("weeks", []):
                print(f"WEEK {week['week_number']}: {week['focus']}")
                print(f"  Practice: {week['practice_task']}")
                print()
        break

    if msg_data.get("question"):
        print(f"\nCOACH: {msg_data['question']}\n")

    time.sleep(1)