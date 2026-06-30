import requests

BASE = "http://localhost:8000/api/v1"

print("=" * 55)
print("GETHIRED AI — INTERVIEW SESSION")
print("=" * 55)

candidate_id = input("\nEnter your candidate_id: ").strip()
n_questions = input("How many questions? (1-5, default 3): ").strip()
n_questions = int(n_questions) if n_questions.isdigit() else 3

# Start interview
start_resp = requests.post(f"{BASE}/interview/start", json={
    "candidate_id": candidate_id,
    "mode": "commercial",
    "total_questions": n_questions,
})

if start_resp.status_code != 200:
    print("ERROR:", start_resp.text)
    exit()

data = start_resp.json()
session_id = data["session_id"]

print(f"\nCandidate  : {data['candidate_name']}")
print(f"Domain     : {data['domain']}")
print(f"YOE tier   : {data['yoe_tier']}")
print(f"Questions  : {data['total_questions']}")
print("-" * 55)

print(f"\nINTERVIEWER: {data['question']}\n")

# Interview loop
q_num = 1
while True:
    answer = input("YOUR ANSWER: ").strip()
    if not answer:
        print("Please provide an answer.")
        continue

    msg_resp = requests.post(f"{BASE}/interview/message", json={
        "session_id": session_id,
        "message": answer,
    })

    if msg_resp.status_code != 200:
        print("ERROR:", msg_resp.text)
        break

    msg_data = msg_resp.json()

    print(f"\nFEEDBACK   : {msg_data['feedback']}")
    print(f"SCORE      : {msg_data['score']}/10")
    print("-" * 55)

    if msg_data["status"] == "complete":
        report = msg_data["report"]
        print("\n" + "=" * 55)
        print("INTERVIEW COMPLETE — FINAL REPORT")
        print("=" * 55)
        print(f"Grade          : {report.get('grade')}")
        print(f"Score          : {report.get('total_score')}/{report.get('max_score')}")
        print(f"Percentage     : {report.get('percentage')}%")
        print(f"Recommendation : {report.get('recommendation')}")
        print(f"\nSummary: {report.get('summary')}")
        print("\nStrengths:")
        for s in report.get("strengths", []):
            print(f"  + {s}")
        print("\nAreas to improve:")
        for a in report.get("areas_to_improve", []):
            print(f"  - {a}")
        break

    print(f"\nINTERVIEWER: {msg_data['question']}\n")
    q_num += 1