from services.agents.retriever import seed_knowledge_base, retrieve_questions

# Seed all three domains
print("Seeding knowledge base...")
print("SWE:", seed_knowledge_base("software_engineering"), "questions")
print("DS:", seed_knowledge_base("data_science"), "questions")
print("Finance:", seed_knowledge_base("finance"), "questions")

print()

# Test retrieval for your profile
questions = retrieve_questions(
    domain="data_science",
    yoe_tier="junior",
    candidate_skills=["Python", "LangChain", "RAG", "Vector Databases"],
    n=3
)

print("Retrieved questions:")
for i, q in enumerate(questions, 1):
    print(f"{i}. [{q['topic']}] {q['question']}")
    print(f"   Criteria: {q['good_answer_criteria']}")
    print()