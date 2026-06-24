from services.agents.evaluator import evaluate_answer

result = evaluate_answer(
    question="Can you explain how a RAG pipeline works and why it is useful?",
    answer="RAG stands for Retrieval Augmented Generation. You embed documents into a vector database, then when a user asks a question you embed the query, find the most similar documents, and pass them as context to the LLM. This grounds the LLM's response in real data instead of hallucinating.",
    good_answer_criteria="Mentions retrieval step, vector embeddings, context injection, grounding LLM responses",
    domain="data_science",
    yoe_tier="junior",
)

print("Score:", result["score"], "/ 10")
print("Feedback:", result["feedback"])
print("Needs follow-up:", result["needs_followup"])