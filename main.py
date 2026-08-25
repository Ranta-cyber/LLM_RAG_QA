# All RAG and LLM parts together
# This is an exercise I made just to learn all process
# In test_rag.py there is QA work, so some tests

# Installed: ollama (download from site)
# Installed packages: chromadb, ollama, pytest, ragas(v 0.3.9),langchain-ollama, numpy(if want to use cosine)
# Downloaded llama3.2 model (>ollama pull llama3.2) (can be run also like >ollama run llama3.2)
# Downloaded embed model (>ollama pull nomic-embed-text)
# Used PyCharm

from src.ingestion import create_vector_store
from src.rag import answer_question

# from evaluation.evaluation_report import evaluation_report

# Create vector local database
# ============================
collection = create_vector_store()

# here just trying tests in the middle
# Automated tests
# evaluation_report = evaluation_report(collection)

# user can ask a question
# ===================
question = input("\nUser: What is the question?")

answer = answer_question(question)

# Answer
# ======
# print(answer)
print("Answer:", answer.get("answer"))
