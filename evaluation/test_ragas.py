from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ContextPrecision
from langchain_ollama import ChatOllama

from src.rag import answer_question

# The question
question = "How much can I get for my lost luggage?"

# RAG
result = answer_question(question)

# print("Result:", result)

# Create Ragas evaluation dataset
sample = {
    "user_input": question,
    "response": result["answer"],
    "retrieved_contexts": result["contexts"],
    "reference": "The maximum compensation for lost baggage is 2000 euros."
}

dataset = EvaluationDataset.from_list([sample])

# print(dataset)

# Ragas uses Ollama evaluator
# temperature means accuracy
# num_ctx means amount of context
# num_predict means response size

evaluator_llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    timeout=300
)

# Faihfullness
# the amount will be between 0 and 1.0 (1.0 means the better ranking)
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    llm=evaluator_llm
)
"""
result = evaluate(
    dataset=dataset,
    metrics=[ContextPrecision()],
    llm=evaluator_llm
)
"""
print(result)
