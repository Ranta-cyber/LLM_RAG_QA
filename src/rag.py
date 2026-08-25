from ollama import chat
from src.retrieval import retrieve


def answer_question(question):
    relevant_chunk = retrieve(question, n_results=2)

    documents = relevant_chunk["documents"]

    # Create context
    # ==============
    context = "\n\n".join(documents)
    # print("Context is:", context)

    # Crate a prompt
    # ==============

    prompt = f"""
    Answer the user's question using ONLY information provided in the context.
    
    If the answer cannot be found in the context, say then is not information available.
    
    Context:
    {context}
    
    Question:
    {question}
    """

    # Ask LLM
    # =======
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.message.content

    # Answer
    # ======
    # print("n\Answer:", answer)
    # print("\nContext", context)

    return {
        "answer": answer,
        "context": context,
        "contexts": documents,
        "distances": relevant_chunk["distances"]
    }
