import chromadb
from ollama import embed

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "insurance_documents"

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(name=COLLECTION_NAME)


def retrieve(question, n_results=2):
    # Embed the question
    # ==================
    response = embed(
        model="nomic-embed-text",
        input=question
    )
    question_embedding = response["embeddings"][0]

    # Find relevant chunk
    # ===================
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2
    )

    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }
