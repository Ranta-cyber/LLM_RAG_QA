import chromadb
from ollama import embed
from pathlib import Path

DOCUMENT_PATH = Path("data/documents/insurance_policy.txt")

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "insurance_documents"


def load_document():
    return DOCUMENT_PATH.read_text(encoding="utf-8")


# Here creating chunks spliting lines but can do using paragraphs or amount of tokens for example
def create_chunks(document):
    chunks = [
        chunk.strip()
        for chunk in document.split("\n\n")
        if chunk.strip()
    ]
    return chunks


def create_vector_store():
    document = load_document()

    chunks = create_chunks(document)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Embedding chunks
    # ================
    responses = embed(
        model="nomic-embed-text",
        input=chunks
    )
    embeddings_responses = responses["embeddings"]

    # Cleaning data
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    # Adding vectors to database
    # ==========================
    chunk_id_list = []
    for i, chunk in enumerate(chunks):
        chunk_id_list.append("chunk_" + str(i))

    # print(chunk_id_list)
    collection.add(
        ids=chunk_id_list,
        documents=chunks,
        embeddings=embeddings_responses
    )
    # print(f"Added {len(chunks)} chunks to vector database.")


if __name__ == "__main__":
    create_vector_store()
