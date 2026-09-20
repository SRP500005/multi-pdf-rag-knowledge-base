from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model


# --------------------------------------------------
# 1. Location of our existing ChromaDB
# --------------------------------------------------

CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "rag_documents"


# --------------------------------------------------
# 2. Load the existing vector database
# --------------------------------------------------

def load_vector_store():
    """
    Load the existing persistent ChromaDB database.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )

    return vector_store


# --------------------------------------------------
# 3. Search the database
# --------------------------------------------------

def retrieve_documents(question, k=3):
    """
    Retrieve the most relevant document chunks
    for a user's question.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=question,
        k=k,
    )

    return results


# --------------------------------------------------
# 4. Test semantic retrieval
# --------------------------------------------------

if __name__ == "__main__":

    question = "How frequently does Cedar Lab check its equipment?"

    print(f"\nQuestion: {question}")

    results = retrieve_documents(
        question=question,
        k=3,
    )

    print(f"\nRetrieved {len(results)} result(s).")

    for index, document in enumerate(results):

        print("\n" + "=" * 60)

        print(f"RESULT {index + 1}")

        print(
            f"Source: {document.metadata.get('source')}"
        )

        print(
            f"Page: {document.metadata.get('page_label')}"
        )

        print("\nContent:")

        print(document.page_content)

    print("\n" + "=" * 60)