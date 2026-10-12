from pathlib import Path

from langchain_chroma import Chroma
from functools import lru_cache
from embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"

@lru_cache(maxsize=1)
def load_vector_store():
    """Load the existing ChromaDB collection."""

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB was not found at {CHROMA_DIR}. "
            "Run python src/ingest.py first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_model(),
    )


def retrieve_documents_with_scores(question, k=3):
    """Retrieve the closest document chunks and their distance scores."""

    if not question.strip():
        raise ValueError("The question cannot be empty.")

    vector_store = load_vector_store()

    return vector_store.similarity_search_with_score(
        query=question.strip(),
        k=k,
    )


def retrieve_relevant_documents(question, k=3, max_distance=1.05):
    """Return retrieved chunks that satisfy the distance threshold."""

    results = retrieve_documents_with_scores(question, k)

    relevant_results = []

    for document, distance in results:
        if distance <= max_distance:
            relevant_results.append(
                {
                    "document": document,
                    "distance": float(distance),
                }
            )

    return relevant_results


def print_retrieval_results(question, results):
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    if not results:
        print("No sufficiently relevant evidence was found.")
        return

    print(f"Accepted chunks: {len(results)}")

    for index, result in enumerate(results, start=1):
        document = result["document"]
        distance = result["distance"]

        source = Path(
            document.metadata.get("source", "Unknown")
        ).name
        page = document.metadata.get(
            "page_label",
            document.metadata.get("page", "Unknown"),
        )

        print("\n" + "-" * 70)
        print(f"Result {index}")
        print(f"Distance: {distance:.4f}")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print("\nContent:")
        print(document.page_content[:300].strip())


if __name__ == "__main__":
    test_questions = [
        "What is RAG and why is it useful?",
        "Why is chunk overlap important?",
        "How should a RAG system be evaluated?",
        "What is the capital of Australia?",
    ]

    for test_question in test_questions:
        retrieved_results = retrieve_relevant_documents(
            question=test_question,
            k=3,
            max_distance=1.05,
        )

        print_retrieval_results(
            question=test_question,
            results=retrieved_results,
        )

    print("\nRetrieval testing completed.")