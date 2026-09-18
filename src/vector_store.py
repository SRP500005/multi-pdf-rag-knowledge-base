from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model


# --------------------------------------------------
# 1. Define where ChromaDB will be stored
# --------------------------------------------------

CHROMA_DIR = Path("chroma_db")


# --------------------------------------------------
# 2. Create the vector store
# --------------------------------------------------

def create_vector_store(chunks):
    """
    Create a persistent ChromaDB vector store
    from LangChain Document chunks.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_name="rag_documents",
    )

    return vector_store