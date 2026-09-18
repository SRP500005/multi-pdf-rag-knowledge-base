from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding_model

# --------------------------------------------------
# 1. Define the folder containing our PDFs
# --------------------------------------------------

DATA_DIR = Path("data")


# --------------------------------------------------
# 2. Find all PDF files
# --------------------------------------------------

pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in the data directory."
    )


print(f"Found {len(pdf_files)} PDF file(s).")

for pdf_file in pdf_files:
    print(f"- {pdf_file.name}")


# --------------------------------------------------
# 3. Load all PDFs
# --------------------------------------------------

all_documents = []

for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    print(f"Loaded {len(documents)} page(s).")

    all_documents.extend(documents)


print(
    f"\nTotal loaded documents/pages: {len(all_documents)}"
)


# --------------------------------------------------
# 4. Create the text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


# --------------------------------------------------
# 5. Split documents into chunks
# --------------------------------------------------

chunks = text_splitter.split_documents(all_documents)

print(f"\nTotal chunks created: {len(chunks)}")


# --------------------------------------------------
# 6. Inspect the first chunk
# --------------------------------------------------

print("\n--- FIRST CHUNK ---")

print(chunks[0].page_content)


print("\n--- FIRST CHUNK METADATA ---")

print(chunks[0].metadata)


# --------------------------------------------------
# 7. Preview first 5 chunks
# --------------------------------------------------

print("\n--- CHUNK PREVIEW ---")

for index, chunk in enumerate(chunks[:5]):

    print(f"\nChunk {index + 1}")

    print(
        f"Source: {chunk.metadata.get('source')}"
    )

    print(
        f"Page: {chunk.metadata.get('page_label')}"
    )

    print(
        f"Length: {len(chunk.page_content)} characters"
    )

    print(chunk.page_content[:200])

    print("-" * 50)
    
    # --------------------------------------------------
# 8. Create embeddings for all chunks
# --------------------------------------------------

embedding_model = get_embedding_model()

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

vectors = embedding_model.embed_documents(chunk_texts)


print("\n--- EMBEDDINGS ---")

print(f"Number of chunks: {len(chunks)}")
print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimensions: {len(vectors[0])}")
print(f"First 5 values: {vectors[0][:5]}")