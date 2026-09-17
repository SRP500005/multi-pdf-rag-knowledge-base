from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


DATA_DIR = Path("data")


pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in the data directory."
    )


print(f"Found {len(pdf_files)} PDF file(s).")

for pdf_file in pdf_files:
    print(f"- {pdf_file.name}")


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