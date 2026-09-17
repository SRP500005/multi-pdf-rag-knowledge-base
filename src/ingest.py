from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found inside the data folder."
    )


print(f"Found {len(pdf_files)} PDF files.")

for pdf_file in pdf_files:
    print("-", pdf_file.name)