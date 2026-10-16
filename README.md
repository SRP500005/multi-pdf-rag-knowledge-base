# Multi-PDF RAG Knowledge Base

A Python project for answering questions using information retrieved from PDF documents.

The project uses Retrieval-Augmented Generation (RAG) to find relevant document passages and provide context to a language model.

## What the project does

- Extracts text from PDF documents.
- Splits text into smaller, overlapping chunks.
- Creates embeddings for document retrieval.
- Retrieves relevant passages for a question.
- Uses Ollama for local answer generation.
- Provides source information to help users check retrieved evidence.
- Includes a Dockerised API and smoke tests.

## How it works

1. Prepare the PDF documents.
2. Extract and split their text into chunks.
3. Generate embeddings and build a searchable index.
4. Retrieve relevant chunks for a user's question.
5. Pass the retrieved context to the language model.
6. Return an answer based on the available evidence.

## Technologies

- Python
- Text embeddings
- Vector similarity search
- Ollama
- Docker

See the dependency file for the libraries and versions used.

## Example

**Question:** Why is chunk overlap useful?

**Example answer:** Chunk overlap helps preserve context near chunk boundaries. Too much overlap can increase duplicated evidence and index size.

The exact answer depends on the indexed documents and model.

## Testing

The Docker smoke test has been run successfully for:

- API and Ollama health checks.
- A supported question returning an answer.
- An unsupported question being rejected.

Once the API and Ollama are running and the document index is ready, run:

    python tests/smoke_test.py

These checks cover selected scenarios. They do not establish accuracy across every possible question.

## Docker

Build the API image from the project root:

    docker build -t multi-pdf-rag-api .

Running the service also requires the correct Ollama connection,
model configuration, and access to the document index.

## Limitations

- Answer quality depends on PDF text extraction and retrieval quality.
- Scanned PDFs may require OCR before processing.
- Relevant information may be missed during retrieval.
- Generated answers can still contain mistakes.
- The unsupported-question check has only been verified on selected examples.

## Planned improvements

- Document complete installation and startup instructions.
- Add a broader evaluation set for retrieval and answer quality.
- Automate suitable tests with GitHub Actions.
- Improve logging and error handling.

## Author

**Shravankumar Prajapati**

Electrical engineering and data science background, based in Berlin, Germany.
