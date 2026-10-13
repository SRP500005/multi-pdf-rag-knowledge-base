from pathlib import Path

import requests

from retriever import retrieve_relevant_documents


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
MAX_CHUNK_CHARACTERS = 800

NOT_FOUND_MESSAGE = (
    "I could not find enough evidence in the documents."
)


def get_source(document):
    source = document.metadata.get("source", "Unknown")
    return Path(source).name


def get_page(document):
    return document.metadata.get(
        "page_label",
        document.metadata.get("page", "Unknown"),
    )


def build_context(retrieval_results):
    passages = []

    for index, result in enumerate(
        retrieval_results,
        start=1,
    ):
        document = result["document"]
        source = get_source(document)
        page = get_page(document)

        text = document.page_content.strip()
        text = text[:MAX_CHUNK_CHARACTERS]

        passages.append(
            f"[Passage {index} | {source} | page {page}]\n"
            f"{text}"
        )

    return "\n\n".join(passages)


def build_prompt(question, context):
    return f"""
You are answering questions about a collection of documents.

Use only the passages provided below.

Rules:
1. Give a direct answer in complete sentences.
2. Do not use outside knowledge.
3. Do not invent missing information.
4. If the passages do not support the answer, reply exactly:
   "{NOT_FOUND_MESSAGE}"
5. Keep the answer concise.

Question:
{question}

Passages:
{context}

Answer:
""".strip()


def collect_sources(retrieval_results):
    sources = []

    for result in retrieval_results:
        document = result["document"]

        source = {
            "file": get_source(document),
            "page": get_page(document),
            "distance": round(result["distance"], 4),
        }

        if source not in sources:
            sources.append(source)

    return sources


def call_ollama(prompt):
    request_data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 180,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=request_data,
            timeout=180,
        )
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure the Ollama application is running."
        ) from error
    except requests.HTTPError as error:
        raise RuntimeError(
            f"Ollama returned an error: {response.text}"
        ) from error

    answer = response.json().get("response", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


def generate_answer(question):
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    retrieval_results = retrieve_relevant_documents(
        question=question,
        k=3,
        max_distance=1.05,
    )

    if not retrieval_results:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    context = build_context(retrieval_results)
    prompt = build_prompt(question, context)
    answer = call_ollama(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": collect_sources(retrieval_results),
    }


def print_answer(question):
    result = generate_answer(question)

    print("\n" + "=" * 70)
    print(f"Question: {result['question']}")
    print("=" * 70)
    print(f"\nAnswer:\n{result['answer']}")

    if result["sources"]:
        print("\nRetrieved sources:")

        for source in result["sources"]:
            print(
                f"- {source['file']}, "
                f"page {source['page']}, "
                f"distance {source['distance']}"
            )


if __name__ == "__main__":
    test_questions = [
        "What is RAG and why is it useful?",
        "Why is chunk overlap important?",
        "What is the capital of Australia?",
    ]

    for test_question in test_questions:
        print_answer(test_question)