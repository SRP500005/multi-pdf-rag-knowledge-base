import json
from pathlib import Path

from retriever import retrieve_documents_with_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_questions.json"
)
TOP_K = 3


def load_test_cases():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_source_name(document):
    source = document.metadata.get("source", "Unknown")
    return Path(source).name


def evaluate_question(question, expected_source):
    results = retrieve_documents_with_scores(
        question=question,
        k=TOP_K,
    )

    retrieved_sources = []
    relevant_result_count = 0
    first_relevant_rank = None

    for rank, (document, distance) in enumerate(
        results,
        start=1,
    ):
        source = get_source_name(document)
        retrieved_sources.append(source)

        if source == expected_source:
            relevant_result_count += 1

            if first_relevant_rank is None:
                first_relevant_rank = rank

    precision = relevant_result_count / TOP_K
    recall = 1.0 if first_relevant_rank is not None else 0.0
    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )
    top_one_correct = (
        bool(retrieved_sources)
        and retrieved_sources[0] == expected_source
    )

    return {
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "top_one_correct": top_one_correct,
    }


def print_question_result(result):
    print("\n" + "-" * 80)
    print(f"Question: {result['question']}")
    print(f"Expected: {result['expected_source']}")

    for rank, source in enumerate(
        result["retrieved_sources"],
        start=1,
    ):
        print(f"Rank {rank}: {source}")

    print(f"Precision@{TOP_K}: {result['precision']:.3f}")
    print(f"Recall@{TOP_K}: {result['recall']:.3f}")
    print(
        "Reciprocal rank: "
        f"{result['reciprocal_rank']:.3f}"
    )


def main():
    test_cases = load_test_cases()
    results = []

    for test_case in test_cases:
        result = evaluate_question(
            question=test_case["question"],
            expected_source=test_case["expected_source"],
        )

        results.append(result)
        print_question_result(result)

    question_count = len(results)

    mean_precision = sum(
        result["precision"] for result in results
    ) / question_count

    mean_recall = sum(
        result["recall"] for result in results
    ) / question_count

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"] for result in results
    ) / question_count

    top_one_accuracy = sum(
        result["top_one_correct"] for result in results
    ) / question_count

    print("\n" + "=" * 80)
    print("Retrieval evaluation summary")
    print("=" * 80)
    print(f"Questions evaluated: {question_count}")
    print(f"Mean Precision@{TOP_K}: {mean_precision:.3f}")
    print(f"Mean Recall@{TOP_K}: {mean_recall:.3f}")
    print(
        "Mean Reciprocal Rank: "
        f"{mean_reciprocal_rank:.3f}"
    )
    print(f"Top-1 accuracy: {top_one_accuracy:.3f}")


if __name__ == "__main__":
    main()