from retriever import retrieve_documents_with_scores


TEST_CASES = [
    {
        "question": "What is retrieval-augmented generation?",
        "should_match": True,
    },
    {
        "question": "Why is RAG useful for document-based questions?",
        "should_match": True,
    },
    {
        "question": "What is a document chunk?",
        "should_match": True,
    },
    {
        "question": "Why is chunk overlap used?",
        "should_match": True,
    },
    {
        "question": "How can retrieval quality be evaluated?",
        "should_match": True,
    },
    {
        "question": "What should the system do when evidence is missing?",
        "should_match": True,
    },
    {
        "question": "What is the capital of Australia?",
        "should_match": False,
    },
    {
        "question": "How do I cook pasta?",
        "should_match": False,
    },
    {
        "question": "Who won the football World Cup?",
        "should_match": False,
    },
    {
        "question": "What is the weather in Berlin?",
        "should_match": False,
    },
]


def evaluate_threshold(threshold):
    correct_predictions = 0

    print(f"\nThreshold: {threshold:.2f}")
    print("-" * 90)

    for test_case in TEST_CASES:
        question = test_case["question"]
        expected_match = test_case["should_match"]

        results = retrieve_documents_with_scores(
            question=question,
            k=1,
        )

        best_distance = float(results[0][1])
        predicted_match = best_distance <= threshold
        is_correct = predicted_match == expected_match

        if is_correct:
            correct_predictions += 1

        expected_label = "match" if expected_match else "reject"
        predicted_label = "match" if predicted_match else "reject"
        outcome = "correct" if is_correct else "incorrect"

        print(
            f"{best_distance:.4f} | "
            f"expected: {expected_label:<6} | "
            f"predicted: {predicted_label:<6} | "
            f"{outcome:<9} | "
            f"{question}"
        )

    accuracy = correct_predictions / len(TEST_CASES)

    print("-" * 90)
    print(
        f"Correct predictions: "
        f"{correct_predictions}/{len(TEST_CASES)}"
    )
    print(f"Threshold accuracy: {accuracy:.2%}")

    return accuracy


if __name__ == "__main__":
    thresholds = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20]

    scores = {}

    for threshold in thresholds:
        scores[threshold] = evaluate_threshold(threshold)

    best_threshold = max(scores, key=scores.get)

    print("\n" + "=" * 90)
    print(f"Best tested threshold: {best_threshold:.2f}")
    print(f"Best accuracy: {scores[best_threshold]:.2%}")