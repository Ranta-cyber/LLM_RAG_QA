import json
from src.rag import answer_question


def evaluation_report(collection):
    with open("../rag_test_cases.json", encoding="utf-8") as file:
        test_cases = json.load(file)

    results = []

    for test_case in test_cases:
        result = answer_question(test_case["question"], collection)
        answer = result["answer"]

        answer_lower = result["answer"].lower()
        context_lower = result["context"].lower()

        contex_ok = (
            test_case["expected_context"].lower() in context_lower
            if test_case["expected_context"]
            else True
        )

        answer_ok = (
                test_case["expected_answer"].lower() in answer_lower
        )

        results.append(
            {
                "name": test_case["name"],
                "context_ok": contex_ok,
                "answer_ok": answer_ok,
                "answer": answer
            }
        )

    print("\nReport")
    print("=" * 50)
    for result in results:
        print(f"\nTests: {result['name']}")
        print(f"Context: {'PASS' if result['context_ok'] else 'FAIL'}")
        print(f"Answer: {'PASS' if result['answer_ok'] else 'FAIL'}")
        print(f"Actual answer: {result['answer']}")
