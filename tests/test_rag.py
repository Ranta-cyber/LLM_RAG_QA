# Here tested just some of the cases just for learning
from src.rag import answer_question
import json
import pytest

from src.ingestion import create_vector_store

# Create vector local database
# ============================
collection = create_vector_store()

with open("rag_test_cases.json", encoding="utf-8") as file:
    test_cases = json.load(file)


# print("test_cases:", test_cases)

@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=lambda test_case: test_case["name"])
def test_rag_answers(test_case):
    user_question = test_case["question"]
    result = answer_question(user_question)
    answer = result["answer"].lower()
    context = result["context"].lower()

    if test_case["should_answer"]:
        assert test_case["expected_context"].lower() in context
        assert test_case["expected_answer"].lower() in answer
    else:
        # here need to guess some better way because the answer can be
        # not information available
        # no information available
        # not enough information is available
        # so this assert is not yet to´tally correct
        assert "information" in answer
        assert "available" in answer
