def check_faithfulness(answer, context):
    answer_lower = answer.lower()
    context_lower = context.lower()

    important_numbers = [
        "150",
        "200",
        "2000",
        "10000"
    ]

    for number in important_numbers:
        if number in answer_lower and number not in context_lower:
            return False

    return True
