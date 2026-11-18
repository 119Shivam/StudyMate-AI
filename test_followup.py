
import re


def is_follow_up(user_query):
    query = user_query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)

    follow_up_patterns = [
        r"\bit\b",
        r"\bits\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bformer\b",
        r"\blatter\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bgive an example\b",
        r"\bexplain more\b",
        r"\belaborate\b",
        r"\btell me more\b",
        r"\bcontinue\b",
        r"\bsame\b",
        r"\bmore details\b",
        r"\bmore about\b",
        r"\bcompare them\b",
        r"\bdifference between them\b"
    ]

    if any(
        re.search(pattern, query)
        for pattern in follow_up_patterns
    ):
        return True

    short_follow_ups = [
        "why",
        "how",
        "why so",
        "how so",
        "example",
        "another example",
        "explain",
        "elaborate",
        "continue",
        "and then",
        "what next"
    ]

    return query in short_follow_ups


# Test questions
test_questions = [
    "Explain its characteristics.",
    "Give an example.",
    "Why does this happen?",
    "What is entropy?",
    "Define thermodynamics.",
    "How does it work?",
    "Explain the concept of pressure.",
    "Another example.",
    "What about its advantages?",
    "What is the first law of thermodynamics?"
]


for question in test_questions:
    result = is_follow_up(question)

    print(f"Question: {question}")
    print(f"Follow-up: {result}")
    print("-" * 50)
