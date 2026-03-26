from src.cleaner import DocumentCleaner


def test_cleaner_drops_repeated_headers_and_footers():
    cleaner = DocumentCleaner()
    paragraphs = [
        "Financial Literacy Grade 4-5",
        "Budgeting begins with goals.",
        "Page 1",
        "Financial Literacy Grade 4-5",
        "Track savings over time.",
        "Page 2",
    ]

    cleaned = cleaner.clean_paragraphs(paragraphs)

    assert "Page 1" not in cleaned
    assert "Page 2" not in cleaned
    assert cleaned.count("Financial Literacy Grade 4-5") <= 1
