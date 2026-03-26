from src.fixer import _has_positional_reference
from src.models import DocumentMetadata, Paragraph, ParsedDocument
from src.scorer import QualityScorer


def test_self_containment_flags_dangling_reference():
    doc_with_reference = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/test.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Savings Basics", level=1, style="Heading 1", index=0),
            Paragraph(
                text="As discussed above, students should compare needs and wants.", level=0, style="Normal", index=1
            ),
        ],
    )
    scorer = QualityScorer()
    card = scorer.score(doc_with_reference)

    issue = next(i for i in card.all_issues if i.category == "self_containment")
    assert "Dangling" in issue.message
    assert issue.location == 1


def test_fixer_rewrites_reference_to_standalone_sentence():
    assert _has_positional_reference("As discussed above, students should compare needs and wants.")
    assert not _has_positional_reference("Students should compare needs and wants before building a budget.")
