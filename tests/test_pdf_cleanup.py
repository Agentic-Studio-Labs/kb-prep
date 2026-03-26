from src.models import Paragraph
from src.parser import DocumentParser


def test_filter_pdf_noise_removes_copyright_and_page_numbers():
    parser = DocumentParser()
    paragraphs = [
        Paragraph(text="1", level=0, style="pdf-size-10.0", index=0),
        Paragraph(
            text="© 2016 by Junior Achievement USA. All rights reserved. Name:",
            level=0,
            style="pdf-size-9.0",
            index=1,
        ),
        Paragraph(text="Students will define SMART goals.", level=0, style="pdf-size-11.0", index=2),
    ]

    cleaned = parser._filter_pdf_noise(paragraphs)

    assert len(cleaned) == 1
    assert cleaned[0].text == "Students will define SMART goals."
    assert cleaned[0].index == 0
