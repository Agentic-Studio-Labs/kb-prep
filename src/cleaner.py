"""Deterministic cleanup before scoring/chunking."""

import re
from collections import Counter

from .models import Paragraph


class DocumentCleaner:
    """Apply deterministic cleanup rules to parsed paragraphs."""

    def clean_paragraphs(self, paragraphs: list[str]) -> list[str]:
        temp = [Paragraph(text=p, level=0, style="Normal", index=i) for i, p in enumerate(paragraphs)]
        return [p.text for p in self.clean_document(temp)]

    def should_drop(self, text: str, seen: Counter) -> bool:
        if re.fullmatch(r"Page\s+\d+", text):
            return True
        if seen[text] >= 1 and len(text.split()) <= 8:
            return True
        return False

    def clean_document(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        rebuilt: list[Paragraph] = []
        seen: Counter = Counter()
        for para in paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if self.should_drop(text, seen):
                seen[text] += 1
                continue
            rebuilt.append(Paragraph(text=text, level=para.level, style=para.style, index=para.index))
            seen[text] += 1
        return rebuilt
