"""Heuristic RAG readiness scoring for documents.

All scoring is regex + structural analysis — no LLM calls required.
Runs fast and free on any document.
"""

import re

from .models import (
    Issue,
    ParsedDocument,
    ScoreCard,
    ScoringResult,
    Severity,
    SplitRecommendation,
)
from .parser import MAX_FILE_SIZE, WARN_FILE_SIZE


class QualityScorer:
    """Multi-criteria heuristic scorer for RAG readiness.

    Scoring penalties are configurable via class constants.
    """

    # Points deducted per issue in each category
    SELF_CONTAINMENT_PENALTY = 12
    HEADING_QUALITY_PENALTY = 8
    SHORT_PARAGRAPH_PENALTY = 3
    LONG_PARAGRAPH_PENALTY = 8
    FILENAME_PENALTY = 20
    ACRONYM_PENALTY = 8

    # Paragraph length thresholds (words)
    MIN_PARAGRAPH_WORDS = 15
    MAX_PARAGRAPH_WORDS = 300
    IDEAL_MIN_WORDS = 50
    IDEAL_MAX_WORDS = 200

    # Heading density threshold (paragraphs per heading)
    MAX_PARAS_PER_HEADING = 20

    # File focus diversity thresholds
    FOCUS_HIGH_DIVERSITY = 0.85  # Flag as sprawling
    FOCUS_MODERATE_DIVERSITY = 0.70  # Flag as broad

    def __init__(self, graph=None, corpus_analysis=None):
        """Initialize scorer with optional knowledge graph and corpus analysis.

        If a graph is provided, adds Knowledge Completeness scoring
        (orphan references, cross-document gaps).
        If corpus_analysis is provided, enables entropy-based file focus
        and retrieval-aware scoring.
        """
        self.graph = graph
        self.corpus_analysis = corpus_analysis

    def score(self, doc: ParsedDocument) -> ScoreCard:
        """Run all scoring criteria and return a complete ScoreCard."""
        card = ScoreCard(file_path=doc.metadata.file_path)

        card.add_result(self._score_self_containment(doc))
        card.add_result(self._score_heading_quality(doc))
        card.add_result(self._score_paragraph_length(doc))
        card.add_result(self._score_file_focus(doc))
        card.add_result(self._score_filename_quality(doc))
        card.add_result(self._score_acronym_definitions(doc))
        card.add_result(self._score_structure_completeness(doc))
        card.add_result(self._score_file_size(doc))
        card.add_result(self._score_retrieval_aware(doc))

        # Graph-powered scoring (when available)
        # Reduce other weights proportionally so total stays at 1.0
        if self.graph and not self.graph.is_empty:
            graph_result = self._score_knowledge_completeness(doc)
            graph_weight = graph_result.weight
            # Scale existing weights down so total remains ~1.0
            non_zero_results = [r for r in card.results if r.weight > 0]
            if non_zero_results:
                current_total = sum(r.weight for r in non_zero_results)
                scale = (current_total - graph_weight) / current_total
                for r in non_zero_results:
                    r.weight *= scale
            card.add_result(graph_result)

        card.compute_overall()
        return card

    # ------------------------------------------------------------------
    # 1. Self-Containment (20%)
    # ------------------------------------------------------------------

    def _score_self_containment(self, doc: ParsedDocument) -> ScoringResult:
        """Detect dangling references that break paragraph independence."""
        issues: list[Issue] = []

        # Patterns that indicate cross-paragraph dependencies
        patterns = [
            (r"\b(?:as\s+)?mentioned\s+(?:above|below|earlier|previously|before)\b", "cross-reference"),
            (r"\bthe\s+(?:above|below|following|previous|preceding)\s+\w+", "positional reference"),
            (r"\bsee\s+(?:section|figure|table|page|chapter)\s+\w+", "section reference"),
            (r"\b(?:refer|referring)\s+to\s+(?:the\s+)?(?:above|below|previous)", "explicit reference"),
            (r"\bas\s+(?:noted|described|shown|outlined|discussed)\s+(?:above|below|earlier)", "back-reference"),
            (r"\b(?:the\s+)?(?:steps?|instructions?|procedure)\s+(?:above|below)\b", "procedural reference"),
        ]

        for para in doc.body_paragraphs:
            for pattern, ref_type in patterns:
                for match in re.finditer(pattern, para.text, re.IGNORECASE):
                    snippet = para.text[max(0, match.start() - 30) : match.end() + 30]
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category="self_containment",
                            message=f'Dangling {ref_type}: "{match.group()}"',
                            location=para.index,
                            context=f"...{snippet}...",
                            fix="Rewrite so the chunk stands alone without requiring earlier or later context.",
                        )
                    )

        score = max(0.0, 100.0 - len(issues) * self.SELF_CONTAINMENT_PENALTY)
        return ScoringResult(
            category="self_containment",
            label="Self-Containment",
            score=score,
            weight=0.20,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 2. Heading Quality (15%)
    # ------------------------------------------------------------------

    def _score_heading_quality(self, doc: ParsedDocument) -> ScoringResult:
        """Evaluate heading hierarchy and descriptiveness."""
        issues: list[Issue] = []
        headings = doc.headings

        # No headings at all
        if not headings and len(doc.body_paragraphs) > 3:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="heading_quality",
                    message="Document has no headings",
                    fix="Add descriptive headings to organize content into searchable sections.",
                )
            )
            return ScoringResult(
                category="heading_quality",
                label="Heading Quality",
                score=30.0,
                weight=0.15,
                issues=issues,
            )

        # Check hierarchy continuity (no H1 → H3 jumps)
        levels_seen: set[int] = set()
        prev_level = 0
        for h in headings:
            if h.level > prev_level + 1 and prev_level > 0:
                issues.append(
                    Issue(
                        severity=Severity.INFO,
                        category="heading_quality",
                        message=f'Heading level jump: H{prev_level} → H{h.level} at "{h.text[:50]}"',
                        location=h.index,
                        fix=f"Use H{prev_level + 1} instead of H{h.level} for proper hierarchy.",
                    )
                )
            levels_seen.add(h.level)
            prev_level = h.level

        # Check for generic/vague headings
        generic_headings = {
            "content",
            "details",
            "info",
            "information",
            "section",
            "material",
            "data",
            "overview",
            "introduction",
            "conclusion",
            "notes",
            "misc",
            "miscellaneous",
            "other",
            "general",
        }
        for h in headings:
            words = set(h.text.lower().split())
            if words and words.issubset(generic_headings):
                issues.append(
                    Issue(
                        severity=Severity.INFO,
                        category="heading_quality",
                        message=f'Generic heading: "{h.text}"',
                        location=h.index,
                        fix="Use a specific, descriptive heading (e.g. 'Adding Fractions' not 'Content').",
                    )
                )

        # Check heading-to-content ratio
        if headings and len(doc.body_paragraphs) > 0:
            ratio = len(doc.body_paragraphs) / len(headings)
            if ratio > self.MAX_PARAS_PER_HEADING:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="heading_quality",
                        message=f"Low heading density: ~{ratio:.0f} paragraphs per heading",
                        fix="Add more sub-headings to break content into smaller searchable sections.",
                    )
                )

        score = max(0.0, 100.0 - len(issues) * self.HEADING_QUALITY_PENALTY)
        return ScoringResult(
            category="heading_quality",
            label="Heading Quality",
            score=score,
            weight=0.15,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 3. Paragraph Length (10%)
    # ------------------------------------------------------------------

    def _score_paragraph_length(self, doc: ParsedDocument) -> ScoringResult:
        """Check paragraph length distribution for RAG suitability."""
        issues: list[Issue] = []

        too_short = 0
        too_long = 0

        for para in doc.body_paragraphs:
            wc = para.word_count
            if wc < self.MIN_PARAGRAPH_WORDS:
                too_short += 1
                if too_short <= 5:  # Don't flood with issues
                    issues.append(
                        Issue(
                            severity=Severity.INFO,
                            category="paragraph_length",
                            message=f"Very short paragraph ({wc} words)",
                            location=para.index,
                            context=para.text[:80],
                            fix="Expand with more context or merge with adjacent paragraph.",
                        )
                    )
            elif wc > self.MAX_PARAGRAPH_WORDS:
                too_long += 1
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="paragraph_length",
                        message=f"Very long paragraph ({wc} words)",
                        location=para.index,
                        context=para.text[:80] + "...",
                        fix="Split into multiple focused paragraphs (ideal: 50-200 words).",
                    )
                )

        total_body = len(doc.body_paragraphs)
        if total_body > 0:
            short_pct = too_short / total_body
            if short_pct > 0.5:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="paragraph_length",
                        message=f"{short_pct:.0%} of paragraphs are very short (<15 words)",
                        fix="Many paragraphs lack context. Expand or consolidate them.",
                    )
                )

        score = max(0.0, 100.0 - too_short * self.SHORT_PARAGRAPH_PENALTY - too_long * self.LONG_PARAGRAPH_PENALTY)
        return ScoringResult(
            category="paragraph_length",
            label="Paragraph Length",
            score=score,
            weight=0.10,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 4. File Focus (10%)
    # ------------------------------------------------------------------

    def _score_file_focus(self, doc: ParsedDocument) -> ScoringResult:
        """Assess whether the document is focused or sprawling."""
        issues: list[Issue] = []
        headings = doc.headings

        # Use entropy from corpus analysis if available
        if self.corpus_analysis:
            metrics = self.corpus_analysis.doc_metrics.get(doc.metadata.filename)
            if metrics:
                entropy = metrics.entropy
                if entropy > 0.7:
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category="file_focus",
                            message=f"High topic entropy ({entropy:.2f}) — document covers many disparate topics",
                            fix="Split into focused single-topic documents for better RAG retrieval.",
                        )
                    )
                    score = 40.0
                elif entropy > 0.4:
                    issues.append(
                        Issue(
                            severity=Severity.INFO,
                            category="file_focus",
                            message=f"Moderate topic entropy ({entropy:.2f})",
                            fix="Consider splitting into 2-3 topic-focused documents.",
                        )
                    )
                    score = 65.0
                else:
                    score = 90.0
                return ScoringResult(category="file_focus", label="File Focus", score=score, weight=0.10, issues=issues)
        # Fallback: original diversity ratio (when no corpus analysis)

        if len(headings) < 2:
            # Can't assess focus with 0-1 headings
            return ScoringResult(
                category="file_focus",
                label="File Focus",
                score=75.0,
                weight=0.10,
                issues=[],
            )

        # Extract meaningful words from headings
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }
        topic_words: list[str] = []
        for h in headings:
            words = [w.lower() for w in re.findall(r"\b\w+\b", h.text) if len(w) > 3 and w.lower() not in stop_words]
            topic_words.extend(words)

        if not topic_words:
            return ScoringResult(
                category="file_focus",
                label="File Focus",
                score=70.0,
                weight=0.10,
                issues=[],
            )

        # Diversity ratio: unique words / total words
        unique = set(topic_words)
        diversity = len(unique) / len(topic_words)

        if diversity > self.FOCUS_HIGH_DIVERSITY:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="file_focus",
                    message=f"Document covers many disparate topics (diversity: {diversity:.0%})",
                    fix="Split into focused single-topic documents for better RAG retrieval.",
                )
            )
            score = 40.0
        elif diversity > self.FOCUS_MODERATE_DIVERSITY:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="file_focus",
                    message=f"Document is moderately broad (diversity: {diversity:.0%})",
                    fix="Consider splitting into 2-3 topic-focused documents.",
                )
            )
            score = 65.0
        else:
            score = 90.0

        return ScoringResult(
            category="file_focus",
            label="File Focus",
            score=score,
            weight=0.10,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 5. Filename Quality (10%)
    # ------------------------------------------------------------------

    def _score_filename_quality(self, doc: ParsedDocument) -> ScoringResult:
        """Evaluate filename descriptiveness."""
        issues: list[Issue] = []
        stem = doc.metadata.stem
        filename = doc.metadata.filename

        # Generic patterns
        generic_patterns = [
            r"^(?:document|doc|file|lesson|untitled|new|draft|copy)",
            r"^(?:final|v\d+|version|rev|revision)",
            r"^\d+$",  # Just a number
            r"^[a-f0-9]{8,}$",  # Hash-like
        ]
        for pattern in generic_patterns:
            if re.match(pattern, stem, re.IGNORECASE):
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="filename_quality",
                        message=f'Generic filename: "{filename}"',
                        fix='Use descriptive names like "fractions-adding-grade5.docx" not "lesson-v2.docx".',
                    )
                )
                break

        # Too short
        if len(stem) < 8:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="filename_quality",
                    message=f'Short filename: "{filename}" ({len(stem)} chars)',
                    fix="Include topic, subject area, or audience in the filename.",
                )
            )

        # No word separators (likely a single mashed word)
        if len(stem) > 15 and not re.search(r"[-_ ]", stem):
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="filename_quality",
                    message=f'No word separators in filename: "{filename}"',
                    fix="Use hyphens or underscores to separate words for readability.",
                )
            )

        score = max(0.0, 100.0 - len(issues) * self.FILENAME_PENALTY)
        return ScoringResult(
            category="filename_quality",
            label="Filename Quality",
            score=score,
            weight=0.10,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 6. Acronym Definitions (5%)
    # ------------------------------------------------------------------

    def _score_acronym_definitions(self, doc: ParsedDocument) -> ScoringResult:
        """Check for undefined acronyms.

        Uses a conservative approach: only flags sequences that look like
        real acronyms (2-6 uppercase letters, not common English words)
        and appear multiple times without a definition pattern nearby.
        """
        issues: list[Issue] = []
        full_text = doc.full_text

        # Find all uppercase acronyms (2-6 letters) that are surrounded by
        # non-uppercase text (avoids matching ALL-CAPS sentences)
        found_acronyms = set(re.findall(r"(?<![A-Z])\b[A-Z]{2,6}\b(?![A-Z])", full_text))

        # Filter out common acronyms, abbreviations, and English words
        common = {
            # Common acronyms
            "US",
            "UK",
            "EU",
            "AI",
            "ML",
            "IT",
            "HR",
            "CEO",
            "CTO",
            "CFO",
            "API",
            "PDF",
            "URL",
            "HTML",
            "CSS",
            "SQL",
            "FAQ",
            "ID",
            "OK",
            "AM",
            "PM",
            "BC",
            "AD",
            "TV",
            "PC",
            "OS",
            "IP",
            "USB",
            "RAM",
            "PhD",
            "MBA",
            "GPA",
            "SAT",
            "ACT",
            "AP",
            "IB",
            "LLC",
            "INC",
            # English words that look like acronyms
            "AND",
            "THE",
            "BUT",
            "FOR",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "HER",
            "WAS",
            "ONE",
            "OUR",
            "OUT",
            "ARE",
            "HAS",
            "HIS",
            "HOW",
            "ITS",
            "MAY",
            "NEW",
            "NOW",
            "OLD",
            "SEE",
            "WAY",
            "WHO",
            "DID",
            "GET",
            "LET",
            "SAY",
            "SHE",
            "TOO",
            "USE",
            "DAD",
            "MOM",
            "RUN",
            "SET",
        }
        acronyms_to_check = found_acronyms - common

        for acronym in sorted(acronyms_to_check):
            # Check if defined: "XYZ (explanation)" or "(XYZ) explanation"
            defined = bool(
                re.search(rf"{re.escape(acronym)}\s*\([^)]+\)", full_text)
                or re.search(rf"\([^)]*{re.escape(acronym)}[^)]*\)", full_text)
            )
            if not defined:
                # Count occurrences — only flag if used more than once
                count = len(re.findall(rf"\b{re.escape(acronym)}\b", full_text))
                if count >= 2:
                    issues.append(
                        Issue(
                            severity=Severity.INFO,
                            category="acronym_definitions",
                            message=f'Acronym "{acronym}" used {count}x but may not be defined',
                            fix=f'Define on first use: "{acronym} (Full Name Here)".',
                        )
                    )

        score = max(0.0, 100.0 - len(issues) * self.ACRONYM_PENALTY)
        return ScoringResult(
            category="acronym_definitions",
            label="Acronym Definitions",
            score=score,
            weight=0.05,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 7. Structure Completeness (10%)
    # ------------------------------------------------------------------

    def _score_structure_completeness(self, doc: ParsedDocument) -> ScoringResult:
        """Check for basic structural elements."""
        issues: list[Issue] = []
        checks_passed = 0
        total_checks = 4

        # Has any headings?
        if doc.headings:
            checks_passed += 1
        else:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="structure",
                    message="No headings found",
                    fix="Add headings to create searchable sections.",
                )
            )

        # Has substantive body content?
        body_word_count = sum(p.word_count for p in doc.body_paragraphs)
        if body_word_count > 100:
            checks_passed += 1
        else:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="structure",
                    message=f"Very little body text ({body_word_count} words)",
                    fix="Document may be too brief to provide useful retrieval context.",
                )
            )

        # Has multiple sections (headings)?
        if len(doc.headings) >= 2:
            checks_passed += 1
        elif doc.headings:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="structure",
                    message="Only one heading/section",
                    fix="Consider breaking content into multiple sections for granular retrieval.",
                )
            )

        # Reasonable paragraph count?
        if len(doc.body_paragraphs) >= 3:
            checks_passed += 1
        else:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="structure",
                    message=f"Only {len(doc.body_paragraphs)} paragraph(s)",
                    fix="Add more content or structure for effective chunking.",
                )
            )

        score = (checks_passed / total_checks) * 100
        return ScoringResult(
            category="structure",
            label="Structure Completeness",
            score=score,
            weight=0.10,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 8. File Size (bonus — doesn't count toward weighted score but flags issues)
    # ------------------------------------------------------------------

    def _score_file_size(self, doc: ParsedDocument) -> ScoringResult:
        """Check file size against RAG upload limits."""
        issues: list[Issue] = []
        size = doc.metadata.file_size_bytes
        mb = size / (1024 * 1024)

        if size > MAX_FILE_SIZE:
            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category="file_size",
                    message=f"File is {mb:.1f} MB — exceeds 50 MB limit",
                    fix="Convert to Markdown (strips formatting/images) or split into smaller files.",
                )
            )
            score = 0.0
        elif size > WARN_FILE_SIZE:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="file_size",
                    message=f"File is {mb:.1f} MB — approaching 50 MB limit",
                    fix="Consider converting to Markdown to reduce size.",
                )
            )
            score = 50.0
        else:
            score = 100.0

        # File size is a hard constraint, not a quality signal — use 0 weight
        # but still surface the issues
        return ScoringResult(
            category="file_size",
            label="File Size",
            score=score,
            weight=0.0,  # Informational only
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 9. Retrieval-Aware (corpus-powered, when available)
    # ------------------------------------------------------------------

    def _score_retrieval_aware(self, doc: ParsedDocument) -> ScoringResult:
        """Score self-retrieval rate — how findable the document is via search."""
        issues: list[Issue] = []
        if not self.corpus_analysis:
            return ScoringResult(
                category="retrieval_aware", label="Retrieval-Aware", score=75.0, weight=0.20, issues=[]
            )
        metrics = self.corpus_analysis.doc_metrics.get(doc.metadata.filename)
        if not metrics:
            return ScoringResult(
                category="retrieval_aware", label="Retrieval-Aware", score=75.0, weight=0.20, issues=[]
            )
        rate = metrics.self_retrieval_score
        score = rate * 100
        if rate < 0.4:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="retrieval_aware",
                    message=f"Low self-retrieval rate ({rate:.0%}) — document may be hard to find via search",
                    fix="Ensure headings and key terms match likely search queries. Split overly broad documents.",
                )
            )
        elif rate < 0.7:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="retrieval_aware",
                    message=f"Moderate self-retrieval rate ({rate:.0%})",
                    fix="Consider adding more specific terminology or splitting into focused sections.",
                )
            )
        return ScoringResult(
            category="retrieval_aware", label="Retrieval-Aware", score=score, weight=0.20, issues=issues
        )

    # ------------------------------------------------------------------
    # 10. Knowledge Completeness (graph-powered, when available)
    # ------------------------------------------------------------------

    def _score_knowledge_completeness(self, doc: ParsedDocument) -> ScoringResult:
        """Assess document's knowledge completeness using the graph.

        Checks for:
        - Orphan references: entities mentioned in this doc that are never
          defined anywhere in the corpus (broken knowledge links)
        - Isolated content: this doc's entities have zero cross-document
          connections (suggests the doc might need more context or integration)
        - Unresolved relationships: edges pointing to 'unresolved' entities
          that originate from this document

        Only runs when a knowledge graph is available (i.e. LLM analysis
        was performed). Weight is low (5%) since it's supplementary.
        """
        issues: list[Issue] = []
        filename = doc.metadata.filename

        # 1. Check for orphan references originating from this document
        file_entities = self.graph.get_entities_for_file(filename)
        orphan_count = 0
        for entity in file_entities:
            if entity.entity_type == "unresolved":
                orphan_count += 1
                if orphan_count <= 3:  # Don't flood
                    issues.append(
                        Issue(
                            severity=Severity.INFO,
                            category="knowledge_completeness",
                            message=f'Referenced "{entity.name}" but it\'s not defined in any document',
                            fix=(
                                "Either define this term/concept in the document or ensure "
                                "the referenced document is included in the corpus."
                            ),
                        )
                    )

        if orphan_count > 3:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="knowledge_completeness",
                    message=f"{orphan_count} references to undefined entities in total",
                    fix="Many references point to content not found in the corpus. Review for dangling references.",
                )
            )

        # 2. Check cross-document connectivity
        cross_refs = self.graph.get_cross_document_references(filename)
        resolved_entities = [e for e in file_entities if e.entity_type != "unresolved"]

        if resolved_entities and not cross_refs:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="knowledge_completeness",
                    message="Document has no cross-references to other documents in the corpus",
                    fix=(
                        "This document is isolated. Consider adding contextual links "
                        "or ensuring related documents are in the corpus."
                    ),
                )
            )

        # Scoring: start at 100, deduct for issues
        penalty = orphan_count * 10  # 10 points per orphan
        if resolved_entities and not cross_refs:
            penalty += 15  # Isolation penalty

        score = max(0.0, 100.0 - penalty)

        return ScoringResult(
            category="knowledge_completeness",
            label="Knowledge Completeness",
            score=score,
            weight=0.05,  # Supplementary — only when graph is available
            issues=issues,
        )


def generate_split_recommendations(
    docs: list[ParsedDocument], cards: list[ScoreCard], corpus_analysis=None
) -> list[SplitRecommendation]:
    """Build structured split recommendations from file-focus signals."""
    recommendations: list[SplitRecommendation] = []

    for doc, card in zip(docs, cards):
        focus_result = next((r for r in card.results if r.category == "file_focus"), None)
        if not focus_result:
            continue
        if focus_result.score > 65 and not any("split" in issue.fix.lower() for issue in focus_result.issues):
            continue

        metrics = corpus_analysis.doc_metrics.get(doc.metadata.filename) if corpus_analysis else None
        boundaries = list(metrics.topic_boundaries) if metrics and metrics.topic_boundaries else []
        if not boundaries:
            boundaries = [h.index for h in doc.headings[1:]]

        suggested_titles = [h.text.strip() for h in doc.headings[:6] if h.text.strip()]
        reason = (
            focus_result.issues[0].message if focus_result.issues else "Document appears broad across multiple topics."
        )

        recommendations.append(
            SplitRecommendation(
                source_file=doc.metadata.filename,
                reason=reason,
                proposed_boundaries=boundaries,
                suggested_titles=suggested_titles,
            )
        )

    return recommendations
