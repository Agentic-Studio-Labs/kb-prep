"""Data models for anam-prep document analysis and upload pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Document parsing models
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    """A single paragraph or heading extracted from a document."""
    text: str
    level: int  # 0 = normal paragraph, 1-6 = heading level
    style: str  # Original style name ('Normal', 'Heading 1', etc.)
    index: int  # Position in document (0-based)

    @property
    def is_heading(self) -> bool:
        return self.level > 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class HeadingNode:
    """A node in the heading hierarchy tree."""
    text: str
    level: int
    position: int  # Paragraph index
    children: list["HeadingNode"] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Basic metadata extracted from the file itself."""
    file_path: str
    file_type: str  # 'docx', 'pdf', 'txt', or 'md'
    file_size_bytes: int = 0
    page_count: int = 0
    title: Optional[str] = None
    author: Optional[str] = None

    @property
    def filename(self) -> str:
        return Path(self.file_path).name

    @property
    def stem(self) -> str:
        return Path(self.file_path).stem


@dataclass
class ParsedDocument:
    """The result of parsing a DOCX or PDF file."""
    metadata: DocumentMetadata
    paragraphs: list[Paragraph] = field(default_factory=list)
    heading_tree: list[HeadingNode] = field(default_factory=list)

    @cached_property
    def headings(self) -> list[Paragraph]:
        # Cached on first access. Safe because paragraphs are treated as
        # immutable after construction — the fixer works on a copy.
        return [p for p in self.paragraphs if p.is_heading]

    @cached_property
    def body_paragraphs(self) -> list[Paragraph]:
        return [p for p in self.paragraphs if not p.is_heading]

    @cached_property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Readiness(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


@dataclass
class Issue:
    """A single quality issue found during scoring."""
    severity: Severity
    category: str
    message: str
    location: Optional[int] = None  # Paragraph index, if applicable
    context: str = ""  # Snippet of text around the issue
    fix: str = ""  # Actionable recommendation


@dataclass
class ScoringResult:
    """Score for one criterion category."""
    category: str
    label: str  # Human-readable name
    score: float  # 0-100
    weight: float  # Relative importance (weights should sum to ~1.0)
    issues: list[Issue] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class ScoreCard:
    """Complete scoring report for a single document."""
    file_path: str
    results: list[ScoringResult] = field(default_factory=list)
    overall_score: float = 0.0

    def add_result(self, result: ScoringResult):
        self.results.append(result)

    def compute_overall(self):
        total_weight = sum(r.weight for r in self.results)
        if total_weight > 0:
            self.overall_score = sum(r.weighted_score for r in self.results) / total_weight
        else:
            self.overall_score = 0.0

    @property
    def readiness(self) -> Readiness:
        if self.overall_score >= 85:
            return Readiness.EXCELLENT
        elif self.overall_score >= 70:
            return Readiness.GOOD
        elif self.overall_score >= 50:
            return Readiness.FAIR
        else:
            return Readiness.POOR

    @property
    def all_issues(self) -> list[Issue]:
        return [issue for r in self.results for issue in r.issues]

    @property
    def critical_issues(self) -> list[Issue]:
        return [i for i in self.all_issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.all_issues if i.severity == Severity.WARNING]


# ---------------------------------------------------------------------------
# Content analysis models (LLM-powered)
# ---------------------------------------------------------------------------

@dataclass
class ContentAnalysis:
    """LLM-extracted understanding of document content."""
    domain: str = ""  # e.g. "education", "legal", "technical"
    topics: list[str] = field(default_factory=list)
    audience: str = ""
    content_type: str = ""  # "tutorial", "reference", "policy", etc.
    key_concepts: list[str] = field(default_factory=list)
    suggested_tags: list[str] = field(default_factory=list)
    summary: str = ""
    # Graph extraction — entities and relationships found in this document
    entities: list["Entity"] = field(default_factory=list)
    relationships: list["Relationship"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge graph models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named concept, topic, standard, or resource extracted from a document."""
    name: str
    entity_type: str  # "topic", "standard", "skill", "lesson", "assessment", "resource", "person", "concept"
    source_file: str = ""  # Which document this was extracted from
    description: str = ""  # Brief context

    @property
    def key(self) -> str:
        """Normalized key for deduplication."""
        return f"{self.entity_type}:{self.name.lower().strip()}"


@dataclass
class Relationship:
    """A directed edge between two entities."""
    source: str  # Entity name
    target: str  # Entity name
    rel_type: str  # "prerequisite", "covers_standard", "assesses", "part_of", "references", "related_to"
    source_file: str = ""  # Which document this was found in
    context: str = ""  # Sentence or paragraph where this relationship was stated


@dataclass
class GraphSummary:
    """Summary statistics for a knowledge graph."""
    total_entities: int = 0
    total_relationships: int = 0
    entity_types: dict[str, int] = field(default_factory=dict)  # type → count
    relationship_types: dict[str, int] = field(default_factory=dict)  # type → count
    orphan_references: list[str] = field(default_factory=list)  # Mentioned but undefined
    clusters: list[list[str]] = field(default_factory=list)  # Community detection results
    cross_document_edges: int = 0  # Relationships spanning multiple files


# ---------------------------------------------------------------------------
# Folder recommendation models
# ---------------------------------------------------------------------------

@dataclass
class FolderNode:
    """A node in the recommended folder hierarchy."""
    name: str
    description: str
    children: list["FolderNode"] = field(default_factory=list)
    document_files: list[str] = field(default_factory=list)  # Files assigned here

    def add_child(self, child: "FolderNode") -> "FolderNode":
        self.children.append(child)
        return child


@dataclass
class FolderRecommendation:
    """Complete folder structure recommendation."""
    root: FolderNode
    file_assignments: dict[str, str] = field(default_factory=dict)  # file_path → folder_name


# ---------------------------------------------------------------------------
# Upload models
# ---------------------------------------------------------------------------

@dataclass
class UploadSession:
    """Tracks a presigned upload session."""
    session_id: str
    presigned_url: str
    filename: str


class DocumentStatus(str, Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class UploadResult:
    """Result of uploading a single document."""
    file_path: str
    document_id: str
    folder_id: str
    folder_name: str
    status: DocumentStatus
    error: Optional[str] = None


@dataclass
class UploadReport:
    """Summary of all uploads in a batch."""
    results: list[UploadResult] = field(default_factory=list)
    folders_created: list[str] = field(default_factory=list)

    @property
    def successful(self) -> list[UploadResult]:
        return [r for r in self.results if r.status != DocumentStatus.FAILED]

    @property
    def failed(self) -> list[UploadResult]:
        return [r for r in self.results if r.status == DocumentStatus.FAILED]


# ---------------------------------------------------------------------------
# Fix models
# ---------------------------------------------------------------------------

@dataclass
class FixAction:
    """A single fix applied to a document."""
    category: str
    original_text: str
    fixed_text: str
    paragraph_index: Optional[int] = None
    description: str = ""


@dataclass
class FixReport:
    """Summary of all fixes applied to a document."""
    source_path: str
    output_path: str
    actions: list[FixAction] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)  # If file was split
    new_filename: Optional[str] = None  # If renamed
