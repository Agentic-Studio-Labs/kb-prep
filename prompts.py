"""LLM prompt templates for document analysis, fixing, and folder recommendation."""

# ---------------------------------------------------------------------------
# Content Analysis Prompt
# ---------------------------------------------------------------------------

ANALYZE_DOCUMENT = """\
You are a document analyst preparing content for a RAG (Retrieval-Augmented Generation) knowledge base.

Analyze the following document and extract structured metadata AND a knowledge graph of entities and relationships. Return ONLY valid JSON with no other text.

<document>
{document_text}
</document>

Return this exact JSON structure:
{{
  "domain": "<primary domain: education, legal, technical, medical, business, marketing, support, scientific, etc.>",
  "topics": ["<specific topic 1>", "<specific topic 2>", ...],
  "audience": "<who is this for: students, teachers, developers, customers, etc.>",
  "content_type": "<tutorial, reference, policy, narrative, guide, lesson plan, report, manual, etc.>",
  "key_concepts": ["<concept 1>", "<concept 2>", ...],
  "suggested_tags": ["<tag1>", "<tag2>", ...],
  "summary": "<2-3 sentence summary of the document's purpose and content>",
  "entities": [
    {{
      "name": "<entity name>",
      "type": "<topic|standard|skill|lesson|assessment|resource|person|concept|tool|process>",
      "description": "<brief context about this entity>"
    }}
  ],
  "relationships": [
    {{
      "source": "<entity name>",
      "target": "<entity name>",
      "type": "<prerequisite|covers_standard|assesses|part_of|references|related_to|requires|teaches|follows|uses>",
      "context": "<brief sentence explaining this relationship>"
    }}
  ]
}}

Guidelines for entity and relationship extraction:
- Extract SPECIFIC named entities (e.g. "Adding Fractions with Unlike Denominators", not just "math")
- Include any standards, skills, or prerequisites mentioned (even if from other documents)
- Capture cross-references to other lessons, units, or documents
- Include relationships between concepts (what depends on what, what builds on what)
- For education: extract grade levels, standards codes, learning objectives, skills
- For technical: extract APIs, systems, dependencies, configurations
- For legal: extract parties, clauses, obligations, definitions
- Be thorough — the graph powers cross-document understanding
"""

# ---------------------------------------------------------------------------
# Folder Recommendation Prompt
# ---------------------------------------------------------------------------

RECOMMEND_FOLDERS = """\
You are organizing documents into a knowledge base folder structure for a RAG system (anam.ai).

Here are the documents and their analyzed metadata:

<documents>
{documents_json}
</documents>

Design a folder hierarchy that:
1. Groups related documents by domain, then by topic/subtopic
2. Uses 2-3 levels max (too deep = hard to navigate, too flat = no organization)
3. Each folder should contain 2-10 documents (split further if more)
4. Folder names should be clear, short, and descriptive
5. Each folder needs a description explaining what content it holds (this helps the LLM decide when to search it)

Return ONLY valid JSON:
{{
  "folders": [
    {{
      "name": "<folder name>",
      "description": "<what this folder contains — written for an LLM to understand when to search here>",
      "children": [
        {{
          "name": "<subfolder name>",
          "description": "<description>",
          "children": []
        }}
      ]
    }}
  ],
  "assignments": {{
    "<filename>": "<full/folder/path>"
  }}
}}

The assignments map each filename to its recommended folder path (e.g. "Mathematics/Fractions").

Important guidelines:
- Distribute files EVENLY across folders. No single folder should contain more than ~30% of all files.
- If a folder would have too many files, split it into meaningful subfolders.
- Group by FUNCTION/PURPOSE first, then by topic within those groups.
- Folder descriptions are critical — they tell the RAG system's LLM when to search each folder. Be specific about what questions each folder can answer.
{domain_hints}"""

# ---------------------------------------------------------------------------
# Auto-Fix Prompts
# ---------------------------------------------------------------------------

FIX_DANGLING_REFERENCES = """\
You are improving a document for RAG (Retrieval-Augmented Generation) readability.

The following paragraph contains dangling references (like "as mentioned above", "the following steps", etc.) that make it not self-contained. In a RAG system, each paragraph may be retrieved independently, so readers won't have the surrounding context.

<surrounding_context>
{surrounding_context}
</surrounding_context>

<paragraph_to_fix>
{paragraph_text}
</paragraph_to_fix>

Rewrite ONLY the paragraph to be fully self-contained. Include any referenced information inline. Keep the same meaning and level of detail. Do not add new information that wasn't in the surrounding context.

Return ONLY the rewritten paragraph text, no explanation.
"""

FIX_GENERIC_HEADING = """\
The following heading is too generic for a RAG knowledge base. Generate a more descriptive heading based on the content that follows it.

<current_heading>
{heading_text}
</current_heading>

<content_below>
{content_below}
</content_below>

Return ONLY the new heading text (no markdown #, no quotes, just the text). Keep it under 10 words.
"""

FIX_LONG_PARAGRAPH = """\
The following paragraph is too long ({word_count} words) for effective RAG retrieval. Split it into 2-4 focused paragraphs.

Each new paragraph should:
- Be self-contained (understandable without reading the others)
- Focus on one main point
- Be 50-200 words

<paragraph>
{paragraph_text}
</paragraph>

Return the split paragraphs separated by exactly one blank line. No other text or explanation.
"""

FIX_UNDEFINED_ACRONYM = """\
The acronym "{acronym}" is used in this document but never defined. Based on the context, what does it stand for?

<context>
{context}
</context>

Return ONLY the full form, e.g. "Individualized Education Program" (no acronym, no parentheses, just the expansion).
If you cannot determine the meaning from context, return "UNKNOWN".
"""

GENERATE_FILENAME = """\
Generate a descriptive filename for this document based on its content. The filename should:
- Be lowercase with hyphens between words
- Include the main topic and any relevant qualifiers (grade level, audience, etc.)
- Be 3-6 words long
- NOT include the file extension

<document_summary>
{summary}
</document_summary>

<current_filename>
{current_filename}
</current_filename>

Return ONLY the new filename (no extension, no path, no explanation). Example: "adding-fractions-grade5-lesson"
"""

# ---------------------------------------------------------------------------
# Document splitting prompt
# ---------------------------------------------------------------------------

SPLIT_DOCUMENT = """\
This document covers multiple distinct topics and should be split into focused single-topic files for better RAG retrieval.

<document_headings>
{headings}
</document_headings>

Identify the natural split points. Return ONLY valid JSON:
{{
  "splits": [
    {{
      "title": "<descriptive title for this section>",
      "start_heading": "<exact heading text where this section starts>",
      "suggested_filename": "<lowercase-hyphenated-name>"
    }}
  ]
}}

Each split should cover one coherent topic. Aim for 2-5 splits.
"""
