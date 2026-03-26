"""Configuration and API key management.

Loads .env file from the project root if python-dotenv is installed.
Falls back silently if not — env vars and CLI flags still work.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load .env from project root (if python-dotenv is available)
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class Config:
    """Application configuration."""

    # LLM (Anthropic Claude)
    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096

    # Processing defaults
    output_dir: str = "./fixed"
    convert_to_markdown: bool = True  # Convert DOCX/PDF → MD
    concurrency: int = 5  # Max parallel LLM calls
    folder_hints: str = ""  # Domain-specific guidance for folder organization

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            llm_model=os.environ.get("KB_PREP_MODEL", "claude-sonnet-4-20250514"),
        )

    def with_overrides(self, **kwargs) -> "Config":
        """Return a new Config with specified overrides.

        Validates that all keys are actual Config fields to catch typos.
        """
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(self)}
        invalid = set(kwargs.keys()) - valid_fields
        if invalid:
            raise ValueError(f"Invalid config fields: {invalid}. Valid: {sorted(valid_fields)}")
        return dataclasses.replace(self, **{k: v for k, v in kwargs.items() if v is not None})
