import os
from config import Config


def test_config_defaults():
    """Config has expected default values."""
    cfg = Config()
    assert cfg.concurrency == 5
    assert cfg.llm_model == "claude-sonnet-4-20250514"
    assert cfg.output_dir == "./fixed"


def test_config_with_overrides():
    """with_overrides sets concurrency."""
    cfg = Config().with_overrides(concurrency=10)
    assert cfg.concurrency == 10


def test_config_invalid_override_raises():
    """with_overrides rejects unknown fields."""
    import pytest
    with pytest.raises(ValueError, match="Invalid config fields"):
        Config().with_overrides(fake_field="nope")
