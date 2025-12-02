import pytest
import os
from src import labeler

def test_model_env_var_is_set():
    """Verify that the LM_STUDIO_MODEL environment variable is loaded and accessible."""
    assert "LM_STUDIO_MODEL" in os.environ
    assert labeler.LM_STUDIO_MODEL is not None
    assert len(labeler.LM_STUDIO_MODEL) > 0
