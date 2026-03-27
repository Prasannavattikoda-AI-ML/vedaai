import os
import pytest
from unittest.mock import patch
from core.config import load_settings

SAMPLE_YAML = """
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-sonnet-4-6"
owner:
  telegram: "999"
  whatsapp: "+91111"
nested:
  list:
    - "${SOME_VAR}"
    - "plain"
"""

@pytest.fixture
def yaml_file(tmp_path):
    f = tmp_path / "settings.yaml"
    f.write_text(SAMPLE_YAML)
    return str(f)

def test_plain_value_unchanged(yaml_file):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["model"] == "claude-sonnet-4-6"

def test_env_var_substituted(yaml_file):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["api_key"] == "sk-test"

def test_missing_env_var_resolves_to_empty(yaml_file):
    with patch.dict(os.environ, {}, clear=True):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["api_key"] == ""

def test_nested_list_resolved(yaml_file):
    with patch.dict(os.environ, {"SOME_VAR": "hello"}):
        cfg = load_settings(yaml_file)
    assert cfg["nested"]["list"][0] == "hello"
    assert cfg["nested"]["list"][1] == "plain"

def test_empty_yaml_returns_empty_dict(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("")
    cfg = load_settings(str(f))
    assert cfg == {}

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Settings file not found"):
        load_settings(str(tmp_path / "nonexistent.yaml"))
