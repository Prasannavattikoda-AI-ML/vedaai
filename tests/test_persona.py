import pytest
import textwrap
from knowledge.persona_engine import PersonaEngine

PERSONA_YAML = textwrap.dedent("""
    name: "Pandu"
    tone: "friendly, professional"
    language: "English"
    boundaries:
      - "Don't share phone number"
    rules:
      - trigger: "availability"
        response_hint: "Check calendar"
      - trigger: "pricing|cost"
        response_hint: "Redirect to email"
""")

@pytest.fixture
def engine(tmp_path):
    f = tmp_path / "persona.yaml"
    f.write_text(PERSONA_YAML)
    return PersonaEngine(str(f))

def test_name(engine):
    assert engine.name == "Pandu"

def test_build_prompt_contains_name(engine):
    prompt = engine.build_prompt("hello")
    assert "Pandu" in prompt

def test_build_prompt_contains_boundary(engine):
    prompt = engine.build_prompt("hello")
    assert "Don't share phone number" in prompt

def test_trigger_hint_included(engine):
    prompt = engine.build_prompt("what is your availability?")
    assert "Check calendar" in prompt

def test_no_hint_when_no_trigger_match(engine):
    prompt = engine.build_prompt("hello how are you")
    assert "Check calendar" not in prompt
    assert "Redirect to email" not in prompt

def test_multiple_triggers_fire(engine):
    prompt = engine.build_prompt("what is your availability and pricing?")
    assert "Check calendar" in prompt
    assert "Redirect to email" in prompt

def test_trigger_case_insensitive(engine):
    prompt = engine.build_prompt("AVAILABILITY please")
    assert "Check calendar" in prompt

def test_pipe_trigger_alternation(engine):
    prompt = engine.build_prompt("what is the cost?")
    assert "Redirect to email" in prompt
