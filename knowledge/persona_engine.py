import re
import yaml


class PersonaEngine:
    def __init__(self, persona_path: str):
        with open(persona_path) as f:
            data = yaml.safe_load(f)
        if "name" not in data:
            raise ValueError(f"Persona YAML at {persona_path} is missing required field 'name'")
        self._name: str = data["name"]
        self._tone: str = data.get("tone", "friendly")
        self._language: str = data.get("language", "English")
        self._boundaries: list[str] = data.get("boundaries", [])
        self._rules: list[dict] = data.get("rules", [])

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, incoming_text: str) -> str:
        boundaries_text = "\n".join(f"- {b}" for b in self._boundaries)
        hints = self._matching_hints(incoming_text)
        hints_text = (
            "\nContextual guidance:\n" + "\n".join(f"- {h}" for h in hints)
        ) if hints else ""

        return (
            f"You are {self.name}, a personal AI assistant.\n"
            f"Tone: {self._tone}\n"
            f"Language: {self._language}\n"
            f"Boundaries (never violate these):\n{boundaries_text}"
            f"{hints_text}"
        )

    def _matching_hints(self, text: str) -> list[str]:
        """Match triggers using re.search with IGNORECASE. Pipe = regex alternation."""
        hints = []
        for rule in self._rules:
            if re.search(rule["trigger"], text, re.IGNORECASE):
                hints.append(rule["response_hint"])
        return hints
