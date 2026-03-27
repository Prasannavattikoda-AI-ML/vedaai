import os
import re
import yaml
from dotenv import load_dotenv

load_dotenv()

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve(value):
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve(raw)
