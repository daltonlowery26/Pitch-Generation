from pathlib import Path

import yaml


def load_config(name: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / f"{name}.yml"
    with open(path) as f:
        return yaml.safe_load(f)
