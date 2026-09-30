from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None, mode: str = "smoke") -> dict:
    """Charge la configuration et fusionne les paramètres du mode demandé."""
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    result = deepcopy(config)
    modes = result.pop("modes")
    if mode not in modes:
        raise ValueError(f"Mode inconnu {mode!r}; modes possibles: {list(modes)}")
    result.update(modes[mode])
    result["mode"] = mode
    result["root"] = str(ROOT)
    return result
