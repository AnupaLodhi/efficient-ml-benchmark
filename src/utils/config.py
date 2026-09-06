"""Centralized configuration loading.

All experiment scripts should obtain their parameters through `load_config`
rather than hard-coding values. This keeps every experiment reproducible from
a single YAML file and makes it possible to override individual fields from
the command line without touching source code.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class Config(dict):
    """A dict that also supports attribute-style access, recursively.

    Example:
        cfg = load_config()
        cfg.training.lr          # attribute access
        cfg["training"]["lr"]    # still works like a normal dict
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configs/config.yaml (or a custom path) into a Config object.

    Args:
        path: path to a YAML config file.
        overrides: optional dict of dotted-key overrides, e.g.
            {"training.epochs": 5, "project.seed": 123}.

    Returns:
        Config object with dict- and attribute-style access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(copy.deepcopy(raw))

    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(cfg, dotted_key, value)

    return cfg


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    """Persist the *effective* config (after overrides) alongside results.

    Every experiment run should dump its resolved config next to its outputs
    so that any figure or number can be traced back to the exact settings
    that produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
