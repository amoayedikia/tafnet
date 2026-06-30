"""
Config loader.

Loads YAML, supports `--override key.path=value` style overrides, and exposes
nested keys as attributes for clean access (e.g. `config.phase4.epochs`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class AttrDict(dict):
    """Nested dict whose keys are accessible as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


def _parse_scalar(value: str) -> Any:
    """Parse CLI override scalar into bool/int/float/None/str."""
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_config(
    yaml_path: str | os.PathLike,
    overrides: Iterable[str] | None = None,
) -> AttrDict:
    """
    Load a YAML config file and apply `key.path=value` overrides.

    Parameters
    ----------
    yaml_path : str or PathLike
        Path to the YAML file.
    overrides : iterable of str, optional
        Each element of the form 'a.b.c=value'.
    """
    yaml_path = Path(yaml_path)
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override '{ov}' must be of form key.path=value")
            k, v = ov.split("=", 1)
            _set_nested(cfg, k.strip(), _parse_scalar(v.strip()))

    return _to_attrdict(cfg)


def dump_config(cfg: Mapping, out_path: str | os.PathLike) -> None:
    """Write a config back to YAML (handy for archiving the exact run config)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_plain(o: Any) -> Any:
        if isinstance(o, AttrDict):
            return {k: _to_plain(v) for k, v in o.items()}
        if isinstance(o, dict):
            return {k: _to_plain(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_plain(v) for v in o]
        return o

    with out_path.open("w") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False)
