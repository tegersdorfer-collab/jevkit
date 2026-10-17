"""Shared helpers for the standalone Jev upgrade benchmark scripts."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevkit.backends import Backend, OpenRouterBackend, TypeSafeBackend
from jevkit.models import JEV_1_13, JEV_1_14, KNOWN, PINNED, JevModel

BackendFactory = Callable[[JevModel, str], Backend]
DEFAULT_MODELS = tuple(dict.fromkeys((KNOWN[-2], PINNED))) if len(KNOWN) > 1 else (PINNED,)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_MODEL_ALIASES: dict[str, JevModel] = {}
for _model in (*KNOWN, JEV_1_13, JEV_1_14):
    for _alias in (
        f"jev_{_model.version.replace('.', '_')}",
        _model.version,
        _model.typesafe_id,
        _model.openrouter_id,
    ):
        _MODEL_ALIASES[_alias.lower()] = _model


def parse_models(value: str | Sequence[JevModel] | None) -> tuple[JevModel, ...]:
    """Parse comma-separated model labels or validate a sequence of JevModel values."""
    if value is None:
        return DEFAULT_MODELS
    if not isinstance(value, str):
        models = tuple(value)
        if not models or any(not isinstance(model, JevModel) for model in models):
            raise ValueError("models must contain one or more JevModel values")
    else:
        labels = [item.strip() for item in value.split(",") if item.strip()]
        if not labels:
            raise ValueError("provide at least one model")
        try:
            models = tuple(_MODEL_ALIASES[label.lower()] for label in labels)
        except KeyError as exc:
            raise ValueError(f"unknown Jev model {exc.args[0]!r}; use a known model id or version") from exc
    unique = tuple(dict.fromkeys(models))
    if not unique:
        raise ValueError("provide at least one model")
    return unique


def provider_model_id(model: JevModel, provider: str) -> str:
    """Return the pinned identifier for a provider."""
    if provider == "typesafe":
        return model.typesafe_id
    if provider == "openrouter":
        return model.openrouter_id
    raise ValueError("provider must be 'typesafe' or 'openrouter'")


def load_env_file(path: str | Path | None) -> dict[str, str]:
    """Read simple KEY=VALUE env-file lines without mutating the process environment."""
    if path is None:
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid env assignment on line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def create_backend(model: JevModel, provider: str, env_values: dict[str, str] | None = None) -> Backend:
    """Create the real HTTP backend for a provider without exposing its API key."""
    if provider == "typesafe":
        backend_type = TypeSafeBackend
    elif provider == "openrouter":
        backend_type = OpenRouterBackend
    else:
        raise ValueError("provider must be 'typesafe' or 'openrouter'")
    env_var = backend_type.env_var
    key = (env_values or {}).get(env_var) or os.environ.get(env_var)
    if not key:
        raise ValueError(f"set ${env_var} or pass --env-file with {env_var}=...")
    return backend_type(api_key=key, retries=4)


def read_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(case, dict) for case in data):
        raise ValueError("cases file must contain a JSON array of objects")
    return data


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def default_results_path(stem: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("bench/results") / f"{stem}-{timestamp}.jsonl"


def make_parser_models_default() -> str:
    return ",".join(f"JEV_{model.version.replace('.', '_')}" for model in DEFAULT_MODELS)


_HTTP_STATUS = re.compile(r"\bHTTP (\d{3})\b")


def safe_error(exc: BaseException) -> str:
    """Error type plus HTTP status only: provider bodies can echo request headers (API keys)."""
    match = _HTTP_STATUS.search(str(exc))
    return f"{type(exc).__name__}" + (f" HTTP {match.group(1)}" if match else "")
