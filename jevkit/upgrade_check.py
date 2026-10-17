"""Check which Jev version each configured provider serves through its latest alias."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable

from jevkit.backends import HTTPBackend, OpenRouterBackend, TypeSafeBackend
from jevkit.models import PINNED, model_version

_STATE = "Turn on the desk lamp."
_QUESTIONS = {
    "version_check": {"type": "noul", "instructions": "Does this sentence request an action?"},
}


async def check_latest(backends: Iterable[HTTPBackend], *, output: Callable[[str], None] = print) -> int:
    """Probe injected backends and return 0=current, 3=newer or unknown version served, or 1=error."""
    configured = list(backends)
    if not configured:
        output("Error: no API key configured; set TYPESAFE_API_KEY or OPENROUTER_API_KEY.")
        return 1

    drift = False
    errors = False
    for backend in configured:
        provider = getattr(backend, "provider_name", type(backend).__name__)
        alias = getattr(backend, "latest_model", None)
        if alias is None:
            output(f"{provider}: error: backend does not define a latest alias")
            errors = True
            continue
        try:
            response = await backend.ask(_STATE, _QUESTIONS, model=alias, timeout_s=8.0)
        except Exception as exc:
            output(f"{provider}: error checking {alias}: {exc}")
            errors = True
            continue

        served_version = model_version(response.model)
        seen = f"{provider}: {alias} resolved to {response.model} ({served_version or 'unknown version'}); "
        if served_version == PINNED.version:
            output(seen + f"jevkit pins {PINNED.version}.")
        elif served_version and _key(served_version) < _key(PINNED.version):
            # The provider's alias lags behind the pin (TypeSafe kept jev-latest on 1.13 after 1.14 shipped).
            output(seen + f"behind the pinned {PINNED.version}; nothing to upgrade.")
        else:
            output(seen + f"jevkit pins {PINNED.version}. See docs/upgrading-jev.md before upgrading.")
            drift = True
    # A newer model somewhere is the actionable signal, even if another provider errored.
    if drift:
        return 3
    return 1 if errors else 0


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def main() -> int:
    """Probe only providers with an API key in the environment."""
    configured: list[HTTPBackend] = []
    if os.environ.get(TypeSafeBackend.env_var):
        configured.append(TypeSafeBackend())
    if os.environ.get(OpenRouterBackend.env_var):
        configured.append(OpenRouterBackend())
    return asyncio.run(check_latest(configured))


if __name__ == "__main__":
    raise SystemExit(main())
