"""Pinned Jev model ids and provider-independent version parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JevModel:
    version: str
    typesafe_id: str
    openrouter_id: str


JEV_1_13 = JevModel("1.13", "jev-1.13.0", "typesafe/jev-1.13-20260917")
JEV_1_14 = JevModel("1.14", "jev-1.14.0", "typesafe/jev-1.14-20261014")
PINNED = JEV_1_14
KNOWN = (JEV_1_13, JEV_1_14)

_MODEL_VERSION = re.compile(
    r"^(?:(?:~?typesafe/)?jev-)?(?P<major>\d+)\.(?P<minor>\d+)(?:\.\d+)?(?:-\d{8})?$"
)


def model_version(model_id: str) -> str | None:
    """Return the major.minor version from a Jev model id or bare version."""
    match = _MODEL_VERSION.fullmatch(model_id)
    if match is None:
        return None
    return f"{match.group('major')}.{match.group('minor')}"
