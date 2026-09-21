"""
All decisions of an application in ONE place: question, severity, privacy, bands,
calibrated model version. The router picks the backend by `privacy` (LOCAL must never
go to the cloud), falls back to local on a cloud outage, and demotes the band when the
responding model isn't the one the bands were calibrated for. A CLOUD spec that gets
answered via the local fallback backend is ALWAYS considered uncalibrated
(`Verdict.calibrated=False`) - regardless of whether `calibrated_model` is set, because
the local model is never the calibrated one.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from jevkit.answers import Answer
from jevkit.client import Client, Decision, JevUnavailable
from jevkit.gate import DEFAULT_BANDS, Band, Bands, Severity, band, demote
from jevkit.questions import Question


class Privacy(Enum):
    LOCAL = "local"   # state must not leave the device
    CLOUD = "cloud"


@dataclass(frozen=True)
class DecisionSpec:
    id: str
    question: Question
    severity: Severity = Severity.READ
    privacy: Privacy = Privacy.CLOUD
    bands: Bands | None = None
    calibrated_model: str | None = None   # prefix, e.g. "typesafe/jev-1.13"

    @property
    def effective_bands(self) -> Bands:
        return self.bands or DEFAULT_BANDS[self.severity]


@dataclass(frozen=True)
class Verdict:
    spec_id: str
    answer: Answer
    band: Band
    model: str
    calibrated: bool


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, DecisionSpec] = {}

    def register(self, spec: DecisionSpec) -> DecisionSpec:
        if spec.id in self._specs:
            raise ValueError(f"decision {spec.id!r} is already registered")
        self._specs[spec.id] = spec
        return spec

    def get(self, spec_id: str) -> DecisionSpec:
        return self._specs[spec_id]

    def __contains__(self, spec_id: object) -> bool:
        return spec_id in self._specs

    def __iter__(self) -> Iterator[DecisionSpec]:
        return iter(self._specs.values())

    def questions(self, ids: Sequence[str]) -> dict[str, Question]:
        return {i: self._specs[i].question for i in ids}


class Router:
    def __init__(self, registry: Registry, *, cloud: Client | None = None,
                 local: Client | None = None) -> None:
        self.registry = registry
        self.cloud = cloud
        self.local = local

    def _verdicts(self, decision: Decision, ids: Sequence[str], *,
                  fallback: bool) -> dict[str, Verdict]:
        out: dict[str, Verdict] = {}
        for i in ids:
            spec = self.registry.get(i)
            ans = decision[i]
            # A fallback backend never answers as the calibrated model, no matter
            # what `calibrated_model` says: the bands then fundamentally don't fit.
            calibrated = not fallback and (spec.calibrated_model is None or
                                           decision.model.startswith(spec.calibrated_model))
            b = band(ans, spec.effective_bands)
            if not calibrated:
                b = demote(b)
            out[i] = Verdict(i, ans, b, decision.model, calibrated)
        return out

    async def _run(self, state, ids: Sequence[str], privacy: Privacy) -> dict[str, Verdict]:
        qs = self.registry.questions(ids)
        if privacy is Privacy.LOCAL:
            if self.local is None:
                raise JevUnavailable("no local backend for LOCAL decisions")
            return self._verdicts(await self.local.decide(state, qs), ids, fallback=False)
        if self.cloud is not None:
            try:
                return self._verdicts(await self.cloud.decide(state, qs), ids, fallback=False)
            except JevUnavailable:
                if self.local is None:
                    raise
        if self.local is None:
            raise JevUnavailable("no backend available")
        # CLOUD spec, but answered via the local backend (fallback after
        # JevUnavailable, or because no cloud client is configured at all) ->
        # never calibrated, band gets demoted.
        return self._verdicts(await self.local.decide(state, qs), ids, fallback=True)

    async def decide(self, state, ids: Sequence[str]) -> dict[str, Verdict]:
        specs = [self.registry.get(i) for i in ids]   # KeyError for unknown IDs, before any call
        out: dict[str, Verdict] = {}
        for privacy in (Privacy.CLOUD, Privacy.LOCAL):
            group = [s.id for s in specs if s.privacy is privacy]
            if group:
                out.update(await self._run(state, group, privacy))
        return out
