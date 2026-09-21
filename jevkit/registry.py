"""
Alle Entscheidungen einer Anwendung an EINER Stelle: Frage, Schwere, Datenschutz,
Bänder, kalibrierte Modellversion. Der Router wählt das Backend nach `privacy`
(LOCAL darf nie in die Cloud), fällt bei Cloud-Ausfall auf lokal zurück und demotet
das Band, wenn das antwortende Modell nicht das ist, für das die Bänder kalibriert
wurden. Ein CLOUD-Spec, das über das lokale Fallback-Backend beantwortet wird, gilt
dabei IMMER als unkalibriert (`Verdict.calibrated=False`) — unabhängig davon, ob
`calibrated_model` gesetzt ist, weil das lokale Modell nie das kalibrierte ist.
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
    LOCAL = "local"   # State darf das Gerät nicht verlassen
    CLOUD = "cloud"


@dataclass(frozen=True)
class DecisionSpec:
    id: str
    question: Question
    severity: Severity = Severity.READ
    privacy: Privacy = Privacy.CLOUD
    bands: Bands | None = None
    calibrated_model: str | None = None   # Präfix, z.B. "typesafe/jev-1.13"

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
            raise ValueError(f"Entscheidung {spec.id!r} ist schon registriert")
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
            # Ein Fallback-Backend beantwortet nie das kalibrierte Modell, egal
            # was `calibrated_model` sagt: die Bänder passen dann grundsätzlich nicht.
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
                raise JevUnavailable("kein lokales Backend für LOCAL-Entscheidungen")
            return self._verdicts(await self.local.decide(state, qs), ids, fallback=False)
        if self.cloud is not None:
            try:
                return self._verdicts(await self.cloud.decide(state, qs), ids, fallback=False)
            except JevUnavailable:
                if self.local is None:
                    raise
        if self.local is None:
            raise JevUnavailable("kein Backend verfügbar")
        # CLOUD-Spec, aber über das lokale Backend beantwortet (Fallback nach
        # JevUnavailable oder weil gar kein Cloud-Client konfiguriert ist) →
        # nie kalibriert, Band wird demotet.
        return self._verdicts(await self.local.decide(state, qs), ids, fallback=True)

    async def decide(self, state, ids: Sequence[str]) -> dict[str, Verdict]:
        specs = [self.registry.get(i) for i in ids]   # KeyError für unbekannte IDs, vor jedem Call
        out: dict[str, Verdict] = {}
        for privacy in (Privacy.CLOUD, Privacy.LOCAL):
            group = [s.id for s in specs if s.privacy is privacy]
            if group:
                out.update(await self._run(state, group, privacy))
        return out
