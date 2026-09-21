# jevkit

Decision Kernel für [TypeSafe Jev](https://docs.typesafe.ai) (System-One-Modell): typisierte
Fragen rein, kalibrierte Wahrscheinlichkeiten raus — und der Code behält die Kontrolle.

Jev generiert keinen Text. Es beantwortet **Noul** (ja/nein → P), **Choice** (Option +
Verteilung + Confidence) und **Score** (Level + Verteilung + Confidence) parallel gegen
einen State. jevkit baut darum herum, was Jev laut eigener Doku nicht kann: zählen,
rechnen, Daten vergleichen, Fremdtext misstrauen, Antworten begründen.

## Installation

```bash
pip install -e /Users/timoegersdorfer/jevkit
export OPENROUTER_API_KEY=...      # oder TYPESAFE_API_KEY für den Direktzugang
```

## Minimal

```python
import asyncio
from jevkit import Client, OpenRouterBackend, Noul, Choice, band, DEFAULT_BANDS, Severity

client = Client(OpenRouterBackend(), timeout_s=3.0)

async def main():
    d = await client.decide(
        {"message": "Mach die Schreibtischlampe an"},
        {"action": Noul("Does the message ask the assistant to perform an action?"),
         "device": Choice("Which device is meant?", {"lamp": "desk lamp", "fan": "fan", "__none__": "none"})},
    )
    print(d["action"].p, d["device"].choice, band(d["device"], DEFAULT_BANDS[Severity.REVERSIBLE]))

asyncio.run(main())
```

## Bausteine

| Modul | Zweck | Neutralisiert |
|---|---|---|
| `state` | `project`, `bucket`, `count_bucket`, `relative_days`, `untrusted` + `GUARD` | Distraktoren, kein Rechnen, Datum als Text, Prompt-Injection |
| `compose` | `count`, `composite`, `extract` (Choice über Kandidaten + „stated"-Noul) | Zählen, zusammengesetzte Urteile, keine Generierung |
| `tools` | `call_questions` / `resolve_call` — Function Calling, Confidence = Minimum | Ein falsches Argument verdirbt den Call |
| `taxonomy` | `beam_search` über Baum, geometrischer Pfad-Score | Frühe ambige Entscheidung (Greedy) |
| `client` | ein Call pro State, Zeitbudget, Breaker, Typ-Prüfung, Cache | 520/529, Schema-Drift, Kosten |
| `gate` | `Band` ACT / CONFIRM / ESCALATE, `Bands` je `Severity`, `check_pairs` | Accuracy < Frontier, P(A)+P(¬A) ≠ 1 |
| `registry` | `DecisionSpec` (Frage, Severity, Privacy, Bänder, kalibriertes Modell), `Router` | Versionsdrift, Privacy |
| `log`/`calibrate` | JSONL-Log, Outcomes, Brier/ECE, `suggest_bands` | Bänder aus dem Bauch |
| `prompt_backend` | Jev-Emulation über lokales LLM (unkalibriert) | Fallback / lokale Daten |

## Regeln (aus der Jev-Doku, jev-1.13)

1. Inhalt darf deutsch sein, Instructions und Criteria englisch.
2. Nur positive, wörtliche Fragen; Grenzfälle in `criteria` mit `what` / `not_for` / `examples`.
3. Alles Numerische vorher in Buckets; Zählen und Vergleichen im Code.
4. Nur benötigte Felder in den State (`project`).
5. Fremdtext über `untrusted()` + `with_guard()`; schlägt der Guard an → alles ESCALATE.
6. Nie eine Schwelle, immer drei Bänder; destruktive Aktionen brauchen höhere.
7. `Decision.model` loggen; ändert sich die Version, Bänder neu messen.

Design: `docs/superpowers/specs/2026-09-21-jevkit-design.md`.
