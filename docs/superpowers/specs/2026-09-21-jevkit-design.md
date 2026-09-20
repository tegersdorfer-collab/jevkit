# jevkit – Design (Decision Kernel für TypeSafe Jev)

Stand: 2026-09-21. Quelle: docs.typesafe.ai (Konzepte, API, Patterns, Cookbooks, Jaggedness jev-1.13) plus Mantis-Benchmark (bench/jev, 86/92, Brier 0,046).

## Was Jev ist

- Kein Generator. RLCD-trainiertes Modell, das für jede Frage eine Wahrscheinlichkeitsverteilung über *vom Aufrufer vorgegebene* Optionen liefert. Nie ein Wert außerhalb.
- Ein Request = ein `state` (str | JSON-Objekt | Array) + `questions` (Map id → Frage). Alle Fragen sehen denselben State, werden parallel und isoliert ausgewertet. State wird einmal bezahlt → Fan-out fast gratis (13 Fragen in einem Call: 12,2× billiger, 10× schneller als 13 Calls).
- Primitive: **Noul** (P(ja)), **Choice** (≤255 Optionen; `choice`, `probabilities`, `confidence`), **Score** (2–10 geordnete Level; `score` als Erwartungswert, `legend`, `probabilities`, `confidence`).
- Confidence = Spitzigkeit: `(k·p_max − 1)/(k − 1)`. Noul hat keine.
- Limits: 64k Kontext/Request, 32k State + längste Frage. $0,042/1M Input, Output frei. Rate-Limits dynamisch. Fehler: 401/422/429/529 (+ Cloudflare 520 beobachtet).
- Instructions und Criteria dürfen JSON sein (Objekte mit `what`/`not_for`/`examples`, Level mit `summary`/`signals`, Dot-Referenzen in den State).
- Fast deterministisch (Std 0,01 über Wiederholungen), aber Nouls nahe 0,5 kippen.
- Endpunkte mit identischem Body: `POST https://api.typesafe.ai/v1/systemone` (Modell `jev-latest`), `POST https://openrouter.ai/api/alpha/decisions` (Modell `~typesafe/jev-latest`). Antwort: `{model, answers, usage}`.

## Schwächen (offiziell + extern) → Gegenmaßnahme

| # | Schwäche | Gegenmaßnahme im Kernel |
|---|---|---|
| 1 | Liest wörtlich | Exakte Bedingung in `instructions`, Grenzfälle in `criteria` (strukturiert) |
| 2 | Kein Rechnen/Zählen | `compose.count`: eine Noul pro Element, Summe im Code; Zahlen vorab in Buckets |
| 3 | Datum = Text | `state.relative_days` liefert Label, Vergleich im Code |
| 4 | Negation/Indirektion | Nur positive Fragen; Gegenfrage als Konsistenz-Check (`gate.consistent`) |
| 5 | Irrelevanter State | `state.project` (Feld-Projektion), Bin-Packing lieber schlank als fett |
| 6 | Keine Injection-Abwehr | `state.untrusted` + Guard-Noul im selben Fan-out; Guard positiv → alles auf ESCALATE |
| 7 | P(A)+P(¬A) ≠ 1 | Konsistenz-Check demotet auf CONFIRM |
| 8 | Keine Generierung | `compose.extract`: Kandidaten aus Code, Jev wählt (Choice) + „stated“-Noul |
| 9 | Deutsch schwächer | Inhalt deutsch, Instructions/Criteria englisch |
| 10 | Keine Begründung | Nicht im Kernel; Aufrufer fragt bei Bedarf ein LLM |
| 11 | Accuracy < Frontier | Drei Bänder ACT/CONFIRM/ESCALATE statt Schwelle; Bänder pro Severity |
| 12 | 520/529/dynamische Limits | Retry mit Backoff im Timeout-Budget, Circuit-Breaker, `JevUnavailable` |
| 13 | Versionswechsel verschiebt Bänder | `Decision.model` geloggt; Registry warnt bei anderer Version als kalibriert |

## Schichten

0. **Registry** (`registry.py`): `DecisionSpec(id, question, severity, privacy, bands, calibrated_model)`; `Router` wählt Backend nach `privacy` (CLOUD → Jev, LOCAL → lokaler Backend).
1. **State Builder** (`state.py`): `project`, `bucket`, `count_bucket`, `relative_days`, `untrusted`, `GUARD` Frage, `with_guard`.
2. **Compose** (`compose.py`, `tools.py`, `taxonomy.py`): count, composite, extract, function_call, beam_search.
3. **Client** (`client.py`, `cache.py`, `backends.py`): ein Call pro State, Cache auf Hash, Retry/Breaker/Timeout, Backends TypeSafe / OpenRouter / Prompt (lokales LLM, unkalibriert) / Static (Tests).
4. **Gate** (`gate.py`): `Band`, `Bands`, `Severity`, `band()`, `consistent()`.
5. **Escalation**: Ergebnis der Bänder; Kaskade ist Sache des Aufrufers (Kernel liefert Band + Antwort).
6. **Log & Kalibrierung** (`log.py`, `calibrate.py`): JSONL-Log je Frage, Outcome nachtragen, Brier/ECE/Accuracy je Frage, `suggest_bands` aus gelabelten Outcomes.

## Nicht in v1

Autoresearch-Schleife (LLM schlägt Fragen vor, Klassifikator bewertet), Begründungs-LLM, Ollama-Adapter mit kalibrierten Wahrscheinlichkeiten.

## Mantis-Integration

`core/decide.py` behält seine Schnittstelle (`Noul`, `Choice`, `Answer`, `decide()`, `JevUnavailable`, `enabled()`, `reset_for_tests()`, `_transport`) und delegiert an `jevkit.Client` mit `OpenRouterBackend`. `core/decisions.py` bekommt Bänder (statt `SURE`), Guard auf dem Voice-Transkript und ein `DecisionLog` unter `data/decisions.jsonl`.
