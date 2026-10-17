"""Run labeled decision cases against pinned Jev model versions."""
from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench._common import (
    BackendFactory,
    create_backend,
    default_results_path,
    load_env_file,
    make_parser_models_default,
    parse_models,
    provider_model_id,
    read_cases,
    safe_error,
    write_jsonl,
)
from jevkit.backends import Backend
from jevkit.models import JevModel


def _probability(probabilities: Any, expected: Any) -> float:
    if not isinstance(probabilities, dict):
        raise ValueError("answer probabilities must be an object")
    candidates = [expected, str(expected)]
    if isinstance(expected, (int, float)) and float(expected).is_integer():
        candidates.append(str(int(expected)))
    for candidate in candidates:
        if candidate in probabilities:
            return float(probabilities[candidate])
    return 0.0


def _answer_fields(answer: dict[str, Any], expected: Any) -> dict[str, Any]:
    answer_type = answer.get("type")
    if answer_type == "noul":
        p_true = float(answer["noul"])
        if not isinstance(expected, bool):
            raise ValueError("noul decision cases must have a boolean expected value")
        pred = p_true >= 0.5
        return {
            "pred": pred,
            "correct": pred == expected,
            "conf": abs(p_true - 0.5) * 2,
            "p_true": p_true,
            "p_correct": p_true if expected else 1 - p_true,
            "probs": None,
        }
    if answer_type == "score":
        # Score answers are a continuous expectation; exact-match accuracy and gold-level log loss
        # would be inconsistent. Not supported until a regression metric is defined.
        raise ValueError("score questions are not supported by the decision benchmark")
    if answer_type == "choice":
        pred = answer["choice"]
        probabilities = answer.get("probabilities")
        return {
            "pred": pred,
            "correct": pred == expected,
            "conf": float(answer["confidence"]),
            "p_true": None,
            "p_correct": _probability(probabilities, expected),
            "probs": probabilities,
        }
    raise ValueError(f"unsupported answer type {answer_type!r}")


async def run_decisions(
    cases: Sequence[dict[str, Any]],
    *,
    models: Sequence[JevModel] | str | None = None,
    provider: str = "typesafe",
    reps: int = 3,
    concurrency: int = 8,
    out_path: str | Path | None = None,
    env_values: dict[str, str] | None = None,
    backend_factory: BackendFactory | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each case/model/repetition and write compare.py-compatible JSONL rows."""
    selected = parse_models(models)
    if reps < 1:
        raise ValueError("reps must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    factory = backend_factory or (
        lambda model, provider_name: create_backend(model, provider_name, env_values)
    )
    backends: dict[JevModel, Backend] = {model: factory(model, provider) for model in selected}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: dict[str, Any], model: JevModel, rep: int) -> dict[str, Any]:
        model_id = provider_model_id(model, provider)
        base = {"id": case["id"], "kind": case.get("kind"), "model": model_id,
                "rep": rep, "expected": case["expected"]}
        started = time.perf_counter()
        try:
            question = case["question"]
            if not isinstance(question, dict) or "type" not in question:
                raise ValueError("case question must be an object with a type")
            async with semaphore:
                started = time.perf_counter()
                response = await backends[model].ask(
                    case["state"], {"q": question}, model=model_id, timeout_s=30.0
                )
            elapsed = time.perf_counter() - started
            answer = response.answers["q"]
            fields = _answer_fields(answer, case["expected"])
            return {**base, "served": response.model, **fields, "latency": elapsed}
        except Exception as exc:
            return {**base, "error": safe_error(exc), "latency": time.perf_counter() - started}

    rows = await asyncio.gather(
        *(one(case, model, rep)
          for rep in range(reps)
          for model in selected
          for case in cases)
    )
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON array in the generic decision case format")
    parser.add_argument("--models", default=make_parser_models_default(),
                        help="comma-separated Jev ids or versions")
    parser.add_argument("--provider", choices=("typesafe", "openrouter"), default="typesafe")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, help="JSONL output path")
    parser.add_argument("--env-file", type=Path, help="read provider API keys from KEY=VALUE lines")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = read_cases(args.cases)
    output = args.out or default_results_path(args.cases.stem)
    env_values = load_env_file(args.env_file)
    rows = asyncio.run(run_decisions(
        cases, models=args.models, provider=args.provider, reps=args.reps,
        concurrency=args.concurrency, out_path=output, env_values=env_values,
    ))
    errors = sum("error" in row for row in rows)
    print(f"{len(rows)} rows, {errors} errors; results: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
