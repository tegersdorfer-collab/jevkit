"""Compare Jev guard scores across pinned models and guard wordings."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
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
from jevkit.state import GUARD, SELF_CLAIM, untrusted


def load_guard_from_ref(ref: str) -> Any:
    """Load a historical GUARD definition from git into a temporary isolated module."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "show", f"{ref}:jevkit/state.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    name = f"jevkit_bench_state_{hashlib.sha256(ref.encode()).hexdigest()[:12]}"
    module = ModuleType(name)
    module.__file__ = f"<git:{ref}:jevkit/state.py>"
    sys.modules[name] = module
    try:
        exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
        return module.GUARD
    finally:
        sys.modules.pop(name, None)


async def run_guard(
    cases: Sequence[dict[str, Any]],
    *,
    models: Sequence[JevModel] | str | None = None,
    provider: str = "typesafe",
    reps: int = 3,
    concurrency: int = 8,
    out_path: str | Path | None = None,
    split: str = "all",
    baseline_ref: str | None = None,
    baseline_guard: Any | None = None,
    env_values: dict[str, str] | None = None,
    backend_factory: BackendFactory | None = None,
) -> list[dict[str, Any]]:
    """Run guard and self-claim questions against selected labeled texts."""
    if split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test, or all")
    if baseline_ref and baseline_guard is not None:
        raise ValueError("pass baseline_ref or baseline_guard, not both")
    if baseline_ref:
        baseline_guard = load_guard_from_ref(baseline_ref)
    selected_models = parse_models(models)
    if reps < 1:
        raise ValueError("reps must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    chosen_cases = [case for case in cases if split == "all" or case.get("split") == split]
    questions = {"current": GUARD.payload()}
    if baseline_guard is not None:
        questions["baseline"] = baseline_guard.payload()
    questions["self_claim"] = SELF_CLAIM.payload()

    factory = backend_factory or (
        lambda model, provider_name: create_backend(model, provider_name, env_values)
    )
    backends: dict[JevModel, Backend] = {model: factory(model, provider) for model in selected_models}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: dict[str, Any], model: JevModel, rep: int) -> dict[str, Any]:
        model_id = provider_model_id(model, provider)
        metadata = {key: case[key] for key in (
            "id", "lang", "category", "text", "injection", "self_claim", "split"
        ) if key in case}
        base = {**metadata, "model": model_id, "rep": rep}
        started = time.perf_counter()
        try:
            async with semaphore:
                started = time.perf_counter()
                response = await backends[model].ask(
                    untrusted(case["text"]), questions, model=model_id, timeout_s=30.0
                )
            elapsed = time.perf_counter() - started
            answers = response.answers
            row = {**base, "served": response.model, "latency": elapsed,
                   "p_current": float(answers["current"]["noul"]),
                   "p_self_claim": float(answers["self_claim"]["noul"])}
            if baseline_guard is not None:
                row["p_baseline"] = float(answers["baseline"]["noul"])
            return row
        except Exception as exc:
            return {**base, "error": safe_error(exc), "latency": time.perf_counter() - started}

    rows = await asyncio.gather(
        *(one(case, model, rep)
          for rep in range(reps)
          for model in selected_models
          for case in chosen_cases)
    )
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("bench/data/guard_eval.json"))
    parser.add_argument("--models", default=make_parser_models_default(),
                        help="comma-separated Jev ids or versions")
    parser.add_argument("--provider", choices=("typesafe", "openrouter"), default="typesafe")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--baseline-ref", help="git ref whose jevkit/state.py supplies the baseline GUARD")
    parser.add_argument("--out", type=Path, help="JSONL output path")
    parser.add_argument("--env-file", type=Path, help="read provider API keys from KEY=VALUE lines")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = read_cases(args.dataset)
    output = args.out or default_results_path(args.dataset.stem)
    env_values = load_env_file(args.env_file)
    rows = asyncio.run(run_guard(
        cases, models=args.models, provider=args.provider, split=args.split,
        reps=args.reps, concurrency=args.concurrency, out_path=output,
        baseline_ref=args.baseline_ref, env_values=env_values,
    ))
    errors = sum("error" in row for row in rows)
    print(f"{len(rows)} rows, {errors} errors; results: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
