"""
jevkit - Decision kernel for TypeSafe Jev (System One).

Jev answers typed questions with calibrated probabilities; jevkit makes sure the code
stays in control: prepare the state (state), compose questions (compose/tools/taxonomy),
one call per state (client), translate the answer into a band (gate/registry), and
recalibrate from outcomes (log/calibrate).
"""
from jevkit.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer, parse_answer
from jevkit.backends import BackendError, OpenRouterBackend, StaticBackend, TypeSafeBackend
from jevkit.cache import MemoryCache
from jevkit.calibrate import accuracy, brier, ece, noul_pairs, pairs_by_question, suggest_bands
from jevkit.client import Client, Decision, JevUnavailable
from jevkit.compose import NONE_OPTION, composite, count, count_questions, extract, extract_questions
from jevkit.gate import DEFAULT_BANDS, Band, Bands, Severity, band, check_pairs, consistent, demote
from jevkit.log import DecisionLog, Record, state_hash
from jevkit.prompt_backend import PromptBackend
from jevkit.questions import Choice, Noul, Score
from jevkit.registry import DecisionSpec, Privacy, Registry, Router, Verdict
from jevkit.state import (
    GUARD,
    GUARD_ID,
    SELF_CLAIM,
    SELF_CLAIM_ID,
    bucket,
    count_bucket,
    injected,
    project,
    relative_days,
    untrusted,
    with_guard,
)
from jevkit.taxonomy import Path, beam_search
from jevkit.tools import TOOL_QID, Arg, Call, ToolSpec, call_questions, resolve_call

__version__ = "0.1.3"

__all__ = [
    "Noul", "Choice", "Score", "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "parse_answer",
    "Client", "Decision", "JevUnavailable", "MemoryCache",
    "TypeSafeBackend", "OpenRouterBackend", "StaticBackend", "PromptBackend", "BackendError",
    "Band", "Bands", "Severity", "DEFAULT_BANDS", "band", "demote", "consistent", "check_pairs",
    "project", "bucket", "count_bucket", "relative_days", "untrusted", "GUARD", "GUARD_ID",
    "SELF_CLAIM", "SELF_CLAIM_ID",
    "with_guard", "injected",
    "count_questions", "count", "composite", "extract_questions", "extract", "NONE_OPTION",
    "Arg", "ToolSpec", "Call", "call_questions", "resolve_call", "TOOL_QID",
    "Path", "beam_search",
    "Privacy", "DecisionSpec", "Registry", "Verdict", "Router",
    "DecisionLog", "Record", "state_hash",
    "brier", "ece", "accuracy", "pairs_by_question", "noul_pairs", "suggest_bands",
]
