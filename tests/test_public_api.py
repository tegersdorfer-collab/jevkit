import jevkit


def test_alle_exporte_vorhanden():
    names = ["Noul", "Choice", "Score", "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "parse_answer",
             "Client", "Decision", "JevUnavailable", "MemoryCache",
             "TypeSafeBackend", "OpenRouterBackend", "StaticBackend", "PromptBackend", "BackendError",
             "Band", "Bands", "Severity", "DEFAULT_BANDS", "band", "demote", "consistent", "check_pairs",
             "project", "bucket", "count_bucket", "relative_days", "untrusted", "GUARD", "GUARD_ID",
             "with_guard", "injected",
             "count_questions", "count", "composite", "extract_questions", "extract", "NONE_OPTION",
             "Arg", "ToolSpec", "Call", "call_questions", "resolve_call", "TOOL_QID",
             "Path", "beam_search",
             "Privacy", "DecisionSpec", "Registry", "Verdict", "Router",
             "DecisionLog", "Record", "state_hash",
             "brier", "ece", "accuracy", "pairs_by_question", "noul_pairs", "suggest_bands"]
    missing = [n for n in names if not hasattr(jevkit, n)]
    assert missing == []
    assert set(names) <= set(jevkit.__all__)
    assert jevkit.__version__ == "0.1.1"
