"""
Function calling without generation (docs.typesafe.ai/cookbooks/function_calling):
tool selection is a Choice, each argument is a Choice over allowed values, and each
optional argument gets a "stated" Noul ("did the user even say that?"). All questions
live in ONE fan-out. Confidence of the call = MINIMUM of all involved judgments - one
wrong argument is enough to spoil the call.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jevkit.answers import ChoiceAnswer, NoulAnswer
from jevkit.client import Decision
from jevkit.compose import NONE_OPTION
from jevkit.questions import Choice, Json, Noul, Question

TOOL_QID = "__tool__"


@dataclass(frozen=True)
class Arg:
    name: str
    instructions: Json
    options: Mapping[str, Json]
    required: bool = True
    stated: Json | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: Json
    args: tuple[Arg, ...] = ()


@dataclass(frozen=True)
class Call:
    tool: str
    args: dict[str, str]
    confidence: float
    omitted: tuple[str, ...]


def _arg_qid(tool: str, arg: str) -> str:
    return f"{tool}.{arg}"


def call_questions(
    tools: Sequence[ToolSpec],
    instructions: Json = "What is the user asking the assistant to do?",
) -> dict[str, Question]:
    if not tools:
        raise ValueError("call_questions: at least one tool is required")
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("call_questions: tool names must be unique")
    criteria: dict[str, Json] = {t.name: t.description for t in tools}
    criteria[NONE_OPTION] = "None of the tools fits the request"
    qs: dict[str, Question] = {TOOL_QID: Choice(instructions, criteria)}
    for t in tools:
        for a in t.args:
            qs[_arg_qid(t.name, a.name)] = Choice(
                a.instructions, dict(a.options)
            )
            if not a.required:
                stated_default = {
                    "question": f"Does the user explicitly mention `{a.name}`?",
                    "argument": a.name,
                }
                qs[f"{_arg_qid(t.name, a.name)}:stated"] = Noul(
                    a.stated or stated_default
                )
    return qs


def resolve_call(decision: Decision, tools: Sequence[ToolSpec], *, stated_at: float = 0.5) -> Call | None:
    tool_ans = decision[TOOL_QID]
    if not isinstance(tool_ans, ChoiceAnswer):
        raise TypeError("resolve_call: __tool__ must be a Choice")
    if tool_ans.choice == NONE_OPTION:
        return None
    spec = next((t for t in tools if t.name == tool_ans.choice), None)
    if spec is None:
        raise ValueError(f"resolve_call: tool {tool_ans.choice!r} is not in tools")
    conf = tool_ans.confidence
    args: dict[str, str] = {}
    omitted: list[str] = []
    for a in spec.args:
        qid = _arg_qid(spec.name, a.name)
        if not a.required:
            stated = decision[f"{qid}:stated"]
            if not isinstance(stated, NoulAnswer):
                raise TypeError(f"resolve_call: {qid}:stated must be a Noul")
            conf = min(conf, stated.confidence)
            if stated.p < stated_at:
                omitted.append(a.name)
                continue
        ans = decision[qid]
        if not isinstance(ans, ChoiceAnswer):
            raise TypeError(f"resolve_call: {qid} must be a Choice")
        args[a.name] = ans.choice
        conf = min(conf, ans.confidence)
    return Call(spec.name, args, conf, tuple(omitted))
