"""
Function Calling ohne Generierung (docs.typesafe.ai/cookbooks/function_calling):
Werkzeugwahl ist eine Choice, jedes Argument eine Choice über erlaubte Werte, jedes
optionale Argument bekommt eine "stated"-Noul ("hat der Nutzer das überhaupt gesagt?").
Alle Fragen liegen in EINEM Fan-out. Confidence des Calls = MINIMUM aller beteiligten
Urteile — ein falsches Argument reicht, um den Call zu verderben.
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


def call_questions(tools: Sequence[ToolSpec],  # noqa: E501
                   instructions: Json = "What is the user asking the assistant to do?") -> dict[str, Question]:  # noqa: E501
    if not tools:
        raise ValueError("call_questions: mindestens ein Tool nötig")
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("call_questions: Tool-Namen müssen eindeutig sein")
    criteria: dict[str, Json] = {t.name: t.description for t in tools}
    criteria[NONE_OPTION] = "None of the tools fits the request"
    qs: dict[str, Question] = {TOOL_QID: Choice(instructions, criteria)}
    for t in tools:
        for a in t.args:
            qs[_arg_qid(t.name, a.name)] = Choice(a.instructions, dict(a.options))
            if not a.required:
                qs[f"{_arg_qid(t.name, a.name)}:stated"] = Noul(
                    a.stated or {"question": f"Does the user explicitly mention `{a.name}`?", "argument": a.name})  # noqa: E501
    return qs


def resolve_call(decision: Decision, tools: Sequence[ToolSpec], *, stated_at: float = 0.5) -> Call | None:
    tool_ans = decision[TOOL_QID]
    if not isinstance(tool_ans, ChoiceAnswer):
        raise TypeError("resolve_call: __tool__ muss eine Choice sein")
    if tool_ans.choice == NONE_OPTION:
        return None
    spec = next(t for t in tools if t.name == tool_ans.choice)
    conf = tool_ans.confidence
    args: dict[str, str] = {}
    omitted: list[str] = []
    for a in spec.args:
        qid = _arg_qid(spec.name, a.name)
        if not a.required:
            stated = decision[f"{qid}:stated"]
            if not isinstance(stated, NoulAnswer):
                raise TypeError(f"resolve_call: {qid}:stated muss eine Noul sein")
            conf = min(conf, stated.confidence)
            if stated.p < stated_at:
                omitted.append(a.name)
                continue
        ans = decision[qid]
        if not isinstance(ans, ChoiceAnswer):
            raise TypeError(f"resolve_call: {qid} muss eine Choice sein")
        args[a.name] = ans.choice
        conf = min(conf, ans.confidence)
    return Call(spec.name, args, conf, tuple(omitted))
