import pytest

from jevkit.answers import ChoiceAnswer, NoulAnswer
from jevkit.client import Decision
from jevkit.compose import NONE_OPTION
from jevkit.questions import Choice, Noul
from jevkit.tools import TOOL_QID, Arg, ToolSpec, call_questions, resolve_call

LAMP = ToolSpec("lamp", "Switch the desk lamp on or off",
                args=(Arg("power", "Should the lamp be on or off?", {"on": None, "off": None}),))
TIMER = ToolSpec("timer", "Set a countdown timer",
                 args=(Arg("minutes", "How many minutes?", {"5": None, "10": None, "30": None}),
                       Arg("label", "What is the timer for?", {"tea": None, "laundry": None},
                           required=False, stated="Does the user say what the timer is for?")))


def _c(choice, probs, conf):
    return ChoiceAnswer(choice, probs, conf)


def test_call_questions_form():
    qs = call_questions([LAMP, TIMER])
    assert set(qs) == {TOOL_QID, "lamp.power", "timer.minutes", "timer.label", "timer.label:stated"}
    tool = qs[TOOL_QID]
    assert isinstance(tool, Choice) and list(tool.criteria) == ["lamp", "timer", NONE_OPTION]
    assert tool.criteria["lamp"] == "Switch the desk lamp on or off"
    assert isinstance(qs["timer.label:stated"], Noul)
    assert qs["timer.label:stated"].instructions == "Does the user say what the timer is for?"
    with pytest.raises(ValueError):
        call_questions([])
    with pytest.raises(ValueError):
        call_questions([LAMP, LAMP])


def test_resolve_call_min_confidence_und_optional():
    d = Decision({
        TOOL_QID: _c("timer", {"timer": 0.9, "lamp": 0.05, NONE_OPTION: 0.05}, 0.85),
        "lamp.power": _c("on", {"on": 0.6, "off": 0.4}, 0.2),
        "timer.minutes": _c("10", {"5": 0.1, "10": 0.8, "30": 0.1}, 0.7),
        "timer.label": _c("tea", {"tea": 0.55, "laundry": 0.45}, 0.1),
        "timer.label:stated": NoulAnswer(0.2),
    }, "m", {}, False, 0.0)
    call = resolve_call(d, [LAMP, TIMER])
    assert (call.tool, call.args, call.omitted) == ("timer", {"minutes": "10"}, ("label",))
    assert abs(call.confidence - 0.6) < 1e-9          # min(0.85, 0.7, stated-conf 0.6)

    d2 = Decision({**d.answers, "timer.label:stated": NoulAnswer(0.95)}, "m", {}, False, 0.0)
    call2 = resolve_call(d2, [LAMP, TIMER])
    assert (
        call2.args == {"minutes": "10", "label": "tea"}
        and call2.omitted == ()
        and abs(call2.confidence - 0.1) < 1e-9
    )


def test_resolve_call_none():
    d = Decision({TOOL_QID: _c(NONE_OPTION, {"lamp": 0.1, NONE_OPTION: 0.9}, 0.8),
                  "lamp.power": _c("on", {"on": 1.0, "off": 0.0}, 1.0)}, "m", {}, False, 0.0)
    assert resolve_call(d, [LAMP]) is None
