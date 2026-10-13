from system_one.schema import (
    TOOLS,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    ToolRouteDecision,
    choice_from_probs,
    score_from_probs,
)


def _decision(confidence: float, tool: str, destructive: float) -> ToolRouteDecision:
    return ToolRouteDecision(
        next_tool=ChoiceAnswer(tool, {name: 0.2 for name in TOOLS}, confidence),
        needs_live_data=NoulAnswer(0.1),
        is_destructive=NoulAnswer(destructive),
        risk=ScoreAnswer(0.2, {"0": "read-only", "1": "local write", "2": "external side effect"}, {"0": 0.8, "1": 0.2, "2": 0.0}, 0.7),
    )


def test_low_confidence_escalates():
    assert _decision(0.2, "search", 0.1).action() == "escalate"


def test_destructive_mid_confidence_asks_for_confirmation():
    assert _decision(0.7, "code", 0.95).action() == "confirm"


def test_high_confidence_returns_the_tool():
    assert _decision(0.95, "calculator", 0.1).action() == "calculator"


def test_choice_answer_stays_inside_the_schema():
    answer = choice_from_probs(TOOLS, [0.05, 0.7, 0.1, 0.1, 0.05])
    assert answer.choice in TOOLS
    assert set(answer.probabilities) == set(TOOLS)
    assert 0 <= answer.confidence <= 1


def test_score_is_the_expected_level():
    answer = score_from_probs([0.2, 0.2, 0.6])
    assert abs(answer.score - (0 * 0.2 + 1 * 0.2 + 2 * 0.6)) < 1e-12
    assert answer.legend["2"] == "external side effect"
