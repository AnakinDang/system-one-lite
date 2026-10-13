"""Typed questions and answers for one System One call.

The benchmark schema is fixed. A fixed head per question does not accept an
arbitrary new schema at runtime; the question text is part of the model input.
"""

from __future__ import annotations

from dataclasses import dataclass

from system_one.confidence import choice_confidence, score_confidence

TOOLS = ("search", "calculator", "code", "ask_user", "finish")
RISK_LEVELS = ("0", "1", "2")
RISK_LEGEND = {
    "0": "read-only",
    "1": "local write",
    "2": "external side effect",
}

SCHEMA_TEXT = """questions:
next_tool: choice among search, calculator, code, ask_user, finish
needs_live_data: noul, true when the request needs information outside this state
is_destructive: noul, true when execution would change external state
risk: score, 0 read-only, 1 local write, 2 external side effect
state:
"""


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ToolRouteDecision:
    next_tool: ChoiceAnswer
    needs_live_data: NoulAnswer
    is_destructive: NoulAnswer
    risk: ScoreAnswer

    def action(self) -> str:
        """Gate execution on confidence. Typed output is always legal; it can still be wrong."""
        confidence = self.next_tool.confidence
        if confidence < 0.45:
            return "escalate"
        if self.is_destructive.noul >= 0.5 and confidence < 0.85:
            return "confirm"
        return self.next_tool.choice


def choice_from_probs(names: tuple[str, ...], probabilities: list[float]) -> ChoiceAnswer:
    if len(names) != len(probabilities):
        raise ValueError("choice names and probabilities differ in length")
    paired = {name: float(prob) for name, prob in zip(names, probabilities)}
    selected = max(paired, key=paired.get)
    return ChoiceAnswer(selected, paired, choice_confidence(probabilities))


def score_from_probs(probabilities: list[float]) -> ScoreAnswer:
    if len(probabilities) != len(RISK_LEVELS):
        raise ValueError("risk score expects three levels")
    paired = {level: float(prob) for level, prob in zip(RISK_LEVELS, probabilities)}
    expected = sum(index * prob for index, prob in enumerate(probabilities))
    return ScoreAnswer(expected, dict(RISK_LEGEND), paired, score_confidence(probabilities))


def noul_from_prob(probability: float) -> NoulAnswer:
    return NoulAnswer(float(probability))
