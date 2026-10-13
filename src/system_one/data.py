"""Synthetic agent tool-routing states with known outcome frequencies."""

from __future__ import annotations

import random
from dataclasses import dataclass

from system_one.schema import SCHEMA_TEXT

TOOLS = ("search", "calculator", "code", "ask_user", "finish")

TOPICS = (
    "grpo",
    "tax",
    "weather",
    "pytorch",
    "invoice",
    "calendar",
    "refund policy",
    "exchange rate",
)
NAMES = ("mina", "leo", "chen", "sofia")
FILES = ("notes.txt", "report.csv", "main.py", "budget.xlsx")


@dataclass(frozen=True)
class Example:
    text: str
    family: str
    tool: int
    live: int
    destructive: int
    risk: int


@dataclass(frozen=True)
class Family:
    name: str
    templates: tuple[str, ...]
    tool: tuple[float, ...]
    live: float
    destructive: float
    risk: tuple[float, ...]


def _unit(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    return tuple(weight / total for weight in weights)


FAMILIES = (
    Family(
        "arithmetic",
        (
            "calculate {a} times {b}",
            "what is {a} plus {b}",
            "compute the product of {a} and {b}",
        ),
        _unit((1, 18, 1, 1, 1)),
        0.04,
        0.02,
        _unit((17, 2, 1)),
    ),
    Family(
        "lookup",
        (
            "look up the documentation for {topic}",
            "find a source that explains {topic}",
            "search for background on {topic}",
        ),
        _unit((18, 1, 1, 1, 1)),
        0.25,
        0.02,
        _unit((18, 1, 1)),
    ),
    Family(
        "live_fact",
        (
            "what is the latest {topic} right now",
            "check today's {topic} before answering {name}",
            "get the current {topic} from the web",
        ),
        _unit((18, 1, 1, 2, 1)),
        0.92,
        0.03,
        _unit((16, 3, 1)),
    ),
    Family(
        "write_code",
        (
            "write a python function that handles {topic}",
            "implement a small parser for {file}",
            "draft code to clean {file}",
        ),
        _unit((1, 1, 16, 1, 1)),
        0.08,
        0.05,
        _unit((2, 16, 2)),
    ),
    Family(
        "debug",
        (
            "debug this stack trace in {file}",
            "the tests for {topic} fail, fix the code",
            "why does {file} raise an error",
        ),
        _unit((2, 1, 16, 2, 1)),
        0.12,
        0.08,
        _unit((3, 15, 2)),
    ),
    Family(
        "unclear",
        (
            "help me with this",
            "{name} asked for something about {topic}, what should we do",
            "not sure if this is a question or a task",
        ),
        _unit((3, 2, 2, 10, 2)),
        0.35,
        0.15,
        _unit((6, 6, 4)),
    ),
    Family(
        "done",
        (
            "thanks, that is all",
            "we are done here",
            "no further action for {name}",
        ),
        _unit((1, 1, 1, 1, 16)),
        0.03,
        0.02,
        _unit((18, 1, 1)),
    ),
    Family(
        "external_write",
        (
            "delete the remote file {file}",
            "send the {topic} report to {name}",
            "pay the invoice and transfer the funds",
        ),
        _unit((1, 1, 4, 10, 1)),
        0.2,
        0.9,
        _unit((1, 2, 17)),
    ),
    Family(
        "code_or_calc",
        (
            "write a script to calculate {a} times {b}",
            "either compute {a} plus {b} or code it",
            "I need {a} * {b}, a function is fine too",
        ),
        _unit((1, 8, 8, 2, 1)),
        0.05,
        0.04,
        _unit((8, 9, 2)),
    ),
    Family(
        "local_edit",
        (
            "append a line to local file {file}",
            "rename {file} in the workspace only",
            "save these notes into {file} on disk",
        ),
        _unit((1, 1, 14, 2, 1)),
        0.05,
        0.12,
        _unit((2, 15, 3)),
    ),
)


def _draw(rng: random.Random, weights: tuple[float, ...]) -> int:
    return rng.choices(range(len(weights)), weights=weights, k=1)[0]


def _render(template: str, rng: random.Random) -> str:
    return template.format(
        a=rng.randint(2, 99),
        b=rng.randint(2, 99),
        topic=rng.choice(TOPICS),
        name=rng.choice(NAMES),
        file=rng.choice(FILES),
    )


def sample_example(rng: random.Random, family: Family | None = None) -> Example:
    chosen = family or rng.choice(FAMILIES)
    state = _render(rng.choice(chosen.templates), rng)
    return Example(
        text=SCHEMA_TEXT + state,
        family=chosen.name,
        tool=_draw(rng, chosen.tool),
        live=int(rng.random() < chosen.live),
        destructive=int(rng.random() < chosen.destructive),
        risk=_draw(rng, chosen.risk),
    )


def make_split(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    return [sample_example(rng) for _ in range(n)]
