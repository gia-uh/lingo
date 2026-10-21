"""
A client for System One decision models.

A decision model answers typed questions (choice, noul, score) about a state
with a probability per option, without generating text. This module speaks the
System One wire contract over HTTP and knows nothing about who serves it: a
hosted gateway or a local decision server fit behind the same URL.
"""

import inspect
import logging
import os
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

from .llm import Message

log = logging.getLogger(__name__)

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    """A question whose answer is one of the given options (name to description)."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: str) -> dict[str, Any]:
    """A yes/no question, answered with the probability of yes."""
    return {"type": "noul", "instructions": instructions}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    """A question placed on an ordered rubric, lowest level first."""
    return {"type": "score", "instructions": instructions, "criteria": levels}


class DeciderError(Exception):
    """The decision server failed or answered outside the contract."""


class DecisionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


class Answer(BaseModel):
    """One answer. Which value field is set depends on `kind`."""

    kind: str
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float


class Decision(BaseModel):
    """Every answer for one state, keyed by question id."""

    answers: dict[str, Answer]
    model: str | None = None
    usage: DecisionUsage = Field(default_factory=DecisionUsage)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]


def _parse_answer(question: dict[str, Any], raw: dict[str, Any]) -> Answer:
    kind = raw.get("type") or question["type"]
    probabilities = {
        str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()
    }

    if kind == "noul":
        p = float(raw["noul"])
        return Answer(kind=kind, noul=p, confidence=max(p, 1 - p))

    if kind == "choice":
        picked = raw["choice"]
        if picked not in question["criteria"]:
            raise DeciderError(f"choice {picked!r} is not one of the options")
        confidence = raw.get("confidence", probabilities.get(picked))
        return Answer(
            kind=kind,
            choice=picked,
            probabilities=probabilities,
            confidence=float(confidence),
        )

    if kind == "score":
        confidence = raw.get("confidence", max(probabilities.values(), default=0.0))
        return Answer(
            kind=kind,
            score=float(raw["score"]),
            probabilities=probabilities,
            confidence=float(confidence),
        )

    raise DeciderError(f"unsupported answer type {kind!r}")


def render_state(messages: list[Message], history: int) -> str:
    """The last `history` user and assistant messages with text, as `role: text` lines."""
    lines = [
        f"{m.role}: {m.content}"
        for m in messages
        if m.role in ("user", "assistant")
        and isinstance(m.content, str)
        and m.content.strip()
    ]
    return "\n".join(lines[-history:])


class SystemOne:
    """
    A System One decision model reached over HTTP.

    `threshold` is the confidence below which the Engine falls back to its LLM;
    `history` is how many user and assistant messages the Engine sends as state.
    """

    def __init__(
        self,
        model: str,
        url: str = OPENROUTER_DECISIONS_URL,
        api_key: str | None = None,
        threshold: float = 0.8,
        history: int = 6,
        timeout: float = 10.0,
        on_decision: Callable[[Decision], Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.url = url
        self.api_key = (
            api_key or os.getenv("SYSTEMONE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        )
        self.threshold = threshold
        self.history = history
        self.timeout = timeout
        self._on_decision = on_decision
        self._client = client

    async def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> Decision:
        """Ask every question about `state` in one request."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "state": state, "questions": questions}

        try:
            if self._client is not None:
                response = await self._client.post(
                    self.url, json=body, headers=headers, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.url, json=body, headers=headers, timeout=self.timeout
                    )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as e:
            raise DeciderError(f"decision request failed: {e}") from e

        try:
            answers = {
                qid: _parse_answer(q, data["answers"][qid])
                for qid, q in questions.items()
            }
        except (KeyError, TypeError, ValueError) as e:
            raise DeciderError(f"answer outside the contract: {e!r}") from e

        decision = Decision(
            answers=answers,
            model=data.get("model"),
            usage=DecisionUsage(
                **{
                    k: v
                    for k, v in (data.get("usage") or {}).items()
                    if k in DecisionUsage.model_fields
                }
            ),
        )
        if self._on_decision:
            resp = self._on_decision(decision)
            if inspect.iscoroutine(resp):
                await resp
        return decision
