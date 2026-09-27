"""Unit tests for the retry on truncated structured generations.

No provider is contacted: `client.chat.completions.parse` is replaced with an
`AsyncMock` whose side effects we choose. The point is the wiring — which
failures are retried, which are not, and how many attempts are spent.

The failure this exists for: under grammar-constrained decoding a model
occasionally fails to close the JSON object and runs to the token ceiling, which
the OpenAI SDK surfaces as `LengthFinishReasonError`. Measured at roughly one
call in twelve against `qwen/qwen3-32b` on OpenRouter with a discriminated union
of enum-bound branches. Re-rolling succeeds.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from lingo import LLM, Message
from pydantic import BaseModel


class Answer(BaseModel):
    """A trivial structured answer."""

    value: str


def _parsed(value: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=Answer(value=value)))],
        usage=None,
        model="fake",
    )


def _length_error() -> openai.LengthFinishReasonError:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(parsed=None))],
        usage=None,
    )
    return openai.LengthFinishReasonError(completion=completion)  # type: ignore[arg-type]


def _llm(side_effect, **kwargs) -> tuple[LLM, AsyncMock]:
    llm = LLM(model="fake", api_key="none", base_url="http://localhost/v1", **kwargs)
    parse = AsyncMock(side_effect=side_effect)
    llm.client = MagicMock()
    llm.client.chat = MagicMock()
    llm.client.chat.completions = MagicMock()
    llm.client.chat.completions.parse = parse
    return llm, parse


MESSAGES = [Message.user("go")]


@pytest.mark.asyncio
async def test_a_truncated_generation_is_retried_and_the_retry_is_returned():
    llm, parse = _llm([_length_error(), _parsed("second try")])
    answer = await llm.create(Answer, MESSAGES)
    assert answer.value == "second try"
    assert parse.await_count == 2


@pytest.mark.asyncio
async def test_a_first_attempt_that_succeeds_costs_exactly_one_call():
    llm, parse = _llm([_parsed("first try")])
    answer = await llm.create(Answer, MESSAGES)
    assert answer.value == "first try"
    assert parse.await_count == 1


@pytest.mark.asyncio
async def test_exhausting_the_budget_raises_rather_than_returning_nothing():
    llm, parse = _llm([_length_error(), _length_error()])
    with pytest.raises(ValueError, match="token ceiling on all 2 attempts"):
        await llm.create(Answer, MESSAGES)
    assert parse.await_count == 2


@pytest.mark.asyncio
async def test_the_budget_is_configurable():
    llm, parse = _llm([_length_error()] * 3 + [_parsed("fourth")], create_attempts=4)
    assert (await llm.create(Answer, MESSAGES)).value == "fourth"
    assert parse.await_count == 4


@pytest.mark.asyncio
async def test_a_budget_below_one_still_makes_a_single_attempt():
    llm, parse = _llm([_parsed("once")], create_attempts=0)
    assert (await llm.create(Answer, MESSAGES)).value == "once"
    assert parse.await_count == 1


@pytest.mark.asyncio
async def test_other_failures_are_not_retried():
    """A schema the provider rejects fails the same way every time, so retrying
    it only hides the cause and spends tokens."""
    llm, parse = _llm([openai.BadRequestError("bad schema", response=MagicMock(), body=None)])
    with pytest.raises(openai.BadRequestError):
        await llm.create(Answer, MESSAGES)
    assert parse.await_count == 1


@pytest.mark.asyncio
async def test_an_unparseable_response_is_not_retried():
    empty = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))], usage=None, model="fake"
    )
    llm, parse = _llm([empty])
    with pytest.raises(ValueError, match="Failed to parse"):
        await llm.create(Answer, MESSAGES)
    assert parse.await_count == 1


@pytest.mark.asyncio
async def test_the_original_error_is_kept_as_the_cause():
    llm, _ = _llm([_length_error(), _length_error()])
    with pytest.raises(ValueError) as excinfo:
        await llm.create(Answer, MESSAGES)
    assert isinstance(excinfo.value.__cause__, openai.LengthFinishReasonError)
