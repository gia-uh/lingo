"""Tests for the System One decider client and the Engine's use of it."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from lingo import Context, Engine, Lingo, Message, SystemOne
from lingo.decider import DeciderError, choice, noul, render_state, score
from lingo.mock import MockLLM
from lingo.tools import tool

URL = "https://decider.test/v1/systemone"


def server(answers=None, status=200, record=None, payload=None):
    """A SystemOne whose HTTP calls go to a fake server returning `answers`."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if record is not None:
            record.append({"body": body, "headers": dict(request.headers)})
        if payload is not None:
            return httpx.Response(status, json=payload)
        return httpx.Response(
            status,
            json={
                "model": "fake-1",
                "answers": answers,
                "usage": {"input_tokens": 12, "cost": 1e-6},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return SystemOne("fake", url=URL, api_key="k", client=client)


def llm_returning(result):
    """An Engine LLM path that records whether it was used."""
    cot = MagicMock()
    cot.result = result
    return AsyncMock(return_value=cot)


def ctx(*messages):
    return Context(list(messages))


# --- the client ------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_body_follows_the_contract():
    calls = []
    questions = {
        "c": choice("Which?", {"a": "first", "b": "second"}),
        "q": noul("Is it?"),
        "s": score("How much?", ["low", "high"]),
    }
    d = server(
        {
            "c": {
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.3},
                "confidence": 0.7,
            },
            "q": {"noul": 0.9},
            "s": {"score": 1, "probabilities": {"0": 0.2, "1": 0.8}, "confidence": 0.8},
        },
        record=calls,
    )
    await d.decide("user: hi", questions)

    assert calls[0]["body"] == {
        "model": "fake",
        "state": "user: hi",
        "questions": questions,
    }
    assert calls[0]["headers"]["authorization"] == "Bearer k"
    assert questions["s"]["criteria"] == ["low", "high"]


@pytest.mark.asyncio
async def test_answers_are_parsed_per_type():
    d = server(
        {
            "c": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.1, "b": 0.9},
                "confidence": 0.9,
            },
            "q": {"type": "noul", "noul": 0.2},
            "s": {
                "type": "score",
                "score": 2,
                "legend": {"0": "x", "1": "y", "2": "z"},
                "probabilities": {"0": 0, "1": 0.1, "2": 0.9},
                "confidence": 0.9,
            },
        }
    )
    decision = await d.decide(
        "state",
        {
            "c": choice("?", {"a": "", "b": ""}),
            "q": noul("?"),
            "s": score("?", ["x", "y", "z"]),
        },
    )

    assert decision["c"].choice == "b" and decision["c"].confidence == 0.9
    assert decision["q"].noul == 0.2 and decision["q"].confidence == 0.8
    assert decision["s"].score == 2 and decision["s"].probabilities["2"] == 0.9
    assert decision.model == "fake-1"
    assert decision.usage.input_tokens == 12 and decision.usage.cost == 1e-6


@pytest.mark.asyncio
async def test_on_decision_receives_every_decision():
    seen = []
    d = server({"q": {"noul": 0.9}})
    d._on_decision = seen.append
    await d.decide("s", {"q": noul("?")})
    assert len(seen) == 1 and seen[0]["q"].noul == 0.9


@pytest.mark.asyncio
async def test_http_error_raises_decider_error():
    with pytest.raises(DeciderError):
        await server(status=503, payload={"error": "busy"}).decide(
            "s", {"q": noul("?")}
        )


@pytest.mark.asyncio
async def test_choice_outside_the_options_raises_decider_error():
    d = server({"c": {"choice": "zzz", "confidence": 1.0}})
    with pytest.raises(DeciderError):
        await d.decide("s", {"c": choice("?", {"a": "", "b": ""})})


@pytest.mark.asyncio
async def test_missing_answer_raises_decider_error():
    with pytest.raises(DeciderError):
        await server({}).decide("s", {"q": noul("?")})


def test_render_state_keeps_the_last_user_and_assistant_messages():
    messages = [
        Message.system("you are a bot"),
        Message.user("one"),
        Message.assistant("two"),
        Message.tool("result", tool_call_id="x"),
        Message.user("three"),
    ]
    assert render_state(messages, 2) == "assistant: two\nuser: three"


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert SystemOne("m").api_key == "from-env"


# --- the Engine --------------------------------------------------------------


@pytest.mark.asyncio
async def test_confident_choose_skips_the_llm():
    calls = []
    engine = Engine(
        MockLLM(),
        decider=server({"q": {"choice": "beta", "confidence": 0.95}}, record=calls),
    )
    engine.create = llm_returning("alpha")

    assert (
        await engine.choose(ctx(Message.user("hi")), ["alpha", "beta"], "Pick one.")
        == "beta"
    )
    engine.create.assert_not_called()
    assert calls[0]["body"]["questions"]["q"] == choice(
        "Pick one.", {"alpha": "alpha", "beta": "beta"}
    )
    assert calls[0]["body"]["state"] == "user: hi"


@pytest.mark.asyncio
async def test_unsure_choose_falls_back_to_the_llm():
    engine = Engine(
        MockLLM(), decider=server({"q": {"choice": "beta", "confidence": 0.5}})
    )
    engine.create = llm_returning("alpha")

    assert await engine.choose(ctx(), ["alpha", "beta"]) == "alpha"
    engine.create.assert_called_once()


@pytest.mark.asyncio
async def test_failing_decider_falls_back_to_the_llm():
    engine = Engine(MockLLM(), decider=server(status=500, payload={}))
    engine.create = llm_returning(True)

    assert await engine.decide(ctx(), "Is it?") is True
    engine.create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("p,expected", [(0.9, True), (0.1, False)])
async def test_confident_decide_skips_the_llm(p, expected):
    engine = Engine(MockLLM(), decider=server({"q": {"noul": p}}))
    engine.create = llm_returning(not expected)

    assert await engine.decide(ctx(), "Is it?") is expected
    engine.create.assert_not_called()


@pytest.mark.asyncio
async def test_equip_sends_tool_descriptions():
    @tool
    async def weather(city: str) -> str:
        """Current weather for a city."""
        return city

    @tool
    async def news(topic: str) -> str:
        """Latest news on a topic."""
        return topic

    calls = []
    engine = Engine(
        MockLLM(),
        tools=[weather, news],
        decider=server({"q": {"choice": "news", "confidence": 0.99}}, record=calls),
    )
    engine.create = llm_returning("weather")

    assert (await engine.equip(ctx())).name == "news"
    criteria = calls[0]["body"]["questions"]["q"]["criteria"]
    assert criteria["weather"].startswith("Current weather") and criteria[
        "news"
    ].startswith("Latest news")
    engine.create.assert_not_called()


@pytest.mark.asyncio
async def test_scope_keeps_the_decider():
    d = server({})
    assert Engine(MockLLM(), decider=d).scope([])._decider is d


@pytest.mark.asyncio
async def test_ask_many_needs_a_decider():
    with pytest.raises(ValueError):
        await Engine(MockLLM()).ask_many(ctx(), {"q": noul("?")})


@pytest.mark.asyncio
async def test_lingo_routes_skills_on_the_decider_alone():
    calls = []
    bot = Lingo(
        llm=MockLLM(),
        decider=server({"q": {"choice": "stores", "confidence": 0.97}}, record=calls),
    )

    @bot.skill
    async def stores(context, engine):
        """Where the shops are."""
        context.append(Message.assistant("store list"))

    @bot.skill
    async def other(context, engine):
        """Anything else."""
        context.append(Message.assistant("other"))

    reply = await bot.chat("¿Dónde hay tiendas?")

    assert reply.content == "store list"
    assert calls[0]["body"]["questions"]["q"]["criteria"]["stores"].startswith(
        "Where the shops are"
    )


# --- live -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for live API",
)
async def test_jev_on_openrouter_live():
    decision = await SystemOne("typesafe/jev-1.14").decide(
        "user: me cobraron dos veces y necesito el dinero hoy",
        {
            "team": choice(
                "¿A qué equipo va?",
                {"billing": "cobros y pagos", "tech": "fallos técnicos"},
            ),
            "urgent": noul("¿Es urgente?"),
        },
    )
    assert decision["team"].choice == "billing"
    assert decision["urgent"].noul > 0.5
