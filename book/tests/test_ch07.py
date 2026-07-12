import pytest
from pydantic import BaseModel
from lingo import Lingo, flow
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM
from lingo.state import State

class GameState(State):
    score: int = 0
    lives: int = 3
    level: str = "beginner"

def test_attribute_read_write():
    s = GameState()
    assert s.score == 0
    s.score = 10
    assert s["score"] == 10  # dict access also works

def test_default_values():
    s = GameState()
    assert s.lives == 3
    assert s.level == "beginner"

class GameSchema(BaseModel):
    score: int
    lives: int
    level: str

def test_schema_validation_passes():
    s = GameState(schema=GameSchema)
    s.score = 5
    s.validate()  # no error — valid value

def test_schema_rejects_bad_type():
    s = GameState(schema=GameSchema)
    try:
        with s.atomic():
            s.score = "not-a-number"
            s.validate()
    except Exception:
        pass
    assert s.score == 0  # reverted to default

def test_atomic_rollback():
    s = GameState()
    try:
        with s.atomic():
            s.score = 99
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    assert s.score == 0

def test_fork_always_rolls_back():
    s = GameState()
    with s.fork():
        s.score = 42
        assert s.score == 42
    assert s.score == 0  # always reverted

def test_atomic_commits_on_success():
    s = GameState()
    with s.atomic():
        s.score = 7
    assert s.score == 7  # committed

def test_render_produces_yaml():
    s = GameState()
    s.score = 7
    yaml_text = s.render("score", "level")
    assert "score: 7" in yaml_text
    assert "level: beginner" in yaml_text

def test_render_all_fields():
    s = GameState()
    yaml_text = s.render()
    assert "lives" in yaml_text
    assert "score" in yaml_text

@pytest.mark.asyncio
async def test_stateful_bot_tracks_state():
    state = GameState()
    bot = Lingo(name="GameBot", llm=MockLLM(["OK!"] * 5), state=state)

    @bot.skill
    async def play(context: Context, engine: Engine):
        """Play the game."""
        state.score += 1
        context.append(f"Score: {state.score}")
        await engine.reply(context)

    await bot.chat("play")
    assert state.score == 1
    await bot.chat("play again")
    assert state.score == 2

