import pytest
import pytest_asyncio
from unittest.mock import MagicMock, Mock, AsyncMock
from pydantic import BaseModel

from lingo import LLM, Message

# We'll need pytest-asyncio to test async functions
pytestmark = pytest.mark.asyncio


class SampleModel(BaseModel):
    """A simple Pydantic model for testing the 'create' method."""

    name: str
    value: int


async def mock_chat_stream(chunks: list[str]):
    """
    A helper async generator to mimic the OpenAI streaming API.
    It yields mock chunks in the structure lingo.llm.LLM expects.
    """
    for content in chunks:
        mock_delta = Mock(content=content)
        mock_choice = Mock(delta=mock_delta)
        mock_chunk = Mock(choices=[mock_choice])
        yield mock_chunk

    # The real API sends a final chunk with None content
    mock_delta_final = Mock(content=None)
    mock_choice_final = Mock(delta=mock_delta_final)
    mock_chunk_final = Mock(choices=[mock_choice_final])
    yield mock_chunk_final


@pytest_asyncio.fixture
def mock_client(mocker):
    """
    This fixture patches 'openai.AsyncOpenAI' at the project level.
    Any time 'LLM()' is initialized, it will get this MagicMock
    instance instead of a real OpenAI client.
    """
    # Patch the class to return a MagicMock instance
    mock_instance = MagicMock()
    mocker.patch("lingo.llm.openai.AsyncOpenAI", return_value=mock_instance)
    return mock_instance


async def test_llm_init(mock_client):
    """Test that the LLM initializes and calls the OpenAI client."""
    llm = LLM(model="gpt-4", api_key="test_key", base_url="test_url")

    assert llm.model == "gpt-4"
    # Verify that our patch was used and the client was instantiated
    # (via the lingo.llm.openai import)
    from lingo.llm import openai

    openai.AsyncOpenAI.assert_called_with(base_url="test_url", api_key="test_key")


async def test_llm_chat_streaming(mock_client):
    """Test the 'chat' method, mocking a streaming response."""
    llm = LLM(model="gpt-4")

    # 1. Define the mock response
    stream_content = ["Hello", ",", " ", "world", "!"]
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_chat_stream(stream_content)
    )

    # 2. Call the method
    messages = [Message.user("Hi")]
    response = await llm.chat(messages)

    # 3. Assert the results
    assert response.role == "assistant"
    assert response.content == "Hello, world!"

    # 4. Verify the mock was called correctly
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )


async def test_llm_chat_with_callback(mock_client):
    """Test that the 'chat' method calls the callback with each chunk."""
    stream_content = ["One", "Two", "Three"]
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_chat_stream(stream_content)
    )

    # Create a callback function to track calls
    callback_chunks = []

    def sync_callback(chunk):
        callback_chunks.append(chunk)

    llm = LLM(model="gpt-4", on_token=sync_callback)

    await llm.chat([Message.user("Count")])

    assert callback_chunks == ["One", "Two", "Three"]


async def test_llm_create_pydantic(mock_client):
    """Test the 'create' method, mocking a Pydantic response."""
    llm = LLM(model="gpt-4")

    # 1. Define the mock response
    expected_result = SampleModel(name="Lingo", value=123)

    # This mocks the response structure: response.choices[0].message.parsed
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = expected_result

    # Set the return value for the 'parse' method
    mock_client.chat.completions.parse = AsyncMock(return_value=mock_response)

    # 2. Call the method
    messages = [Message.user("Create the object")]
    result = await llm.create(SampleModel, messages)

    # 3. Assert the results
    assert result == expected_result
    assert isinstance(result, SampleModel)

    # 4. Verify the mock was called correctly
    mock_client.chat.completions.parse.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Create the object"}],
        response_format=SampleModel,
    )
