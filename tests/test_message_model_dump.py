"""Regression tests for Message.model_dump() — the OpenAI wire shape.

Two bugs were caught by the lovelaice live E2E and fixed in 2d245ac:
the dump method wasn't including `tool_calls` on assistant messages
or `tool_call_id` on tool-role messages. Both are REQUIRED by OpenAI's
chat completions API when replaying a tool-using conversation.
"""

from lingo.llm import Message, ToolCall


def test_assistant_message_dump_includes_tool_calls():
    """OpenAI rejects an assistant turn that emitted tool calls but doesn't
    replay them on the next call. Bug fixed in 2d245ac."""
    tc = ToolCall(id="call_xyz", name="read", arguments={"path": "foo.py"})
    msg = Message.assistant("thinking about it", tool_calls=[tc])
    dump = msg.model_dump()
    assert dump["role"] == "assistant"
    assert "tool_calls" in dump, (
        "assistant.tool_calls MUST be serialized for OpenAI replay"
    )
    assert isinstance(dump["tool_calls"], list)
    assert len(dump["tool_calls"]) == 1
    # OpenAI's expected shape: each entry has id + type + function.{name, arguments}
    serialized_tc = dump["tool_calls"][0]
    # Spot-check the essential fields the API requires.
    assert serialized_tc.get("id") == "call_xyz"
    # Either function.name + function.arguments structure (OpenAI native), or
    # at minimum the name/arguments are reachable somehow:
    assert "read" in str(serialized_tc), (
        f"name 'read' should appear somewhere in serialized tool call: {serialized_tc!r}"
    )


def test_assistant_message_dump_without_tool_calls_omits_field():
    """A plain assistant message (no tool calls) should not carry an empty
    tool_calls field — keeps the wire output clean."""
    msg = Message.assistant("plain reply")
    dump = msg.model_dump()
    # tool_calls should either be absent or None — definitely not [].
    assert dump.get("tool_calls") in (None,), (
        f"empty tool_calls should be omitted/None, got {dump.get('tool_calls')!r}"
    )
    # But it's also acceptable for it to be entirely absent:
    # the assertion above passes for either {} or {"tool_calls": None}.


def test_tool_message_dump_includes_tool_call_id():
    """OpenAI rejects tool-role messages without tool_call_id.
    Bug fixed in 2d245ac."""
    msg = Message.tool("the result content", tool_call_id="call_xyz")
    dump = msg.model_dump()
    assert dump["role"] == "tool"
    assert dump.get("tool_call_id") == "call_xyz", (
        "tool.tool_call_id MUST be serialized for OpenAI to link the result "
        "back to the originating tool call"
    )


def test_tool_call_round_trips_through_dump():
    """A full conversation segment (user → assistant.tool_calls → tool → assistant)
    should dump cleanly and the dumps should be ready to send back to OpenAI."""
    tc = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    msgs = [
        Message.user("please call echo"),
        Message.assistant("I'll call echo.", tool_calls=[tc]),
        Message.tool("hi", tool_call_id="c1"),
        Message.assistant("Done.", stop_reason="stop"),
    ]
    dumps = [m.model_dump() for m in msgs]
    assert dumps[0]["role"] == "user"
    assert dumps[1]["role"] == "assistant"
    assert "tool_calls" in dumps[1]
    assert dumps[2]["role"] == "tool"
    assert dumps[2]["tool_call_id"] == "c1"
    assert dumps[3]["role"] == "assistant"
