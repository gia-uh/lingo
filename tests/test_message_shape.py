from lingo.llm import Message, ToolCall


def test_tool_call_model():
    tc = ToolCall(id="call_xyz", name="read", arguments={"path": "foo.py"})
    assert tc.id == "call_xyz"
    assert tc.name == "read"
    assert tc.arguments == {"path": "foo.py"}


def test_assistant_message_defaults():
    msg = Message.assistant("hello")
    assert msg.tool_calls is None
    assert msg.thinking is None
    assert msg.stop_reason is None


def test_assistant_message_with_tool_calls():
    tc = ToolCall(id="c1", name="read", arguments={"path": "x"})
    msg = Message(role="assistant", content="thinking…", tool_calls=[tc],
                  thinking="some reasoning", stop_reason="tool_calls")
    assert msg.tool_calls == [tc]
    assert msg.thinking == "some reasoning"
    assert msg.stop_reason == "tool_calls"
