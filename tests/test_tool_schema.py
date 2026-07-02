from typing import Literal, Optional

from lingo.llm import _python_type_to_json_schema as j
from lingo.tools import tool as lingo_tool
from lingo.llm import tool_to_openai_schema


# --- Task 1: richer type mapping ---

def test_literal_becomes_enum():
    s = j(Literal["a", "b"])
    assert s["enum"] == ["a", "b"]
    assert s["type"] == "string"


def test_list_becomes_array_with_items():
    s = j(list[str])
    assert s["type"] == "array"
    assert s["items"] == {"type": "string"}


def test_optional_is_nullable():
    s = j(Optional[int])
    assert "null" in s["type"] and "integer" in s["type"]


def test_plain_types_unchanged():
    assert j(str) == {"type": "string"}
    assert j(int) == {"type": "integer"}
    assert j(bool) == {"type": "boolean"}


# --- Task 2: defaults + docstring param docs ---

@lingo_tool
async def grep(pattern: str, path: str = ".") -> str:
    """Search files.

    Args:
        pattern: Regex to search for.
        path: Directory to search under.
    """
    return ""


def test_default_param_not_required_and_documented():
    s = tool_to_openai_schema(grep)["function"]
    props = s["parameters"]["properties"]
    assert s["parameters"]["required"] == ["pattern"]
    assert props["path"]["default"] == "."
    assert props["pattern"]["description"] == "Regex to search for."
    assert props["path"]["description"] == "Directory to search under."


# --- Task 3: pass-through pre-built schema ---

def test_prebuilt_schema_passthrough():
    grep.json_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string", "description": "rich"}},
        "required": ["pattern"],
    }
    try:
        s = tool_to_openai_schema(grep)["function"]
        assert s["parameters"]["properties"]["pattern"]["description"] == "rich"
    finally:
        grep.json_schema = None  # reset
