import pytest
from pydantic import BaseModel
from lingo.utils import render, RenderStyle


# --- Setup Models ---
class Specs(BaseModel):
    cpu: str
    ram: str


class Machine(BaseModel):
    name: str
    specs: Specs
    tags: list[str]


@pytest.fixture
def sample_data():
    return {
        "user": "Alice",
        "roles": ["admin", "editor"],
        "meta": {"active": True, "login_count": 42},
    }


@pytest.fixture
def sample_model():
    return Machine(
        name="Server-01", specs=Specs(cpu="M1", ram="16GB"), tags=["prod", "web"]
    )


# --- Tests ---


def test_render_defaults(sample_data):
    """Test default markdown rendering."""
    output = render(sample_data, title="User Profile")

    assert "## User Profile" in output
    assert "- **user**: Alice" in output
    assert "- **meta**:" in output
    assert "  - **active**: True" in output  # Check indentation
    assert "  - admin" in output  # Check list rendering


def test_render_pydantic(sample_model):
    """Test rendering a Pydantic model directly."""
    output = render(sample_model)

    # Keys should be bolded by default
    assert "- **name**: Server-01" in output
    assert "- **specs**:" in output
    assert "  - **cpu**: M1" in output
    # Lists
    assert "  - prod" in output


def test_custom_string_format(sample_data):
    """Test custom string templates (e.g., plain text / email style)."""
    email_style = RenderStyle(
        title_format="=== {title} ===\n",
        leaf_format="{indent}{key} -> {value}",
        node_format="{indent}[{key}]",
        indent_string="..",
    )

    output = render(sample_data, title="REPORT", style=email_style)

    assert "=== REPORT ===" in output
    assert "user -> Alice" in output
    assert "[meta]" in output
    assert "..active -> True" in output  # Check custom indent


def test_dynamic_callable_format():
    """Test using functions for dynamic formatting logic."""

    data = {"Section 1": {"Subsection A": ["Content"]}}

    # Logic: Top level keys are #, nested are ##, etc.
    # Note: 'level' starts at 0 for top-level keys
    def dynamic_header(key, value, level, indent):
        hashes = "#" * (level + 1)
        return f"\n{hashes} {key}\n"

    style = RenderStyle(
        node_format=dynamic_header,
        leaf_format="{indent}{key}: {value}",
        indent_string="",
    )

    output = render(data, style=style)

    assert "# Section 1" in output
    assert "## Subsection A" in output  # Nested deeper gets more hashes
    assert "- Content" in output  # Leaf uses string format


def test_list_formatting():
    """Test complex list handling."""
    data = ["Simple", {"Complex": "Item"}]

    style = RenderStyle(list_item_format="* {value}")

    output = render(data, style=style)

    assert "* Simple" in output
    assert "- **Complex**: Item" in output  # Default node format for inner dict


def test_empty_input():
    """Test safe handling of empty structures."""
    assert render({}) == ""
    assert render([], title="Empty") == "## Empty"
