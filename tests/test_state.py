import pytest
from pydantic import BaseModel, ValidationError
from lingo.state import State


def test_init_and_access():
    """Test basic initialization and attribute access."""
    # 1. Init with dict and kwargs
    s = State({"a": 1}, b=2)

    # 2. Check dict access
    assert s["a"] == 1
    assert s["b"] == 2

    # 3. Check attribute access
    assert s.a == 1
    assert s.b == 2

    # 4. Check assignment
    s.c = 3
    assert s["c"] == 3

    # 5. Check deletion
    del s.c
    with pytest.raises(AttributeError):
        _ = s.c


def test_schema_validation():
    """Test Pydantic schema validation."""

    class Memory(BaseModel):
        count: int
        name: str = "Guest"

    # 1. Valid Init
    s = State({"count": 10}, schema=Memory)
    assert s.count == 10
    assert s.name == "Guest"  # Default applied

    # 2. Invalid Init
    with pytest.raises(ValueError, match="State validation failed"):
        State({"count": "not-a-number"}, schema=Memory)

    # 3. Validation on manual call
    s.count = "20"  # String assigned
    s.validate()  # Should coerce to int (20)
    assert s.count == 20
    assert isinstance(s.count, int)


def test_deep_copy_isolation():
    """Test that clone() creates a deep copy for normal keys."""
    original = State({"nested": {"x": 1}})
    clone = original.clone()

    # Modify clone
    clone.nested["x"] = 999

    # Verify original is untouched
    assert original.nested["x"] == 1
    assert clone.nested["x"] == 999


def test_shared_keys():
    """Test that shared_keys are passed by reference."""
    # A mutable shared resource (e.g., a list or object)
    shared_resource = ["initial"]

    s1 = State({"local": 1, "db": shared_resource}, shared_keys={"db"})
    s2 = s1.clone()

    # 1. Modify local (should be isolated)
    s2.local = 999
    assert s1.local == 1

    # 2. Modify shared (should reflect in both)
    s2.db.append("modified")
    assert s1.db == ["initial", "modified"]
    assert s1.db is s2.db  # Same object identity


def test_atomic_transaction_success():
    """Test atomic() commits changes on success."""
    s = State({"count": 0})

    with s.atomic():
        s.count += 1

    assert s.count == 1


def test_atomic_transaction_rollback():
    """Test atomic() rolls back changes on exception."""
    s = State({"count": 0})

    with pytest.raises(ValueError):
        with s.atomic():
            s.count = 100
            raise ValueError("Oops")

    # Should be rolled back to 0
    assert s.count == 0


def test_fork_scope():
    """Test fork() context manager (always rollback)."""
    s = State({"mode": "prod"})

    # 1. Fork block
    with s.fork():
        s.mode = "test"
        assert s.mode == "test"

    # 2. Verify rollback on exit
    assert s.mode == "prod"


def test_subclassing_ide_support():
    """Test the pattern used for IDE autocompletion."""

    class AgentState(State):
        score: int = 0
        user: str

    # Initialize with required field
    s = AgentState(user="Alice", score=10)

    assert s.score == 10
    assert s.user == "Alice"

    # Verify it still behaves like a state/dict
    s.score = 20
    assert s["score"] == 20

    # Verify cloning works with subclass
    clone = s.clone()
    assert isinstance(clone, AgentState)
    assert clone.score == 20
