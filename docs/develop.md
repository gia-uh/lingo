# Development & Testing Guide

This document outlines the development discipline, coding standards, and testing strategies for `lingo-ai`.

## 👩‍💻 Coding Standards

- **Python Version**: Use Python 3.12 or newer.
- **Type Safety**: All functions and classes must be type-hinted. We use Pydantic for data models.
- **Async First**: Lingo is built on the asynchronous execution model. Prefer `async/await` whenever possible.
- **Formatting**: Use `black` and `ruff` for code formatting and linting.
- **Dependency Injection**: Use `registry.depends()` to manage components like the LLM, the Engine, and the State.

## 🧪 Testing Strategy

Lingo applications should be tested thoroughly to ensure deterministic behavior and correct integration with the LLM. We recommend using `pytest` and `pytest-asyncio`.

### Unit Testing Skills & Tools
You can test individual skills and tools by providing mock versions of the `Context` and `Engine`.

```python
import pytest
from lingo import Message, Context
from lingo.mock import MockEngine

@pytest.mark.asyncio
async def test_onboarding_skill():
    # Setup
    ctx = Context()
    eng = MockEngine()
    eng.add_input("Alice") # Pre-configure the mock engine to respond to an 'ask'

    # Execute
    await onboarding(ctx, eng)

    # Validate
    assert "Alice" in ctx.render()
    assert any("Welcome" in msg.content for msg in ctx.history)
```

### Integration Testing (The Complete Loop)
For full application tests, you can mock the OpenAI API using the `lingo.mock` utilities or provide a controlled set of user inputs to the `chat()` method.

## 🤝 Contribution Guidelines

We welcome contributions from the community!

1.  **Fork the Repository**: Create your own fork and clone it locally.
2.  **Use `uv`**: Initialize your environment with `uv sync`.
3.  **Create a Branch**: Use descriptive branch names (e.g., `feat/add-new-node-type`, `fix/issue-123`).
4.  **Run Tests**: Ensure all tests pass before submitting a pull request.
    ```bash
    pytest tests/
    ```
5.  **Submit a PR**: Provide a clear description of your changes, including the rationale and any breaking changes.

### Git Workflow
We follow the standard GitHub Flow:
- Maintain a clean commit history.
- Use meaningful commit messages (e.g., `feat: implement retry node with exponential backoff`).
- Link your pull requests to related issues.
