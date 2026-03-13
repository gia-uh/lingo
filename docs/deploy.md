# Execution & Deployment Guide

This guide describes how to configure, run, and deploy Lingo-powered applications.

## 🛠️ Environment Setup

Lingo requires an OpenAI-compatible API to function.

1.  **Dependencies**: We recommend using `uv` for fast, reproducible dependency management.
    ```bash
    uv add lingo-ai
    ```

2.  **Environment Variables**: Create a `.env` file in your project root or export these variables in your shell:
    ```bash
    OPENAI_API_KEY=sk-your-key-here
    OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o-mini
    OPENAI_BASE_URL=...   # Optional, for custom endpoints
    ```

---

## 🚀 Running Your Application

### CLI Loop (Development)
For quick testing and prototyping, use the built-in `loop` function to run your bot in the terminal.

```python
from lingo.cli import loop
from my_app import bot

if __name__ == "__main__":
    loop(bot)
```

### Library Mode (Production)
In a production environment (e.g., a web server), you should use the asynchronous `chat()` method to process messages manually.

```python
import asyncio
from lingo import Message

async def handle_request(user_input: str):
    # Initialize your bot and context
    # ...
    # Send a message and get the response
    response_messages = await bot.chat(Message.user(user_input), context)
    return response_messages[-1].content
```

---

## 🐳 Dockerization

To deploy a Lingo application as a containerized service, use the following `Dockerfile` template.

```dockerfile
# Use a slim Python 3.12 image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev groups)
RUN uv sync --frozen --no-dev

# Copy the rest of the application
COPY . .

# Set environment variables (prefer passing these at runtime)
# ENV OPENAI_API_KEY=...

# Run the application
CMD ["uv", "run", "python", "main.py"]
```

### Deployment Best Practices
- **Security**: Never hardcode your `OPENAI_API_KEY` in the Dockerfile. Use Docker Secrets or environment variable injection at runtime.
- **Statelessness**: If your application uses a custom `State` class, remember that Lingo's in-memory state is tied to the current process. For multi-node deployments, you may need to implement a persistence layer (e.g., Redis or a database) to sync state across instances.
- **Monitoring**: Use Lingo's `verbose=True` flag during deployment testing to log the internal flow execution and LLM interactions.
