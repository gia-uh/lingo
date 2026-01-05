import os
from typing import overload, Any
import openai


class Embedder:
    """
    A client for generating text embeddings using an OpenAI-compatible API.
    Handles automatic batching of requests to respect token limits.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens_per_batch: int = 8000,
        **extra_kwargs,
    ):
        """
        Initializes the Embedder.

        Args:
            model: The embedding model to use. Defaults to EMBEDDING_MODEL or text-embedding-3-small.
            api_key: API key for the client.
            base_url: Base URL for the API.
            max_tokens_per_batch: Maximum estimated tokens per API call.
            **extra_kwargs: Additional arguments for the embedding request.
        """
        self.model = model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        self.client = openai.AsyncOpenAI(
            base_url=base_url or os.getenv("BASE_URL"),
            api_key=api_key or os.getenv("API_KEY"),
        )
        self.max_tokens_per_batch = max_tokens_per_batch
        self.extra_kwargs = extra_kwargs

    def _estimate_tokens(self, text: str) -> int:
        """
        Provides a sensible approximation of token count.
        Standard heuristic: ~4 characters per token.
        """
        return len(text) // 4

    @overload
    async def embed(self, text: str) -> list[float]: ...

    @overload
    async def embed(self, text: list[str]) -> list[list[float]]: ...

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """
        Generates embeddings for the provided text.
        If a list of strings is provided, they are automatically batched.
        """
        if isinstance(text, str):
            response = await self.client.embeddings.create(
                model=self.model, input=text, **self.extra_kwargs
            )
            return response.data[0].embedding

        # Batching logic for list[str]
        all_embeddings: list[list[float]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for item in text:
            item_tokens = self._estimate_tokens(item)

            # Check if adding this item exceeds the batch limit
            if (
                current_tokens + item_tokens > self.max_tokens_per_batch
                and current_batch
            ):
                batch_response = await self.client.embeddings.create(
                    model=self.model, input=current_batch, **self.extra_kwargs
                )
                all_embeddings.extend([d.embedding for d in batch_response.data])
                current_batch = []
                current_tokens = 0

            current_batch.append(item)
            current_tokens += item_tokens

        # Process the final batch
        if current_batch:
            batch_response = await self.client.embeddings.create(
                model=self.model, input=current_batch, **self.extra_kwargs
            )
            all_embeddings.extend([d.embedding for d in batch_response.data])

        return all_embeddings
