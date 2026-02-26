"""Thin wrapper around the Anthropic SDK."""

import os

from anthropic import Anthropic


class AnthropicClient:
    """Client for Anthropic's Claude API.

    Initializes the Anthropic SDK once and reuses it across calls.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use. Defaults to claude-sonnet-4-20250514.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required: provide api_key or set ANTHROPIC_API_KEY"
            )
        self._client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def create_message(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 2048,
    ) -> str:
        """Send a message to Claude and return the response text.

        Args:
            system: System prompt.
            user_message: User message content.
            max_tokens: Maximum tokens in response.

        Returns:
            The text content of Claude's response.

        Raises:
            ValueError: If Claude returns an empty response.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        if not response.content:
            raise ValueError("Empty response from Claude API")
        return response.content[0].text
