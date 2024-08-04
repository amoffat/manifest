from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifest.types.service import Service


class NoLLMFoundError(Exception):
    """When the automatic LLM initialization fails to find any LLM api keys in
    the environment."""

    pass


class UnknownLLMServiceError(Exception):
    """When the user specifies an unknown LLM service."""

    def __init__(self, service: str):
        self.service = service
        super().__init__(f"Unknown LLM service: {service}")


class NoApiKeyError(Exception):
    """When the automatic LLM initialization fails to find an api key for a
    specific LLM service."""

    def __init__(self, service: "Service"):
        self.service = service
        super().__init__(f"No API key found for {service}")
