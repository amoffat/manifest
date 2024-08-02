from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifest.types.service import Service


class LLM(ABC):
    """The base class for all LLM clients."""

    @abstractmethod
    def call(self, *, prompt: str, system_msg: str) -> str:
        """Run an LLM completion"""
        ...

    @staticmethod
    @abstractmethod
    def service() -> "Service":
        """Return the service that this LLM client is using"""
        ...
