from abc import ABC, abstractmethod
from io import BytesIO
from types import UnionType
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from manifest.types.service import Service


class LLM(ABC):
    """The base class for all LLM clients."""

    @abstractmethod
    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        response_schema: dict[str, Any],
        images: list[BytesIO] | None = None,
    ) -> str:
        """Run an LLM completion"""

    @staticmethod
    @abstractmethod
    def service() -> "Service":
        """Return the service that this LLM client is using"""

    @staticmethod
    @abstractmethod
    def serialize(
        *,
        return_type: Type | UnionType,
        caller_ns: dict[str, Any],
    ) -> Any:
        """Serialize the return type of a function into jsonschema"""

    @staticmethod
    @abstractmethod
    def deserialize(
        schema: dict,
        data: Any,
        registry: dict[str, Type],
    ) -> Any:
        """Deserialize the response from an LLM call into the expected return
        type"""
