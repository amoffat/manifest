import os
from typing import TYPE_CHECKING, Callable

from manifest.types.service import Service

if TYPE_CHECKING:
    from manifest.llm.base import LLM


# Will be replaced during manual initialization
def make_llm() -> "LLM":
    make = env_init()
    return make()


def init(make: Callable[[], "LLM"]) -> None:
    """Manually initialize the LLM client as an alternative to env_init()."""
    global make_llm
    make_llm = make


def env_init() -> Callable[[], "LLM"]:
    """Examines the .env and environment variables and returns a function that
    produces the appropriate llm client."""

    from dotenv import load_dotenv

    load_dotenv()

    service = Service(os.getenv("LLM_SERVICE", "auto"))

    if service == Service.AUTO:
        openai = os.getenv("OPENAI_API_KEY")
        if openai:
            service = Service.OPENAI
        else:
            raise ValueError("No LLM service discovered")

    elif service == Service.OPENAI:
        openai = os.getenv("OPENAI_API_KEY")
        if not openai:
            raise ValueError("OPENAI_API_KEY is required for OpenAI service")

    # Now we know what service we want to use

    if service == Service.OPENAI:

        def make_llm(**kwargs) -> "LLM":
            from manifest.llm.openai import OpenAILLM

            return OpenAILLM(
                api_key=openai,
                model="gpt-4o",
                **kwargs,
            )

        return make_llm

    raise ValueError(f"Unknown service: {service}")
