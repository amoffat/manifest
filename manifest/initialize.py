import os
import sys
from typing import TYPE_CHECKING, Callable

from manifest import constants, exc
from manifest.types.service import Service

if TYPE_CHECKING:
    from manifest.llm.base import LLM


# Will be replaced during manual initialization
def make_llm() -> "LLM":
    header = """manifest.py error:"""
    manual_clause = """
For advanced users, you may manually initialize the LLM client in your code by
calling `manifest.init(client_maker)`, where `client_maker` is a function that
returns an LLM client.
""".strip()

    valid_services = "\n".join([f"  - {s.value}" for s in Service])
    valid_envs = "\n".join(
        [f"  - {constants.ENV_KEY_NAMES[s]}" for s in Service if s != Service.AUTO]
    )

    try:
        make = env_init()
        return make()
    except exc.UnknownLLMServiceError as e:
        print(
            f"""
{header}

Unknown LLM service: "{e.service}". Please specify one of the following
services instead:

{valid_services}

Exiting.
""",
            file=sys.stderr,
        )

    except exc.NoApiKeyError as e:
        print(
            f"""
{header}

No api key found for {e.service}, try defining the environment variable
{constants.ENV_KEY_NAMES[e.service]} in a .env file or in your environment, then
re-running the program.

{manual_clause}

Exiting.
""",
            file=sys.stderr,
        )

    except exc.NoLLMFoundError:
        print(
            f"""
{header}

No LLM api keys found, try defining one of the following environment variables
in a .env file or in your environment, then re-running the program:

{valid_envs}

{manual_clause}

Exiting.
""",
            file=sys.stderr,
        )

    exit(1)


def init(make: Callable[[], "LLM"]) -> None:
    """Manually initialize the LLM client as an alternative to env_init()."""
    global make_llm
    make_llm = make


def env_init() -> Callable[[], "LLM"]:
    """Examines the .env and environment variables and returns a function that
    produces the appropriate llm client."""

    from dotenv import load_dotenv

    load_dotenv()

    # Do you have multiple LLM keys? Allow the user to choose
    service_name = os.getenv("MANIFEST_SERVICE", "auto").lower()
    try:
        service = Service(service_name)
    except ValueError:
        raise exc.UnknownLLMServiceError(service_name)

    key_names = constants.ENV_KEY_NAMES

    if service == Service.AUTO:
        openai_key = os.getenv(key_names[Service.OPENAI])
        if openai_key:
            service = Service.OPENAI
        else:
            raise exc.NoLLMFoundError

    elif service == Service.OPENAI:
        openai_key = os.getenv(key_names[Service.OPENAI])
        if not openai_key:
            raise exc.NoApiKeyError(service)

    # Now we know what service we want to use

    if service == Service.OPENAI:

        def make_llm(**kwargs) -> "LLM":
            from manifest.llm.openai_service import OpenAILLM

            return OpenAILLM(
                api_key=openai_key,
                model="gpt-4o",
                **kwargs,
            )

        return make_llm

    raise exc.UnknownLLMServiceError(service_name)
