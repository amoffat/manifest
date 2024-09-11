import base64
import json
from dataclasses import make_dataclass
from hashlib import sha1
from io import BytesIO
from types import UnionType
from typing import TYPE_CHECKING, Any, Type

import openai

from manifest import serde
from manifest.llm.base import LLM
from manifest.types.service import Service

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionContentPartParam,
        ChatCompletionMessageParam,
    )


class OpenAILLM(LLM):
    def __init__(self, *, api_key: str, model: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    @staticmethod
    def service() -> Service:
        return Service.OPENAI

    @staticmethod
    def serialize(
        *,
        return_type: Type | UnionType,
        caller_ns: dict[str, Any],
    ) -> Any:
        """Serialize our return type to jsonschema, while also wrapping it in
        an envelope because OpenAI does not support top-level simple types
        (type: string), only objects."""
        ResponseEnvelope = make_dataclass(
            "ResponseEnvelope",
            [("contents", return_type)],
        )
        return serde.serialize(
            data_type=ResponseEnvelope,
            caller_ns=caller_ns,
        )

    @staticmethod
    def deserialize(
        schema: dict,
        data: Any,
        registry: dict[str, Type],
    ) -> Any:
        """Deserialize our data from OpenAI into the expected return type,
        while taking care to strip off the envelope beforehand."""
        data = data["contents"]
        old_defs = schema.pop("$defs", None)
        schema = schema["properties"]["contents"]
        if old_defs:
            schema["$defs"] = old_defs

        obj = serde.deserialize(
            schema=schema,
            data=data,
            registry=registry,
        )
        return obj

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        response_schema: dict[str, Any],
        images: list[BytesIO] | None = None,
    ) -> str:

        user_message: "ChatCompletionMessageParam" = {
            "role": "user",
            "content": prompt,
        }

        if images:
            content: list["ChatCompletionContentPartParam"] = [
                {"type": "text", "text": prompt}
            ]
            for image in images:
                b64_data = base64.b64encode(image.getvalue()).decode("utf-8")
                img_block: "ChatCompletionContentPartParam" = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_data}",
                    },
                }
                content.append(img_block)

            user_message = {
                "role": "user",
                "content": content,
            }

        messages: list["ChatCompletionMessageParam"] = [
            {
                "role": "system",
                "content": system_msg,
            },
            user_message,
        ]

        # Not exactly sure what the response name is for on OpenAI's end, the
        # jsonschema launch post seems to suggest it is for caching purposes, so
        # we'll make it predictable at least.
        response_name = sha1(
            json.dumps(response_schema, sort_keys=True).encode("utf8")
        ).hexdigest()

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_name,
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )

        resp = completion.choices[0].message.content or ""
        return resp
