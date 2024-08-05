# renamed due to name collision with the python package
import base64
from io import BytesIO
from typing import TYPE_CHECKING

import openai

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

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
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

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # Somehow produces worse outcomes
            # response_format={"type": "json_object"},
        )

        resp = completion.choices[0].message.content or ""
        return resp
