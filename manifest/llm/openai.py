from typing import TYPE_CHECKING

import openai

from manifest.llm.base import LLM
from manifest.types.service import Service

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class OpenAILLM(LLM):
    def __init__(self, *, api_key: str, model: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    @staticmethod
    def service() -> Service:
        return Service.OPENAI

    def call(self, *, prompt: str, system_msg: str) -> str:
        messages: list["ChatCompletionMessageParam"] = [
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        resp = completion.choices[0].message.content or ""
        return resp
