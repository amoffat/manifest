import base64
from typing import Optional, Union
from enum import Enum
from io import BytesIO

import requests

from manifest.llm.base import LLM
from manifest.llm.models import ModelConfig, supports_vision_from_model_config
from manifest.types.service import Service

# "Open" AI - GARBAGE because of this:
# What you cannot do. You may not use our Services for any
# illegal, harmful, or abusive activity. For example,
# you may not: Use Output to develop models that compete with OpenAI.
# (HINT: THIS INCLUDES EVERYTHING)
from manifest.llm import openai_service

OpenAI = openai_service.OpenAILLM


# Mistral Remote LLM Client - GARBAGE because of this:
# You must: Not use the Services for a benefit of a third party
# unless agreed otherwise in a separate contract with Us.
# (HINT: THIS MEANS YOU CAN ONLY BENEFIT YOURSELF)
class MistralRemoteModelConfig(Enum):
    OPEN_CODESTRAL_MAMBA = ModelConfig(
        model_name="open-codestral-mamba",
        context_limit=256_000,
        supports_vision=False,
    )
    OPEN_MISTRAL_NEMO = ModelConfig(
        model_name="open-mistral-nemo",
        context_limit=128_000,
        supports_vision=False,
    )


class MistralRemote(LLM):
    __slots__ = ("model_config", "session", "api_key")
    model_config: MistralRemoteModelConfig
    session: requests.Session
    api_key: str

    def __init__(
        self,
        *,
        api_key: str,
        model_config: MistralRemoteModelConfig = MistralRemoteModelConfig.OPEN_CODESTRAL_MAMBA,  # noqa: 501
    ):
        assert isinstance(
            model_config, MistralRemoteModelConfig
        ), f"{model_config} not for MistralRemote"
        self.api_key = api_key
        self.model_config = model_config
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @staticmethod
    def service() -> Service:
        return Service.MISTRAL_REMOTE

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images:
            raise ValueError("Image input is not supported for Mistral Remote LLM.")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        response = self.request_completion(messages=messages)
        return response["choices"][0]["message"]["content"]

    def request_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1,
        max_tokens: int = 512,
        stream: bool = False,
        safe_prompt: bool = False,
        random_seed: Optional[int] = None,
    ) -> dict:
        url = "https://api.mistral.ai/v1/chat/completions"
        data = {
            "model": self.model_config.value.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
            "safe_prompt": safe_prompt,
            "random_seed": random_seed,
        }

        response = self.session.post(url, json=data)
        response.raise_for_status()
        completion = response.json()
        return completion


# Gemini LLM Client - GARBAGE because of this:
# You may not use the Services to develop models that compete with the Services
# (HINT: THIS INCLUDES EVERYTHING)
# (e.g., Gemini API or Google AI Studio).
# You also may not attempt to extract or replicate the underlying models
# (e.g., parameter weights).
class GeminiModelConfig(Enum):
    PRO = ModelConfig(
        model_name="gemini-1.5-pro-exp-0801",
        context_limit=2_097_152,
        output_limit=8_192,
        supports_vision=True,
    )
    FLASH = ModelConfig(
        model_name="gemini-1.5-flash",
        context_limit=1_048_576,
        output_limit=8_192,
        supports_vision=True,
    )


class Gemini(LLM):
    _slots__ = ("model_config", "session", "api_key")
    model_config: GeminiModelConfig
    session: requests.Session
    api_key: str

    def __init__(
        self,
        *,
        api_key: str,
        model_config: GeminiModelConfig = GeminiModelConfig.PRO,
    ):
        self.model_config = model_config
        self.session = requests.Session()
        self.api_key = api_key

    @staticmethod
    def service() -> Service:
        return Service.GEMINI

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        # Handle images if supported by the model
        if images and supports_vision_from_model_config(self.model_config):
            image_content = []
            for image in images:
                b64_data = base64.b64encode(image.getvalue()).decode("utf-8")
                image_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_data,
                        },
                    }
                )

            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [*image_content, {"type": "text", "text": prompt}],
                },
            ]
        elif images:
            raise ValueError(
                "Image input is not supported for model:"
                f" {self.model_config.value.model_name}"
            )
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
        response = self._generate_content(messages=messages)  # type: ignore
        return response

    def _generate_content(
        self,
        messages: list[dict[str, Union[str, list]]],
        temperature: float = 1.0,
        max_output_tokens: int = 800,
        top_p: float = 0.8,
        top_k: int = 10,
    ) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_config.value.model_name}:generateContent?key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"parts": [{"text": message["content"]} for message in messages]}
            ],
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": top_p,
                "topK": top_k,
            },
        }
        response = self.session.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        # Extract and return the generated text
        return response_data["candidates"][0]["content"]


# Anthropic LLM Client - GARBAGE because of this:
# You may not access or use,
# or help another person to access or use,
# our Services in the following ways:
# To develop any products or services that compete with our Services,
# (HINT: THIS INCLUDES EVERYTHING)
# including to develop or train
# any artificial intelligence or machine learning algorithms or models.
class AnthropicModelConfig(Enum):
    CLAUDE_3_5_SONNET = ModelConfig(
        model_name="claude-3-5-sonnet-20240620",
        context_limit=200_000,
        output_limit=8_192,
        supports_vision=True,
    )


class Anthropic(LLM):
    __slots__ = ("model_config", "session", "api_key")
    model_config: AnthropicModelConfig
    session: requests.Session
    api_key: str

    def __init__(
        self,
        *,
        api_key: str,
        model_config: AnthropicModelConfig = AnthropicModelConfig.CLAUDE_3_5_SONNET,
    ):
        self.api_key = api_key
        self.model_config = model_config
        self.session = requests.Session()
        self.session.headers.update(
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        )

    @staticmethod
    def service() -> Service:
        return Service.ANTHROPIC

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images and supports_vision_from_model_config(self.model_config):
            image_content = []
            for image in images:
                b64_data = base64.b64encode(image.getvalue()).decode("utf-8")
                image_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_data,
                        },
                    }
                )

            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [*image_content, {"type": "text", "text": prompt}],
                },
            ]
        elif images:
            raise ValueError(
                "Image input is not supported for model:"
                f" {self.model_config.value.model_name}"
            )
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
        response = self.request_completion(messages=messages)  # type: ignore
        return response["content"][0]["text"]

    def request_completion(
        self,
        messages: list[dict[str, Union[str, list]]],
        max_tokens: int = 1024,
    ) -> dict:
        url = "https://api.anthropic.com/v1/messages"
        data = {
            "model": self.model_config.value.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
