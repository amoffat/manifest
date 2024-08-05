from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
from io import BytesIO

import requests
import pathlib
import sys

from manifest.types.service import Service
from manifest.llm.base import LLM

# === Lazy Imports ===

# Mistral Inference
MistralLocalChatCompletionRequest = None
MistralLocalMessageRoles = None
MistralLocalSystemMessage = None
MistralLocalUserMessage = None
mistral_local_generate = None
MistralModel = None
MistralArgs = None

# Apple MLX
mlx_generate = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    model_name: str
    context_limit: int
    supports_vision: bool
    output_limit: int = -1


# "VARIANT = (model_name: str, context_limit: int, supports_vision: bool)"
class GroqModelConfig(Enum):
    "Docs: https://console.groq.com/docs/models"

    LLAMA_3_1_INSTANT = ModelConfig(
        model_name="llama-3.1-8b-instant",
        context_limit=131_072,
        output_limit=8_000,
        supports_vision=False,
    )
    LLAMA_3_1_70B_VERSATILE = ModelConfig(
        model_name="llama-3.1-70b-versatile",
        context_limit=131_072,
        output_limit=8_000,
        supports_vision=False,
    )


class MLXModelConfig(Enum):
    "Community: https://huggingface.co/mlx-community"

    META_LLAMA_3_1_8B = ModelConfig(
        model_name="mlx-community/Meta-Llama-3.1-8B-Instruct",
        context_limit=128_000,
        supports_vision=False,
    )
    GEMMA_2_2B = ModelConfig(
        model_name="mlx-community/gemma-2-2b-it",
        context_limit=8_192,
        supports_vision=False,
    )


class HFModelConfig(Enum):
    "Models: https://huggingface.co/models?pipeline_tag=text-generation&sort=trending"

    META_LLAMA_3_1_8B = ModelConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        context_limit=128_000,
        supports_vision=False,
    )
    PHI_3_SMALL = ModelConfig(
        model_name="microsoft/Phi-3-small-128k-instruct",
        context_limit=128_000,
        supports_vision=False,
    )
    MISTRAL_7B = ModelConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        context_limit=32_768,
        supports_vision=False,
    )
    QWEN_2_7B = ModelConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        context_limit=128_000,
        supports_vision=False,
    )
    GEMMA_2_9B = ModelConfig(
        model_name="google/gemma-2-9b-it",
        context_limit=8_192,
        supports_vision=False,
    )
    # NOTE: gemma 2b 10m may require special setup (DIY)
    GEMMA_2B_10M = ModelConfig(
        model_name="mustafaaljadery/gemma-2B-10M",
        context_limit=10_000_000,
        supports_vision=False,
    )


class MistralLocalModelConfig(Enum):
    CODESTRAL_MAMBA_7B = ModelConfig(
        model_name="codestral-mamba-7B-v0.1",
        context_limit=256_000,
        supports_vision=False,
    )
    MATHSTRAL_7B = ModelConfig(
        model_name="mathstral-7B-v0.1",
        context_limit=32_768,
        supports_vision=False,
    )
    MISTRAL_7B = ModelConfig(
        model_name="mistral-7B-Instruct-v0.3",
        context_limit=32_768,
        supports_vision=False,
    )


def model_name_from_model_config(model: Enum) -> str:
    return model.value.model_name


def context_limit_from_model_config(model: Enum) -> int:
    return model.value.context_limit


def supports_vision_from_model_config(model: Enum) -> bool:
    return model.value.supports_vision


# Groq LLM Client
class Groq(LLM):
    __slots__ = ("model_config", "session", "api_key")
    model_config: GroqModelConfig
    session: requests.Session
    api_key: str

    def __init__(
        self,
        *,
        api_key: str,
        model_config: GroqModelConfig = GroqModelConfig.LLAMA_3_1_INSTANT,
    ):
        assert isinstance(
            model_config, GroqModelConfig
        ), f"{model_config=} not for Groq"
        self.api_key = api_key
        self.model_config = model_config
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @staticmethod
    def service() -> Service:
        return Service.GROQ

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images:
            raise ValueError("Image input is not supported for Groq LLM.")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = self._chat_completion(messages=messages)
        # TODO: verify groq response json structure more carefully
        reply = response.choices[0]["message"]["content"]  # type: ignore
        return reply

    def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict[str, str]] = None,
        seed: Optional[int] = None,
        stop: Optional[str] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[dict[str, Any]] = None,
        tools: Optional[list] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
    ) -> dict:
        "Docs: https://console.groq.com/docs/api-reference#chat-create"
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "model": model_name_from_model_config(self.model_config),
            "messages": messages,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "stream_options": stream_options,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tools": tools,
            "top_p": top_p,
            "user": user,
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data


# MLX LLM Client (for Apple Silicon)
class MLX(LLM):
    """Example:
    https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/examples/generate_response.py
    """

    _slots__ = ("model_config", "_tokenizer", "_model")
    model_config: MLXModelConfig
    _tokenizer: Any  # BUG: missing type hints for MLX
    _model: Any  # BUG: missing type hints for MLX

    def __init__(
        self,
        *,
        model_config: MLXModelConfig = MLXModelConfig.META_LLAMA_3_1_8B,
    ):
        assert isinstance(model_config, MLXModelConfig), f"{model_config} not for MLX"
        if not sys.platform == "darwin":
            raise ValueError(
                "MLX LLM is only supported on Apple Silicon (darwin) platform."
            )
        global mlx_generate
        from mlx_lm import generate as mlx_generate  # type: ignore
        from mlx_lm import load  # type: ignore

        self.model_config = model_config
        self._model, self._tokenizer = load(
            path_or_hf_repo=model_config.value.model_name
        )

    @staticmethod
    def service() -> Service:
        return Service.MLX_LM

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images:
            raise ValueError("Image input is not supported for MLX LLM.")

        global mlx_generate
        if mlx_generate is None:
            raise ValueError(f"Missing {mlx_generate=}")

        conversation = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        _prompt = self._tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        generation_args = {
            "temp": 0.7,
            "repetition_penalty": 1.2,
            "repetition_context_size": 20,
            "top_p": 0.95,
        }

        response = mlx_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=_prompt,
            max_tokens=1_000,
            verbose=False,
            **generation_args,
        )

        return response


# HuggingFace LLM Client (for NVIDIA GPUs)
class HuggingFace(LLM):
    """
    Docs:
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
    """

    _slots__ = ("model_config", "_tokenizer", "text_streamer", "_model", "template")
    model_config: HFModelConfig
    _tokenizer: Any  # BUG: no types for hf w/ lazy load
    text_streamer: Any
    _model: Any
    template: str

    def __init__(
        self,
        *,
        model_config: HFModelConfig = HFModelConfig.META_LLAMA_3_1_8B,
    ):
        assert isinstance(
            model_config, HFModelConfig
        ), f"{model_config} not for HuggingFace"
        # TODO: fix this for non-nvidia
        import torch
        from transformers import (
            AutoTokenizer,
            TextStreamer,
            BitsAndBytesConfig,
            AutoModelForCausalLM,
        )

        self.model_config = model_config
        model_name = model_name_from_model_config(model_config)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        self.text_streamper = TextStreamer(self._tokenizer, skip_prompt=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_from_model_config(
                self.model_config
            ),
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self.template = """{% for message in messages %}
{{ message['role']}}: {{ message['content'] }}
{% endfor %}
{{ eos_token }}"""

    @staticmethod
    def service() -> Service:
        return Service.HUGGINGFACE

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images:
            raise ValueError("Image input is not supported for HuggingFace LLM.")

        chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        processed_chat = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.template,
        )

        input_ids = self._tokenizer(processed_chat, return_tensors="pt").to(
            self._model.device
        )
        n_tokens = len(input_ids.tokens())

        context_limit = context_limit_from_model_config(self.model_config)
        max_new_tokens = context_limit - n_tokens

        response = self._model.generate(
            **input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            streamer=self.text_streamer,
        )

        convo = self._tokenizer.batch_decode(response, skip_special_tokens=False)[0]
        n = len(processed_chat)

        _, convo_output = convo[:n], convo[n:]
        return convo_output


# Mistral Local LLM Client
class MistralLocal(LLM):
    __slots__ = (
        "model_config",
        "base_path",
        "tokenizer_file_name",
        "_tokenizer",
        "_model",
    )
    config: MistralLocalModelConfig
    base_path: pathlib.Path
    tokenizer_file_name: str
    _tokenizer: Any
    _model: Any

    def __init__(
        self,
        *,
        model_config: MistralLocalModelConfig = MistralLocalModelConfig.CODESTRAL_MAMBA_7B,  # noqa: E501
        base_path: pathlib.Path = pathlib.Path().resolve(),
        tokenizer_file_name: str = "tokenizer.model.v3",
    ):
        assert isinstance(
            model_config, MistralLocalModelConfig
        ), f"{model_config} not for MistralLocal"
        global mistral_local_generate
        global MistralArgs
        global MistralModel
        global MistralLocalUserMessage
        global MistralLocalSystemMessage
        global MistralLocalMessageRoles
        global MistralLocalChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.protocol.instruct.messages import (
            SystemMessage as MistralLocalSystemMessage,
            UserMessage as MistralLocalUserMessage,
            Roles as MistralLocalMessageRoles,
        )
        from mistral_common.protocol.instruct.request import (
            ChatCompletionRequest as MistralLocalChatCompletionRequest,
        )

        if model_config is MistralLocalModelConfig.MISTRAL_7B:
            from mistral_inference.transformer import Transformer as MistralModel
            from mistral_inference.transformer import TransformerArgs as MistralArgs
            from mistral_inference.main import generate as mistral_local_generate
        elif model_config is MistralLocalModelConfig.CODESTRAL_MAMBA_7B:
            from mistral_inference.mamba import Mamba as MistralModel
            from mistral_inference.mamba import MambaArgs as MistralArgs
            from mistral_inference.generate import (
                generate_mamba as mistral_local_generate,
            )
        assert MistralModel is not None, "failed to import MistralModel"

        # BUG: they're using a string path, so the slashes will crash on windows
        # BUG: the model and tokenizer may not exist, need to idempotently get if needed
        self.tokenizer_file_name = tokenizer_file_name
        self.model_config = model_config
        self.base_path = base_path
        model_name = model_name_from_model_config(model_config)
        model_folder_path = base_path / model_name
        tokenizer_path = model_folder_path / tokenizer_file_name

        tokenizer_file_nonexistent = False
        model_folder_nonexistent = False
        if not model_folder_path.exists():
            model_folder_nonexistent = True
        if not tokenizer_path.exists():
            tokenizer_file_nonexistent = True

        # TODO: implement mistral local idempotent downloading
        if tokenizer_file_nonexistent or model_folder_nonexistent:
            print("ERROR: you need to get this model from mistral")
            print(f"{model_config=}")
            print(f"{model_folder_path=}")
            print(f"{tokenizer_file_nonexistent=}")
            print(f"{model_folder_nonexistent=}")
            print(f"{tokenizer_file_nonexistent=}")
            raise AssertionError(
                f"{model_config=} not downloaded to {model_folder_path=}"
            )
        self._tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
        self._model = MistralModel.from_folder(model_folder_path)

    # https://github.com/mistralai/mistral-inference?tab=readme-ov-file#model-download
    # link,link,md5,md5,1,2,1,2
    # https://models.mistralcdn.com/codestral-mamba-7b-v0-1/codestral-mamba-7B-v0.1.tar
    # https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar
    # d3993e4024d1395910c55db0d11db163
    # 80b71fcb6416085bcb4efad86dfb4d52
    # wget \
    #  https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar
    # mkdir -p $12B_DIR
    # tar -xf mistral-nemo-instruct-v0.1.tar -C $12B_DIR
    def get_model(self) -> bool:
        print(
            "https://github.com/mistralai/mistral-inference"
            "?tab=readme-ov-file#model-download"
        )
        print("use Makefile")
        return False

    @staticmethod
    def service() -> Service:
        return Service.MISTRAL_LOCAL

    def call(
        self,
        *,
        prompt: str,
        system_msg: str,
        images: list[BytesIO] | None = None,
    ) -> str:
        if images:
            raise ValueError("Image input is not supported for Mistral Local LLM.")

        global MistralLocalChatCompletionRequest
        global MistralLocalSystemMessage
        global MistralLocalMessageRoles
        global MistralLocalUserMessage
        global mistral_local_generate
        if mistral_local_generate is None:
            raise ValueError(f"{mistral_local_generate=}")
        if MistralLocalUserMessage is None:
            raise ValueError(f"{MistralLocalUserMessage=}")
        if MistralLocalChatCompletionRequest is None:
            raise ValueError(f"{MistralLocalChatCompletionRequest=}")

        # note: these are on separate lines so error messages are more clear
        assert MistralLocalSystemMessage is not None
        assert MistralLocalMessageRoles is not None
        assert MistralLocalUserMessage is not None
        system_message = MistralLocalSystemMessage(
            role=MistralLocalMessageRoles.system, content=system_msg
        )
        prompt_message = MistralLocalUserMessage(
            role=MistralLocalMessageRoles.user, content=prompt
        )
        completion_request = MistralLocalChatCompletionRequest(
            messages=[system_message, prompt_message]
        )
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = mistral_local_generate(
            [tokens],
            self._model,
            max_tokens=1024,
            temperature=0.35,
            eos_id=self._tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        result = self._tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result
