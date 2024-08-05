import pathlib
import sys
from manifest.llm import models


def test_models_can_create_mlx():
    if sys.platform == "darwin":
        model_config = models.GroqModelConfig.LLAMA_3_1_INSTANT
        api_key = "FAKE"
        llm = models.Groq(
            model_config=model_config,
            api_key=api_key,
        )
        assert llm is not None, "failed to create mlx lm on mac"


def test_models_can_create_hf_qwen2():
    model_config = models.HFModelConfig.QWEN_2_7B
    llm = models.HuggingFace(model_config=model_config)
    assert llm is not None, "failed to create qwen2 llm with hf, \n{model_config=}"


def test_models_can_create_hf_phi3_small():
    model_config = models.HFModelConfig.PHI_3_SMALL
    llm = models.HuggingFace(model_config=model_config)
    assert llm is not None, "failed to create phi3 small with hf, \n{model_config=}"


def test_models_can_create_hf_llama_3_1():
    model_config = models.HFModelConfig.META_LLAMA_3_1_8B
    llm = models.HuggingFace(model_config=model_config)
    assert llm is not None, "failed to create llama3 8b with hf, \n{model_config=}"


def test_models_can_create_gemma_2b_10m():
    model_config = models.HFModelConfig.GEMMA_2B_10M
    llm = models.HuggingFace(model_config=model_config)
    assert llm is not None, "failed to create gemma_2b_10m with hf, \n{model_config=}"


mistral_models_base_path = (
    pathlib.Path().resolve() / "manifest" / "llm" / "mistral_models"
)


def test_models_can_create_codestral_mamba_local():
    mamba_config = models.MistralLocalModelConfig.CODESTRAL_MAMBA_7B
    mamba = models.MistralLocal(
        model_config=mamba_config,
        base_path=mistral_models_base_path,
    )
    assert mamba is not None, "failed to create mistral local mamba llm"


def test_models_can_create_mistral_local():
    model_config = models.MistralLocalModelConfig.MISTRAL_7B
    model = models.MistralLocal(
        model_config=model_config,
        base_path=mistral_models_base_path,
    )
    assert model is not None, "failed to create mistral local llm"


def test_models_can_create_mathstral_local():
    model_config = models.MistralLocalModelConfig.MATHSTRAL_7B
    model = models.MistralLocal(
        model_config=model_config,
        base_path=mistral_models_base_path,
    )
    assert model is not None, "failed to create mistral local llm"


# Remote:
def test_models_can_create_groq_remote():
    model_config = models.GroqModelConfig.LLAMA_3_1_INSTANT
    api_key = "FAKE"
    llm = models.Groq(
        model_config=model_config,
        api_key=api_key,
    )
    assert llm is not None
    assert "Authorization" in llm.session.headers
    assert llm.session.headers["Authorization"] == f"Bearer {api_key}"
