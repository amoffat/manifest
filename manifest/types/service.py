from enum import Enum


class Service(Enum):
    AUTO = "auto"
    GROQ = "groq"
    MLX_LM = "mlx_lm"
    MISTRAL_LOCAL = "mistral_local"
    HUGGINGFACE = "huggingface"
    MISTRAL_REMOTE = "mistral_remote"
    # BUG: the following have legal terms which prohibit use of output for work:
    # see llm/GARBAGE.py:
    ANTHROPIC = "anthropic"  # meh
    GEMINI = "gemini"  # meh
    OPENAI = "openai"  # meh
