"""Runtime configuration. Thresholds are first guesses, NOT tuned on data (see README)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    # ordered fallback chain, same models as the speech agent
    llm_models: list[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv(
            "LLM_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",") if m.strip()])
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "20"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_max_wait_s: float = float(os.getenv("LLM_MAX_WAIT_S", "10"))
    asr_model: str = os.getenv("ASR_MODEL", "whisper-large-v3-turbo")
    # escalation gate
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.7"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.2"))
    top_k: int = int(os.getenv("TOP_K", "3"))


settings = Settings()
