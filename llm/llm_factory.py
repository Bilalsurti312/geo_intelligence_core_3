# llm/llm_factory.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

load_dotenv()


# -----------------------------
# Core LLM factory
# -----------------------------
def get_llm(provider: str):
    provider = provider.lower()

    if provider == "openai":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
            max_tokens=1500,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
            max_output_tokens=1024,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


# -----------------------------
# Discovery LLM (SAFE + FALLBACK)
# -----------------------------
def get_discovery_llm(provider: str | None = None):
    """
    Discovery MUST be resilient.
    Default: ACTIVE_LLM
    Fallback: gemini
    """

    provider = (provider or os.getenv("ACTIVE_LLM", "gemini")).lower()

    try:
        return get_llm(provider)
    except Exception as e:
        # HARD fallback — discovery should never crash the system
        return get_llm("gemini")