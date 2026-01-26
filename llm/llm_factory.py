# llm/llm_factory.py
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
load_dotenv()

# Perplexity Wrapper
class PerplexityLLM:
    def __init__(self, api_key: str, model="sonar"):
        self.api_key = api_key
        self.model = model

    def invoke(self, prompt):
        # Convert LangChain messages to plain text
        if isinstance(prompt, list):
            prompt = "\n".join([m.content for m in prompt if hasattr(m, "content")])
        elif hasattr(prompt, "content"):
            prompt = prompt.content
        else:
            prompt = str(prompt)

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512   # REQUIRED by Perplexity
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        # Helpful debugging if it fails
        if response.status_code != 200:
            raise Exception(f"Perplexity API Error {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"]

        # Match LangChain-style response
        class R:
            def __init__(self, content):
                self.content = content

        return R(content)

# Core LLM Factory
def get_llm(provider: str | None = None):
    """
    Default LLM = OpenAI
    Other providers must be explicitly requested.
    """

    provider = (provider or "openai").lower()

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

    if provider == "perplexity":
        return PerplexityLLM(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            model="sonar"  # cheapest chat-compatible model
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

# Discovery LLM (OpenAI Default)
def get_discovery_llm(provider: str | None = None):
    """
    Discovery uses OpenAI by default.
    Gemini / Perplexity only if explicitly requested.
    """

    provider = (provider or "openai").lower()
    print("ACTIVE DISCOVERY LLM =", provider)

    return get_llm(provider)