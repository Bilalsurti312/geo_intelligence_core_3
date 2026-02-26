import os
import requests
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Set up a basic logger for fallback notifications
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Perplexity Wrapper
# ---------------------------------------------------------
class PerplexityLLM:
    def __init__(self, api_key: str, model="sonar"):
        if not api_key:
            raise ValueError("Perplexity API key is missing.")
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


# ---------------------------------------------------------
# Core LLM Factory
# ---------------------------------------------------------
def get_llm(provider: str):
    """
    Initializes and returns the requested LLM provider.
    """
    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Azure OpenAI API key is missing.")
            
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=api_key,
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
            max_tokens=1500,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is missing.")
            
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=1024,
        )

    if provider == "perplexity":
        return PerplexityLLM(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            model="sonar"  # cheapest chat-compatible model
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------
# Resilient Fallback Wrapper
# ---------------------------------------------------------
class ResilientLLM:
    """
    Wraps multiple LLM providers. Tries them in order.
    If one fails (initialization or invoke), it catches the error and moves to the next.
    """
    def __init__(self, providers: list[str]):
        if not providers:
            raise ValueError("Must provide at least one LLM provider.")
        self.providers = providers

    def invoke(self, prompt):
        errors = {}
        
        for provider in self.providers:
            try:
                logger.info(f"Attempting to generate response with: {provider.upper()}")
                
                # 1. Try to initialize the model
                llm = get_llm(provider)
                
                # 2. Try to invoke the model
                response = llm.invoke(prompt)
                
                logger.info(f"Successfully generated response using: {provider.upper()}")
                return response
                
            except Exception as e:
                # Catch the error, log it, and loop to the next provider
                logger.warning(f"{provider.upper()} failed. Error: {str(e)}")
                errors[provider] = str(e)
                continue

        # 3. If the loop finishes and all providers failed, raise a comprehensive error
        error_msg = f"All LLM providers failed. Diagnostic details: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


# ---------------------------------------------------------
# Discovery LLM (With Automatic Fallbacks)
# ---------------------------------------------------------
def get_discovery_llm(primary_provider: str | None = None):
    """
    Returns a ResilientLLM that defaults to the primary provider, 
    but automatically falls back to others if it fails.
    """
    # Define your default fallback hierarchy
    default_hierarchy = ["openai", "gemini", "perplexity"]
    
    primary = (primary_provider or "openai").lower()
    
    # Reorder the hierarchy so the requested primary provider is always first
    if primary in default_hierarchy:
        default_hierarchy.remove(primary)
    hierarchy = [primary] + default_hierarchy

    logger.info(f"ACTIVE DISCOVERY LLM HIERARCHY = {' -> '.join(hierarchy)}")

    return ResilientLLM(providers=hierarchy)