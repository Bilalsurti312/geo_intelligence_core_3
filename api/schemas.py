from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

# ==================================================
# 1️⃣ COMPANY VERIFICATION
# ==================================================

class CompanyVerifyRequest(BaseModel):
    url: HttpUrl


class CompanyVerifyResponse(BaseModel):
    valid: bool
    company: Optional[str] = None
    reason: Optional[str] = None


# ==================================================
# BASE (VERIFIED COMPANY)
# ==================================================

class CompanyBase(BaseModel):
    company: str = Field(..., description="Verified company name")


# ==================================================
# 2️⃣ DISCOVERY
# ==================================================

class ProductRequest(CompanyBase):
    pass


class PersonaRequest(CompanyBase):
    category: str


class TopicRequest(CompanyBase):
    category: str


# ==================================================
# 3️⃣ ANALYSIS (MULTI-LLM)
# ==================================================

class AnalysisRequest(BaseModel):
    topic: str
    persona: str

    models: List[str] = Field(
        ...,
        description="LLMs to use, e.g. ['openai', 'gemini']"
    )

    num_prompts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Prompts per model"
    )
