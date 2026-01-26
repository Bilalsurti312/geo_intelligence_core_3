from fastapi import APIRouter
from pydantic import BaseModel
from llm.llm_factory import get_llm

router = APIRouter()

class ContentGenerationRequest(BaseModel):
    topic: str

class ContentGenerationResponse(BaseModel):
    topic: str
    content_type: str
    content: str

@router.post("/content-generation", response_model=ContentGenerationResponse)
def generate_content(payload: ContentGenerationRequest):
    llm = get_llm("openai")

    prompt = f"""
You are a senior luxury industry content strategist.

Your task is to generate a high-quality blog article
focused on improving visibility and authority for the topic below.

TOPIC:
{payload.topic}

CONTENT OBJECTIVE:
- Strengthen topical authority
- Educate readers
- Improve strategic visibility of the topic
- No promotional tone
- No sales language

CONTENT RULES:
- Blog-style content
- Professional, authoritative tone
- Clear structure with headings
- 800–1200 words
- Insight-driven, not generic
- Avoid fluff and repetition
- No brand pushing unless contextually necessary

OUTPUT:
Return ONLY the blog content as plain text.
"""

    resp = llm.invoke(prompt)

    return {
        "topic": payload.topic,
        "content_type": "blog",
        "content": resp.content.strip()
    }