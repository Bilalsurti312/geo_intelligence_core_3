from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.schemas import *
from discovery.products import extract_products
from discovery.personas import generate_personas
from discovery.topics import generate_topics
from discovery.company import verify_company_from_url
from analysis.analyze import run_analysis
from analysis.report import generate_report
from analysis.prompts import generate_prompts
from llm.llm_factory import get_llm

# App Initialization
app = FastAPI(title="GEO Intelligence Core")

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.0.102:8080",
        "*"  # DEV ONLY — remove later in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BASE MODELS
class CompanyVerifyRequest(BaseModel):
    url: str

class ContentGenerationRequest(BaseModel):
    topic: str

class ContentGenerationResponse(BaseModel):
    topic: str
    content_type: str
    content: str

# ROOT
@app.get("/")
def greeting():
    return "Welcome to GEO Intelligence Application"

# DISCOVERY
@app.post("/verify-company")
def verify_company(req: CompanyVerifyRequest):
    return verify_company_from_url(req.url)

@app.post("/products")
def products(req: ProductRequest):
    return {"products": extract_products(req.company)}

@app.post("/personas")
def personas(req: PersonaRequest):
    return {"personas": generate_personas(req.company, req.product)}

@app.post("/topics")
def topics(req: TopicRequest):
    return {"topics": generate_topics(req.company, req.product, req.persona)}

# PROMPT GENERATION
@app.post("/prompts")
def prompts(req: AnalysisRequest):
    results = []
    total = 0

    for model in req.models:
        llm = get_llm(model)

        prompts = generate_prompts(
            product=req.product,
            persona=req.persona,
            topic=req.topic,
            num=req.num_prompts,
            llm=llm,
        )

        results.append({
            "model": model,
            "prompts": prompts
        })

        total += len(prompts)

    return {
        "total_prompts": total,
        "results": results
    }

# REPORT
@app.post("/report")
def report(payload: ReportRequest):
    return generate_report(payload.dict())

# CONTENT GENERATION MODULE
@app.post("/content-generation", response_model=ContentGenerationResponse)
def content_generation(payload: ContentGenerationRequest):
    """
    Generates blog-style content to improve visibility for a given topic.
    """

    llm = get_llm("openai")

    prompt = f"""
You are a senior industry content strategist.

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

    # Clean markdown & newlines
    raw_content = resp.content.strip()
    clean_content = (
        raw_content
        .replace("**", "")
        .replace("\n\n", "\n")
        .replace("\n", " ")
    )

    return {
        "topic": payload.topic,
        "content_type": "blog",
        "content": clean_content
    }