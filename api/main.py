from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.schemas import *
from discovery.products import extract_products
from discovery.personas import generate_personas
from discovery.topics import generate_topics
from discovery.company import verify_company_from_url
from analysis.report import generate_report
from analysis.prompts import generate_prompts
from llm.llm_factory import get_llm
from api.response import success_response, error_response

# App Initialization
app = FastAPI(title="GEO Intelligence Core")

# Global Exception Handler (Catches Unhandled Errors)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "errors": [str(exc)]
        }
    )

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.0.102:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EXTRA MODELS
class ContentGenerationRequest(BaseModel):
    topic: str

# ROOT
@app.get("/")
def greeting():
    return success_response(
        message="API is running successfully",
        data={"info": "Welcome to GEO Intelligence Application"}
    )

# DISCOVERY MODULE
@app.post("/verify-company")
def verify_company(req: CompanyVerifyRequest):
    try:
        result = verify_company_from_url(req.url)

        if not result.get("valid"):
            return error_response(
                message="Company verification failed",
                errors=[result.get("reason", "Invalid company")]
            )

        return success_response(
            message="Company verified successfully",
            data=result
        )

    except Exception as e:
        return error_response(
            message="Verification process failed",
            errors=[str(e)]
        )

@app.post("/products")
def products(req: ProductRequest):
    try:
        result = extract_products(req.company)

        return success_response(
            message="Products generated successfully",
            data={"products": result}
        )

    except Exception as e:
        return error_response(
            message="Failed to generate products",
            errors=[str(e)]
        )

@app.post("/personas")
def personas(req: PersonaRequest):
    try:
        result = generate_personas(req.company, req.product)

        return success_response(
            message="Personas generated successfully",
            data={"personas": result}
        )

    except Exception as e:
        return error_response(
            message="Failed to generate personas",
            errors=[str(e)]
        )

@app.post("/topics")
def topics(req: TopicRequest):
    try:
        result = generate_topics(
            req.company,
            req.product,
            req.personas
        )

        return success_response(
            message="Topics generated successfully",
            data={"topics": result}
        )

    except Exception as e:
        return error_response(
            message="Failed to generate topics",
            errors=[str(e)]
        )

# PROMPT GENERATION
@app.post("/prompts")
def prompts(req: AnalysisRequest):
    try:
        results = []
        total = 0

        personas_text = ", ".join(req.personas)
        topics_text = ", ".join(req.topics)

        for model in req.models:

            llm = get_llm(model)

            generated = generate_prompts(
                brand=req.brand,
                product=req.product,
                persona=personas_text,
                topic=topics_text,
                num=req.num_prompts,
                llm=llm,
            )

            results.append({
                "model": model,
                "prompts": generated
            })

            total += len(generated)

        return success_response(
            message="Prompts generated successfully",
            data={
                "total_prompts": total,
                "results": results
            }
        )

    except Exception as e:
        return error_response(
            message="Failed to generate prompts",
            errors=[str(e)]
        )

# REPORT GENERATION
@app.post("/report")
def report(payload: ReportRequest):
    try:
        result = generate_report(payload.dict())

        return success_response(
            message="Report generated successfully",
            data=result
        )

    except Exception as e:
        return error_response(
            message="Failed to generate report",
            errors=[str(e)]
        )

# CONTENT GENERATION
@app.post("/content-generation")
def content_generation(payload: ContentGenerationRequest):
    try:
        llm = get_llm("openai")

        prompt = f"""
You are a senior industry content strategist.

Generate a high-quality blog article
focused on improving visibility and authority for the topic below.

TOPIC:
{payload.topic}

CONTENT OBJECTIVE:
- Strengthen topical authority
- Educate readers
- Improve strategic visibility
- No promotional tone

CONTENT RULES:
- Blog style
- Professional tone
- Clear headings
- Insight-driven
- No fluff

Return ONLY plain text content.
"""

        resp = llm.invoke(prompt)

        raw_content = resp.content.strip()
        clean_content = (
            raw_content
            .replace("**", "")
            .replace("\n\n", "\n")
            .replace("\n", " ")
        )

        return success_response(
            message="Content generated successfully",
            data={
                "topic": payload.topic,
                "content_type": "blog",
                "content": clean_content
            }
        )

    except Exception as e:
        return error_response(
            message="Failed to generate content",
            errors=[str(e)]
        )