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
        model_errors = [] # Track errors for specific models
        total = 0

        personas_text = ", ".join(req.personas)
        topics_text = ", ".join(req.topics)

        for model in req.models:
            try:
                # 1. Try to initialize the LLM and generate prompts
                llm = get_llm(model)

                generated = generate_prompts(
                    brand=req.brand,
                    product=req.product,
                    persona=personas_text,
                    topic=topics_text,
                    num=req.num_prompts,
                    llm=llm,
                )

                # 2. If successful, append to results
                results.append({
                    "model": model,
                    "prompts": generated
                })

                total += len(generated)

            except Exception as model_error:
                # 3. If THIS model fails, log it and keep the loop going
                print(f"Error with model '{model}': {model_error}")
                model_errors.append({
                    "model": model, 
                    "error": str(model_error)
                })
                continue # Skip to the next model in req.models

        # 4. If ALL models failed, return an error response
        if not results and model_errors:
            return error_response(
                message="Failed to generate prompts for all requested models",
                errors=[e["error"] for e in model_errors]
            )

        # 5. Return success (even if partial)
        # Optional: You can pass model_errors into the data dict so the frontend knows what failed
        return success_response(
            message="Prompts generated successfully" if not model_errors else "Prompts generated with some model failures",
            data={
                "total_prompts": total,
                "results": results,
                "failed_models": model_errors # Lets the client know Perplexity failed while OpenAI succeeded
            }
        )

    except Exception as e:
        # This now only catches major outer errors (e.g., bad request data)
        return error_response(
            message="An unexpected error occurred",
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