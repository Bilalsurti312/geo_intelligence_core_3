from fastapi import FastAPI
from api.schemas import *
from discovery.products import extract_products
from discovery.personas import generate_personas
from discovery.topics import generate_topics
from discovery.company import verify_company_from_url
from analysis.analyze import run_analysis

app = FastAPI(title="GEO Intelligence Core")

class CompanyVerifyRequest(BaseModel):
    url: str
@app.post("/verify-company")
def verify_company(req: CompanyVerifyRequest):
    return verify_company_from_url(req.url)
@app.post("/products")
def products(req: ProductRequest):
    return {"products": extract_products(req.company)}

@app.post("/personas")
def personas(req: PersonaRequest):
    return {"personas": generate_personas(req.company, req.category)}

@app.post("/topics")
def topics(req: TopicRequest):
    return {"topics": generate_topics(req.company, req.category)}

@app.post("/prompts")
def analyze(req: AnalysisRequest):
    return run_analysis(
        req.topic,
        req.persona,
        req.models,
        req.num_prompts
    )
