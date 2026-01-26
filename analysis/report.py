from typing import Dict
from collections import defaultdict
import json
import re
import os
from llm.llm_factory import get_llm

# Helper: Extract JSON safely
def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}

# Per-Model Evaluation
def evaluate_per_model(payload: Dict, model: str, answers: list) -> Dict:
    """
    LLM-only evaluation for a SINGLE model.
    NO model_visibility here.
    """

    llm = get_llm("openai")

    formatted_answers = "\n".join(
        [f"Answer {i+1}: {a}" for i, a in enumerate(answers)]
    )

    prompt = f"""
You are a senior market intelligence analyst reviewing insights from ONE AI model.

INPUT CONTEXT:
Brand: {payload["brand"]}
Category / Product: {payload.get("product")}
Personas: {payload.get("personas")}
Topics: {payload.get("topics")}

MODEL UNDER REVIEW: {model.upper()}

MODEL ANSWERS:
{formatted_answers}

TASK:
Evaluate how strong, relevant, and strategically useful this model's intelligence is.

EVALUATION RULES:

1. BRAND VISIBILITY  
Score how prominently the INPUT BRAND appears in strategic context.

2. BRAND MENTIONS  
Rank competitors by market relevance and innovation presence.  
You MUST include the input brand + 4–7 major competitors.

3. PERSONA VISIBILITY  
Score which roles appear most influential for decisions in this domain.

4. TOPIC VISIBILITY  
Score which strategic themes dominate the analysis.

SCORING INTENSITY:
- Internal leadership report, not marketing
- Be critical and realistic
- Mid-tier relevance should score in the 60s
- Only strong dominance deserves 90+
- Score range: 45–98
- Avoid rounded numbers
- Use analyst-style variance (91, 87, 82, 76, 69)

ORDERING RULE:
All sections must be sorted from highest to lowest score.

OUTPUT FORMAT (JSON ONLY):
{{
  "brand_visibility": {{ "<brand>": <score> }},
  "brand_mentions": {{ "<brand>": <score> }},
  "persona_visibility": {{ "<persona>": <score> }},
  "topic_visibility": {{ "<topic>": <score> }}
}}
"""
    resp = llm.invoke(prompt)
    return _extract_json(str(resp.content))

# Combined Evaluation
def evaluate_combined(payload: Dict, answers_by_model: Dict[str, list]) -> Dict:
    """
    LLM-only evaluation across ALL models.
    model_visibility is INCLUDED here.
    """

    llm = get_llm("openai")

    # Format outputs clearly by model
    formatted_outputs = ""
    for model, answers in answers_by_model.items():
        formatted_outputs += f"\n=== MODEL: {model.upper()} ===\n"
        for i, ans in enumerate(answers, 1):
            formatted_outputs += f"Answer {i}: {ans}\n"

    prompt = f"""
You are a senior competitive intelligence lead preparing a FINAL strategic report.

You have received analysis outputs from MULTIPLE AI MODELS.

Your job is NOT to repeat their scores.
Your job is to SYNTHESIZE, COMPARE, and DIFFERENTIATE.

INPUT CONTEXT:
Brand: {payload["brand"]}
Category / Product: {payload.get("product")}
Personas: {payload.get("personas")}
Topics: {payload.get("topics")}

MODEL OUTPUTS (KEY = MODEL NAME):
{formatted_outputs}

EVALUATION RULES:

1. BRAND VISIBILITY  
Score how strongly the INPUT BRAND dominates the strategic narrative overall.

2. BRAND MENTIONS  
Rank competitors based on market relevance, innovation presence, and strategic weight.  
Do NOT mirror individual model rankings blindly.

3. PERSONA VISIBILITY  
Score which roles are most influential for decision-making in this domain.

4. TOPIC VISIBILITY  
Score which strategic themes dominate the competitive landscape.

5. MODEL VISIBILITY  
Score how useful, insightful, and strategically valuable each model’s contribution was.

SCORING PHILOSOPHY:
- Internal strategy review, not marketing
- Be honest and discriminative
- Weak items should score lower
- Strong dominance should stand out
- Score range: 45–98
- Avoid rounded numbers
- Use realistic variance (91, 87, 82, 76, 69)

SCORING INTENSITY:
- You are allowed to be critical
- Mid-tier relevance should score in the 60s
- Only top strategic dominance deserves 90+
- Do NOT inflate scores for balance

ORDERING RULE:
All sections must be sorted from highest to lowest score.

OUTPUT FORMAT (JSON ONLY):
{{
  "brand_visibility": {{ "<brand>": <score> }},
  "brand_mentions": {{ "<brand>": <score> }},
  "persona_visibility": {{ "<persona>": <score> }},
  "topic_visibility": {{ "<topic>": <score> }},
  "model_visibility": {{ "<model>": <score> }}
}}
"""

    resp = llm.invoke(prompt)
    return _extract_json(str(resp.content))

# Main Report Generator
def generate_report(payload: Dict) -> Dict:
    """
    Final LLM-only report generator.
    """

    prompts = payload.get("prompts", [])
    models = payload.get("models", [])

    prompts_by_model = defaultdict(list)

    for p in prompts:
        if isinstance(p, dict):
            prompts_by_model[p["model"]].append(p["prompt"])
        else:
            for m in models:
                prompts_by_model[m].append(p)

    answers_by_model = {}

    for model in models:
        llm = get_llm(model)
        answers = []

        for prompt in prompts_by_model.get(model, []):
            resp = llm.invoke(prompt)
            answers.append(
                resp.content if hasattr(resp, "content") else str(resp)
            )

        answers_by_model[model] = answers

    per_model = {
        model: evaluate_per_model(payload, model, answers)
        for model, answers in answers_by_model.items()
    }

    combined = evaluate_combined(payload, answers_by_model)

    return {
        "per_model": per_model,
        "combined": combined
    }