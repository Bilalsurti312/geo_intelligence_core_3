from typing import Dict
from collections import defaultdict
import json
import re
import os
from llm.llm_factory import get_llm


# ---------------- LLM Evaluators ----------------
def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def evaluate_per_model(payload: Dict, model: str, answers: list) -> Dict:
    """
    LLM-only evaluation for a SINGLE model.
    NO model_visibility here.
    """

    llm = get_llm(os.getenv("ACTIVE_LLM", "gemini"))

    prompt = f"""
You are a senior market intelligence analyst.

INPUT CONTEXT:
Brand: {payload["brand"]}
Category / Product: {payload.get("product")}
Personas selected by user: {payload.get("personas")}
Topics selected by user: {payload.get("topics")}

ANSWERS FROM MODEL [{model}]:
{answers}

TASK:
Evaluate the intelligence from THIS MODEL ONLY.

RULES:
- ALL scores must be reasoned, not mathematical
- Score range: 40–100
- Scores must be relative (avoid ties)
- ALWAYS include the input brand in brand_visibility

BRAND MENTIONS RULES (IMPORTANT):
- brand_mentions represents the COMPETITIVE LANDSCAPE
- You MUST include the input brand AND its major competitors
- Infer competitors using general market knowledge of the category
- Do NOT limit brand_mentions to only brands explicitly mentioned
- Include at least 4–7 brands total
- Output ONLY JSON

SCORING RULES (VERY IMPORTANT):
- Score range: 40–100
- Scores MUST NOT be rounded
- Avoid multiples of 5 or 10 (e.g., 80, 85, 90) unless absolutely necessary
- Use realistic analyst-style variance (e.g., 91, 88, 83, 79, 74, 68)
- Scores must be meaningfully different, not evenly spaced
- Higher score = stronger relevance or influence
- Lower score = weaker but still relevant

ORDERING RULE:
- For EACH section (brand_mentions, persona_visibility, topic_visibility, model_visibility):
  return entries SORTED from highest score to lowest score

FORMAT:
{{
  "brand_visibility": {{ "<brand>": <score> }},
  "brand_mentions": {{ "<brand>": <score> }},
  "persona_visibility": {{ "<persona>": <score> }},
  "topic_visibility": {{ "<topic>": <score> }}
}}
"""

    resp = llm.invoke(prompt)
    return _extract_json(str(resp.content))


def evaluate_combined(payload: Dict, answers_by_model: Dict[str, list]) -> Dict:
    """
    LLM-only evaluation across ALL models.
    model_visibility is INCLUDED here.
    """

    llm = get_llm(os.getenv("ACTIVE_LLM", "openai"))

    prompt = f"""
You are a senior competitive intelligence lead.

INPUT CONTEXT:
Brand: {payload["brand"]}
Category / Product: {payload.get("product")}
Personas selected by user: {payload.get("personas")}
Topics selected by user: {payload.get("topics")}

MODEL OUTPUTS:
{answers_by_model}

TASK:
Produce a FINAL COMBINED intelligence report.

RULES:
- ALL scores must be reasoned, not mathematical
- Score range: 40–100
- Scores must be relative (avoid ties)
- ALWAYS include the input brand in brand_visibility

BRAND MENTIONS RULES (IMPORTANT):
- brand_mentions represents the COMPETITIVE LANDSCAPE
- You MUST include the input brand AND its major competitors
- Infer competitors using general market knowledge of the category
- Do NOT limit brand_mentions to only brands explicitly mentioned
- Include at least 4–7 brands total
- Output ONLY JSON

SCORING RULES (VERY IMPORTANT):
- Score range: 40–100
- Scores MUST NOT be rounded
- Avoid multiples of 5 or 10 (e.g., 80, 85, 90) unless absolutely necessary
- Use realistic analyst-style variance (e.g., 91, 88, 83, 79, 74, 68)
- Scores must be meaningfully different, not evenly spaced
- Higher score = stronger relevance or influence
- Lower score = weaker but still relevant

ORDERING RULE:
- For EACH section (brand_mentions, persona_visibility, topic_visibility, model_visibility):
  return entries SORTED from highest score to lowest score

FORMAT:
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

# ---------------- Main Report Generator ----------------
def generate_report(payload: Dict) -> Dict:
    """
    Final LLM-only report generator.
    """

    prompts = payload.get("prompts", [])
    models = payload.get("models", [])

    # ---- Group prompts by model ----
    prompts_by_model = defaultdict(list)

    for p in prompts:
        if isinstance(p, dict):
            prompts_by_model[p["model"]].append(p["prompt"])
        else:
            for m in models:
                prompts_by_model[m].append(p)

    # ---- Run models ----
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

    # ---- Per-model evaluation (NO model_visibility) ----
    per_model = {
        model: evaluate_per_model(payload, model, answers)
        for model, answers in answers_by_model.items()
    }

    # ---- Combined evaluation (WITH model_visibility) ----
    combined = evaluate_combined(payload, answers_by_model)

    return {
        "per_model": per_model,
        "combined": combined
    }