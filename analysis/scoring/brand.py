# analysis/scoring/brand.py
# Scores brand visibility & competitor mentions

from typing import Dict
import json, re

from langchain_core.messages import HumanMessage
from llm.llm_factory import get_llm
from llm.response_utils import extract_text


def score_brand(company: str, category: str, corpus: str) -> Dict:
    """
    Returns:
    {
      "brand_visibility": { "Brand": percent },
      "brand_mentions": { "Brand": score, ... }
    }
    """

    llm = get_llm()

    prompt = f"""
You are a competitive market analyst.

Identify REAL competitor brands for:
Company: "{company}"
Category: "{category}"

TEXT:
{corpus}

Rules:
- Output ONLY JSON object
- Keys = brand names
- Values = integer relevance score (0–100)
- Include "{company}"
- Max 5 brands
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = extract_text(resp)

    try:
        match = re.search(r"\{{.*\}}", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except:
        data = {}

    # Normalize
    normalized = {}
    for k, v in data.items():
        try:
            v = int(float(v))
        except:
            v = 0
        v = max(0, min(v, 100))
        normalized[k.title()] = v

    if company.title() not in normalized:
        normalized[company.title()] = 1

    total = sum(normalized.values()) or 1
    visibility = round((normalized[company.title()] / total) * 100)

    return {
        "brand_visibility": {company.title(): visibility},
        "brand_mentions": normalized
    }
