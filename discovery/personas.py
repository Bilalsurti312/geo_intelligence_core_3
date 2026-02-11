# discovery/personas.py
from typing import List, Dict
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
import json, re

def generate_personas(company: str, category: str, num: int = 6) -> List[Dict]:
    """
    Returns structured personas with name + description.
    Domain-specific, strategic, non-generic.
    """

    prompt = f"""
You are an industry intelligence strategist.

Generate {num} DISTINCT professional personas that are SPECIFICALLY
relevant to the following company and product combination:

Company: {company}
Product Category: {category}

CRITICAL RULES:
- Personas MUST be relevant to BOTH the company and the product
- If product changes, personas must logically change
- Do NOT generate generic cross-industry roles
- Roles must reflect realistic decision-makers or influencers
- 2–5 words for name
- 1–2 sentence professional description
- No fluff
- No brand names inside description
- Output ONLY JSON list

Example:
[
  {{
    "name": "Mobile UX Optimization Lead",
    "description": "Improves smartphone user interface performance to increase engagement and retention metrics."
  }}
]
"""

    llm = get_discovery_llm()
    resp = llm.invoke([HumanMessage(content=prompt)])

    raw = resp.content

    if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))

    raw = str(raw).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)

    try:
        return json.loads(match.group(0)) if match else []
    except:
        return []