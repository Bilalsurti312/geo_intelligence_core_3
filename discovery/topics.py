# discovery/topics.py
from typing import List, Dict
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
import json
import re

def generate_topics(company: str, product: str, personas: List[str], num: int = 6) -> List[Dict]:
    """
    Generates structured strategic topics based on company, product, and multiple personas.
    """
    
    personas_text = ", ".join(personas)

    prompt = f"""
You are a strategic market intelligence analyst.

Generate exactly {num} HIGH-LEVEL dashboard topic themes.

Company: {company}
Product: {product}
Personas: {personas_text}

CRITICAL RULES:
- Topics MUST reflect company + product + personas together
- Different personas must influence topic selection
- No generic topics
- 3–8 words max
- Each topic must include:
    - name
    - description (1–2 strategic sentences)
- Output ONLY JSON list

Example:
[
  {{
    "name": "Mobile UX Optimization",
    "description": "Improves usability and retention through interface innovation and behavioral insights."
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