# discovery/topics.py
from typing import List, Dict
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
import json
import re

def generate_topics(company: str, product: str, persona: str, num: int = 6) -> List[Dict]:
    """
    Generates structured high-level strategic topics
    with name + description.
    """

    prompt = f"""
You are a strategic market intelligence analyst.

Generate exactly {num} HIGH-LEVEL dashboard topic themes
for the following context:

Company: {company}
Product: {product}
Persona: {persona}

CRITICAL RULES:
- Topics MUST logically reflect company + product + persona together
- If persona changes, topics must shift accordingly
- No generic corporate themes
- 3–8 words max
- No questions
- No analysis verbs
- No brand names
- Each topic must include:
    - name
    - description (1–2 strategic sentences)
- Output ONLY JSON list

Example:
[
  {{
    "name": "Smartphone Market Penetration",
    "description": "Analyzes regional adoption rates and consumer switching behavior for mobile devices."
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