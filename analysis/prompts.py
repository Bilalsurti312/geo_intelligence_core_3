# analysis/prompts.py

from typing import List
import json
import re

from langchain_core.messages import HumanMessage


def generate_prompts(
    topic: str,
    persona: str,
    num: int,
    llm
) -> List[str]:
    """
    Generate analytical prompts using a provided LLM instance.
    (LLM is injected → supports multi-model analysis)
    """

    # -------- SAFETY: persona must be string --------
    if isinstance(persona, list):
        persona = persona[0]

    persona = persona.strip().lower()

    persona_hints = {
        "founder": "strategy, market positioning, long-term vision",
        "product manager": "feature tradeoffs, roadmap, user value",
        "marketing analyst": "market demand, segmentation, messaging",
        "technical lead": "architecture, scalability, system risks",
        "investor": "ROI, growth potential, competitive advantage",
        "designer": "UX clarity, usability, accessibility",
        "researcher": "evidence, metrics, validation methodology",
    }

    hint = persona_hints.get(persona, "analytical thinking")

    prompt = f"""
Generate exactly {num} high-quality analytical prompts.

Context:
- Topic: "{topic}"
- Persona: "{persona}"
- Thinking style: {hint}

Rules:
- Each prompt must be ONE sentence
- No brand names
- No generic filler
- Insightful and analysis-driven
- Output ONLY a JSON list

Example format:
["Prompt 1", "Prompt 2"]
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    raw = response.content
    if isinstance(raw, list):
        raw = "".join(
            part.get("text", "") for part in raw if isinstance(part, dict)
        )

    raw = str(raw).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else []
