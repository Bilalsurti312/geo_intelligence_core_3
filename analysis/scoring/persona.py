# analysis/scoring/persona.py
# Scores persona visibility in corpus

from typing import Dict, List
import json, re

from langchain_core.messages import HumanMessage
from llm.llm_factory import get_llm
from llm.response_utils import extract_text


def score_personas(personas: List[str], corpus: str) -> Dict[str, int]:
    """
    Returns: { persona: score }
    """

    llm = get_llm()

    prompt = f"""
Analyze how strongly each persona perspective appears.

TEXT:
{corpus}

PERSONAS:
{json.dumps(personas)}

Rules:
- Output ONLY JSON dict
- Score 0–100 (integer)
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = extract_text(resp)

    try:
        match = re.search(r"\{{.*\}}", text, re.DOTALL)
        raw = json.loads(match.group(0)) if match else {}
    except:
        raw = {}

    result = {}
    for k, v in raw.items():
        try:
            v = int(float(v))
        except:
            v = 0
        result[k] = max(0, min(v, 100))

    # Sort descending
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
