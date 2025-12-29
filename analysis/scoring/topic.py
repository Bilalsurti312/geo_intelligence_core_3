# analysis/scoring/topic.py
# Scores relevance of topics in analysis corpus

from typing import Dict, List
import json, re

from langchain_core.messages import HumanMessage
from llm.llm_factory import get_llm
from llm.response_utils import extract_text


def score_topics(topics: List[str], corpus: str) -> Dict[str, int]:
    """
    Returns: { topic: score }
    """

    llm = get_llm()

    prompt = f"""
Evaluate relevance of these topics based on the text.

TEXT:
{corpus}

TOPICS:
{json.dumps(topics)}

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

    return result
