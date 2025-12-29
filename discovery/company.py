# discovery/company.py

import requests
import socket
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm


def _domain_resolves(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
        return True
    except:
        return False


def verify_company_from_url(url: str) -> dict:
    # ---------- URL validation ----------
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return {"valid": False, "reason": "Invalid URL"}

    if not _domain_resolves(hostname):
        return {"valid": False, "reason": "Domain does not resolve"}

    # ---------- Website fetch ----------
    try:
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0"
        })
    except:
        return {"valid": False, "reason": "Website unreachable"}

    if resp.status_code >= 400:
        return {"valid": False, "reason": "Website returned error"}

    page_text = resp.text[:6000]  # enough for LLM

    # ---------- LLM verification (Gemini – discovery fixed) ----------
    llm = get_discovery_llm()

    prompt = f"""
You are a strict company verification system.

Analyze the website content below and answer:

1. Is this a REAL operating company?
2. If yes, extract the OFFICIAL company name.

Website domain: {hostname}

CONTENT:
{page_text}

Rules:
- If NOT a real company → return {{ "valid": false, "reason": "<short reason>" }}
- If real company → return {{ "valid": true, "company": "<company name>" }}
- Output ONLY JSON
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content

    if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))

    raw = str(raw).strip()

    import json, re
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    try:
        result = json.loads(match.group(0)) if match else {}
    except:
        return {"valid": False, "reason": "LLM parse failure"}

    return result
