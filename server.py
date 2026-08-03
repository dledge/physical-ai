"""Web server for the physical AI visualization with Haiku-powered descriptions."""

import os
import sys

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

# Lazy import so the repo-level config doesn't need to import at server startup
# in unusual PYTHONPATH setups. config lives one dir up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    MONGO_URL, MONGO_DB, COMPANIES_COL, SCORE_FIELD, THEME, get_api_key,
)

_mongo = MongoClient(MONGO_URL)
_companies = _mongo[MONGO_DB][COMPANIES_COL]

llm = anthropic.Anthropic(api_key=get_api_key())

app = FastAPI(title="Physical AI Visualization")


class DescribeRequest(BaseModel):
    companies: list[dict]  # [{name, type, description, cluster_name}, ...]


class DescribeResponse(BaseModel):
    description: str


@app.post("/api/describe")
async def describe_view(req: DescribeRequest):
    if not req.companies:
        return DescribeResponse(description="")

    # Limit to 30 companies for the prompt
    companies = req.companies[:30]

    company_list = "\n".join(
        f"- {c['name']} ({c.get('type', '')}) — {c.get('description', '')[:100]}"
        for c in companies
    )

    prompt = f"""You are looking at a cluster of {len(companies)} {THEME} companies plotted by semantic similarity. Companies near each other have similar business profiles.

Here are the companies currently visible:

{company_list}

In 1-2 concise sentences, describe what these companies share in common and what this region of the {THEME} landscape represents. Be specific about the theme — don't just say "robotics companies" or "AI companies." Name the specific layer of the stack, sector, technology, or market niche. Return plain text only — no markdown, no headers, no bullet points."""

    try:
        response = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return DescribeResponse(description=response.content[0].text.strip())
    except Exception as e:
        return DescribeResponse(description=f"Error: {e}")


@app.get("/api/company/{name}")
async def get_company_detail(name: str):
    """Return extended details for a single company — used by the deep-hover
    panel in the viz. Looks up by name first, then by name_key for robustness."""
    doc = _companies.find_one(
        {"$or": [{"name": name}, {"name_key": name.lower()}]},
        {"_id": 0, "name": 1, "type": 1, "location": 1, "tags": 1,
         "short_blurb": 1, "clustering_bullets": 1, SCORE_FIELD: 1,
         "mention_count": 1, "research_citations": 1},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc


# Serve static files
web_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
