"""Web server for the physical AI visualization with LLM-powered descriptions."""

import os
import sys

import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

# Lazy import so the repo-level config doesn't need to import at server startup
# in unusual PYTHONPATH setups. config lives one dir up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    MONGO_URL, MONGO_DB, COMPANIES_COL, ARTICLES_COL, SCORE_FIELD, THEME,
    get_openrouter_key,
)

from dedup_semantic import normalize_key, KNOWN_ALIASES  # noqa: E402

_mongo = MongoClient(MONGO_URL)
_companies = _mongo[MONGO_DB][COMPANIES_COL]
_articles = _mongo[MONGO_DB][ARTICLES_COL]

# Alias-key redirects for graph nodes, e.g. "maxim integrated" -> "analog devices"
_ALIAS_KEYS = {
    normalize_key(alias): normalize_key(canonical)
    for canonical, aliases in KNOWN_ALIASES.items()
    for alias in aliases
}


def _graph_key(name: str) -> str:
    """Canonical node key: same normalization + alias folding the entity dedup
    uses, so raw extraction variants ("Analog Devices Inc.") collapse into one
    graph node instead of appearing beside the canonical entity."""
    key = normalize_key(name)
    return _ALIAS_KEYS.get(key, key)


def _set_label(labels: dict, key: str, name: str) -> None:
    """Display label for a node: direct variants of the canonical key beat
    alias-folded names (so "Analog Devices" wins over "Maxim"); among equals,
    the shortest form wins (so "Analog Devices" beats "Analog Devices Inc.")."""
    cur = labels.get(key)
    if cur is None:
        labels[key] = name
        return
    if (normalize_key(name) != key, len(name)) < (normalize_key(cur) != key, len(cur)):
        labels[key] = name

# /api/describe LLM — GLM-5.2 via OpenRouter (was Haiku until 2026-08-04)
DESCRIBE_MODEL = "z-ai/glm-5.2"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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
        resp = _requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {get_openrouter_key()}"},
            json={
                "model": DESCRIBE_MODEL,
                "max_tokens": 150,
                "reasoning": {"enabled": False},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return DescribeResponse(description=(msg.get("content") or msg.get("reasoning") or "").strip())
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


# --- Knowledge graph -------------------------------------------------------
#
# Built live from articles.entities.relationships, so it reflects whatever the
# most recent scan has ingested. Nothing downstream persists these edges — the
# company pipeline consumes entity descriptions and discards the graph — so
# this endpoint is the only thing that surfaces them.
#
# Edge direction is NOT uniform in the extracted data. The relationship type
# sometimes names entity_1's role ("Amazon --acquirer--> RIVR") and sometimes
# entity_2's ("Reflection --investor--> Nvidia", i.e. Nvidia invests). Three
# types are inconsistent even with themselves. Rather than guess, each type is
# tagged with how much its direction can be trusted:
#
#   "forward"   — entity_1 is the actor; arrow is meaningful as stored
#   "reverse"   — entity_2 is the actor; arrow should be drawn flipped
#   "symmetric" — mutual; draw undirected
#   "ambiguous" — convention varies per article; draw undirected and do not
#                 infer causality. The prose in `description` holds the truth.
EDGE_DIRECTION = {
    "investor": "reverse",
    "acquirer": "forward",
    "founder": "forward",
    "employee": "forward",
    "developer": "forward",
    "manufacturer": "forward",
    "partner": "symmetric",
    "technology partner": "symmetric",
    "competitor": "symmetric",
    "collaborator": "symmetric",
    "supplier": "ambiguous",
    "customer": "ambiguous",
    "subsidiary": "ambiguous",
}


def _entity_rows(entities: dict, key: str) -> list[tuple[str, str]]:
    """Yield (name, one-line description) out of one entity bucket.

    Organizations and products carry `description` directly. People don't —
    they carry `title` and `company` — so a sentence is composed from those.
    """
    out = []
    for item in entities.get(key) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("person")
        if not (isinstance(name, str) and name.strip()):
            continue

        desc = item.get("description")
        if not (isinstance(desc, str) and desc.strip()):
            title, company = item.get("title"), item.get("company")
            if isinstance(title, str) and title.strip():
                desc = title.strip()
                if isinstance(company, str) and company.strip():
                    desc += f" at {company.strip()}"
            else:
                desc = ""
        out.append((name.strip(), desc.strip()))
    return out


def _score_lookup() -> dict[str, float]:
    """Map lowercase company name (and every alias) → relevance_score.

    The graph is built from articles.entities, which predates scoring; this
    join is what lets the UI filter graph nodes by the pipeline's relevance
    judgment. Only organizations ever match — people and products aren't
    scored, so their nodes come back with score=None and are exempt from the
    score filter.
    """
    scores: dict[str, float] = {}
    for doc in _companies.find(
        {SCORE_FIELD: {"$exists": True}},
        {"name": 1, "aliases": 1, SCORE_FIELD: 1},
    ):
        s = doc.get(SCORE_FIELD)
        if s is None:
            continue
        for name in [doc.get("name", "")] + (doc.get("aliases") or []):
            if isinstance(name, str) and name.strip():
                scores[_graph_key(name)] = s
    return scores


# Rebuilding the whole graph from articles costs a full collection scan, which
# is fine once but not per-click while exploring. Cache it briefly and refresh
# when the article count moves — the scan only ever adds documents, so the
# count is a sufficient staleness signal.
_graph_cache: dict = {"stamp": None, "built": None}


def _build_graph() -> dict:
    stamp = _articles.estimated_document_count()
    cached = _graph_cache["built"]
    if cached is not None and _graph_cache["stamp"] == stamp:
        return cached

    kinds: dict[str, str] = {}
    labels: dict[str, str] = {}
    descs: dict[str, str] = {}
    mentions: dict[str, int] = {}
    arts: dict[str, list] = {}
    edges: dict[tuple, dict] = {}

    cursor = _articles.find(
        {"entities": {"$nin": [None, {}]}},
        {"entities.relationships": 1, "entities.organizations": 1,
         "entities.people": 1, "entities.products": 1,
         "entities.datasets": 1, "url": 1},
    )
    for doc in cursor:
        ents = doc.get("entities") or {}
        url = doc.get("url")
        for bucket, kind in (("organizations", "organization"),
                             ("people", "person"),
                             ("products", "product"),
                             ("datasets", "dataset")):
            for name, desc in _entity_rows(ents, bucket):
                key = _graph_key(name)
                _set_label(labels, key, name)
                mentions[key] = mentions.get(key, 0) + 1
                kinds.setdefault(key, kind)
                if desc and len(desc) > len(descs.get(key, "")) and len(desc) <= 320:
                    descs[key] = desc
                if url:
                    lst = arts.setdefault(key, [])
                    if url not in lst and len(lst) < 8:
                        lst.append(url)

        for rel in ents.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            a, b = rel.get("entity_1"), rel.get("entity_2")
            if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
                continue
            rtype = (rel.get("type") or "related").strip().lower()
            direction = EDGE_DIRECTION.get(rtype, "ambiguous")
            src, tgt = a.strip(), b.strip()
            if direction == "reverse":
                src, tgt = tgt, src
            sk, tk = _graph_key(src), _graph_key(tgt)
            if sk == tk:
                continue
            _set_label(labels, sk, src)
            _set_label(labels, tk, tgt)

            ekey = (sk, tk, rtype)
            edge = edges.get(ekey)
            if edge is None:
                edges[ekey] = {
                    "source": sk, "target": tk, "type": rtype,
                    "direction": direction, "count": 1,
                    "description": (rel.get("description") or "")[:300],
                    "articles": [doc.get("url")],
                }
            else:
                edge["count"] += 1
                if len(edge["articles"]) < 5:
                    edge["articles"].append(doc.get("url"))

    adjacency: dict[str, set] = {}
    degree: dict[str, int] = {}
    for sk, tk, _ in edges:
        degree[sk] = degree.get(sk, 0) + 1
        degree[tk] = degree.get(tk, 0) + 1
        adjacency.setdefault(sk, set()).add(tk)
        adjacency.setdefault(tk, set()).add(sk)

    built = {
        "labels": labels, "kinds": kinds, "descs": descs,
        "mentions": mentions, "arts": arts, "edges": edges,
        "degree": degree, "adjacency": adjacency,
        "scores": _score_lookup(),
    }
    _graph_cache.update(stamp=stamp, built=built)
    return built


def _node_payload(g: dict, key: str, **extra) -> dict:
    return {
        "id": key,
        "label": g["labels"].get(key, key),
        "kind": g["kinds"].get(key, "concept"),
        "degree": g["degree"].get(key, 0),
        "mentions": g["mentions"].get(key, 0),
        "score": g["scores"].get(key),
        "desc": g["descs"].get(key, ""),
        "articles": g["arts"].get(key, []),
        **extra,
    }


@app.get("/api/graph/search")
async def graph_search(q: str, limit: int = 12):
    """Autocomplete over entity names, best match first."""
    ql = q.strip().lower()
    if len(ql) < 2:
        return {"results": []}
    g = _build_graph()

    hits = []
    for key, label in g["labels"].items():
        pos = key.find(ql)
        if pos < 0:
            continue
        # exact, then prefix, then earliest substring; break ties on degree
        rank = 0 if key == ql else 1 if pos == 0 else 2
        hits.append((rank, pos, -g["degree"].get(key, 0), key))
    hits.sort()
    return {"results": [_node_payload(g, k) for _, _, _, k in hits[:limit]]}


@app.get("/api/graph/ego")
async def graph_ego(center: str, depth: int = 2, limit: int = 300):
    """Return the neighbourhood around one entity.

    Each node carries the hop count from the centre so the client can render
    depth 1 solid and depth 2 faded, per the focus+context pattern.
    """
    g = _build_graph()
    key = _graph_key(center)
    if key not in g["labels"]:
        raise HTTPException(status_code=404, detail="Unknown entity")

    depth = max(1, min(depth, 3))
    hops = {key: 0}
    frontier = [key]
    for d in range(1, depth + 1):
        nxt = []
        for node in frontier:
            for nb in sorted(g["adjacency"].get(node, ()),
                             key=lambda n: -g["degree"].get(n, 0)):
                if nb not in hops:
                    hops[nb] = d
                    nxt.append(nb)
            if len(hops) > limit:
                break
        frontier = nxt
        if len(hops) > limit:
            break

    # trim the outermost ring first if we overshot — never the centre or depth 1
    if len(hops) > limit:
        keep = {n for n, d in hops.items() if d <= 1}
        outer = sorted((n for n, d in hops.items() if d > 1),
                       key=lambda n: -g["degree"].get(n, 0))
        for n in outer:
            if len(keep) >= limit:
                break
            keep.add(n)
        hops = {n: d for n, d in hops.items() if n in keep}

    links = [e for (s, t, _), e in g["edges"].items() if s in hops and t in hops]
    nodes = [_node_payload(g, n, hop=hops[n]) for n in hops]
    nodes.sort(key=lambda n: (n["hop"], -n["degree"]))

    return {
        "center": key,
        "nodes": nodes,
        "links": links,
        "meta": {
            "node_count": len(nodes),
            "link_count": len(links),
            "truncated": len(hops) >= limit,
            "total_neighbours": len(g["adjacency"].get(key, ())),
        },
    }


@app.get("/api/graph")
async def get_graph(
    min_degree: int = 1,
    min_mentions: int = 1,
    min_score: float = 0.0,
    types: str | None = None,
    sources: str | None = None,
    limit_nodes: int = 2000,
    directed_only: bool = False,
):
    """Return the entity relationship graph as {nodes, links, meta}.

    min_degree   — drop nodes with fewer than this many edges (the long tail of
                   single-mention orgs is what turns the view into a hairball)
    min_mentions — drop nodes mentioned in fewer than this many articles
    min_score    — drop *organizations* whose pipeline relevance_score is below
                   this (unscored orgs count as 0). People, products, and
                   concepts are never scored, so they are exempt — they
                   survive as long as a kept node still links to them.
    types        — comma-separated edge types to include, e.g. "investor,acquirer"
    sources      — comma-separated article discovery stages to build from, e.g.
                   "search,news" (keyword scan only) or "citation" (deep-research
                   backfill only). Default: all stages.
    limit_nodes  — keep only the top-N nodes by degree
    directed_only— exclude edges whose direction cannot be trusted
    """
    wanted = {t.strip().lower() for t in types.split(",")} if types else None
    scores = _score_lookup()

    article_query: dict = {"entities": {"$nin": [None, {}]}}
    if sources:
        article_query["search_source"] = {
            "$in": [s.strip().lower() for s in sources.split(",")]
        }

    # kind lookup drives node colouring; a name can appear in several buckets
    # (a company that is also a product line), so the first one wins by priority
    kinds: dict[str, str] = {}
    edges: dict[tuple, dict] = {}
    mentions: dict[str, int] = {}
    labels: dict[str, str] = {}
    descs: dict[str, str] = {}
    arts: dict[str, list] = {}

    cursor = _articles.find(
        article_query,
        {"entities.relationships": 1, "entities.organizations": 1,
         "entities.people": 1, "entities.products": 1,
         "entities.datasets": 1, "url": 1},
    )

    for doc in cursor:
        ents = doc.get("entities") or {}
        url = doc.get("url")
        for bucket, kind in (("organizations", "organization"),
                             ("people", "person"),
                             ("products", "product"),
                             ("datasets", "dataset")):
            for name, desc in _entity_rows(ents, bucket):
                key = _graph_key(name)
                _set_label(labels, key, name)
                mentions[key] = mentions.get(key, 0) + 1
                kinds.setdefault(key, kind)
                # the same entity is described in every article that mentions
                # it; keep the fullest one that still reads as a single line
                if desc and len(desc) > len(descs.get(key, "")) and len(desc) <= 320:
                    descs[key] = desc
                if url:
                    lst = arts.setdefault(key, [])
                    if url not in lst and len(lst) < 8:
                        lst.append(url)

        for rel in ents.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            a, b = rel.get("entity_1"), rel.get("entity_2")
            if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
                continue
            rtype = (rel.get("type") or "related").strip().lower()
            if wanted and rtype not in wanted:
                continue
            direction = EDGE_DIRECTION.get(rtype, "ambiguous")
            if directed_only and direction in ("symmetric", "ambiguous"):
                continue

            src, tgt = a.strip(), b.strip()
            if direction == "reverse":
                src, tgt = tgt, src
            sk, tk = _graph_key(src), _graph_key(tgt)
            if sk == tk:
                continue
            _set_label(labels, sk, src)
            _set_label(labels, tk, tgt)

            ekey = (sk, tk, rtype)
            edge = edges.get(ekey)
            if edge is None:
                edges[ekey] = {
                    "source": sk, "target": tk, "type": rtype,
                    "direction": direction, "count": 1,
                    "description": (rel.get("description") or "")[:300],
                    "articles": [doc.get("url")],
                }
            else:
                edge["count"] += 1
                if len(edge["articles"]) < 5:
                    edge["articles"].append(doc.get("url"))

    # degree drives both filtering and node size
    degree: dict[str, int] = {}
    for sk, tk, _ in edges:
        degree[sk] = degree.get(sk, 0) + 1
        degree[tk] = degree.get(tk, 0) + 1

    keep = {n for n, d in degree.items() if d >= min_degree}
    if min_mentions > 1:
        keep = {n for n in keep if mentions.get(n, 0) >= min_mentions}
    if min_score > 0:
        keep = {
            n for n in keep
            if kinds.get(n) != "organization" or (scores.get(n) or 0) >= min_score
        }
    if len(keep) > limit_nodes:
        keep = set(sorted(keep, key=lambda n: -degree[n])[:limit_nodes])

    links = [e for e in edges.values() if e["source"] in keep and e["target"] in keep]

    # A score/mention filter can strand exempt nodes (people, products) whose
    # only links pointed at dropped organizations — prune those floaters.
    if min_score > 0 or min_mentions > 1:
        linked = {e["source"] for e in links} | {e["target"] for e in links}
        keep &= linked
        links = [e for e in links if e["source"] in keep and e["target"] in keep]

    nodes = [
        {
            "id": n,
            "label": labels.get(n, n),
            "kind": kinds.get(n, "concept"),
            "degree": degree.get(n, 0),
            "mentions": mentions.get(n, 0),
            "score": scores.get(n),
            "desc": descs.get(n, ""),
            "articles": arts.get(n, []),
        }
        for n in keep
    ]
    nodes.sort(key=lambda n: -n["degree"])

    type_counts: dict[str, int] = {}
    for e in links:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "node_count": len(nodes),
            "link_count": len(links),
            "edge_types": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
            "direction_legend": EDGE_DIRECTION,
            "filters": {
                "min_degree": min_degree,
                "min_mentions": min_mentions,
                "min_score": min_score,
                "types": sorted(wanted) if wanted else None,
                "sources": sources,
                "limit_nodes": limit_nodes,
                "directed_only": directed_only,
            },
        },
    }


# Serve static files
web_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
