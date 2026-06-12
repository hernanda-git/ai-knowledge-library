#!/usr/bin/env python3
"""AI Library of Alexandria — Full Static Site Generator
Reads ALL markdown files from the library and generates a complete static web portal
with full document content rendered as HTML.
"""

import os, shutil, re, json
import markdown as md_lib

BASE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_ROOT = os.path.dirname(BASE) if os.path.basename(os.path.dirname(BASE)) != "ai-library" else os.path.dirname(os.path.dirname(BASE))
# Walk up until we find the library root
for _ in range(5):
    if os.path.exists(os.path.join(LIBRARY_ROOT, "01-Foundations")):
        break
    LIBRARY_ROOT = os.path.dirname(LIBRARY_ROOT)

OUT = os.path.join(BASE, "out")

# ============================================================
# CATEGORY DEFINITIONS
# ============================================================
CATEGORIES = {
    "01-Foundations": {"id": "foundations", "icon": "📐", "title": "Foundations", "color": "#6366f1",
                       "desc": "ML, Deep Learning, Training, Data Engineering, RL, GNNs, Math, Federated, Causal"},
    "02-LLMs": {"id": "llms", "icon": "🧠", "title": "LLMs", "color": "#22c55e",
                "desc": "Transformer, Model Families, Tokenization, Quantization, NLP"},
    "03-Agents": {"id": "agents", "icon": "🤖", "title": "Agents", "color": "#a855f7",
                  "desc": "Agent Architectures, Multi-Agent, Frameworks, MCP/ACP, Tools"},
    "04-RAG": {"id": "rag", "icon": "🔍", "title": "RAG", "color": "#06b6d4",
               "desc": "RAG Architectures, Advanced RAG, Vector Databases"},
    "05-Enterprise": {"id": "enterprise", "icon": "🏭", "title": "Enterprise", "color": "#f59e0b",
                      "desc": "AI Deployment, Fine-Tuning, Infrastructure"},
    "06-Advanced": {"id": "advanced", "icon": "⚡", "title": "Advanced", "color": "#eab308",
                    "desc": "Multimodal, Diffusion, Evaluation, Prompt Engineering, Interpretability, RecSys, TS, Adv, UX, AutoML"},
    "07-Emerging": {"id": "emerging", "icon": "🔬", "title": "Emerging", "color": "#ef4444",
                    "desc": "Research Frontiers, AI Safety, AI Governance"},
    "08-Reference": {"id": "reference", "icon": "📚", "title": "Reference", "color": "#71717a",
                     "desc": "Glossary, Future Roadmap, SOUL.md/SKILL.md"},
    "09-Papers": {"id": "papers", "icon": "📄", "title": "Papers", "color": "#06b6d4",
                  "desc": "50+ Foundational AI Papers with Summaries"},
    "10-Industry": {"id": "industry", "icon": "🏢", "title": "Industry", "color": "#f59e0b",
                    "desc": "Industry Applications, AI Economics, Robotics"},
}


# Special standalone files
STANDALONE = {
    "09-Evolution-of-AI-Adoption.md": {"id": "evolution", "icon": "🔄", "title": "Evolution", "color": "#22c55e",
                                       "desc": "7-Stage AI Adoption Timeline: ChatGPT to AI Economy"},
    "Hermes-Agent-Sharing-Session-Agenda.md": {"id": "hermes", "icon": "⚡", "title": "Hermes", "color": "#a855f7",
                                               "desc": "Full Reference: CLI, Slash Commands, Config, Providers, Toolsets"},
}

# ============================================================
# SCAN ALL MARKDOWN FILES
# ============================================================
def scan_library():
    docs = []  # list of dicts: id, title, description, lines, category, icon, color, html_content, tags
    
    # Scan numbered folders
    for folder_name in sorted(os.listdir(LIBRARY_ROOT)):
        folder_path = os.path.join(LIBRARY_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith(".") or folder_name == "ai-library":
            continue
        if folder_name in CATEGORIES:
            cat = CATEGORIES[folder_name]
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(folder_path, fname)
                doc = process_md_file(fpath, fname, cat)
                if doc:
                    docs.append(doc)
    
    # Scan standalone files in root
    for fname, cat in STANDALONE.items():
        fpath = os.path.join(LIBRARY_ROOT, fname)
        if os.path.exists(fpath):
            doc = process_md_file(fpath, fname, cat)
            if doc:
                docs.append(doc)
    
    return docs

def process_md_file(fpath, fname, cat):
    """Read a markdown file, extract metadata, convert to HTML."""
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    
    lines = len(raw.split("\n"))
    
    # Extract title (first # heading)
    title = fname.replace(".md", "").replace("-", " ").title()
    title_match = re.search(r'^#\s+(.+)$', raw, re.MULTILINE)
    if title_match:
        t = title_match.group(1).strip()
        # Remove leading number like "01 - " or "1. "
        t = re.sub(r'^\d+\s*[\.\-]\s*', '', t)
        # Remove leading "## " if somehow matched
        title = t
    
    # Extract description (blockquote after title)
    desc = ""
    desc_match = re.search(r'^>\s*(.+?)$', raw, re.MULTILINE)
    if desc_match:
        desc = desc_match.group(1).strip()
    
    # Extract tags from content
    tags = extract_tags(raw)
    
    # Generate document ID
    doc_id = os.path.splitext(fname)[0].lower().replace(" ", "-").replace("_", "-")
    # Remove leading numbers
    doc_id = re.sub(r'^\d+-', '', doc_id)
    
    # Convert full markdown to HTML
    html_content = md_to_html(raw)
    
    return {
        "id": doc_id,
        "title": title,
        "description": desc,
        "lines": lines,
        "category": cat["id"],
        "icon": cat["icon"],
        "color": cat["color"],
        "category_title": cat["title"],
        "cat_desc": cat["desc"],
        "html_content": html_content,
        "tags": tags,
    }

def extract_tags(raw):
    """Extract meaningful tags from document content."""
    tags = set()
    # Keywords to look for
    keywords = [
        "Transformer", "Attention", "LLM", "GPT", "BERT", "RLHF", "DPO", "PPO", "LoRA", "QLoRA",
        "RAG", "MCP", "ACP", "Agent", "Orchestrator", "ReAct", "Multi-Agent", "Fine-tuning",
        "Quantization", "Tokenization", "Prompt Engineering", "Chain-of-Thought", "Scaling Laws",
        "Diffusion", "Multimodal", "CNN", "RNN", "GNN", "GAN", "VAE", "SVM", "XGBoost",
        "RL", "Supervised", "Unsupervised", "Self-Supervised", "Transfer Learning",
        "Federated Learning", "Differential Privacy", "Causal Inference", "AutoML", "NAS",
        "GPU", "Inference", "Deployment", "Evaluation", "Benchmark", "Safety", "Alignment",
        "Governance", "Economics", "Robotics", "CLIP", "Whisper", "Stable Diffusion",
        "Vector Database", "GraphRAG", "HYDE", "SLAM", "SHAP", "LIME",
        "SOUL.md", "SKILL.md", "Hermes Agent", "Claude Code", "OpenCode", "CrewAI",
        "LangGraph", "AutoGen", "Mamba", "MoE", "FlashAttention", "GQA", "RoPE",
        "Data Engineering", "Synthetic Data", "Ensemble", "Backpropagation",
        "Recommendation Systems", "Time-Series", "Adversarial ML", "Interpretability",
    ]
    content_lower = raw.lower()
    for kw in keywords:
        if kw.lower() in content_lower:
            tags.add(kw)
    
    # Also extract from special markdown elements like **bold** terms
    bold_terms = re.findall(r'\*\*([A-Za-z0-9+\-./ ]{3,40})\*\*', raw)
    for t in bold_terms:
        t = t.strip()
        if len(t) > 3 and len(t) < 30 and t[0].isupper():
            tags.add(t)
    
    return sorted(list(tags))[:12]  # max 12 tags

def md_to_html(raw):
    """Convert markdown content to full HTML."""
    md_extras = ["fenced_code", "codehilite", "tables", "nl2br", "sane_lists"]
    html = md_lib.markdown(raw, extensions=md_extras)
    
    # Post-process: add IDs to headings for anchor links
    html = re.sub(
        r'<h(\d)>(.*?)</h(\d)>',
        lambda m: f'<h{m.group(1)} id="{slugify(m.group(2))}">{m.group(2)}</h{m.group(3)}>',
        html
    )
    
    return html

def slugify(text):
    """Convert text to URL-friendly slug."""
    s = text.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[-\s]+', '-', s)
    return s

# ============================================================
# HTML GENERATION
# ============================================================
def html_base(title):
    fonts = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — AI Library of Alexandria</title>
<link href="{fonts}" rel="stylesheet">
<link rel="stylesheet" href="/style.css">
</head><body>'''

def gen_nav(all_docs, current_cat=None, current_doc=None):
    """Generate navigation bar with categories and document links."""
    # Build category list
    cats_seen = set()
    nav_links = []
    for d in all_docs:
        if d["category"] not in cats_seen:
            cats_seen.add(d["category"])
            active = ' class="act"' if d["category"] == current_cat else ""
            nav_links.append(f'<a href="/{d["category"]}/" class="cat-link"{active}>{d["icon"]} {d["category_title"]}</a>')
    
    total_docs = len(all_docs)
    nav_html = "\n".join(nav_links)
    
    return f'''<nav class="nv"><div class="brand"><span>📖 AI Library</span><span class="badge">{total_docs} Docs</span></div>
<a href="/" class="home-link">🏠 Home</a>
{nav_html}
</nav>'''

def gen_sidebar(all_docs, current_cat, current_doc_id=None, current_doc_title=None):
    """Generate sidebar for category/doc pages."""
    cat_docs = [d for d in all_docs if d["category"] == current_cat]
    cat = cat_docs[0] if cat_docs else None
    if not cat:
        return ""
    
    title_html = f'{cat["icon"]} {cat["category_title"]}'
    links = []
    
    for d in cat_docs:
        active = ' class="act"' if d["id"] == current_doc_id else ""
        doc_label = d["id"].replace("-", " ").title() if d["id"] == current_doc_id else d["title"]
        links.append(f'<a href="/{d["category"]}/{d["id"]}/"{active}>📄 {d["title"]}</a>')
    
    items = "\n".join(links)
    return f'''<aside class="sb">
<div class="sc">{title_html}</div>
<a href="/">🏠 Home</a>
<a href="/{current_cat}/" class="cat-overview">📂 Overview</a>
<div class="sc">Documents</div>
{items}
</aside>'''

def gen_footer(text):
    return f'<footer class="ft"><p><strong>AI Library of Alexandria</strong></p><p>{text}</p></footer>'

def gen_home(all_docs):
    """Home page with all categories and document cards."""
    # Group docs by category
    cat_groups = {}
    for d in all_docs:
        cat_groups.setdefault(d["category"], []).append(d)
    
    # Build category cards
    cards = []
    for cat_id in sorted(cat_groups.keys()):
        docs_in_cat = cat_groups[cat_id]
        d0 = docs_in_cat[0]
        total_lines = sum(d["lines"] for d in docs_in_cat)
        cards.append(f'''<a href="/{cat_id}/" class="gc" style="text-decoration:none">
<div class="ic">{d0["icon"]}</div>
<h4>{d0["category_title"]}</h4>
<p>{len(docs_in_cat)} docs · {total_lines:,} lines</p>
<p style="font-size:10px;margin-top:4px;color:var(--tx3)">{d0["cat_desc"]}</p>
</a>''')
    
    total_docs = len(all_docs)
    total_lines = sum(d["lines"] for d in all_docs)
    
    # Stats table
    rows_list = []
    for cat_id in sorted(cat_groups.keys()):
        docs_in_cat = cat_groups[cat_id]
        d0 = docs_in_cat[0]
        total_lines_cat = sum(d["lines"] for d in docs_in_cat)
        rows_list.append(f'<tr><td>{d0["icon"]} {d0["category_title"]}</td><td>{len(docs_in_cat)}</td><td>{total_lines_cat:,}</td><td style="font-size:11px">{d0["cat_desc"]}</td></tr>')
    
    rows = "\n".join(rows_list)
    
    # Study pathways
    pathways = [
        ("🧠", "rgba(99,102,241,0.15)", "ML Engineer", "Math → ML Foundations → Deep Learning → Training", "/foundations/"),
        ("🔧", "rgba(34,197,94,0.1)", "LLM Engineer", "Transformer → Model Families → Tokenization → Quantization → Prompt Engineering", "/llms/"),
        ("🤖", "rgba(168,85,247,0.1)", "Agent Developer", "Agent Architectures → MCP/ACP → Multi-Agent → Agentic Frameworks → SOUL/SKILL", "/agents/"),
        ("📚", "rgba(6,182,212,0.1)", "RAG Specialist", "RAG Architectures → Advanced RAG → Vector Databases", "/rag/"),
        ("🏭", "rgba(245,158,11,0.1)", "Enterprise Architect", "Enterprise Deployment → Fine-Tuning → AI Infrastructure", "/enterprise/"),
        ("🚀", "rgba(234,179,8,0.1)", "AI Researcher", "Diffusion → Multimodal → Interpretability → Safety → Emerging Research → Papers", "/advanced/"),
        ("🛡️", "rgba(239,68,68,0.1)", "Safety Practitioner", "AI Safety → Interpretability → Governance → Adversarial ML → Federated Learning", "/emerging/"),
    ]
    
    pw_html = ""
    for icon, bg, role, path_desc, link in pathways:
        pw_html += f'''<div class="kp"><div class="ki" style="background:{bg}">{icon}</div>
<div class="kt"><strong>{role}:</strong> {path_desc}. <a href="{link}" style="color:var(--accent-light)">Start →</a></div></div>'''
    
    page = html_base("AI Library of Alexandria")
    page += gen_nav(all_docs)
    page += f'''<div style="max-width:1100px;margin:0 auto;padding:80px 32px 60px">
<div class="hero">
<div class="t">AI Library of Alexandria</div>
<h1>Complete AI Study — <span class="hl">{total_docs} Documents</span></h1>
<p>A unified, cross-referenced knowledge base covering the entire AI stack — from mathematical foundations and deep learning through LLMs, agents, RAG, enterprise deployment, multimodal AI, safety, interpretability, and the future roadmap.</p>
<div class="stats">
<div class="stat"><div class="n" style="color:#818cf8">{total_docs}</div><div class="l">Documents</div></div>
<div class="stat"><div class="n" style="color:#22c55e">{total_lines:,}</div><div class="l">Total Lines</div></div>
<div class="stat"><div class="n" style="color:#f59e0b">{len(cat_groups)}</div><div class="l">Categories</div></div>
</div></div>
<div class="g2">{"".join(cards)}</div>
<div class="card" style="margin-top:32px"><h2>🧭 Study Pathways</h2>
<p>Choose your learning path based on your role:</p>
{pw_html}</div>
<div class="card" style="margin-top:32px"><h2>📊 Library Statistics</h2>
<div class="tw"><table><thead><tr><th>Category</th><th>Docs</th><th>Lines</th><th>Key Topics</th></tr></thead>
<tbody>{rows}</tbody></table></div></div>
{gen_footer(f"{total_docs} Documents · {total_lines:,} Lines · Auto-enriching every 12 hours")}
</div>'''
    page += '</body></html>'
    return page

def gen_category_overview(all_docs, cat_id):
    """Category overview page with document list."""
    cat_docs = [d for d in all_docs if d["category"] == cat_id]
    if not cat_docs:
        return None
    
    d0 = cat_docs[0]
    total_lines = sum(d["lines"] for d in cat_docs)
    
    doc_cards = []
    for d in cat_docs:
        tags = "".join(f'<span class="tag ac">{t}</span>' for t in d["tags"])
        doc_cards.append(f'''<div class="card" onclick="window.location='/{cat_id}/{d["id"]}/'" style="cursor:pointer">
<h3>📄 <a href="/{cat_id}/{d["id"]}/" style="color:inherit;text-decoration:none">{d["title"]}</a> <span class="doc-lines">— {d["lines"]} lines</span></h3>
<p>{d["description"]}</p>
<div style="margin:6px 0">{tags}</div>
</div>''')
    
    page = html_base(f"{d0['category_title']} — AI Library")
    page += gen_nav(all_docs, current_cat=cat_id)
    page += gen_sidebar(all_docs, cat_id)
    page += f'''<div class="ct">
<div class="hero">
<div class="t">{d0["icon"]} {d0["category_title"]}</div>
<h1>{d0["category_title"]} — <span class="hl">{len(cat_docs)} Documents</span></h1>
<p>{d0["cat_desc"]}</p>
<div class="stats">
<div class="stat"><div class="n" style="color:{d0['color']}">{len(cat_docs)}</div><div class="l">Documents</div></div>
<div class="stat"><div class="n" style="color:#22c55e">{total_lines:,}</div><div class="l">Total Lines</div></div>
</div></div>
{"".join(doc_cards)}
{gen_footer(f"{d0['category_title']} · {len(cat_docs)} Docs")}
</div>'''
    page += '</body></html>'
    return page

def gen_doc_page(all_docs, doc):
    """Full document page with ALL markdown content rendered."""
    cat_docs = [d for d in all_docs if d["category"] == doc["category"]]
    d0 = doc
    total_lines_cat = sum(d["lines"] for d in cat_docs)
    
    tags = "".join(f'<span class="tag ac">{t}</span>' for t in doc["tags"])
    
    page = html_base(f"{doc['title']} — {doc['category_title']} — AI Library")
    page += gen_nav(all_docs, current_cat=doc["category"])
    page += gen_sidebar(all_docs, doc["category"], current_doc_id=doc["id"], current_doc_title=doc["title"])
    
    page += f'''<div class="ct">
<div class="doc-header">
<div class="breadcrumb"><a href="/">Home</a> › <a href="/{doc['category']}/">{doc['category_title']}</a> › {doc['title']}</div>
<h1>{doc['icon']} {doc['title']}</h1>
<div class="doc-meta"><span class="doc-lines">{doc['lines']} lines</span>{tags}</div>
<div class="doc-desc"><p>{doc['description']}</p></div>
</div>
<div class="doc-content">
{doc['html_content']}
</div>
{gen_footer(f"{doc['category_title']} · {doc['title']}")}
</div>'''
    page += '</body></html>'
    return page

# ============================================================
# BUILD
# ============================================================
def build():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    
    # Scan all markdown files
    print("🔍 Scanning library...")
    all_docs = scan_library()
    print(f"   Found {len(all_docs)} documents")
    
    # Copy CSS
    css_path = os.path.join(BASE, "app", "globals.css")
    if os.path.exists(css_path):
        shutil.copy(css_path, os.path.join(OUT, "style.css"))
        print("✅ style.css copied")
    
    # Generate home page
    home_html = gen_home(all_docs)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(home_html)
    print("✅ index.html")
    
    # Group by category
    cat_groups = {}
    for d in all_docs:
        cat_groups.setdefault(d["category"], []).append(d)
    
    # Generate category and doc pages
    for cat_id, docs_in_cat in cat_groups.items():
        # Category overview
        cat_dir = os.path.join(OUT, cat_id)
        os.makedirs(cat_dir, exist_ok=True)
        
        cat_html = gen_category_overview(all_docs, cat_id)
        if cat_html:
            with open(os.path.join(cat_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(cat_html)
            print(f"✅ /{cat_id}/")
        
        # Individual doc pages
        for doc in docs_in_cat:
            doc_dir_name = doc["id"]
            doc_dir = os.path.join(cat_dir, doc_dir_name)
            os.makedirs(doc_dir, exist_ok=True)
            
            doc_html = gen_doc_page(all_docs, doc)
            with open(os.path.join(doc_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(doc_html)
            print(f"   ├ {doc['title']}")
    
    # Stats
    total_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUT) for f in fn)
    total_files = sum(len(fn) for _, _, fn in os.walk(OUT))
    print(f"✅ Build complete: {total_bytes:,} bytes | {total_files} files | {len(all_docs)} docs in {len(cat_groups)} categories")

if __name__ == "__main__":
    build()
