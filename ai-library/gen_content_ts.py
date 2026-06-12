#!/usr/bin/env python3
"""Generate lib/content.ts with full section data from the actual markdown files."""
import os, re
import markdown as md_lib

BASE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_ROOT = os.path.dirname(BASE)

for _ in range(5):
    if os.path.exists(os.path.join(LIBRARY_ROOT, "01-Foundations")):
        break
    LIBRARY_ROOT = os.path.dirname(LIBRARY_ROOT)

OUT_FILE = os.path.join(BASE, "lib", "content.ts")

CATEGORIES = {
    "01-Foundations": {"id": "foundations", "icon": "📐", "title": "Foundations", "color": "#6366f1",
                       "description": "ML, Deep Learning, Training, Data Engineering, RL, GNNs, Math, Federated, Causal"},
    "02-LLMs": {"id": "llms", "icon": "🧠", "title": "LLMs", "color": "#22c55e",
                "description": "Transformer, Model Families, Tokenization, Quantization, NLP"},
    "03-Agents": {"id": "agents", "icon": "🤖", "title": "Agents", "color": "#a855f7",
                  "description": "Agent Architectures, Multi-Agent, Frameworks, MCP/ACP, Tools"},
    "04-RAG": {"id": "rag", "icon": "🔍", "title": "RAG", "color": "#06b6d4",
               "description": "RAG Architectures, Advanced RAG, Vector Databases"},
    "05-Enterprise": {"id": "enterprise", "icon": "🏭", "title": "Enterprise", "color": "#f59e0b",
                      "description": "AI Deployment, Fine-Tuning, Infrastructure"},
    "06-Advanced": {"id": "advanced", "icon": "⚡", "title": "Advanced", "color": "#eab308",
                    "description": "Multimodal, Diffusion, Evaluation, Prompt Engineering, Interpretability, RecSys, TS, Adv, UX, AutoML"},
    "07-Emerging": {"id": "emerging", "icon": "🔬", "title": "Emerging", "color": "#ef4444",
                    "description": "Research Frontiers, AI Safety, AI Governance"},
    "08-Reference": {"id": "reference", "icon": "📚", "title": "Reference", "color": "#71717a",
                     "description": "Glossary, Future Roadmap, SOUL.md/SKILL.md"},
    "09-Papers": {"id": "papers", "icon": "📄", "title": "Papers", "color": "#06b6d4",
                  "description": "50+ Foundational AI Papers with Summaries"},
    "10-Industry": {"id": "industry", "icon": "🏢", "title": "Industry", "color": "#f59e0b",
                    "description": "Industry Applications, AI Economics, Robotics"},
}

STANDALONE = {
    "09-Evolution-of-AI-Adoption.md": {"id": "evolution", "icon": "🔄", "title": "Evolution", "color": "#22c55e",
                                       "description": "7-Stage AI Adoption Timeline: ChatGPT to AI Economy"},
    "Hermes-Agent-Sharing-Session-Agenda.md": {"id": "hermes", "icon": "⚡", "title": "Hermes", "color": "#a855f7",
                                               "description": "Full Reference: CLI, Slash Commands, Config, Providers, Toolsets"},
}

def js_escape(s):
    """Escape string for TypeScript double-quoted strings."""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    s = s.replace("\n", "\\n").replace("\r", "").replace("\t", "\\t").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return s.strip()

def extract_tags(raw):
    keywords = [
        "Transformer", "Attention", "LLM", "GPT", "BERT", "RLHF", "DPO", "PPO", "LoRA", "QLoRA",
        "RAG", "MCP", "ACP", "Agent", "Orchestrator", "ReAct", "Multi-Agent", "Fine-tuning",
        "Quantization", "Tokenization", "Prompt Engineering", "Scaling Laws",
        "Diffusion", "Multimodal", "CNN", "RNN", "GNN", "RL", "Supervised", "Unsupervised",
        "Federated Learning", "Differential Privacy", "Causal Inference", "AutoML", "NAS",
        "GPU", "Inference", "Deployment", "Evaluation", "Benchmark", "Safety", "Alignment",
        "Governance", "Robotics", "CLIP", "Whisper", "Stable Diffusion", "Vector Database",
        "SOUL.md", "SKILL.md", "Hermes Agent", "Claude Code", "OpenCode",
        "Data Engineering", "Synthetic Data", "Ensemble", "Recommendation Systems",
        "Adversarial ML", "Interpretability", "Constitutional AI",
    ]
    tags = set()
    content_lower = raw.lower()
    for kw in keywords:
        if kw.lower() in content_lower:
            tags.add(kw)
    return sorted(list(tags))[:10]

def extract_sections(raw):
    """Extract h2 sections with full content (up to 3000 chars per section)."""
    sections = []
    h2_matches = list(re.finditer(r'^##\s+(.+?)$', raw, re.MULTILINE))
    for i, m in enumerate(h2_matches):
        sec_title = m.group(1).strip()
        if any(x in sec_title.lower() for x in ["table of contents", "further reading", "references", "see also", "cross-references"]):
            continue
        
        start = m.end()
        end = h2_matches[i+1].start() if i+1 < len(h2_matches) else len(raw)
        sec_content = raw[start:end].strip()
        
        # Clean up: remove empty lines at start, limit length
        lines = sec_content.split("\n")
        # Find first non-empty content line (skip ---, [toc], empty)
        content_start = 0
        for j, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("---") and not stripped.startswith("[") and not stripped.startswith("```"):
                content_start = j
                break
        
        content_text = "\n".join(lines[content_start:])
        # Convert markdown to HTML for clean rendering
        content_html = md_lib.markdown(content_text, extensions=["fenced_code", "codehilite", "tables", "nl2br", "sane_lists"])
        # Limit HTML length to keep file manageable (20000 chars covers most sections fully)
        if len(content_html) > 20000:
            content_html = content_html[:20000] + "..."
        
        if sec_title:
            sections.append({"title": sec_title, "content": content_html})
    return sections[:12]  # Max 12 sections per doc

def get_title(raw, fname):
    title = fname.replace(".md", "").replace("-", " ").title()
    t = re.sub(r'^\d+\s*[\.\-]\s*', '', title)
    title_match = re.search(r'^#\s+(.+)$', raw, re.MULTILINE)
    if title_match:
        t2 = title_match.group(1).strip()
        t2 = re.sub(r'^\d+\s*[\.\-]\s*', '', t2)
        if t2 and not t2.startswith("##"):
            return t2
    return t if len(t) > 3 else title

def get_desc(raw):
    desc_match = re.search(r'^>\s*(.+?)$', raw, re.MULTILINE)
    if desc_match:
        return desc_match.group(1).strip()
    return ""

# Collect all docs
all_docs = []

for folder_name in sorted(os.listdir(LIBRARY_ROOT)):
    folder_path = os.path.join(LIBRARY_ROOT, folder_name)
    if not os.path.isdir(folder_path) or folder_name.startswith(".") or folder_name in ("ai-library",):
        continue
    if folder_name in CATEGORIES:
        cat = CATEGORIES[folder_name]
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            
            lines = len(raw.split("\n"))
            title = get_title(raw, fname)
            desc = get_desc(raw)
            doc_id = os.path.splitext(fname)[0].lower().replace(" ", "-").replace("_", "-")
            doc_id = re.sub(r'^\d+-', '', doc_id)
            
            all_docs.append({
                "id": doc_id,
                "title": title,
                "lines": lines,
                "description": desc,
                "tags": extract_tags(raw),
                "sections": extract_sections(raw),
                "category": cat["id"],
            })

# Add standalone files
for fname, cat in STANDALONE.items():
    fpath = os.path.join(LIBRARY_ROOT, fname)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        lines = len(raw.split("\n"))
        title = get_title(raw, fname)
        desc = get_desc(raw)
        doc_id = os.path.splitext(fname)[0].lower().replace(" ", "-").replace("_", "-")
        doc_id = re.sub(r'^\d+-', '', doc_id)
        
        all_docs.append({
            "id": doc_id,
            "title": title,
            "lines": lines,
            "description": desc,
            "tags": extract_tags(raw),
            "sections": extract_sections(raw),
            "category": cat["id"],
        })

# Group by category
cat_groups = {}
for d in all_docs:
    cat_groups.setdefault(d["category"], []).append(d)

# Generate content.ts
lines = []
lines.append('import { TopicCategory, KnowledgeDoc } from "@/lib/types";')
lines.append('')

# Categories
lines.append("export const categories: TopicCategory[] = [")
all_cat_items = list(CATEGORIES.items()) + list(STANDALONE.items())
for folder_name, cat in sorted(all_cat_items):
    docs_in_cat = cat_groups.get(cat["id"], [])
    total_lines = sum(d["lines"] for d in docs_in_cat)
    lines.append(f'  {{ id: "{cat["id"]}", icon: "{cat["icon"]}", title: "{cat["title"]}", description: "{cat["description"]}", totalLines: {total_lines}, docCount: {len(docs_in_cat)}, color: "{cat["color"]}" }},')
lines.append("];")
lines.append("")

# allDocs
lines.append("export const allDocs: Record<string, KnowledgeDoc[]> = {")
for cat_id, docs in sorted(cat_groups.items()):
    lines.append(f'  {cat_id}: [')
    for d in docs:
        tags_str = ", ".join(f'"{t}"' for t in d["tags"])
        desc_esc = js_escape(d["description"])
        title_esc = js_escape(d["title"])
        sections_str = ", ".join(f'{{ title: "{js_escape(s["title"])}", content: "{js_escape(s["content"])}" }}' for s in d["sections"])
        lines.append(f'    {{ id: "{d["id"]}", title: "{title_esc}", lines: {d["lines"]}, description: "{desc_esc}", tags: [{tags_str}], sections: [{sections_str}] }},')
    lines.append('  ],')
lines.append("};")
lines.append("")

output = "\n".join(lines)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(output)

total_secs = sum(len(d["sections"]) for d in all_docs)
print(f"✅ Generated lib/content.ts")
print(f"   {len(all_docs)} docs, {len(cat_groups)} categories, {total_secs} sections")
print(f"   File size: {len(output):,} bytes")
