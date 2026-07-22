# Contributing to the AI Knowledge Library

> **"The only thing greater than knowledge is shared knowledge."**

Thank you for helping grow the Great Library. This guide explains how to contribute effectively.

---

## 📋 Quick Checklist

- [ ] I've read this contributing guide
- [ ] My document follows the five-file pattern (if adding a new category)
- [ ] My file name follows the naming convention
- [ ] I've placed my file in the correct directory (01–10 are portal-scanned, 11+ are free)
- [ ] I've added "See also" cross-links to related documents
- [ ] I've updated `CATALOG.md` to reflect changes
- [ ] I've verified no duplicate or overlapping content exists

---

## 📁 Repository Structure

```
ai-knowledge-library/
├── 01-Foundations/           ← Portal-scanned (01-10 are read by ai-library/generate.py)
├── 02-LLMs/                  ← Portal-scanned
├── ...
├── 10-Industry/              ← Portal-scanned
├── 11-AI-Applications/       ← Markdown-only (not scanned by portal generator)
├── ...
├── 77-Beyond-Human-AI/       ← Markdown-only
├── CATALOG.md                ← Auto-updated structure index
├── README.md                 ← This file
├── LICENSE                   ← CC BY 4.0
└── _Meta/                    ← Operational artifacts (cron reports, etc.)
```

**Important:** Folders `01`–`10` are frozen for the web portal generator (`ai-library/generate.py`). If you add or rename files in these directories, the portal will break. Folders `11`+ are freely rearrangeable.

---

## 📝 Document Structure

### New Category (full wing)
Each category should follow the five-file pattern:

| File | Content | Purpose |
|:-----|:--------|:--------|
| `01-Overview.md` | High-level intro, why it matters, scope map | Orientation |
| `02-Core-Topics.md` | Foundational concepts, taxonomy, key ideas | Education |
| `03-Technical-Deep-Dive.md` | Detailed mechanisms, architectures, equations | Reference |
| `04-Tools-and-Frameworks.md` | Software, hardware, platforms, benchmarks | Practical use |
| `05-Future-Outlook.md` | Roadmaps, timelines, emerging trends, risks | Strategic view |

### Single Document (improvement)
- Place it in the most relevant existing category
- Use the naming pattern: `##-Topic-Name.md` (two-digit number + kebab-case)
- Start with `# ## | Topic Name — Subtitle`

### Nested Sub-Topics
If a topic deserves 3+ documents but logically belongs under a parent category:

```
04-RAG/
├── 01-RAG-Architectures.md
├── 02-Advanced-RAG.md
├── 03-Vector-Databases.md
└── 69-GraphRAG-and-Knowledge-Graph-Retrieval/   ← Nested sub-topic
    ├── 01-Overview.md
    ├── 02-Core-Topics.md
    └── ...
```

---

## ✍️ Writing Style

### Tone
- **Technical but accessible** — write for an AI practitioner with ~1 year of industry experience
- **Neutral and factual** — present competing views fairly
- **Actionable** — include specific tools, metrics, configurations where relevant

### Format
- GitHub-flavored Markdown
- Tables for structured comparisons
- Code blocks with language tags for examples
- Blockquotes sparingly for key insights or expert quotes
- ASCII diagrams for architectures (works in raw markdown AND the web portal)

### Required Sections Per Document
Every document should have:
1. A **title** (`# ## | Category — Topic`)
2. A **brief description or epigraph** under the title
3. **Clear section headings** (not just walls of text)
4. **"See also" footer** linking to related documents

---

## 🔗 Cross-Referencing Convention

Add a "See also" footer to every document:

```markdown
---
**See also:**
- [01-Foundations/03-Deep-Learning](01-Foundations/03-Deep-Learning.md)
- [06-Advanced/01-Multimodal-AI](06-Advanced/01-Multimodal-AI.md)
- [13-Top-Demand/02-AI-Agent-Development](13-Top-Demand/02-AI-Agent-Development.md)
```

Keep links relative to the repo root. Link to the `.md` file, not the directory (GitHub renders it inline).

---

## 🧪 Verifying Your Contribution

```bash
# 1. Check your document renders
pandoc 75-Quantum-AI/01-Overview.md -o /dev/null

# 2. Run the portal generator (optional, only for 01-10 changes)
cd ai-library && python generate.py

# 3. Update the catalog
# Edit CATALOG.md to add your new category to the listing
# Update document counts

# 4. Check for broken links
grep -r "](.*\.md)" your-file.md | grep -v "http" | while read -r line; do
  file=$(echo "$line" | grep -oP '(?<=\]\()([^)]+\.md)')
  [ ! -f "$file" ] && echo "BROKEN: $file"
done
```

---

## 💻 Pull Request Process

1. **Fork the repo** on GitHub
2. **Create a branch**: `git checkout -b docs/your-topic`
3. **Make your changes** following the conventions above
4. **Update CATALOG.md** with your additions
5. **Commit** with a descriptive message:
   ```
   docs: add AI-for-Agriculture section (5 documents)
   ```
6. **Push and open a PR**
7. In the PR description, explain:
   - What gap you're filling
   - Key sources used
   - Any conventions you weren't sure about

---

## 🐛 Reporting Issues

- **Missing topic**: Use the "Category Gap" template
- **Inaccuracy or outdated info**: Open an issue with the file path and correction
- **Broken link**: Include the file path and the broken reference
- **Suggestions**: Open a discussion — all ideas welcome

---

## ❓ Questions?

Open an issue with the "question" label, or start a GitHub Discussion.

---

*"The Great Library grows through collective effort. Every contribution — from a single footnote to a new wing — makes the flame burn brighter."*
