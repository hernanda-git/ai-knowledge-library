# 01 — Overview: Community Resources & Templates

> **Purpose:** Central index and contribution guide for the AI Community Resources & Templates directory.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [What Belongs Here](#what-belongs-here)
3. [File Index & Descriptions](#file-index--descriptions)
4. [How to Contribute](#how-to-contribute)
5. [Style Guide](#style-guide)
6. [Cross-Reference Conventions](#cross-reference-conventions)
7. [Review Process](#review-process)
8. [License & Attribution](#license--attribution)
9. [FAQ](#faq)
10. [Maintainers](#maintainers)

---

## Directory Structure

This directory (`15-Community-Resources-Templates/`) is the **curated hub** for reusable community materials, templates, and reference documents used across the AI knowledge base. Every file here serves as both a reference and a starting point for new contributions.

```
15-Community-Resources-Templates/
├── 01-Overview.md                 # You are here
├── 02-SOUL-SKILL-Templates.md     # SOUL.md & SKILL.md authoring guide
├── 03-Prompt-Libraries.md         # Curated prompt templates by use case
├── 04-Agent-Toolkits.md           # Agent framework comparisons & code
├── 05-Fine-Tuning-Datasets.md     # Curated dataset list & tools
├── 06-Awesome-AI-Repos.md         # Awesome-list of AI GitHub repos
├── 07-AI-2026-Roadmap.md          # 2026 learning path & conference calendar
├── 08-Contribution-Templates.md   # PR/issue templates & conventions
├── 09-Community-Forums-Events.md  # Discord/Reddit/hackathons/meetups
└── 10-Tools-Ecosystem.md          # Full dev/deploy/monitor tool map
```

Each file follows a **consistent structure**:
- A `## Table of Contents` section for navigation
- Curated, annotated, and dated resource listings
- Cross-references to other files in the knowledge base
- A `## Further Reading` section at the end

---

## What Belongs Here

This directory is the **community-facing surface** of the AI Knowledge Base. It includes:

| Category | Examples |
|----------|----------|
| **Templates** | SOUL.md, SKILL.md, prompt templates, contribution PR templates |
| **Curated Lists** | Awesome repos, datasets, tools, frameworks |
| **Guides** | How to contribute, how to write a SKILL file |
| **Roadmaps** | 2026 learning plans, conference schedules |
| **Community** | Forums, Discord servers, hackathons, meetups |

Files here should be **immediately useful** to someone building, learning, or contributing in the AI space. They are not deep research papers — they are practical reference documents.

### What Does NOT Belong Here

- **Research papers** → See `03-Papers-Research/`
- **Raw datasets** → See `05-Datasets`
- **Tutorial walkthroughs** → See individual skill or project directories
- **Personal notes** → Use personal wiki or notes directory

---

## File Index & Descriptions

### 01-Overview.md (this file)
The entry point for the entire directory. Explains structure, contribution process, and cross-reference system.

### 02-SOUL-SKILL-Templates.md
The **authoritative reference** for writing SOUL.md and SKILL.md files. Covers:
- Frontmatter field specifications
- Complete SOUL.md example for a code-generation agent
- Complete SKILL.md example for a code-review skill
- Best practices, validation rules, extension points
- Common pitfalls and troubleshooting

### 03-Prompt-Libraries.md
A **curated prompt template library** organized by task type:
- Chain-of-thought, few-shot, role-playing, structured output
- Code generation, RAG, agent orchestration, safety
- Each template includes placeholders and usage notes
- Prompt engineering best practices

### 04-Agent-Toolkits.md
**Framework-by-framework comparison** of agent-building toolkits:
- LangChain/LangGraph, CrewAI, AutoGen, Semantic Kernel
- Eliza, Haystack, Dify, Flowise, n8n
- Architecture overviews, pros/cons, quick-start code snippets
- Decision flowchart for choosing a framework

### 05-Fine-Tuning-Datasets.md
**Comprehensive dataset reference** for fine-tuning:
- Instruction datasets (OpenAssistant, Dolly, ShareGPT, Alpaca)
- Preference datasets (HH-RLHF, UltraFeedback, HelpSteer)
- Domain-specific datasets (medical, legal, code)
- Synthetic data generation (Evol-Instruct, Self-Instruct, GLAN)
- Quality control and filtering tools

### 06-Awesome-AI-Repos.md
An **awesome-list** of GitHub repositories organized by category:
- LLMs, Agents, RAG, Vision, Audio, Deployment
- Evaluation, Safety, Fine-tuning, Data
- Each entry: description, stars, activity status, notes

### 07-AI-2026-Roadmap.md
A **practitioner's roadmap** for 2026:
- Monthly learning tracks (Jan–Dec)
- Conference calendar (NeurIPS, ICML, ICLR, AI Engineer Summit, etc.)
- Certification timelines
- Open-source contribution cycles
- 2025 retrospective

### 08-Contribution-Templates.md
**Templates and conventions** for contributing to the knowledge base:
- Document template with frontmatter
- Cross-reference format specification
- Code block standards
- Issue templates, PR template, commit message conventions
- Review checklist

### 09-Community-Forums-Events.md
**Active community hubs** and event calendars:
- Discord servers (AI, ML, open-source communities)
- Reddit communities
- Slack groups, hackathons, meetups
- 2026 conference calendar with dates
- Online courses and bootcamps

### 10-Tools-Ecosystem.md
**End-to-end AI tool ecosystem map**:
- Development (VS Code extensions, Cursor, Jupyter)
- Deployment (Docker, K8s, BentoML, Ray)
- Monitoring (W&B, MLflow, WhyLabs)
- Data (Label Studio, Scale AI, Snorkel)
- Vector Databases (Pinecone, Weaviate, Qdrant, Chroma)
- Compute (Lambda, RunPod, Vast.ai, JarvisLabs)

---

## How to Contribute

We welcome contributions from the community! Here is the process:

### Step 1: Identify What to Add

Check the existing files to avoid duplication. Look for:
- **Missing entries** in curated lists (a great new tool, dataset, or repo)
- **Outdated information** (stale links, deprecated tools)
- **New categories** not yet covered
- **Better examples** for templates

### Step 2: Use the Templates

Every new document should follow the standard template defined in `08-Contribution-Templates.md`. Key requirements:
- Frontmatter with `title`, `description`, `last_updated`, `maintainer`
- A `## Table of Contents` at the top
- Consistent heading hierarchy (`#` title, `##` sections, `###` subsections)
- Cross-references using the `[see:XX-FileName.md]` format
- Code blocks must specify language

### Step 3: Create a Pull Request

1. Fork the repository
2. Create a feature branch: `add/new-resource-category`
3. Make your changes following the style guide
4. Submit a PR using the template in `08-Contribution-Templates.md`
5. A maintainer will review within 5 business days

### Step 4: Respond to Feedback

Reviewers may request:
- Additional entries or sources
- Better formatting or cross-references
- Corrections to technical details

Once approved, your contribution will be merged. Thank you!

---

## Style Guide

### Markdown Conventions

| Element | Convention |
|---------|-----------|
| **Headers** | ATX-style (`#`, `##`, `###`). No closing `#`. Space after `#`. |
| **Bold** | `**text**` — for emphasis, file names, UI labels |
| **Italic** | `*text*` — for terms, titles, secondary emphasis |
| **Code inline** | Backticks for paths, commands, short snippets |
| **Code blocks** | Fenced with language tag: ` ```python ` |
| **Lists** | `-` for unordered, `1.` for ordered. Indent 2 spaces. |
| **Tables** | `|` pipe tables with header row and divider |
| **Links** | `[text](url)` — prefer descriptive text over bare URLs |
| **Blockquotes** | `> ` for notes, warnings, key callouts |
| **Horizontal rules** | `---` with blank lines before and after |
| **Cross-references** | `[see:XX-FileName.md#section]` |

### Frontmatter (YAML)

Every document MUST start with YAML frontmatter:

```yaml
---
title: Document Title
description: One-sentence summary of the document's purpose
last_updated: 2026-06-12
maintainer: AI Knowledge Base Team
status: active  # active | draft | deprecated
---
```

### Naming Conventions

- Files: `NN-Descriptive-Name.md` (NN = 01–99, sorted by category)
- Assets: `asset-name.ext` (images, diagrams in adjacent `assets/` dir)
- Internal anchors: lowercase, hyphenated, matching section headers

### Language & Tone

- **Be concise.** Prefer short paragraphs and bullet points.
- **Be specific.** Provide exact names, versions, and links.
- **Be neutral.** No marketing language or vendor hype.
- **Be inclusive.** Use they/them pronouns, avoid ableist terms.
- **Be accurate.** Verify URLs before submitting.

---

## Cross-Reference Conventions

To maintain a connected knowledge base, use these cross-reference patterns:

| Pattern | Example |
|---------|---------|
| Cross-directory file | `[see:../02-Frameworks/README.md]` |
| Within-directory file | `[see:02-SOUL-SKILL-Templates.md]` |
| Specific section | `[see:02-SOUL-SKILL-Templates.md#frontmatter-fields]` |
| External resource | `[OpenAI Cookbook](https://github.com/openai/openai-cookbook)` |
| Term definition | `[term:SOUL.md]` — defined first occurrence |

When you move or rename a file, update all cross-references in the repository.

---

## Review Process

All submissions go through a two-stage review:

### Stage 1: Automated Checks
- Frontmatter validation (required fields present)
- Link checking (all URLs respond 200)
- Markdown linting (MD013 line length, MD033 inline HTML, etc.)
- Minimum word count per section

### Stage 2: Human Review

A maintainer checks:
- **Accuracy**: Are the facts and links correct?
- **Relevance**: Does the content belong in this directory?
- **Completeness**: Are all sections filled out?
- **Consistency**: Does it follow the style guide?
- **Cross-references**: Are references to other files correct?

### Review Timeline

| Priority | Response Time | Merge Time |
|----------|---------------|------------|
| 🟢 Standard | 3 business days | 7 days |
| 🔵 Fast-track | 1 business day | 3 days |
| 🔴 Urgent (fix errors only) | 4 hours | 24 hours |

---

## License & Attribution

All content in this directory is licensed under:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/

### Contributor Agreement

By submitting content, you agree that:
1. You have the rights to share the content
2. You license it under CC BY 4.0
3. Your contributions may be edited for clarity and style

---

## FAQ

### Can I add entries to the awesome list directly?

Yes! Open a PR adding your entry to `06-Awesome-AI-Repos.md`. Follow the existing format and include a brief description, star count, and why it's notable.

### How do I suggest a new template?

Open an issue with the `template-proposal` label. Include a draft of the template and explain which use case it addresses.

### A link is broken — what do I do?

Open an issue with the `broken-link` label. Include the file path, broken URL, and (if possible) the replacement URL.

### Can I translate these documents?

We welcome translations! Open an issue first to coordinate, then submit a PR with translated files in a `translations/` subdirectory.

### How often are files updated?

We aim to review every file at least once per quarter. Time-sensitive content (conference dates, tool versions) is updated as changes occur.

### My PR wasn't reviewed — what happened?

First, check that the automated checks pass. If they do and it's been more than 7 business days, ping the maintainers in the associated issue.

---

## Maintainers

| Name | Role | Contact |
|------|------|---------|
| AI Knowledge Base Team | Primary maintainers | kb-team@nousresearch.com |
| Community Contributors | Content reviewers | Via PR system |

### On-Call Rotation

For urgent issues (broken links, security concerns, factual errors), the on-call maintainer can be reached via the repository's GitHub Discussions or by tagging `@kb-maintainers` in an issue.

---

## Further Reading

- [08-Contribution-Templates.md](08-Contribution-Templates.md) — Templates and PR process
- [02-SOUL-SKILL-Templates.md](02-SOUL-SKILL-Templates.md) — Writing effective SOUL/SKILL files
- Knowledge Base Root README — Top-level directory explanation
- [Markdown Guide](https://www.markdownguide.org/) — Reference for markdown syntax

---

*Document version 1.0 — Last updated 2026-06-12*
