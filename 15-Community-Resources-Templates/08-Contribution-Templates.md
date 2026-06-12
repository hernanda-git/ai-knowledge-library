# 08 — Contribution Templates

> **Purpose:** Standardized templates and conventions for contributing to the AI Knowledge Base — documents, issues, pull requests, and code.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Document Template with Frontmatter](#document-template-with-frontmatter)
3. [Cross-Reference Format](#cross-reference-format)
4. [Code Block Standards](#code-block-standards)
5. [Issue Templates](#issue-templates)
6. [Pull Request Template](#pull-request-template)
7. [Commit Message Conventions](#commit-message-conventions)
8. [Review Checklist](#review-checklist)
9. [File Naming & Organization](#file-naming--organization)
10. [Style Enforcement](#style-enforcement)
11. [Contributor License Agreement](#contributor-license-agreement)
12. [Further Reading](#further-reading)

---

## Introduction

Consistent contribution formats make the knowledge base easier to maintain, navigate, and automate. This document provides the canonical templates and conventions for all contributions.

### What This Covers

| Artifact | Template/Convention | Section |
|----------|--------------------|---------|
| New markdown documents | Document template with YAML frontmatter | [Document Template](#document-template-with-frontmatter) |
| Cross-references | Standardized reference format | [Cross-Reference Format](#cross-reference-format) |
| Code in documents | Fenced code blocks with language tags | [Code Block Standards](#code-block-standards) |
| Bug reports | Issue template | [Issue Templates](#issue-templates) |
| Feature requests | Issue template | [Issue Templates](#issue-templates) |
| Pull requests | PR template | [Pull Request Template](#pull-request-template) |
| Commits | Conventional commit format | [Commit Message Conventions](#commit-message-conventions) |

### Before You Contribute

1. Read [01-Overview.md](01-Overview.md) for directory structure and scope
2. Check for existing documents to avoid duplication
3. Review the style guide in [01-Overview.md#style-guide](01-Overview.md#style-guide)
4. Use the appropriate template from this document
5. Validate with the automated tooling (see [Style Enforcement](#style-enforcement))

---

## Document Template with Frontmatter

Every markdown document in this knowledge base MUST start with YAML frontmatter.

### Minimal Template

```markdown
---
title: Document Title
description: One-sentence summary of the document's purpose
last_updated: 2026-06-12
maintainer: AI Knowledge Base Team
status: active
---

# Document Title

> **Purpose:** Brief summary of what this document covers.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

## Table of Contents

1. [Section 1](#section-1)
2. [Section 2](#section-2)
3. [Further Reading](#further-reading)

---

## Section 1

Content here.

## Section 2

Content here.

## Further Reading

- [Related Document](related-file.md) — Description

---

*Document version 1.0 — Last updated 2026-06-12*
```

### Full Template with All Optional Fields

```markdown
---
title: "Document Title"
description: >
  A longer description that can span multiple lines using YAML block
  scalar syntax. This describes the document's scope and intended
  audience in 2-3 sentences.
last_updated: 2026-06-12
maintainer: AI Knowledge Base Team
status: active  # active | draft | deprecated | archived
version: "1.0.0"
license: CC-BY-4.0
tags:
  - reference
  - template
  - ai-community
language: en
audience: developers  # developers | researchers | all
difficulty: intermediate  # beginner | intermediate | advanced
word_count_estimate: 1500
review_by: 2026-09-12
supersedes: old-file.md
related:
  - 01-Overview.md
  - 09-Community-Forums-Events.md
---

# Document Title

> **Purpose:** One-sentence description of the document.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team
> **Status:** Active | Draft | Deprecated
> **Audience:** {{audience}}
> **Difficulty:** {{difficulty}}

## Table of Contents

1. [Introduction](#introduction)
2. [Section 1](#section-1)
   - [Subsection 1.1](#subsection-11)
   - [Subsection 1.2](#subsection-12)
3. [Section 2](#section-2)
4. [FAQ](#faq)
5. [Troubleshooting](#troubleshooting)
6. [Further Reading](#further-reading)

---

## Introduction

Context about why this document exists and who should read it.

## Section 1

Content with [cross-reference](related-file.md#section).

### Subsection 1.1

Detailed content.

## FAQ

### Q: Common question?
A: Answer.

## Further Reading

- [Related Document 1](path/to/file.md) — Description of relationship
- [Related Document 2](path/to/file.md#section) — Specific section reference

---

*Document version 1.0 — Last updated 2026-06-12*
```

### Frontmatter Field Reference

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `title` | ✅ | string | Document title (should match filename topic) |
| `description` | ✅ | string | 1-3 sentence summary of document scope |
| `last_updated` | ✅ | date | ISO 8601 date (YYYY-MM-DD) |
| `maintainer` | ✅ | string | Who is responsible for this document |
| `status` | ✅ | enum | `active`, `draft`, `deprecated`, `archived` |
| `version` | ❌ | string | Semver version number |
| `license` | ❌ | string | SPDX license identifier |
| `tags` | ❌ | string[] | Classification tags |
| `language` | ❌ | string | Primary language code (en, zh, etc.) |
| `audience` | ❌ | enum | `developers`, `researchers`, `all` |
| `difficulty` | ❌ | enum | `beginner`, `intermediate`, `advanced` |
| `review_by` | ❌ | date | Next review deadline |
| `supersedes` | ❌ | string | Path to document this replaces |
| `related` | ❌ | string[] | Paths to related documents |

---

## Cross-Reference Format

### Within This Directory

```
[see:02-SOUL-SKILL-Templates.md]
[see:02-SOUL-SKILL-Templates.md#frontmatter-fields]
```

### Cross-Directory References

```
[see:../02-Frameworks/README.md]
[see:../05-Datasets/README.md#structure]
```

### External References

```
[OpenAI Cookbook](https://github.com/openai/openai-cookbook)
[LangChain Documentation](https://python.langchain.com/)
```

### Reference Types

| Type | Syntax | Example |
|------|--------|---------|
| Internal file | `[see:NN-FileName.md]` | `[see:02-SOUL-SKILL-Templates.md]` |
| Internal section | `[see:NN-FileName.md#section-name]` | `[see:02-SOUL-SKILL-Templates.md#frontmatter-fields]` |
| External link | `[Display Text](url)` | `[OpenAI](https://openai.com)` |
| Term definition | `[term:SOUL.md]` | First occurrence in document |
| Acronym expansion | `[acronym:RAG:Retrieval-Augmented Generation]` | First occurrence |

### Rules

1. Use **`see:` prefix** for internal references — this enables automated link checking
2. Use **display text** for external URLs — never bare URLs in running text
3. Use **lowercase, hyphenated anchors** matching section headers
4. Validate all cross-references before committing
5. When renaming a file, update ALL cross-references to it

---

## Code Block Standards

### Language Tags

Always specify the language after the opening fence:

````
```python
print("hello world")
```

```javascript
console.log("hello world");
```

```bash
echo "hello world"
```

```yaml
key: value
```

```json
{"key": "value"}
```

```sql
SELECT * FROM users;
```

```
No language tag for plain text / output
```
````

### Supported Languages

| Language | Tag | Notes |
|----------|-----|-------|
| Python | `python` | |
| JavaScript | `javascript` | Also: `js` |
| TypeScript | `typescript` | Also: `ts` |
| Bash/Shell | `bash` | Also: `shell`, `sh` |
| YAML | `yaml` | Also: `yml` |
| JSON | `json` | |
| SQL | `sql` | |
| HTML | `html` | |
| CSS | `css` | |
| Dockerfile | `dockerfile` | Also: `docker` |
| Go | `go` | |
| Rust | `rust` | |
| C# | `csharp` | Also: `c#`, `cs` |
| Java | `java` | |
| Diff | `diff` | For showing changes |
| Text | (none) | For output/examples |

### Code Block Guidelines

1. **Maximum line length:** 100 characters per line (horizontal scrolling is bad UX)
2. **Use comments** to explain non-obvious code
3. **Start each code block** with a comment describing what it does
4. **Prefer runnable examples** over pseudocode
5. **Include output** after code where helpful

### Example with Output

```python
# Calculate factorial recursively
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Expected: 120
```

```
Output:
120
```

### Inline Code

Use inline code for:
- **File paths**: `/path/to/file.md`
- **Commands**: `pip install torch`
- **Variable names**: `model_name`, `batch_size`
- **Short snippets**: `os.path.join(a, b)`
- **CLI flags**: `--verbose`, `-o output.json`

---

## Issue Templates

### Bug Report Template

```markdown
---
name: Bug Report
about: Report an error, broken link, or incorrect information
title: "[Bug] Short description"
labels: bug
---

## Bug Description

A clear and concise description of the bug.

## Location

**File:** `path/to/file.md`
**Section:** Section name (if applicable)
**Line:** ~ Line number (if applicable)

## Current Behavior

What currently appears:
```
{{current content}}
```

## Expected Behavior

What should appear instead:
```
{{expected content}}
```

## Evidence

- Link to authoritative source: {{url}}
- Screenshot: {{if applicable}}

## Suggested Fix

(Optional) If you know how to fix it:

````markdown
```markdown
{{corrected content}}
```
````

## Additional Context

Add any other context here.
```

### Content Addition Template

```markdown
---
name: Content Addition
about: Suggest adding new content to an existing document
title: "[Addition] Short description"
labels: addition
---

## Addition Request

**File to update:** `path/to/file.md`
**Section to add to:** Section name

## Proposed Content

```markdown
{{your proposed content}}
```

## Why This Should Be Added

- **Relevance:** Why is this valuable?
- **Distinction:** How is this different from existing content?
- **Source:** Where does this information come from?

## Authoritative Sources

- {{source_url_1}}
- {{source_url_2}}

## Checklist

- [ ] I have checked that this content does not duplicate existing entries
- [ ] I have verified the information is accurate
- [ ] I have confirmed the sources are reliable
- [ ] This content follows the style guide
```

### New Document Proposal

```markdown
---
name: New Document
about: Propose a new document for the knowledge base
title: "[New Doc] Proposed title"
labels: new-document
---

## Proposal

**Suggested title:** {{title}}
**Suggested filename:** `NN-Descriptive-Name.md`
**Category:** {{category from 01-Overview.md directory structure}}

## Document Purpose

What gap does this document fill? Why isn't existing content sufficient?

## Outline

1. {{Section 1}}
2. {{Section 2}}
3. {{Section 3}}

## Target Audience

{{developer | researcher | practitioner | all}}

## Related Documents

- {{related document 1}}
- {{related document 2}}

## Would You Like to Write This?

- [ ] Yes, I will write a draft
- [ ] I'm suggesting it for someone else
```

---

## Pull Request Template

When you create a pull request, use this template:

```markdown
## Description

Please include a summary of the change and which issue is fixed.

Fixes #(issue number)

## Type of Change

- [ ] 📝 New document
- [ ] 🔧 Update to existing document
- [ ] 🐛 Bug fix (broken link, typo, factual error)
- [ ] ✨ New feature (new section, new list entry)
- [ ] ♻️ Refactoring (formatting, reorganization)
- [ ] 🚀 Performance improvement
- [ ] 📖 Documentation only

## Checklist

- [ ] I have read the [contribution guide](01-Overview.md#how-to-contribute)
- [ ] My PR title follows the commit message convention
- [ ] I have added/updated frontmatter as needed
- [ ] I have verified all links are valid
- [ ] I have checked for spelling and grammar
- [ ] I have added cross-references where appropriate
- [ ] I have used language-tagged code blocks where applicable
- [ ] My changes are consistent with the style guide

## Testing

Describe how you tested your changes:

- [ ] Ran markdown linting
- [ ] Verified links with link checker
- [ ] Previewed rendering

## Screenshots (if applicable)

## Additional Notes

Any additional context or deployment notes.
```

---

## Commit Message Conventions

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Usage | Example |
|------|-------|---------|
| `feat` | New content | `feat(fine-tuning): add DPO dataset list` |
| `fix` | Bug/error fix | `fix(roadmap): correct ICLR 2026 dates` |
| `docs` | Documentation only | `docs(overview): clarify contribution process` |
| `style` | Formatting, typos | `style(prompts): fix markdown formatting` |
| `refactor` | Restructure | `refactor(agents): reorganize framework sections` |
| `chore` | Maintenance | `chore: update link checker config` |
| `test` | Testing | `test: add frontmatter validation test` |

### Scopes

| Scope | Document Area |
|-------|---------------|
| `overview` | 01-Overview.md |
| `soul-skill` | 02-SOUL-SKILL-Templates.md |
| `prompts` | 03-Prompt-Libraries.md |
| `agents` | 04-Agent-Toolkits.md |
| `fine-tuning` | 05-Fine-Tuning-Datasets.md |
| `awesome` | 06-Awesome-AI-Repos.md |
| `roadmap` | 07-AI-2026-Roadmap.md |
| `templates` | 08-Contribution-Templates.md |
| `community` | 09-Community-Forums-Events.md |
| `tools` | 10-Tools-Ecosystem.md |
| `*` | Multiple files / global |

### Examples

```
feat(fine-tuning): add safety dataset section
fix(roadmap): correct NeurIPS 2026 date (Dec 6-12)
docs(overview): add FAQ entry for translation requests
style(prompts): fix indentation in CoT template
refactor(agents): consolidate framework comparison tables
chore: update link checker to skip known dead URLs
```

### Body Guidelines

- Use the body to explain **what** changed and **why**, not **how**
- Reference issues with `#123`
- Breaking changes should be marked with `BREAKING CHANGE:`

---

## Review Checklist

### For Contributors (before submitting)

```
## Pre-Submission Checklist

### Content
[ ] All facts are accurate and verified
[ ] All URLs are reachable and correct
[ ] No duplicate content with existing documents
[ ] Content adds value beyond existing resources
[ ] Language is clear, concise, and professional

### Formatting
[ ] YAML frontmatter is complete and valid
[ ] Headers follow proper hierarchy (# title, ## sections, ### subsections)
[ ] Code blocks have language tags
[ ] Tables have header rows and alignment
[ ] Cross-references use [see:...] format
[ ] No bare URLs (all wrapped in markdown links)

### Organization
[ ] File is in the correct directory
[ ] File name follows NN-Descriptive-Name.md convention
[ ] File is listed in 01-Overview.md if it's a new addition
[ ] Related cross-references are updated

### Quality
[ ] Spell-checked
[ ] Grammar-checked
[ ] Readability — can a newcomer understand this?
[ ] Length — appropriate for the topic (not too long/short)
[ ] Tone — neutral, factual, helpful
```

### For Reviewers

```
## Review Checklist

### Required Checks
[ ] Frontmatter is valid and complete
[ ] No factual errors
[ ] All links are functional
[ ] No duplicate content
[ ] Follows style guide

### Quality Checks
[ ] Is the content well organized?
[ ] Are code examples correct and runnable?
[ ] Are cross-references accurate?
[ ] Is the language clear and accessible?
[ ] Does it add value to the knowledge base?

### Recommendations
- [ ] Approve
- [ ] Changes requested (details below)
- [ ] Close (with reason)
```

---

## File Naming & Organization

### File Naming Convention

```
NN-Descriptive-Name.md
```

- **`NN`** — Two-digit number (01–99) for ordering
- **`Descriptive-Name`** — PascalCase-with-hyphens, descriptive of content
- **`.md`** — Markdown extension

### Directory Organization

```
15-Community-Resources-Templates/
├── 01-Overview.md
├── 02-SOUL-SKILL-Templates.md
├── ...
├── assets/            # Images, diagrams, other media
│   ├── images/
│   ├── diagrams/
│   └── other/
└── translations/      # Translated versions (optional)
    ├── zh/
    ├── es/
    └── ...
```

### Adding a New File

1. Choose the next available number (or a logical insertion point)
2. If inserting between existing files, use decimal: `02.5-Topic.md`
3. Add the file entry to `01-Overview.md#file-index--descriptions`
4. Update related cross-references in other files

### Deprecating or Archiving a File

1. Change `status` to `deprecated` or `archived` in frontmatter
2. Add a `supersedes` field pointing to the replacement
3. Update cross-references in `01-Overview.md`
4. Keep the file in place for backward compatibility

---

## Style Enforcement

### Automated Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **markdownlint** | Markdown style checking | `.markdownlint.json` |
| **lychee** | Link checking | `lychee.toml` |
| **vale** | Prose/style checking | `.vale.ini` |
| **prettier** | Automatic formatting | `.prettierrc` |
| **frontmatter-validator** | Custom frontmatter checks | `.frontmatter.yml` |

### Running Checks

```bash
# Markdown linting
markdownlint "**/*.md" --ignore node_modules

# Link checking
lychee "**/*.md" --exclude "https://twitter.com"

# Prose style
vale "**/*.md"

# Formatting
prettier --check "**/*.md"

# Frontmatter validation
python scripts/validate_frontmatter.py
```

### CI/CD Integration

All pull requests are checked with:
1. **markdownlint** — Style conformance
2. **lychee** — Link validity
3. **Frontmatter validation** — Required fields present
4. **Spell check** — No misspellings in new/changed content

PRs that fail automated checks will be blocked until resolved.

---

## Contributor License Agreement

By submitting content to this repository, you agree to the following:

### Grant of License

You grant the AI Knowledge Base project an irrevocable, worldwide, royalty-free license to use, copy, modify, distribute, and sublicense your contributions under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

### Representations

You represent that:
1. You have the right to make the contribution
2. The contribution does not infringe any third-party rights
3. The contribution is accurate to the best of your knowledge

### Sign-off

When committing, use the `--signoff` flag to acknowledge:

```
Signed-off-by: Your Name <your.email@example.com>
```

This confirms you agree to the terms above.

---

## Further Reading

- [01-Overview.md](01-Overview.md) — Directory structure and contribution process
- [02-SOUL-SKILL-Templates.md](02-SOUL-SKILL-Templates.md) — Document content style reference
- [GitHub Docs: PR Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
- [Conventional Commits](https://www.conventionalcommits.org/) — Commit message specification
- [Semantic Versioning](https://semver.org/) — Version numbering

---

*Document version 1.0 — Last updated 2026-06-12*
