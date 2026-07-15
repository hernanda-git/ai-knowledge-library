# 04 — Generative UI & AI-Native Design: Tools and Frameworks

> **Category:** 73 — Generative UI and AI-Native Design
> **Last Updated:** July 2026
> **Cross-references:** [01-Overview](01-Overview.md), [02-Core-Topics](02-Core-Topics.md), [33-AI-Native-Software-Development/04-Vibe-Coding](../33-AI-Native-Software-Development/04-Vibe-Coding-and-Low-Code-AI.md)

---

## Table of Contents

1. [Product Landscape](#1-product-landscape)
2. [Detailed Profiles](#2-detailed-profiles)
3. [Open-Source Tooling](#3-open-source-tooling)
4. [Component & Design-System Foundations](#4-component--design-system-foundations)
5. [Evaluation & Preview Tooling](#5-evaluation--preview-tooling)
6. [Model Backends](#6-model-backends)
7. [Build Tutorial: GenUI from Scratch](#7-build-tutorial-genui-from-scratch)
8. [Selection Guide](#8-selection-guide)
9. [Integration Patterns](#9-integration-patterns)
10. [Watchlist](#10-watchlist)

---

## 1. Product Landscape

| Product | Vendor | Inputs | Output | Deploy | Notes |
|---|---|---|---|---|---|
| **v0** | Vercel | NL, image | React + Tailwind | 1-click Vercel | Tight shadcn/ui integration |
| **Lovable** | Lovable | NL | Full-stack app | Built-in | Backend + DB + UI |
| **Bolt.new** | StackBlitz | NL | Full-stack (in-browser) | Instant | WebContainer runtime |
| **Replit Agent** | Replit | NL | App + deploy | Built-in | End-to-end |
| **Cursor** | Anysphere | Code + NL | Edited code | Your infra | Agentic editing |
| **Windsurf** | Codeium | Code + NL | Edited code | Your infra | Cascade agent |
| **Figma AI** | Figma | NL, image | Design layer | Figma | Handoff-oriented |
| **Galileo / Uizard** | — | Sketch, NL | Mockups | Export | Design-first |
| **Canva Magic** | Canva | NL | Visual designs | Canva | Marketing assets |

---

## 2. Detailed Profiles

### v0 (Vercel)
- **Strength:** Production-quality React/Tailwind; component reuse from shadcn/ui.
- **Workflow:** prompt → multiple variants → pick → refine in chat → deploy.
- **Best for:** Front-end engineers who want a head start with maintainable code.

### Lovable
- **Strength:** Full-stack (UI + Supabase backend + auth) from one prompt.
- **Best for:** Non-developers shipping a real product.
- **Relation:** Extends [Vibe Coding (33-04)](../33-AI-Native-Software-Development/04-Vibe-Coding-and-Low-Code-AI.md) to the whole stack.

### Bolt.new
- **Strength:** Runs entirely in the browser via WebContainers — no local setup.
- **Best for:** Instant prototyping and demos.

### Replit Agent
- **Strength:** Generates, runs, and deploys in one environment.
- **Best for:** Learners and solo builders.

---

## 3. Open-Source Tooling

| Tool | What it does |
|---|---|
| **screenshot-to-code** | Screenshot → HTML/React/Vue via VLM |
| **OpenUI** | Chat-based UI generation (BYO model) |
| **shadcn/ui + `npx shadcn add`** | Component source as generator vocabulary |
| **tldraw "make real"** | Whiteboard sketch → UI |
| **GPT-Engineer / OpenHands** | NL → project scaffold incl. UI |
| **Aider** | Git-aware agentic code editing |

Example — screenshot-to-code style flow:

```bash
git clone https://github.com/abi/screenshot-to-code
cd screenshot-to-code && pip install -r requirements.txt
# Set OPENAI_API_KEY, then:
python main.py  # then paste a screenshot URL in the UI
```

---

## 4. Component & Design-System Foundations

Your generator is only as good as its vocabulary:

| Foundation | Role |
|---|---|
| **shadcn/ui** | Copy-paste components, generator-friendly |
| **Radix UI** | Accessible headless primitives |
| **Tailwind CSS** | Utility classes = compact prompts |
| **Material UI (MUI)** | Enterprise component set |
| **Storybook** | Component catalog the model can browse |
| **Figma Variables** | Export tokens → generator context |

**Tip:** Expose your component catalog as a markdown manifest the LLM can read:

```md
# Component Catalog
- Button(variant: primary|secondary|ghost, size: sm|md|lg)
- Card(padding, elevated: bool)
- Input(type: text|email|password, label)
```

---

## 5. Evaluation & Preview Tooling

| Tool | Use |
|---|---|
| **Vite / Webpack dev server** | Hot-reload preview |
| **Playwright / Puppeteer** | Screenshot + smoke tests |
| **Chromatic / Percy** | Visual regression |
| **axe-core / Lighthouse** | Accessibility + perf |
| **Storybook Test** | Component-level checks |

A screenshot pipeline:

```bash
npm run dev &          # Vite on :5173
npx playwright screenshot http://localhost:5173 out.png
npx axe http://localhost:5173      # a11y report
```

---

## 6. Model Backends

| Backend | Strength for GenUI |
|---|---|
| **Claude (Anthropic)** | Strong code + long-context spec |
| **GPT-4o / 5-class (OpenAI)** | Multimodal image→UI |
| **Gemini (Google)** | Native vision, long context |
| **Open weights (Qwen-VL, DeepSeek-V3)** | Self-hosted, private ([23-Local-AI](../23-Local-AI-Inference-Self-Hosting/)) |

Model routing (cheap draft → strong polish) is covered in [03-Technical-Deep-Dive §10](03-Technical-Deep-Dive.md).

---

## 7. Build Tutorial: GenUI from Scratch

A 6-step minimal pipeline you can run today:

**Step 1 — Scaffold**
```bash
npm create vite@latest my-genui -- --template react-ts
cd my-genui && npm i && npx shadcn@latest init
```

**Step 2 — Add a component vocabulary**
```bash
npx shadcn@latest add button card input
```

**Step 3 — Write the generator (Node/TS)**
```ts
import OpenAI from "openai";
const oai = new OpenAI();
const SYS = "React+Tailwind engineer. Use ./components/ui only. Return ```tsx.";
export async function gen(intent: string) {
  const r = await oai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "system", content: SYS },
                { role: "user", content: intent }],
  });
  return extractTsx(r.choices[0].message.content!);
}
```

**Step 4 — Preview**
```bash
echo "export { default } from './Generated'" > src/App.tsx
npm run dev   # open localhost:5173
```

**Step 5 — Evaluate** (axe + Playwright per §5).

**Step 6 — Deploy**
```bash
npm run build && npx vercel deploy --prod
```

---

## 8. Selection Guide

| If you are… | Choose |
|---|---|
| A front-end dev wanting a head start | v0 + shadcn |
| A non-dev shipping a product | Lovable / Bolt |
| Building inside an existing repo | Cursor / Windsurf / Aider |
| Privacy-sensitive enterprise | Self-hosted model + OpenUI ([23](../23-Local-AI-Inference-Self-Hosting/)) |
| Design-team handoff | Figma AI / Galileo |

---

## 9. Integration Patterns

- **GenUI inside an app** — let end-users generate dashboards/reports (personalization via [32-Memory](../32-Agent-Memory-Systems/)).
- **Design-ops pipeline** — generator → brand judge → Storybook → production.
- **Agent UI** — an agent ([03](../03-Agents/)) calls the generator as a tool to build its own control surface.

---

## 10. Watchlist

- **Agentic UI that self-improves from telemetry** (click-through → regenerate).
- **Multimodal design agents** that go sketch → brand → code → deploy unattended.
- **Standardized UI spec formats** (like an "HTML for agents") to make generators interchangeable.
- **On-device GenUI** for privacy ([62-Edge-AI](../62-Edge-AI-and-On-Device-Inference/)).

---

*Part of AI Knowledge Library auto-enrichment (2026-07-15).*
