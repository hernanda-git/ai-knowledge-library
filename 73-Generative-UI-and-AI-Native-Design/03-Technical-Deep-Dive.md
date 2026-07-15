# 03 — Generative UI & AI-Native Design: Technical Deep Dive

> **Category:** 73 — Generative UI and AI-Native Design
> **Last Updated:** July 2026
> **Cross-references:** [01-Overview](01-Overview.md), [02-Core-Topics](02-Core-Topics.md), [68-Context-Engineering](../68-Context-Engineering/), [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/), [03-Agents](../03-Agents/)

---

## Table of Contents

1. [Architecture of a GenUI System](#1-architecture-of-a-genui-system)
2. [The Generator Prompt Contract](#2-the-generator-prompt-contract)
3. [Code Generation Strategies](#3-code-generation-strategies)
4. [Constrained Decoding & Tool Use](#4-constrained-decoding--tool-use)
5. [Rendering & Hot-Reload Loop](#5-rendering--hot-reload-loop)
6. [Agentic UI Generation](#6-agentic-ui-generation)
7. [Evaluation Harness](#7-evaluation-harness)
8. [Failure Taxonomy](#8-failure-taxonomy)
9. [A Minimal Reference Implementation](#9-a-minimal-reference-implementation)
10. [Performance & Cost Engineering](#10-performance--cost-engineering)

---

## 1. Architecture of a GenUI System

```
                         ┌────────────────────────────┐
        USER PROMPT ───► │        ORCHESTRATOR        │
        IMAGE / REF      │  (plans, routes, validates)│
                         └──────────┬─────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                        ▼
     [PLANNER]              [GENERATOR]              [RENDERER]
     intent → spec          spec → code              code → live preview
            │                       │                        │
            └───────────┬───────────┴───────────┬──────────┘
                        ▼                         ▼
                  [CONSTRAINTS]             [EVALUATOR]
                  tokens, components         visual + functional gate
                        │                         │
                        └─────────► REGENERATE (if fail)
```

The **orchestrator** is often an agent ([03-Agents](../03-Agents/)) that decomposes a vague brief into a structured spec before any code is written. This "plan first, then generate" pattern dramatically improves quality.

---

## 2. The Generator Prompt Contract

A robust prompt template (context-engineered per [68](../68-Context-Engineering/)):

```md
# ROLE
You are a senior React + Tailwind engineer.

# CONTEXT (constraints — DO NOT VIOLATE)
- Use ONLY components from: ./components/ui (Button, Card, Input, ...)
- Design tokens: { "primary": "#2563EB", "radius": "8px" }
- Target: responsive, mobile-first.
- Accessibility: WCAG AA, real semantic elements, alt text.

# TASK
Generate a {page_type} for: "{user_intent}"

# OUTPUT FORMAT
Return ONLY a single TSX file wrapped in ```tsx. No prose.

# EXAMPLE (few-shot)
<example component from the library>
```

**Why structured?** Free-form "make me a nice page" yields inconsistent, non-idiomatic output. A contract aligns the model with your stack.

---

## 3. Code Generation Strategies

| Strategy | Description | When to use |
|---|---|---|
| **Whole-file** | Generate one complete component | Simple pages, clones |
| **Section-by-section** | Generate each section, then compose | Complex, multi-section UIs |
| **Diff-based** | Edit existing code via unified diff | Iterative refinement |
| **Spec-driven** | Model emits a JSON spec → renderer builds code | Maximum control / type safety |

**Spec-driven example** (separation of reasoning from rendering):

```json
{
  "layout": "two-column",
  "sections": [
    { "type": "hero", "title": "Ship faster", "cta": "Get started" },
    { "type": "feature-grid", "items": 3 }
  ],
  "theme": "dark"
}
```

A deterministic renderer turns this spec into code — cheaper and more reliable than letting the LLM write every line.

---

## 4. Constrained Decoding & Tool Use

Instead of hoping the model uses your library, **force** it via tool-calling:

```ts
// Pseudocode: generator constrained to component tools
const tools = [
  { name: "use_button",  description: "Render a Button", input: ButtonSchema },
  { name: "use_card",    description: "Render a Card",   input: CardSchema },
  { name: "use_input",   description: "Render an Input", input: InputSchema },
];

const plan = await model.chat({
  messages: [{ role: "user", content: intent }],
  tools,
  tool_choice: "required",   // model MUST call a tool
});

// Translate tool calls → JSX via a renderer
const jsx = renderPlanToJSX(plan.toolCalls, designSystem);
```

This guarantees every element maps to a known, accessible component.

---

## 5. Rendering & Hot-Reload Loop

```ts
import { createServer } from "vite";

async function preview(code: string) {
  const server = await createServer({ server: { port: 5173 } });
  writeFileSync("Generated.tsx", code);   // HMR picks it up
  return `http://localhost:5173/Generated`; // screenshot for eval
}

// Eval loop
const url = await preview(generated);
const shot = await screenshot(url);
const ok = await visualGate(shot, intent);
if (!ok) {
  const feedback = await critique(shot, intent);
  generated = await regenerate(generated, feedback);
}
```

The **live preview** is the feedback channel that makes the loop work — the model sees its own output rendered, not just text.

---

## 6. Agentic UI Generation

The frontier: an agent that **critiques and revises its own UI** using vision.

```ts
async function agenticGenerate(intent: string): Promise<string> {
  let code = await generate(intent);
  for (let i = 0; i < MAX_ITERS; i++) {
    const shot = await screenshot(await preview(code));
    const issues = await visionCritic(shot, intent);
    if (issues.length === 0) break;          // converged
    code = await regenerate(code, issues);    // self-repair
  }
  return code;
}
```

This is **agentic UI** — the UI emerges through self-correction against a visual goal, converging toward [70-World-Models](../70-World-Models/)-style simulation of "what the user will see."

---

## 7. Evaluation Harness

A layered gate before any UI ships:

```python
def evaluate(ui_code: str, intent: str) -> dict:
    results = {}
    # 1. Static: lint + typecheck
    results["lint"] = run(["eslint", ui_code]) == 0
    results["types"] = run(["tsc", "--noEmit", ui_code]) == 0
    # 2. Functional: headless click-through
    results["smoke"] = playwright_smoke(ui_code)
    # 3. Accessibility
    results["a11y"] = axe_score(ui_code) >= 90
    # 4. Visual diff vs. intent (CLIP / VLM similarity)
    results["visual"] = vlm_similarity(screenshot(ui_code), intent) >= 0.8
    # 5. Brand rubric
    results["brand"] = llm_judge(ui_code, brand_rubric) >= 7
    return results
```

See [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/) for the underlying eval methodology.

---

## 8. Failure Taxonomy

| # | Failure | Symptom | Mitigation |
|---|---|---|---|
| F1 | Hallucinated component | Imports a nonexistent `ButtonX` | Constrain to library manifest |
| F2 | Non-responsive | Looks fine on desktop, breaks mobile | Mandate mobile-first + viewport test |
| F3 | Inaccessible | Missing labels, low contrast | Axe gate in eval |
| F4 | Brand drift | Wrong palette | Token lock + style judge |
| F5 | Broken wiring | Button calls undefined handler | Typecheck + smoke test |
| F6 | Verbose code | 2000-line single file | Section-based generation + lint |
| F7 | Layout overflow | Content spills container | Flex/grid constraints + visual gate |
| F8 | Inconsistent across runs | Same prompt → different UI | Seed + version pinning |

---

## 9. A Minimal Reference Implementation

A runnable Node/TS sketch (illustrative):

```ts
// genui.ts — minimal orchestrator
import { openai } from "openai";                 // or any LLM SDK
import { renderToStaticMarkup } from "react-dom/server";
import { writeFileSync } from "fs";

const SYS = `You are a React+Tailwind engineer.
Use ONLY components from ./components/ui.
Return a single ```tsx block. No prose.`;

export async function genUI(intent: string, fewShot: string) {
  const client = new openai();
  const res = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYS },
      { role: "user", content: `Example:\n${fewShot}` },
      { role: "user", content: `Now build: ${intent}` },
    ],
  });
  const code = extractTsx(res.choices[0].message.content!);
  writeFileSync("out/Generated.tsx", code);
  return code;
}

// Wire to a preview server (Vite) + screenshot + gate as in §5/§7.
```

> This is intentionally minimal. Production systems add the planner, constraint layer, and eval harness from §1–§7.

---

## 10. Performance & Cost Engineering

| Lever | Impact | How |
|---|---|---|
| Spec-driven rendering | High cost save | LLM emits JSON, not code |
| Diff-based edits | Medium | Edit in place vs. full regen |
| Model routing | High | Cheap model for drafts, strong for final |
| Caching | Medium | Cache by (intent hash + tokens version) |
| Streaming UI | UX | Show partial render immediately |
| Few-shot reuse | Medium | Reuse solved examples as context |

```ts
// Model routing: cheap draft, strong polish
const draft = await cheapModel.generate(intent);
const polished = await strongModel.refine(draft, brandRubric);
```

---

*Part of AI Knowledge Library auto-enrichment (2026-07-15).*
