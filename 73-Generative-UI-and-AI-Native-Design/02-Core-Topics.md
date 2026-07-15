# 02 — Generative UI & AI-Native Design: Core Topics

> **Category:** 73 — Generative UI and AI-Native Design
> **Last Updated:** July 2026
> **Cross-references:** [01-Overview](01-Overview.md), [33-AI-Native-Software-Development/04-Vibe-Coding](../33-AI-Native-Software-Development/04-Vibe-Coding-and-Low-Code-AI.md), [68-Context-Engineering](../68-Context-Engineering/)

---

## Table of Contents

1. [The Generation Loop](#1-the-generation-loop)
2. [Prompt-to-Interface: Input Modalities](#2-prompt-to-interface-input-modalities)
3. [The Constraint Layer (Design Systems)](#3-the-constraint-layer-design-systems)
4. [Component Libraries as Vocabularies](#4-component-libraries-as-vocabularies)
5. [Image-to-UI Translation](#5-image-to-ui-translation)
6. [Personalization and Memory](#6-personalization-and-memory)
7. [Conversational Iteration](#7-conversational-iteration)
8. [Accessibility in Generative UI](#8-accessibility-in-generative-ui)
9. [Brand Consistency](#9-brand-consistency)
10. [Evaluation Signals](#10-evaluation-signals)

---

## 1. The Generation Loop

A generative UI system is a loop, not a one-shot:

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
   USER INTENT ──► GENERATE ──► RENDER PREVIEW    │
   (prompt/img)     (LLM)        (live iframe)    │
                          ▲            │            │
                          │            ▼            │
                          └──── REVIEW ◄── USER EDIT / AUTO CHECK
                                 (chat or eval)
```

Each cycle adds **constraints** (feedback) that steer the next generation. The loop is what makes GenUI feel *collaborative* rather than *autonomous*.

### One-Shot vs. Iterative

| Mode | Pros | Cons | Use |
|---|---|---|---|
| One-shot | Fast, deterministic seed | May miss intent | Clone a known page |
| Iterative | Refines to taste | More tokens, latency | Net-new product UI |

---

## 2. Prompt-to-Interface: Input Modalities

### (a) Natural Language

```
"Build a pricing page with three tiers, a toggle for
 monthly/annual, and a sticky 'Start free trial' button.
 Use our brand blue #2563EB."
```

The model must parse: layout structure, components, interaction (toggle), styling (color).

### (b) Image / Screenshot

The model performs **visual understanding** then **code synthesis**. See [50-Multimodal-AI](../50-Multimodal-AI/).

### (c) Sketch / Wireframe

Lower visual fidelity → model fills gaps from constraints.

### (d) Reference + Diff

"Make it like Stripe's dashboard but for our IoT data." The reference grounds style; the diff targets content.

---

## 3. The Constraint Layer (Design Systems)

The single biggest lever for quality. A design system gives the generator:

- **Color tokens** — primary, secondary, semantic (success/danger).
- **Typography scale** — font families, sizes, weights.
- **Spacing scale** — 4px / 8px grids.
- **Elevation & radius** — shadows, border radii.
- **Component specs** — button variants, input states.

```json
// design-tokens.json — fed to the generator as context
{
  "color": {
    "primary": "#2563EB",
    "background": "#0B0F19",
    "text": "#E5E7EB"
  },
  "radius": { "sm": "4px", "md": "8px", "lg": "16px" },
  "spacing": { "unit": 8 },
  "font": { "body": "Inter", "display": "Sora" }
}
```

> **See [68-Context-Engineering](../68-Context-Engineering/)** for how to package tokens + component docs into the model's context window.

---

## 4. Component Libraries as Vocabularies

Constraining the model to a known library produces idiomatic, maintainable code.

| Library | Style | Why use it |
|---|---|---|
| **shadcn/ui** | Tailwind + Radix | Copy-paste, fully editable, popular with v0 |
| **Radix / Headless** | Unstyled primitives | Accessibility built-in |
| **MUI** | Material Design | Enterprise familiarity |
| **Chakra / Mantine** | Opinionated | Fast, consistent |
| **Tailwind** | Utility-first | Compact, generator-friendly |

**Best practice:** Instruct the model *"only use components from ./components/ui"* so it imports rather than reinvents.

---

## 5. Image-to-UI Translation

The pipeline:

```
IMAGE ─► VISION ENCODER ─► LAYOUT GRAPH ─► CODE SYNTHESIS ─► PREVIEW
        (Gemini/Claude)     (boxes, text)    (HTML/React)
```

Common failure modes:
- Mis-detecting a decorative element as interactive.
- Wrong hierarchy (treating a subtitle as a heading).
- Pixel-perfect but non-responsive.

Mitigation: generate **semantic** HTML (correct landmarks, headings) and apply responsive classes afterward.

---

## 6. Personalization and Memory

A generative UI can adapt per user when wired to memory ([32-Agent-Memory-Systems](../32-Agent-Memory-Systems/)):

```ts
// Pseudocode: personalize layout from user memory
const prefs = await memory.get(userId, "ui_prefs");
const ui = await generateUI({
  intent: "dashboard",
  constraints: designSystem,
  personalize: {
    density: prefs.density,        // "compact" | "comfortable"
    accent: prefs.accentColor,
    modules: prefs.pinnedModules,
  },
});
```

This moves from *one-size-fits-all generation* to *one-size-fits-this-user generation*.

---

## 7. Conversational Iteration

The chat is the editing surface:

| User says | System does |
|---|---|
| "Make the hero bigger" | Adjusts font scale + spacing tokens |
| "Put the nav on the side" | Restructures layout grid |
| "Use our Q2 campaign colors" | Swaps token palette |
| "Add a testimonials section" | Inserts component from library |

**Prompt engineering tip:** keep prior generated code in context so edits are *diffs*, not regenerations — saving tokens and preserving untoucheds parts.

---

## 8. Accessibility in Generative UI

Generative UIs fail accessibility by default unless constrained. Add to every generation brief:

```md
ACCESSIBILITY REQUIREMENTS (mandatory):
- All images need alt text
- Color contrast ≥ WCAG AA (4.5:1 text)
- Interactive elements are real <button>/<a>, keyboard reachable
- Form inputs have associated <label>
- Headings follow a logical h1→h2→h3 order
```

Evaluate output with **axe-core** or **Lighthouse CI** in the review step ([69](../69-AI-Evaluation-and-LLM-Testing/)).

---

## 9. Brand Consistency

Brand drift is the #1 complaint about naive GenUI. Defenses:

1. **Token lock** — palette/spacing come only from the design system.
2. **Component lock** — only approved components are used.
3. **Style judge** — a second LLM pass scores output against brand guidelines (0–10) and rejects < 7.
4. **Human gate** — brand owner approves before publish.

```python
# Style judge (pseudocode)
score = judge_model(
    ui_code,
    rubric="Does this match brand guidelines? Score 0-10 with reasons."
)
if score < 7:
    regenerate_with_feedback(score.reasons)
```

---

## 10. Evaluation Signals

What to measure for a generated UI:

| Signal | Method | Tooling |
|---|---|---|
| Visual correctness | Screenshot diff vs. intent | Percy, Chromatic |
| Functional | Click-through smoke test | Playwright |
| Accessibility | Rule scan | axe-core, Lighthouse |
| Brand match | LLM rubric judge | Custom |
| Code quality | Lint + typecheck | ESLint, tsc |
| Performance | Lighthouse score | Lighthouse CI |

> Detailed evaluation harness design is in [03-Technical-Deep-Dive](03-Technical-Deep-Dive.md).

---

*Part of AI Knowledge Library auto-enrichment (2026-07-15).*
