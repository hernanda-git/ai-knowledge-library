# Cron Gap Report — Generative UI & AI-Native Design (2026-07-15)

## Gap
No dedicated category existed for **Generative UI (GenUI) & AI-Native Design** — the practice of using AI (LLMs/multimodal models) to produce user-interface artifacts (code, design layers, or both) from natural-language prompts, images, or sketches. The capability was only mentioned tangentially inside `33-AI-Native-Software-Development/04-Vibe-Coding-and-Low-Code-AI.md` (as the "interface half" of vibe coding) and in coding-assistant docs. It was explicitly listed as the **#1 next candidate** in the 2026-07-15 Agentic Search report.

## Why high priority
- Strong, sustained real-world demand signal: v0 (Vercel), Lovable, Bolt.new, Replit Agent, Cursor/Windsurf, Figma AI, Galileo/Uizard are a major 2025–2026 product category and a key frontier-lab + startup battleground.
- Directly converges multiple existing categories (Vibe Coding/33, Multimodal/50, Agents/03, Memory/32, Context Engineering/68, Evaluation/69) but was not itself a first-class topic.
- Enterprise adoption growing fast; accessibility, brand consistency, and eval are active pain points.
- Web search backend was unavailable this run (PARALLEL_API_KEY missing), so gap was identified via internal library analysis + the previous report's candidate list.

## Action taken
Created new top-level category `73-Generative-UI-and-AI-Native-Design/` with 5 files:
- 01-Overview.md (~190 lines) — definitions, spectrum, stack, market landscape, cross-refs
- 02-Core-Topics.md (~210 lines) — generation loop, constraint layer, image-to-UI, personalization, a11y, brand, eval signals
- 03-Technical-Deep-Dive.md (~280 lines) — architecture, prompt contract, constrained decoding, agentic generation, eval harness, failure taxonomy, reference impl
- 04-Tools-and-Frameworks.md (~230 lines) — product landscape, OSS tooling, build tutorial, selection guide
- 05-Future-Outlook.md (~150 lines) — trajectories, agentic/personalized UI, standardization, risks, 12-month forecast

## Excluded (already covered)
- AI Video Generation (50-Multimodal-AI/28-AI-Video-Audio-Generation) — distinct, has its own subdir.
- Vibe Coding (33-04) — parent practice; GenUI is the interface subset carved out here.
- Agentic Search (72-) — created earlier the same day.

## Next candidate gaps (not yet created)
1. Agentic Process Automation (APA) vs traditional RPA — only passing mentions.
2. AI Wearables / Ambient Intelligence consolidation (50- has one subdir but fragmented).
3. Human-AI Alignment evaluation at scale.
4. AI for Scientific Discovery workflow automation (beyond drug discovery, 42-).
5. AI-Native Interface asset generation (generative icons/illustration as a sub-topic).
