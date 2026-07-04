# Gap Report — July 4, 2026 (Auto-Enrichment — AI Model Routing and Smart Selection)

## Auto-Enrichment Summary

### What Was Done
- **New document added**: 02-LLMs/10-AI-Model-Routing-and-Smart-Selection.md
- **Total lines added**: 1,232
- **Git commit**: pending
- **Pushed to**: GitHub (main branch)

### Gap Identified: AI Model Routing and Smart Selection

**Why this gap?** Web research on Hacker News (July 2-4, 2026) confirmed extremely strong demand signals around the emerging discipline of dynamic model routing:

**Strong signals from HN:**
- **"GLM5.2 on AMD MI355X at 2626 tok/s/node at over 2x lower cost than Blackwell"** (154pts) — Inference cost competition between providers is the dominant theme
- **"AI inference is obviously profitable"** (11pts) — Discussion of inference economics
- **"Save Claude Code Tokens with Smart Routing"** (9pts) — Direct reference to smart routing
- **"GPT 5.5 (high) is as good at coding as Claude Fable (medium) at a lower cost"** (4pts) — Cost-quality tradeoff across models
- **"Did you know your code is overpaying for AI?"** (3pts) — Cost optimization need
- **"Kontext – Move an AI chat's full context to another AI"** (8pts) — Context portability across models

**Library gap:** The library had existing coverage of:
- 02-LLMs/02-Model-Families.md — Model capabilities and comparisons
- 02-LLMs/06-AI-Model-Providers-Free-Tiers.md — Provider pricing and free tiers
- 41-AI-Cost-Optimization-and-Enterprise-ROI/ — Enterprise cost optimization

But **NO dedicated guide** to the practice of model routing — dynamically selecting the optimal model for each request based on task type, cost, quality, and latency requirements.

### What Was Created

A comprehensive 1,232-line document covering:

1. **Why Model Routing Matters** — Cost explosion data, task-dependent quality, provider diversity
2. **Model Landscape** — Tiers, selection matrix, task-model affinity
3. **Routing Architectures** — Static, classifier-based, hybrid, ML-powered
4. **Cost-Performance Optimization** — Tiered routing, adaptive routing, prompt-aware optimization
5. **Task-Based Selection** — Task classification, detection patterns, context-aware selection
6. **Real-Time Techniques** — Pre-flight estimation, speculative routing, cascade routing
7. **Implementation Patterns** — OpenRouter, cost-aware, quality-gated, multi-provider failover (with code)
8. **Quality Assurance** — Evaluation framework, A/B testing
9. **Production Deployment** — Microservice patterns, configuration, monitoring
10. **Case Studies** — Real-world cost savings (73% and 66% reductions)
11. **Tools & Frameworks** — OpenRouter, Portkey, LiteLLM, Not Diamond
12. **Common Pitfalls** — Over-routing, latency, feedback loops, provider lock-in
13. **Future Outlook** — Self-tuning routers, inference mesh, multi-model responses

### Remaining Gaps (Priority Order)

| Priority | Gap | Evidence | Status |
|----------|-----|----------|--------|
| 1 | **AI Agent Fleet Management** | Multiple HN posts about managing agent fleets (OpenLegion, Axon, Tonkotsu) | Not covered |
| 2 | **AI Model Vulnerability Disclosure** | CVE spike around Mythos release (68pts on HN) | Partially covered in 22 |
| 3 | **AI Agent Configuration as Code** | Skillsaw (3pts), theta-spec (5pts), Kontext (8pts) | Not covered |
| 4 | **AI Agent Incident Response** | AI Agent ransomware (3pts), credential crisis (3pts) | Not covered |
| 5 | **AI Model Benchmarking at Scale** | GPT5.5=Fable comparison (4pts), evaluation frameworks | Partially covered |
