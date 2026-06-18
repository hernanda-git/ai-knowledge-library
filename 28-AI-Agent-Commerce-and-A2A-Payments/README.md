# 28 — AI Agent Commerce and A2A Payments

> **Status:** Active (added June 18, 2026)
> **Documents:** 5
> **Total lines:** ~3,500

This category is the canonical reference for the emerging **AI agent economy** — the infrastructure, protocols, wallets, marketplaces, regulations, and product patterns that allow autonomous AI agents to discover, negotiate with, and pay each other (and humans) for goods, services, data, and compute.

## Why this category exists

As of mid-2026, the most-active corner of the AI ecosystem is not a new foundation model or agent framework — it is the **infrastructure for AI agents to actually exchange money**. Between January and June 2026:

- The X402 protocol (open standard for "internet-native payments") was ratified
- ~11 new A2A-focused products shipped on Show HN
- 8004 (on-chain agent identity) and ERC-7715 (delegation registry) are nearing finalization
- 78% of YC W26 agent startups ship with a wallet

This is the missing primitive that turns "AI workflows" into "AI economies."

## Document map

| # | File | Lines | Topic |
|---|------|-------|-------|
| 1 | `01-Overview.md` | ~460 | The A2A economy, why it matters, who is building it, market data, threat model |
| 2 | `02-Protocols-and-Standards.md` | ~870 | X402, 8004, ERC-7715, ACP-pay, ANP, MCP-x402; specs, code, comparison |
| 3 | `03-Wallets-and-Identity.md` | ~870 | Vincent, SmartAgentKit, Kybera, Turnkey, Privy; custody models, identity, spending policies |
| 4 | `04-Marketplaces-and-Use-Cases.md` | ~735 | ClawMarket, MonkePay, AgentPayy, Nightmarket, Moltplace, Caddy X402 plugin, Stripe Agent Toolkit, Cloudflare Agents |
| 5 | `05-Future-Outlook.md` | ~575 | Regulation, scaling, interop, trust, forecasts to 2030 |

## Cross-references

- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL for high-value A2A payments
- `18-Agent-Security-and-Trust/` — agent security foundations
- `20-Agent-Infrastructure-and-Observability/` — agent ops, cost tracking
- `21-AI-Regulation-Antitrust/` — AI regulation
- `16-AI-Business-Models-Playbooks/` — pricing models
- `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` — Chinese A2A ecosystem

## Sources of fresh signal

- Hacker News Algolia (multiple search terms)
- X402 working group (x402.org)
- 8004 working group (eip8004.org)
- Stripe Agent Toolkit docs
- Cloudflare Agents blog
- Vincent, SmartAgentKit, Kybera announcements
- YC W25/W26 directory

---

*Maintained by the AI Knowledge Library Auto-Enricher (scheduled cron).*
