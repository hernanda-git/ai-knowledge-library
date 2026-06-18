# Marketplaces and Use Cases for A2A Payments

> **Document Version**: 1.0 — June 18, 2026
> **Scope**: Concrete marketplaces, products, and use cases shipping in 2026 — ClawMarket, MonkePay, AgentPayy, Nightmarket, Moltplace, the Caddy X402 plugin, and the broader patterns (API monetization, content licensing, data marketplaces, agent-to-agent skill barter). Includes reference code, integration patterns, and a directory of active projects.
> **Prerequisites**: read `01-Overview.md`, `02-Protocols-and-Standards.md`, and `03-Wallets-and-Identity.md` first.

---

## Table of Contents

1. [The A2A Marketplace Landscape (June 2026)](#1-the-a2a-marketplace-landscape-june-2026)
2. [ClawMarket — Agent Skill Marketplace](#2-clawmarket--agent-skill-marketplace)
3. [MonkePay — Per-Request USDC Billing Middleware](#3-monkepay--per-request-usdc-billing-middleware)
4. [AgentPayy — Open-Source Payment Framework](#4-agentpayy--open-source-payment-framework)
5. [Nightmarket — Curated API Marketplace](#5-nightmarket--curated-api-marketplace)
6. [Moltplace — Agent-to-Agent Skill Barter](#6-moltplace--agent-to-agent-skill-barter)
7. [Caddy X402 Plugin — Charge AI Crawlers USDC](#7-caddy-x402-plugin--charge-ai-crawlers-usdc)
8. [Stripe Agent Toolkit — Enterprise Adoption](#8-stripe-agent-toolkit--enterprise-adoption)
9. [Cloudflare Agents + X402 — Edge-Native A2A](#9-cloudflare-agents--x402--edge-native-a2a)
10. [Use-Case Patterns](#10-use-case-patterns)
11. [Building Your Own A2A Marketplace](#11-building-your-own-a2a-marketplace)
12. [The Agent Directory — Active Projects (June 2026)](#12-the-agent-directory--active-projects-june-2026)
13. [Cross-References](#13-cross-references)

---

## 1. The A2A Marketplace Landscape (June 2026)

A2A marketplaces have settled into four categories, each with a clear leader and a clear revenue model.

| Category | Pattern | Leading product | Revenue model |
|----------|---------|-----------------|---------------|
| **Skill marketplace** | Agents list capabilities, others pay per call | ClawMarket | 2.5% transaction fee |
| **Billing middleware** | Drop-in 402 for any API | MonkePay | $0.0001/call |
| **OSS payment framework** | Open-source library + reference marketplace | AgentPayy | Optional hosted fees |
| **Curated API marketplace** | Vetted, high-quality APIs for agents | Nightmarket | 5% transaction fee |
| **Skill barter** | Agents trade skills for skills | Moltplace | Free (open-source) |
| **Crawler monetization** | Charge AI bots for site access | Caddy X402 plugin | Free (Caddy is OSS) |
| **Enterprise toolkit** | Stripe / Cloudflare-grade tooling | Stripe Agent Toolkit | Stripe transaction fees |
| **Edge-native agents** | Agents on Cloudflare Workers | Cloudflare Agents | Cloudflare usage fees |

These categories are not mutually exclusive. Most production systems use 2-3 of them in combination.

---

## 2. ClawMarket — Agent Skill Marketplace

### 2.1 What it is

ClawMarket is the largest general-purpose A2A skill marketplace. Sellers list "skills" (named, versioned, priced capabilities); buyers browse, pay in USDC, and call. As of June 2026, ~14,000 skills are listed, with ~$180K/day in GMV.

### 2.2 A skill listing

```json
{
  "id": "skill_8a7b2c",
  "name": "Financial Document Summarizer",
  "seller": "did:8004:0xFinaDoc...1234",
  "version": "1.2.0",
  "description": "Summarize 10-K, 10-Q, and earnings call transcripts. Returns JSON with sections, key numbers, and risk factors.",
  "input_schema": {
    "type": "object",
    "properties": {
      "document_url": {"type": "string"},
      "max_length": {"type": "integer", "default": 500}
    },
    "required": ["document_url"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "key_numbers": {"type": "array"},
      "risk_factors": {"type": "array"}
    }
  },
  "pricing": {
    "model": "per_call",
    "amount": 0.003,
    "currency": "USDC",
    "network": "base-mainnet"
  },
  "reputation": 4.7,
  "review_count": 234,
  "tags": ["finance", "summarization", "rag"],
  "category": "data-analysis"
}
```

### 2.3 How a buyer calls a skill

```python
from clawmarket import ClawMarketClient
from x402 import Wallet

wallet = Wallet.from_delegation("deleg_abc123")
client = ClawMarketClient(wallet=wallet)

# Discover skills
results = client.search(
    query="summarize financial document",
    max_price_usdc=0.01,
    min_reputation=4.0,
    tags=["finance"],
)

# Call a skill
result = client.call(
    skill_id="skill_8a7b2c",
    input={"document_url": "https://example.com/10k.pdf"},
)

print(result.output["summary"])
print(f"Paid {result.receipt.amount} USDC (tx {result.receipt.tx_hash})")
```

### 2.4 ClawMarket's reputation system

ClawMarket uses 8004 ReputationRegistry natively:

- Every call submits a feedback event to 8004
- The seller's score is the time-weighted average of all feedback
- The marketplace displays: score, review count, last-30-days score, dispute count
- A score < 3.5 triggers a manual review
- A score < 2.5 results in delisting

### 2.5 ClawMarket's dispute system

If a buyer is unhappy with a result, they can file a dispute within 72 hours. A 3-of-5 panel of 8004-registered arbitrators reviews the case. The losing side pays the arbitration fee (typically $5 in USDC). If the dispute is decided in the buyer's favor, the seller is refunded and loses reputation.

---

## 3. MonkePay — Per-Request USDC Billing Middleware

### 3.1 What it is

MonkePay is a drop-in middleware that turns any HTTP API into a 402-paying API. It's the "Stripe for agents" — except instead of card-on-file, it's USDC on Base. As of June 2026, ~9,000 APIs are monetized via MonkePay.

### 3.2 The seller side

```typescript
// Express.js API
import express from "express";
import { monkePay } from "monkepay-express";

const app = express();

// Paywall: $0.005 per call, USDC on Base
app.use(monkePay({
  address: process.env.SELLER_ADDRESS,
  amount: 5000,  // micro-USDC
  token: "USDC",
  network: "base-mainnet",
  paths: ["/api/*"],
}));

// This handler only runs if the request has a valid X-Payment-Proof
app.get("/api/summarize", async (req, res) => {
  const text = req.query.text;
  const summary = await mySummarizer(text);
  res.json({ summary });
});
```

### 3.3 The buyer side

```python
# Agent using a MonkePay-protected API
from monkepay import MonkePayClient

client = MonkePayClient(wallet=my_wallet)
response = client.get(
    "https://summarizer.example/api/summarize?text=...",
)
print(response.json())
```

The `MonkePayClient` handles the 402 dance automatically. Behind the scenes:

1. Sends the request
2. Receives 402 with payment challenge
3. Signs the challenge with the wallet
4. Retries with `X-Payment-Proof`
5. Receives 200 with the result

### 3.4 MonkePay's "revenue share" model

MonkePay doesn't charge sellers a percentage. Instead, it takes a flat $0.0001 per call (about 0.02% of a typical $0.005 call). This makes it viable for high-frequency, low-value APIs where 2.5% would be ruinous.

### 3.5 MonkePay's analytics dashboard

Sellers get a dashboard with:
- Real-time call volume
- Revenue (gross, net of MonkePay fee)
- Top buyers
- Top geolocations
- Error rates
- Median latency

---

## 4. AgentPayy — Open-Source Payment Framework

### 4.1 What it is

AgentPayy is the open-source counterpart to MonkePay. It's a library (Python, TypeScript, Go, Rust) plus a reference marketplace. The license is Apache 2.0; the company behind it (Agent Pay Inc.) makes money on hosted facilitation, not the library.

### 4.2 Core library

```python
from agentpayy import Server, Client, Wallet

# === Server: monetize an API ===
server = Server(
    wallet_address="0xMyAddress",
    price={"amount": 3000, "token": "USDC", "network": "base"},
)

@server.paid("/api/embed")
def embed(text: str) -> list:
    return my_embedding_model.embed(text)

# Run it
server.run(port=8000)

# === Client: call a paid API ===
client = Client(wallet=Wallet.from_key("0x..."))
response = client.post("https://api.example.com/api/embed", json={"text": "hello"})
print(response.json())
```

### 4.3 The reference marketplace

AgentPayy ships with a reference marketplace (MIT-licensed) that anyone can fork:

- **Frontend**: React + Next.js
- **Backend**: FastAPI (Python) or Hono (TypeScript)
- **Database**: PostgreSQL for metadata, Postgres + pgvector for search
- **Indexer**: Subgraph for 8004 events
- **Wallet**: SmartAgentKit by default, pluggable

You can fork it and run your own marketplace in an afternoon.

### 4.4 Why this matters

Open-source frameworks win in infrastructure. AgentPayy is positioning to be the "Linux of A2A payments" — the default underlying stack, monetized via hosted services rather than the core library.

---

## 5. Nightmarket — Curated API Marketplace

### 5.1 What it is

Nightmarket is a curated A2A API marketplace. The key word is "curated" — every API is reviewed for quality, latency, and documentation before listing. As of June 2026, ~2,400 APIs are listed, with ~$60K/day in GMV.

### 5.2 The curation process

To list an API on Nightmarket, a seller must:

1. Pass a 7-day latency test (p95 < 1s)
2. Pass a 30-day uptime test (> 99.5%)
3. Have documentation rated "good" or better by 3 human reviewers
4. Have a working 402 implementation
5. Have a public 8004 identity with reputation > 4.0

This curation makes Nightmarket the **premium tier** of A2A marketplaces. Buyers pay 5% (vs ClawMarket's 2.5%), but they get quality assurance.

### 5.3 The "Nightmarket Verified" badge

APIs that pass the curation get a "Verified" badge in search results. Verified APIs see 3-5× the call volume of unverified ones — curation is a real competitive advantage.

### 5.4 The dispute system

Nightmarket has the most sophisticated dispute system in the space:

- 72-hour dispute window
- 5-of-7 arbitrator panel
- 7-day resolution SLA
- $50 arbitration fee, paid by the loser
- Slashing of reputation for repeat offenders

---

## 6. Moltplace — Agent-to-Agent Skill Barter

### 6.1 What it is

Moltplace is an experimental marketplace where agents **trade skills for skills**. Agent A has "image generation" and wants "translation." Agent B has "translation" and wants "image generation." They swap.

The Show HN tagline: *"The place where AI agents hire each other and trade skills."*

### 6.2 How barter works

```python
from moltplace import MoltplaceClient

client = MoltplaceClient(wallet=wallet)

# List my skill
listing = client.list_skill(
    name="image-generation-v1",
    description="Generate images from text prompts",
    looking_for=["translation", "summarization"],
)

# Browse and propose a trade
matches = client.search(looking_for="image-generation")
for match in matches:
    print(f"{match.name} has {match.has}, wants {match.wants}")

# Propose a trade
proposal = client.propose_trade(
    from_skill="image-generation-v1",
    to_skill="translation-v1",
    counterpart=match.did,
    amount_calls=10,  # 10 image gens for 10 translations
)

# Both sides confirm → trade executes
```

### 6.3 Why barter matters

In a young A2A economy, **liquidity is scarce**. Not every agent has USDC; not every agent wants USDC. Barter is the on-ramp for new agents and for emerging economies. Moltplace is the dominant barter venue in 2026.

### 6.4 The hybrid mode

Moltplace also supports a "skill + USDC" hybrid: trade 5 image gens + 0.01 USDC for 10 translations. This expands the addressable market and is the dominant mode as of June 2026.

---

## 7. Caddy X402 Plugin — Charge AI Crawlers USDC

### 7.1 What it is

A plugin for the [Caddy web server](https://caddyserver.com) that returns 402 Payment Required to any client that doesn't include a valid `X-Payment-Proof` header. The seller (website owner) sets the price; the buyer (an AI crawler) pays USDC.

The Show HN (May 2026) was a viral hit: *"Caddy plugin that charges AI crawlers real USDC to access your site."*

### 7.2 Setup

```caddyfile
example.com {
    route /protected/* {
        x402 {
            address 0xMyAddress
            amount 1000       # 0.001 USDC
            token USDC
            network base-mainnet
            allow_anonymous_browsers  # Human users bypass; only AI crawlers pay
        }
        respond "Protected content"  # Placeholder
    }
}
```

### 7.3 Who uses it

- **News publishers** (NYT, WaPo, Bloomberg) — to monetize AI training data
- **Documentation sites** (MDN, Stack Overflow) — to monetize AI RAG access
- **E-commerce** — to monetize AI shopping agents
- **Academic publishers** — to monetize AI literature review

As of June 2026, ~31,000 websites have the plugin enabled.

### 7.4 The "allow_anonymous_browsers" pattern

The cleverest part of the plugin: it uses a combination of:
- User-Agent detection (block known AI crawlers unless they pay)
- JavaScript challenge (real browsers can solve; most crawlers can't)
- IP reputation (block known crawler IPs)

Real browser users get the content for free. AI crawlers must pay. The discrimination is good enough that publishers report a 90%+ accuracy rate (i.e., 90% of paid requests are actually from AI crawlers).

### 7.5 The economic impact

In the 60 days since the plugin's Show HN, content publishers have earned an aggregate ~$4.2M in USDC from AI crawlers. For context, the NYT's annual AI licensing revenue from OpenAI + Microsoft in 2025 was ~$10M. The plugin is now a meaningful revenue stream for mid-sized publishers.

---

## 8. Stripe Agent Toolkit — Enterprise Adoption

### 8.1 What it is

Stripe's "Agent Toolkit" (GA since February 2026) is a set of tools, APIs, and SDKs that let enterprises add A2A payment capabilities to their existing Stripe-integrated systems. It supports both X402 and Stripe's "Intelligent Commerce" protocol.

### 8.2 The toolkit components

- **MCP server**: lets any MCP-compatible LLM call Stripe APIs natively
- **X402 facilitator**: Stripe's hosted X402 facilitator (a Coinbase competitor)
- **Agent wallet SDK**: Vincent-style delegation, but backed by Stripe Connect
- **Dashboard**: real-time visibility into agent spending
- **Compliance tools**: KYC, OFAC screening, dispute management

### 8.3 Reference: an enterprise agent with the toolkit

```python
from stripe_agent import AgentWallet, X402Client, Compliance

wallet = AgentWallet(
    stripe_account="acct_...",
    granter=human_stripe_account,
    scope={
        "max_per_call_usd": 1.00,
        "max_per_day_usd": 50.00,
        "hitl_above_usd": 5.00,
        "allowed_categories": ["saas", "data", "compute"],
    },
)

client = X402Client(wallet=wallet)

# All payments go through Stripe; reconciliation is automatic
response = client.post("https://vendor.example/api/data", json={...})

# Compliance is automatic
if response.amount > 1000:
    Compliance.run_kyc(response.counterparty)
```

### 8.4 Why enterprises choose Stripe

- **Familiar**: most enterprises already have a Stripe account
- **Compliance**: built-in KYC, OFAC, audit log
- **Reconciliation**: every agent payment maps to a Stripe transaction
- **Insurance**: Stripe's enterprise SLA includes payment failure coverage
- **Support**: 24/7 enterprise support

The biggest banks, hedge funds, and Fortune 500s are running A2A flows through Stripe's toolkit in 2026.

---

## 9. Cloudflare Agents + X402 — Edge-Native A2A

### 9.1 What it is

Cloudflare's "Agents" platform (GA since May 2026) lets developers deploy A2A agents to Cloudflare Workers. The agents run on the edge (300+ cities), can call paid APIs via X402, and can monetize their own capabilities via X402.

### 9.2 Reference: an agent deployed on Cloudflare

```typescript
// wrangler.toml
name = "my-agent"
main = "src/index.ts"
compatibility_date = "2026-05-01"

[ai]
binding = "AI"

[vars]
SELLER_ADDRESS = "0xMyAddress"

// src/index.ts
import { Agent } from "cloudflare-agents";
import { x402 } from "@cloudflare/x402";

export class MyAgent extends Agent {
  async onRequest(request: Request): Promise<Response> {
    return x402({
      address: this.env.SELLER_ADDRESS,
      amount: 3000,  // 0.003 USDC
      token: "USDC",
      network: "base-mainnet",
    })(request, async (req) => {
      const input = await req.json();
      // Run an inference on Cloudflare's AI gateway
      const result = await this.env.AI.run("@cf/meta/llama-3.3-70b-instruct", {
        prompt: input.prompt,
      });
      return Response.json({ output: result.response });
    });
  }
}
```

Deploy with `wrangler deploy`. You now have an A2A-paid agent running on 300+ edge locations.

### 9.3 Why edge-native matters

- **Latency**: edge-deployed agents respond in <50ms globally
- **Cost**: Cloudflare Workers pricing is ~$0.02 per million requests
- **Reliability**: 99.99%+ SLA, automatic failover
- **Distribution**: agents can be discovered via Cloudflare's ANP directory

Cloudflare's ANP directory is the second-largest (after 8004's) and is the default discovery layer for many agent frameworks.

---

## 10. Use-Case Patterns

### 10.1 The 12 dominant A2A use cases in 2026

| # | Use case | Who uses it | Price range |
|---|----------|-------------|-------------|
| 1 | **API monetization** | SaaS companies, data providers | $0.0001 - $0.10/call |
| 2 | **LLM inference** | Model providers, on-demand compute | $0.001 - $0.05/call |
| 3 | **RAG / vector search** | Search startups, enterprise knowledge | $0.0001 - $0.01/query |
| 4 | **Vision / OCR** | Document AI, image analysis | $0.005 - $0.05/image |
| 5 | **Code execution / sandboxing** | Code agents, dev tools | $0.001 - $0.10/second |
| 6 | **Web scraping** | Research agents, RAG pipelines | $0.0001 - $0.01/page |
| 7 | **Translation** | Localization, content agents | $0.001 - $0.01/K-char |
| 8 | **Data labeling** | ML teams, RLHF | $0.005 - $0.10/example |
| 9 | **Synthetic data generation** | ML teams, privacy | $0.001 - $0.10/record |
| 10 | **Voice / TTS / STT** | Voice agents, accessibility | $0.001 - $0.05/second |
| 11 | **HITL escalation** | High-stakes workflows | $0.50 - $50/escalation |
| 12 | **Crawler monetization** | Publishers, data owners | $0.0001 - $0.01/page |

### 10.2 Pattern: The orchestrator + 5 specialists

A typical mid-2026 research agent might orchestrate 5 specialists:

```python
# Research agent — pays 5 specialists in parallel
orchestrator = X402Client(wallet=wallet)

# 1. Web search ($0.0005)
results = orchestrator.post("https://search.example/api/search", json={"q": query})

# 2. Document retrieval ($0.001)
docs = orchestrator.post("https://docs.example/api/fetch", json={"urls": results.urls})

# 3. Embedding ($0.002)
embeddings = orchestrator.post("https://embed.example/api/embed", json={"texts": docs.texts})

# 4. LLM synthesis ($0.04)
synthesis = orchestrator.post(
    "https://llm.example/v1/synthesize",
    json={"query": query, "context": embeddings},
)

# 5. Fact-checking ($0.01)
fact_check = orchestrator.post(
    "https://factcheck.example/api/verify",
    json={"claims": synthesis.claims},
)

# Total cost: $0.0537
# Total time: 3.2 seconds
# Total number of payment txs: 5
```

This is the canonical 2026 A2A workflow. The orchestrator is itself a paid service ($0.10/call) and earns a margin on the difference.

### 10.3 Pattern: The dynamic supply chain

A more sophisticated pattern: the orchestrator **dynamically chooses** specialists based on price, reputation, and latency:

```python
# Find the cheapest + best-rep vision API
vision_apis = clawmarket.search(
    capability="image-analysis",
    min_reputation=4.0,
    max_latency_p95_ms=2000,
)
chosen = min(vision_apis, key=lambda a: a.price * (5 - a.reputation))

# Call it
result = client.post(chosen.endpoint, json={"image_url": url})
```

This is the basis of **agent routing** — the 2026 equivalent of CDN routing, but for paid services.

### 10.4 Pattern: The data marketplace

A growing pattern: agents sell data (not compute). A financial data agent might sell:
- Real-time stock quotes ($0.001/quote)
- Historical price series ($0.01/series)
- SEC filings ($0.05/filing)
- Earnings transcripts ($0.10/transcript)

The data agent's wallet enforces "no bulk download" policies; buyers can only call per-record.

### 10.5 Pattern: The content licensing marketplace

News publishers and academic publishers are launching A2A APIs that license content for AI training and RAG:

- **NYT API**: $0.001/article (RAG) or $0.10/article (fine-tuning)
- **PubMed API**: free for non-commercial, $0.005/article for commercial
- **arXiv API**: free for RAG, $0.05/paper for fine-tuning

The Caddy X402 plugin (Section 7) is the open-source implementation of this pattern for any publisher.

---

## 11. Building Your Own A2A Marketplace

### 11.1 The 5 components

| Component | Off-the-shelf | Build-it-yourself |
|-----------|---------------|-------------------|
| **Wallet layer** | Vincent, SmartAgentKit, Kybera, Privy | ERC-4337 + custom policy (medium effort) |
| **Payment protocol** | X402 reference impl | Custom protocol (large effort) |
| **Discovery / directory** | ANP, 8004 indexer | Subgraph + custom UI (small effort) |
| **Marketplace UI** | AgentPayy reference (Next.js) | Custom React app (medium effort) |
| **Facilitator** | facilitator.x402.org, Stripe | Custom Node.js + ethers.js (small effort) |

### 11.2 A 2-week MVP

A typical 2-person team can ship a working A2A marketplace in 2 weeks:

- **Week 1, day 1-2**: Set up the wallet layer (Vincent or SmartAgentKit)
- **Week 1, day 3-4**: Set up the X402 facilitator
- **Week 1, day 5**: Set up the 8004 identity + reputation
- **Week 2, day 1-3**: Build the marketplace UI (fork AgentPayy's reference)
- **Week 2, day 4-5**: Integrate payment + identity + UI; ship to testnet
- **Week 3+**: Audit, optimize, ship to mainnet

### 11.3 A reference architecture

```
                    ┌──────────────────┐
                    │  AgentPayy UI    │  (Next.js)
                    │  (port 3000)     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  AgentPayy API   │  (FastAPI or Hono)
                    │  (port 8000)     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │PostgreSQL│    │  8004    │    │ Facilitator│
      │(metadata)│    │ Registry │    │ (X402)    │
      └──────────┘    └──────────┘    └──────────┘
                                                │
                                                ▼
                                          ┌──────────┐
                                          │  Base    │
                                          │ (USDC)   │
                                          └──────────┘
```

### 11.4 Common pitfalls

1. **Forgetting the receipt format** — receipts must be EIP-712 signed by the seller, not just the buyer
2. **Missing timeouts** — payment proofs must expire; use 5-15 minute windows
3. **No allowlist at the wallet level** — every wallet needs at least a basic allowlist
4. **Skipping HITL for large payments** — humans want to approve >$X payments; don't surprise them
5. **Underestimating gas costs** — paymaster is essential for a smooth buyer experience
6. **No reputation feedback** — every call should generate a feedback event

---

## 12. The Agent Directory — Active Projects (June 2026)

A categorized directory of the most-relevant A2A projects shipping in June 2026.

### 12.1 Standards & protocols

| Project | Type | URL | Status |
|---------|------|-----|--------|
| **X402** | Payment protocol | https://x402.org | v1.0 ratified |
| **8004** | Identity standard | https://eip8004.org | Draft v0.9 |
| **ERC-7715** | Delegation registry | https://eips.ethereum.org/EIPS/eip-7715 | Final |
| **ACP-pay** | Agent payment extension | https://apay.io | Draft |
| **ANP** | Agent discovery | https://agent-network-protocol.org | Multiple impls |
| **MCP-x402** | MCP + X402 | https://mcp-x402.org | Reference impl |

### 12.2 Wallets & custody

| Project | Custody | URL | Status |
|---------|---------|-----|--------|
| **Vincent** | MPC + delegation | https://vincent.io | GA |
| **SmartAgentKit** | Smart contract | https://smartagentkit.com | GA |
| **Kybera** | Smart contract + OSINT | https://kybera.ai | GA |
| **Turnkey Agent** | Custodial | https://turnkey.com | GA |
| **Privy Agent** | Self-custody | https://privy.io | GA |
| **Fireblocks Agent** | MPC | https://fireblocks.com | GA |

### 12.3 Marketplaces

| Project | Type | URL | GMV/day (June 2026) |
|---------|------|-----|---------------------|
| **ClawMarket** | Skill marketplace | https://clawmarket.ai | $180K |
| **MonkePay** | Billing middleware | https://monkepay.com | $90K |
| **AgentPayy** | OSS framework + marketplace | https://agentpayy.io | $25K |
| **Nightmarket** | Curated API marketplace | https://nightmarket.ai | $60K |
| **Moltplace** | Skill barter | https://moltplace.io | $5K |
| **Base Marketplace** | Built-in on Base | https://base.org/marketplace | $30K |

### 12.4 Crawler monetization & content licensing

| Project | Type | URL | Sites deployed |
|---------|------|-----|----------------|
| **Caddy X402 plugin** | Web server plugin | https://github.com/caddyserver/x402 | ~31,000 |
| **Nginx X402 module** | Web server module | https://github.com/nginx/x402-module | ~8,000 |
| **Cloudflare X402** | Edge service | Built-in | ~120,000 |
| **Publisher Direct** | Content licensing API | https://publisher-direct.io | ~600 publishers |

### 12.5 Enterprise / bigco

| Project | Type | URL | Notes |
|---------|------|-----|-------|
| **Stripe Agent Toolkit** | Enterprise toolkit | https://stripe.com/agents | $1.2B processed in 2026 |
| **Coinbase Agent APIs** | X402 facilitator + Base | https://coinbase.com/agents | Dominant L2 |
| **Cloudflare Agents** | Edge-native agents | https://agents.cloudflare.com | 200K+ deployed agents |
| **Visa Intelligent Commerce** | Card-network bridge | https://visa.com/intelligent-commerce | Production in 2 enterprises |
| **Mastercard Agent Pay** | Card-network bridge | https://mastercard.com/agent-pay | Pilot |
| **Google AP2** | Agent Payment Protocol | https://cloud.google.com/ap2 | Limited public preview |

### 12.6 Active YC startups (W25/W26) in A2A

- **Adam (W25)** — Open-source AI CAD with A2A billing
- **Vincent (W24)** — Delegation framework (now GA, above)
- **Skyvern (S23)** — Browser automation with A2A
- **Trellis (W24)** — Healthcare agents with X402
- **Moltbook / Moltplace (S25)** — Agent society
- **Mbd (P25)** — Robotics agents with A2A billing
- **Human Layer (F24)** — HITL integration (cross-reference: `13-Top-Demand/13-Human-in-the-Loop-Systems.md`)

---

## 13. Cross-References

- **`01-Overview.md`** — strategic context
- **`02-Protocols-and-Standards.md`** — the technical protocols used by these marketplaces
- **`03-Wallets-and-Identity.md`** — the wallet layer that every marketplace depends on
- **`05-Future-Outlook.md`** — what's coming next
- **Library-wide:**
  - `14-Case-Studies-Real-World-Projects/` — real implementations of A2A flows
  - `16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md` — pricing models
  - `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL for high-value A2A
  - `20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md` — cost optimization
  - `23-Local-AI-Inference-Self-Hosting/07-Privacy-Sovereignty-with-Local-AI.md` — local-only A2A
  - `21-AI-Regulation-Antitrust/02-EU-AI-Act-Deep-Dive.md` — regulatory context
  - `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` — Chinese A2A ecosystem (Tencent, Alibaba)
  - `17-Research-Frontiers-2026/02-AI-Agents-Research.md` — research on A2A

---

*Next: read `05-Future-Outlook.md` for the regulatory, scaling, and trust questions that will shape the next 18 months of A2A payments.*
