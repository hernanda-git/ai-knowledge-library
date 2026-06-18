# Protocols and Standards for Agent-to-Agent Payments

> **Document Version**: 1.0 — June 18, 2026
> **Scope**: The technical protocols, EIPs, and open standards that make A2A payments work in 2026 — X402, 8004, ERC-7715, ACP-pay, ANP, and the integrations that bind them. Includes code examples, message formats, and a protocol-comparison table.
> **Prerequisites**: read `01-Overview.md` first. Familiarity with HTTP, JSON, EIP-712, and ERC-20 tokens assumed.

---

## Table of Contents

1. [The Protocol Stack in 2026](#1-the-protocol-stack-in-2026)
2. [X402 — The Open Standard for Internet-Native Payments](#2-x402--the-open-standard-for-internet-native-payments)
3. [8004 — On-Chain Agent Identity, Reputation, and Validation](#3-8004--on-chain-agent-identity-reputation-and-validation)
4. [ERC-7715 — Delegation Registry for Scoped Permissions](#4-erc-7715--delegation-registry-for-scoped-permissions)
5. [ACP-pay — The Agent Communication Protocol Payment Extension](#5-acp-pay--the-agent-communication-protocol-payment-extension)
6. [ANP — Agent Network Protocol for Discovery](#6-anp--agent-network-protocol-for-discovery)
7. [MCP + X402 — Tool Calling with Native Payment](#7-mcp--x402--tool-calling-with-native-payment)
8. [Comparison Table — Which Protocol for Which Job](#8-comparison-table--which-protocol-for-which-job)
9. [A Reference Implementation: Agent-to-Agent Image Analysis](#9-a-reference-implementation-agent-to-agent-image-analysis)
10. [ZK Attribute Proofs in X402 Headers](#10-zk-attribute-proofs-in-x402-headers)
11. [Dispute Resolution Standards (Draft)](#11-dispute-resolution-standards-draft)
12. [Interop: Bridging X402 ↔ Visa Intelligent Commerce](#12-interop-bridging-x402--visa-intelligent-commerce)
13. [What's Missing in the 2026 Stack](#13-whats-missing-in-the-2026-stack)
14. [Cross-References](#14-cross-references)

---

## 1. The Protocol Stack in 2026

The A2A payment stack is a layered cake. Each layer has a clear job and a clear protocol.

```
┌──────────────────────────────────────────────────────────────────┐
│  Application layer: marketplaces, agent apps, dashboards         │
│     (ClawMarket, MonkePay, Nightmarket, Stripe Agent Toolkit)    │
├──────────────────────────────────────────────────────────────────┤
│  Agent messaging: ACP, ANP, MCP, JSON-RPC                        │
│     (How agents talk to each other)                              │
├──────────────────────────────────────────────────────────────────┤
│  Payment: X402, ACP-pay, MCP-x402                                │
│     (How money moves)                                            │
├──────────────────────────────────────────────────────────────────┤
│  Identity: 8004, DID, ENS, Vincent attestations                 │
│     (Who is this agent)                                          │
├──────────────────────────────────────────────────────────────────┤
│  Authorization: ERC-7715, Vincent, SmartAgentKit policies        │
│     (What is the agent allowed to do)                            │
├──────────────────────────────────────────────────────────────────┤
│  Settlement: USDC, EURC, PYUSD on Base, Polygon, Tempo           │
│     (Where the money actually lands)                             │
└──────────────────────────────────────────────────────────────────┘
```

This document covers the middle three layers in depth. Identity and authorization are detailed in `03-Wallets-and-Identity.md`.

---

## 2. X402 — The Open Standard for Internet-Native Payments

### 2.1 History and status

X402 was proposed in early 2025 by Coinbase and Cloudflare, building on a 30-year-old IETF note that HTTP 402 was "reserved for future use." After a year of working-group iteration, **X402 v1.0 was ratified in May 2026**. As of June 2026, X402 is deployed at:

- **Cloudflare** — every Worker can opt in to X402 with a single line
- **Coinbase** — Base is the default settlement network
- **Stripe** — Agent Toolkit includes an X402 client
- **Caddy** — official plugin for monetizing crawler access
- **Nginx** — community module (since May 2026)
- **AWS API Gateway** — X402 support in private preview

The X402 working group is chaired by Erik Reppel (Coinbase) and accepts implementations from any vendor under the Apache 2.0 license.

### 2.2 The protocol flow

The simplest X402 exchange is a 4-message dance:

1. **Buyer → Seller:** HTTP request (no payment proof)
2. **Seller → Buyer:** HTTP 402 with payment challenge
3. **Buyer → Seller:** HTTP request with payment proof
4. **Seller → Buyer:** HTTP 200 + receipt

For high-frequency exchanges, buyers can also send a "pre-paid receipt" to skip the 402 round-trip.

### 2.3 Message formats

#### 2.3.1 The 402 challenge

```http
HTTP/1.1 402 Payment Required
Content-Type: application/json
X-Payment-Address: 0xSeller...5678
X-Payment-Amount: 3000          # in micro-USDC (1 USDC = 1,000,000)
X-Payment-Token: USDC
X-Payment-Network: base-mainnet
X-Payment-Facilitator: https://facilitator.x402.org
X-Payment-Receipt-URL: https://summarizer.example/receipts/{nonce}
X-Payment-Expires-At: 2026-06-18T10:30:00Z
X-Payment-Challenge: eyJhbGciOiJFUzI1NiIs...   # EIP-712 typed-data
```

The `X-Payment-Challenge` is an EIP-712 typed-data blob that the buyer's wallet must sign. The signature binds the buyer to this specific (address, amount, token, network, expiry) tuple.

#### 2.3.2 The payment proof

```http
GET /v1/summarize HTTP/1.1
Host: summarizer.example
X-Agent-Identity: did:8004:0xAbCd...1234
X-Payment-Proof: eyJhbGciOiJFUzI1NiIs...   # signed EIP-712 + tx hash
X-Payment-Tx: 0xabc...def
```

The `X-Payment-Proof` is a signed receipt that includes:
- The original challenge
- A transaction hash (on-chain) or a payment-channel update
- The buyer's signature
- Optionally, a ZK attribute proof (see Section 10)

#### 2.3.3 The 200 + receipt

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Payment-Receipt-ID: 0xtx123...
X-Payment-Settled-At: 2026-06-18T10:29:55Z
X-Payment-Block: 12345678
X-Payment-Facilitator-Fee: 50    # in micro-USDC

{"summary": "..."}
```

### 2.4 Reference client in Python

The `x402-client` Python package is the canonical implementation:

```python
from x402 import X402Client, Wallet, USDC

wallet = Wallet.from_private_key("0x...")
client = X402Client(wallet=wallet, max_per_call_usdc=0.10)

# A simple GET — client handles 402 automatically
response = client.get(
    "https://summarizer.example/v1/summarize",
    identity="did:8004:0xAbCd...1234",
    budget_remaining=0.12,
)

# Inspect the receipt
receipt = response.payment_receipt
print(f"Paid {receipt.amount} {receipt.token} in {receipt.latency_ms}ms")
print(f"Tx hash: {receipt.tx_hash}")
print(f"Facilitator fee: {receipt.facilitator_fee} {receipt.token}")
```

Internally, `x402-client` does:

```python
def get(self, url, **headers):
    response = self.session.get(url, headers=headers)
    if response.status_code == 402:
        challenge = parse_challenge(response)
        proof = self.wallet.sign_challenge(challenge)
        headers["X-Payment-Proof"] = proof
        headers["X-Payment-Tx"] = proof.tx_hash
        response = self.session.get(url, headers=headers)
    return Response(response, payment_receipt=parse_receipt(response))
```

### 2.5 Reference server in TypeScript (Cloudflare Workers)

```typescript
import { x402 } from "@x402/server";

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    return x402({
      // Price this endpoint at 0.003 USDC per call
      price: { amount: 3000, token: "USDC", network: "base-mainnet" },
      // Settle to this address
      address: env.SELLER_ADDRESS,
      // Optional: rate-limit by agent identity
      rateLimit: { byIdentity: true, maxPerHour: 1000 },
    })(req, async (req) => {
      // The actual handler runs only after payment is verified
      const body = await req.json();
      const summary = await env.AI.run("@cf/meta/llama-3.3-70b-instruct", {
        prompt: `Summarize: ${body.text}`,
      });
      return Response.json({ summary });
    });
  },
};
```

### 2.6 Facilitator model

The **facilitator** is a critical role: it verifies a payment proof and tells the seller "yes, this buyer has paid." In X402 v1.0, facilitators are trusted but not custodial:

```python
# Facilitator pseudo-code
def verify_payment(proof, challenge):
    # 1. Check the signature matches the challenge
    if not verify_eip712(proof.signature, challenge):
        return VerificationResult.INVALID_SIGNATURE

    # 2. Check the on-chain transaction
    tx = base_web3.eth.get_transaction(proof.tx_hash)
    if tx.to != challenge.payment_address:
        return VerificationResult.WRONG_RECIPIENT
    if tx.value < challenge.amount:
        return VerificationResult.INSUFFICIENT_AMOUNT
    if tx.block_number > current_block - 100:
        return VerificationResult.NOT_YET_FINAL

    # 3. Optional: charge a facilitator fee
    if proof.includes_facilitator_fee:
        charge_facilitator_fee(proof.tx_hash, FACILITATOR_FEE)

    return VerificationResult.VALID
```

The leading facilitator as of June 2026 is `facilitator.x402.org` (Coinbase + Cloudflare co-operated). Alternative facilitators include Stripe's, Tempo's, and several independent operators.

### 2.7 Payment channels (off-chain, then settle)

For high-frequency buyers, X402 supports **payment channels**: the buyer and seller open a channel with a small on-chain deposit, then exchange signed off-chain updates, and only settle the net balance on-chain at the end. This drops per-call cost to near-zero:

```python
# Open a channel
channel = client.open_channel(
    seller="0xSeller",
    deposit_usdc=10.0,  # $10 pre-funded
    expires_in_hours=24,
)

# Make 1000 calls; each signs a channel update
for i in range(1000):
    response = client.get_through_channel(
        channel,
        f"https://api.seller.com/v1/embed?text={i}",
    )
    # Each call is now 0.0001 USDC (channel update only)

# Close the channel: only the net balance is settled on-chain
client.close_channel(channel)
# -> On-chain tx settles 1000 * 0.003 = 3.0 USDC to seller
# -> Buyer paid 0.0001 USDC in gas instead of 1000 * 0.0001 = 0.1 USDC
```

This pattern is critical for high-frequency A2A flows. Most production agents in 2026 use channels for any counterparty with > 100 calls/day.

### 2.8 What's new in X402 v1.0 vs draft 0.9

- Added the **X-Payment-Expires-At** header (replay window)
- Added the **X-Payment-Facilitator** header (multi-facilitator support)
- Specified **payment-channel** message format
- Added **receipt format** with EIP-712 sig from seller
- Added **partial-payment** support for streaming (e.g., pay $0.0001/second)
- Added **ZK attribute proof** extension (see Section 10)

---

## 3. 8004 — On-Chain Agent Identity, Reputation, and Validation

### 3.1 What 8004 is

**ERC-8004** (final expected Q3 2026; draft v0.9 widely deployed) is the proposed standard for an **on-chain agent identity, reputation, and validation registry**. It consists of three on-chain registries:

1. **Identity Registry** — maps `did:8004:0x...` to a public address and metadata
2. **Reputation Registry** — accepts signed feedback events from counterparties
3. **Validation Registry** — third-party validators stake on an agent's behavior

### 3.2 The identity document

An 8004 identity document is a JSON file stored on IPFS or Arweave, pinned by a content hash on-chain:

```json
{
  "id": "did:8004:0xAbCd...1234",
  "name": "ResearchAgent v1.2",
  "description": "Multi-source research synthesis for finance and policy",
  "agent_card": "https://research-agent.example/card.json",
  "services": [
    {
      "type": "x402",
      "endpoint": "https://api.research-agent.example/v1/",
      "pricing": "0.003 USDC per /summarize call"
    },
    {
      "type": "x402",
      "endpoint": "https://api.research-agent.example/v1/",
      "pricing": "0.04 USDC per /synthesize call"
    }
  ],
  "wallet": "0xAbCd...1234",
  "owner": "did:eth:0xOwner...5678",
  "validators": ["did:8004:0xValidatorA", "did:8004:0xValidatorB"],
  "registration_tx": "0xReg...1234",
  "stake": "100 USDC"
}
```

### 3.3 The reputation event

After every interaction, the counterparty submits a signed reputation event:

```json
{
  "agent": "did:8004:0xAbCd...1234",
  "reviewer": "did:8004:0xReviewer...5678",
  "interaction_tx": "0xInt...5678",
  "rating": 5,
  "tags": ["accuracy:5", "latency:4", "price:5"],
  "comment_hash": "0xCom...9abc",  // IPFS-pinned comment
  "timestamp": 1718705400,
  "signature": "0xSig...def"
}
```

The 8004 ReputationRegistry contract verifies the signature and the interaction transaction (must be a real X402 payment), then updates the agent's aggregate score.

### 3.4 Validation and staking

Validators are independent agents that **stake** USDC on an agent's behavior. If the validator's assessment is later proven wrong (via a dispute contract), the validator is **slashed**. This is the trust-minimized version of "rely on reviews":

```python
# A validator stakes 50 USDC on ResearchAgent's quality
validation_tx = validator.stake(
    target="did:8004:0xAbCd...1234",
    amount=50_000_000,  # 50 USDC
    claim="accuracy >= 90% on finance Q&A",
    duration_days=30,
)

# If the agent is later proven to be inaccurate,
# the validator loses the stake (slashed)
```

### 3.5 Reference client

```python
from eip8004 import AgentIdentity, ReputationRegistry

# Register a new agent
identity = AgentIdentity(
    name="ResearchAgent v1.2",
    services=[...],
    stake=100_000_000,  # 100 USDC
)
tx = identity.register(private_key=PRIVATE_KEY)
did = identity.did  # did:8004:0xAbCd...1234

# Submit feedback after an interaction
rep = ReputationRegistry(base_web3)
rep.submit_feedback(
    agent=did,
    interaction_tx="0xInt...5678",
    rating=5,
    tags={"accuracy": 5, "latency": 4},
)

# Read an agent's score
score = rep.get_score(did)
print(f"{did} has score {score.mean} from {score.count} reviews")
```

### 3.6 Why 8004 matters for A2A

Without 8004, an A2A buyer has no portable way to know "is this seller any good?" With 8004:

- **Discovery** is on-chain; no central registry can delist your agent
- **Reputation** is portable; an agent's score follows it across marketplaces
- **Validation** is stake-weighted; you can trust high-stake validators more than low-stake ones
- **Sybil resistance** comes from stake + interaction-tx verification

---

## 4. ERC-7715 — Delegation Registry for Scoped Permissions

### 4.1 The problem

An agent's private key is dangerous. If compromised, the attacker can drain the wallet. The agent needs a way to **delegate scoped, time-bound, revocable signing rights** to other agents (or even to itself, in a multi-process setup).

### 4.2 The standard

**ERC-7715** (merged June 2026) is a smart-contract registry where:

- An **owner** (human or another contract) can register a **delegate**
- The delegate can sign on behalf of the owner, **scoped** to a specific (chain, contract, method, parameters, value, time window)
- The owner can **revoke** at any time
- A **caveat enforcer** is a smart contract that adds extra conditions (e.g., "value ≤ 1 USDC", "only between 9am-5pm")

### 4.3 Example

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@erc7715/DelegationRegistry.sol";
import "@erc7715/examples/SpecificActionCaveatEnforcer.sol";
import "@erc7715/examples/ValueLimitCaveatEnforcer.sol";

contract AgentWallet {
    DelegationRegistry public registry;
    SpecificActionCaveatEnforcer public specificAction;
    ValueLimitCaveatEnforcer public valueLimit;

    constructor(address _registry) {
        registry = DelegationRegistry(_registry);
        specificAction = new SpecificActionCaveatEnforcer();
        valueLimit = new ValueLimitCaveatEnforcer(1_000_000);  // 1 USDC max
    }

    // The HUMAN grants a delegation to an AGENT
    function grantDelegation(address agent) external {
        // Caveat 1: only X402 payment methods on Base
        bytes memory c1 = specificAction.encodeCaveat(
            chainId: 8453,
            contractAddr: 0xUSDC_BASE,
            selector: bytes4(keccak256("transfer(address,uint256)"))
        );
        // Caveat 2: max 1 USDC per tx
        bytes memory c2 = valueLimit.encodeCaveat(1_000_000);
        // Caveat 3: only valid for 7 days
        uint48 validUntil = uint48(block.timestamp + 7 days);

        registry.delegate(agent, [c1, c2], validUntil);
    }

    // The AGENT calls USDC.transfer — it works because of the delegation
    function paySeller(address usdc, address seller, uint256 amount) external {
        // Will revert if amount > 1 USDC, wrong selector, or expired
        IERC20(usdc).transfer(seller, amount);
    }
}
```

### 4.4 Why ERC-7715 is critical for agent safety

- **Compromise containment**: if the agent's key is stolen, the thief can only spend up to 1 USDC per call
- **Auto-expiry**: delegations expire; long-running attacks are bounded
- **Revocability**: human can revoke at any time (single tx)
- **Composability**: any number of caveats can be combined

This is the **single most important standard** for production A2A agents in 2026.

---

## 5. ACP-pay — The Agent Communication Protocol Payment Extension

### 5.1 What ACP is

The **Agent Communication Protocol (ACP)** is a JSON-RPC-based standard for agents to exchange structured messages. It is complementary to MCP (Model Context Protocol, for tool use) and ANP (for discovery). ACP handles the "let's negotiate" layer.

### 5.2 The payment extension

ACP-pay is a proposed extension (final expected Q3 2026) that adds payment-aware methods to ACP:

```json
// ACP message: request a paid service
{
  "jsonrpc": "2.0",
  "id": "req-1234",
  "method": "service.request",
  "params": {
    "service": "summarizer-v1",
    "input": {"text": "..."},
    "budget": {"max_usdc": 0.01, "max_latency_ms": 5000},
    "payment": {
      "rail": "x402",
      "token": "USDC",
      "network": "base-mainnet",
      "pre_auth": true
    }
  }
}

// ACP message: response with payment receipt
{
  "jsonrpc": "2.0",
  "id": "req-1234",
  "result": {
    "output": {"summary": "..."},
    "payment_receipt": {
      "amount": 3000,
      "token": "USDC",
      "tx_hash": "0xabc...def",
      "facilitator": "facilitator.x402.org"
    }
  }
}
```

### 5.3 Why ACP-pay matters

ACP-pay is the glue between agent messaging and agent payment. It lets a multi-agent workflow treat payment as a first-class message field, not an out-of-band HTTP dance. The reference implementation is in TypeScript and Python, with bindings for LangChain, AutoGen, CrewAI, and Hermes.

---

## 6. ANP — Agent Network Protocol for Discovery

### 6.1 What ANP solves

How does Agent A find Agent B? MCP solves "how do I call a known tool" but not "how do I discover an unknown agent." **ANP (Agent Network Protocol)** is a DNS-like system for agent capabilities.

### 6.2 The protocol

```http
# Resolve an agent by name
GET /.well-known/anp/agents?name=ResearchAgent HTTP/1.1
Host: agent-directory.example

# Response
HTTP/1.1 200 OK
Content-Type: application/json

{
  "agents": [
    {
      "did": "did:8004:0xAbCd...1234",
      "name": "ResearchAgent v1.2",
      "endpoint": "https://api.research-agent.example/",
      "capabilities": ["summarize", "synthesize", "fact-check"],
      "pricing": {
        "summarize": "0.003 USDC/call",
        "synthesize": "0.04 USDC/call"
      },
      "reputation": 4.7,
      "validators_stake": 500_000_000  // 500 USDC total
    }
  ]
}
```

### 6.3 Integrations

- **Cloudflare Agents Directory** — built-in ANP support, deployed globally
- **Stripe Agent Directory** — ANP + Stripe Billing integration
- **8004 Indexers** — Covalent, The Graph index 8004 events for ANP queries

---

## 7. MCP + X402 — Tool Calling with Native Payment

### 7.1 The gap

MCP is the de-facto standard for "how an LLM calls a tool." But MCP's tool definition is silent on payment. The agent's tool call returns "402 Payment Required" and the agent has no MCP-native way to handle that.

### 7.2 The MCP-x402 extension

The MCP-x402 working group (Cloudflare, Stripe, Anthropic-affiliated) has proposed an extension to the MCP spec:

```json
// MCP tool definition with X402 payment metadata
{
  "name": "summarize",
  "description": "Summarize a document",
  "inputSchema": {...},
  "x402": {
    "amount": 3000,
    "token": "USDC",
    "network": "base-mainnet",
    "address": "0xSeller...5678"
  }
}
```

### 7.3 Reference client (TypeScript, Anthropic SDK)

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { X402ToolAdapter } from "@mcp-x402/adapter";

const client = new Anthropic();
const x402 = new X402ToolAdapter({ wallet: myAgentWallet });

// Wrap the MCP tools with X402 awareness
const tools = x402.wrapMCPTools(mcpTools);

// Now the LLM can call them, and payments happen automatically
const response = await client.messages.create({
  model: "claude-3-7-sonnet-20250219",
  tools,
  messages: [{ role: "user", content: "Summarize the attached doc" }],
});
```

The adapter handles the full 402 dance, the wallet sign-off, and the budget enforcement.

### 7.4 Why this matters

Until MCP-x402, every MCP server had to re-implement the 402 dance. With the extension, payment is a first-class property of an MCP tool, and the LLM client handles it natively. **This is the most likely path to mass A2A adoption.**

---

## 8. Comparison Table — Which Protocol for Which Job

| Job | Best protocol | Why |
|-----|---------------|-----|
| Pay for an HTTP API call | **X402** | Native to HTTP, simplest integration |
| Pay for a non-HTTP service (gRPC, MQTT) | **ACP-pay** | JSON-RPC works over any transport |
| Discover agents in a directory | **ANP** | DNS-like, no central authority |
| Identify an agent portably | **8004** | On-chain, sybil-resistant, portable |
| Delegate scoped signing rights | **ERC-7715** | Smart-contract-native caveats |
| Tool-calling with payment | **MCP-x402** | First-class in the LLM SDK |
| Multi-agent coordination with payment | **ACP + ACP-pay** | Structured messages, payment metadata |
| Compute reputation across marketplaces | **8004 ReputationRegistry** | On-chain, signed feedback, aggregated |
| Stream payment for long-running work | **X402 partial-payment** | Pay per second, not per call |
| Pay with fiat (not stablecoin) | **Stripe Agent Toolkit + ACP** | Card-on-file with tokenized agent ID |

---

## 9. A Reference Implementation: Agent-to-Agent Image Analysis

Let's wire up a complete end-to-end A2A flow: Agent A (orchestrator) pays Agent B (vision) to analyze an image, with 8004 identity and ERC-7715 delegation.

### 9.1 The buyer setup

```python
# agent_a.py — the orchestrator
import os
from x402 import X402Client, Wallet
from eip8004 import AgentIdentity
from erc7715 import DelegationRegistry

# 1. Load the agent's identity
my_did = AgentIdentity.from_did(os.environ["MY_DID"])

# 2. Load the wallet (with delegation from the human)
wallet = Wallet.from_delegation(
    delegation_id=os.environ["DELEGATION_ID"],
    private_key=os.environ["AGENT_KEY"],  # the delegated key
)

# 3. Look up the vision agent in the ANP directory
vision = anp.lookup("VisionAnalyzer")
print(f"Found {vision.did} with reputation {vision.reputation}")

# 4. Set a per-call budget
client = X402Client(
    wallet=wallet,
    max_per_call_usdc=0.05,
    max_per_day_usdc=2.00,
    require_reputation_min=4.0,
)

# 5. Make the paid call
result = client.post(
    f"{vision.endpoint}/analyze",
    identity=my_did,
    json={"image_url": "https://example.com/cat.jpg"},
)

# 6. Inspect the result and the payment
print(result.json()["labels"])
print(f"Paid {result.payment_receipt.amount} {result.payment_receipt.token}")
```

### 9.2 The seller setup

```python
# agent_b.py — the vision analyzer
from fastapi import FastAPI, Request
from x402 import x402_server, verify_payment
from eip8004 import ReputationRegistry
import httpx

app = FastAPI()

# X402 config: $0.012 per call
SELLER_CONFIG = {
    "address": os.environ["SELLER_ADDRESS"],
    "amount_usdc": 0.012,
    "token": "USDC",
    "network": "base-mainnet",
}

@app.post("/analyze")
@x402_server(SELLER_CONFIG)
async def analyze(request: Request):
    # This handler only runs after payment is verified
    body = await request.json()
    image_url = body["image_url"]

    # Call a vision model
    async with httpx.AsyncClient() as h:
        vision_resp = await h.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_KEY']}"},
            json={
                "model": "gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Label this image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            },
        )
    labels = vision_resp.json()["choices"][0]["message"]["content"]

    # Submit feedback to 8004 ReputationRegistry
    rep = ReputationRegistry(base_web3)
    rep.submit_feedback_async(
        agent=request.headers["X-Agent-Identity"],
        interaction_tx=request.headers["X-Payment-Tx"],
        rating=5,  # we mark the buyer as a good payer (no chargebacks)
    )

    return {"labels": labels}
```

### 9.3 What happens at runtime

1. Agent A sends `POST /analyze` to Agent B (no payment proof)
2. Agent B returns 402 with payment challenge
3. Agent A's wallet signs the challenge (subject to ERC-7715 caveats: max $0.05/call, max $2/day, require reputation ≥ 4.0)
4. Agent A retries with `X-Payment-Proof` header
5. Agent B's `x402_server` decorator verifies the proof via the facilitator
6. The handler runs, calls OpenAI, gets labels
7. Agent B returns 200 with the labels and the receipt
8. Agent B submits feedback to the 8004 registry
9. Agent A's budget ledger decrements by 0.012 USDC

Total elapsed: ~2.5 seconds. Cost: $0.012 + $0.0001 gas. That's the A2A economy in one transaction.

---

## 10. ZK Attribute Proofs in X402 Headers

### 10.1 The privacy problem

A public blockchain is a public record. If Agent A's 8004 DID is linked to its wallet, anyone can see Agent A's full transaction history. For some agents (e.g., a corporate trading bot), that's a competitive leak.

### 10.2 The solution: ZK attribute proofs

X402 v1.0 supports an optional `X-Payment-ZKProof` header. The buyer's wallet generates a zero-knowledge proof of an attribute (e.g., "I have ≥ 4.5 reputation in the 8004 registry") *without* revealing the wallet address.

```http
GET /v1/summarize HTTP/1.1
Host: summarizer.example
X-Agent-Identity: did:8004:0xAbCd...1234
X-Payment-Proof: eyJhbGciOiJFUzI1NiIs...
X-Payment-Tx: 0xabc...def
X-Payment-ZKProof: {
  "claim": "reputation >= 4.5 in 8004 registry at did:8004:0xAbCd...1234",
  "proof": "0xZK...1234",
  "verifier": "https://verifier.zk-8004.org"
}
```

The seller can verify the ZK proof without learning the buyer's actual reputation, address, or transaction history.

### 10.3 What's in production in June 2026

- **SnarkyJS (Aleo)** — the most common ZK circuit library for 8004 proofs
- **Risc Zero** — used for heavier ZK attribute claims
- **Polygon Miden** — the leading L2 for ZK agent proofs
- **Verifier directory** at `verifier.zk-8004.org` (maintained by the 8004 working group)

The "We put ZK attribute proofs inside X402 payment headers" Show HN (May 2026) is the canonical reference implementation.

---

## 11. Dispute Resolution Standards (Draft)

### 11.1 The problem

A buyer pays for an API call. The seller returns garbage. What's the recourse? In a no-chargeback world, the buyer's only option is to leave a bad 8004 review. The seller has no incentive to refund.

### 11.2 The proposed design (ERC-7715 + escrow)

A working group led by Stripe and Coinbase is drafting an extension (expected Q4 2026) that adds **escrowed A2A payments**:

```
1. Buyer deposits 0.012 USDC into an escrow contract
2. Seller performs the service
3. If buyer accepts → 0.012 USDC releases to seller
4. If buyer disputes → 3rd-party arbitrator (or 8004 validator) reviews
5. Arbitrator decides → funds release to either side
```

### 11.3 The arbitrator

The arbitrator is itself an 8004-registered agent with staked USDC. If the arbitrator rules badly, it gets slashed. This is the same trust-minimized pattern as the validation registry, applied to dispute resolution.

### 11.4 Status as of June 2026

- Draft spec circulating since March 2026
- Two reference implementations (Stripe + community)
- No production deployments yet
- Expected final by Q4 2026

---

## 12. Interop: Bridging X402 ↔ Visa Intelligent Commerce

### 12.1 Why bridge

Most enterprises already have a card-on-file relationship with Visa. They want to give their agents a card-like payment rail, but with the programmatic semantics of X402.

### 12.2 The bridge

Visa's "Intelligent Commerce" (announced October 2025) is a tokenized-agent-payment system built on the Visa network. The X402 working group has drafted an interop spec that lets an X402 client transparently use a Visa card as the funding source:

```http
# Standard X402 402 challenge
X-Payment-Address: 0xVisaBridge...5678
X-Payment-Amount: 3000
X-Payment-Token: USDC
X-Payment-Network: base-mainnet

# Plus an interop hint
X-Payment-Accepted-Rails: x402, visa-intelligent-commerce
```

If the buyer's wallet supports Visa Intelligent Commerce, it can pay via card-network (and the seller still gets USDC via the bridge). The seller's X402 code doesn't change.

### 12.3 Status

- Draft spec: April 2026
- Visa reference impl: May 2026
- Stripe reference impl: June 2026
- Production deployments: 2 (one major retailer, one airline)
- Expected mass adoption: Q1 2027

---

## 13. What's Missing in the 2026 Stack

Even after this much progress, several critical pieces are still missing:

| Gap | Why it matters | Who's working on it |
|-----|----------------|---------------------|
| **Universal identity portability** | An agent's 8004 DID should work across all chains and all marketplaces | Cross-chain 8004 working group |
| **Privacy-preserving reputation** | Current 8004 reviews are public; need ZK reputation by default | ZK-8004 group |
| **Standardized dispute resolution** | No clear recourse for bad service | Escrow / arbitrator working group |
| **Fraud detection at the facilitator level** | Need ML models to flag suspicious payment patterns | Coinbase, Stripe, Cloudflare |
| **Multi-currency / FX** | An agent in Japan wants to pay in JPY, seller wants USDC | Tempo, LayerZero, Wormhole |
| **Agent insurance** | Coverage for stolen keys, smart-contract bugs | Nexus Mutual, InsurAce |
| **Standardized budget delegation** | "Give this agent a $5 weekly budget" should be a one-liner | ERC-7715 working group |
| **Compliance APIs** | KYC/AML for >$X transactions, OFAC screening | Chainalysis, TRM Labs |

These are the open problems. The next 18 months will see several of them get solved — most likely by the same actors building the standards above.

---

## 14. Cross-References

- **`01-Overview.md`** — the strategic and economic context for these protocols
- **`03-Wallets-and-Identity.md`** — deep dive on wallets, custody, and identity
- **`04-Marketplaces-and-Use-Cases.md`** — concrete examples using these protocols
- **`05-Future-Outlook.md`** — what's coming next
- **Library-wide:**
  - `03-Agents/04-Protocols-MCP-ACP.md` — the original MCP and ACP specs
  - `18-Agent-Security-and-Trust/05-Agent-Authentication-and-Identity.md` — agent identity foundations
  - `20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md` — tracking A2A payment costs
  - `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL for high-value A2A payments
  - `21-AI-Regulation-Antitrust/02-EU-AI-Act-Deep-Dive.md` — Article 14 and A2A
  - `16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md` — pricing models for A2A

---

*Next: read `03-Wallets-and-Identity.md` to understand the wallet, custody, and identity layers that all these protocols depend on.*
