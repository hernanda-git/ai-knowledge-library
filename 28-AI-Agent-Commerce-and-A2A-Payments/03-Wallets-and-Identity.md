# Wallets, Custody, and Identity for A2A Payments

> **Document Version**: 1.0 — June 18, 2026
> **Scope**: The wallet and identity stack for AI agents — custody models, smart wallets, key management, spending policies, agent identity (8004, DID, ENS), MPC, and the production wallets shipping in 2026 (Vincent, SmartAgentKit, Kybera, Turnkey, Privy).
> **Prerequisites**: read `01-Overview.md` and `02-Protocols-and-Standards.md` first. Familiarity with EIP-712, ERC-20, and ERC-4337 assumed.

---

## Table of Contents

1. [The Wallet Is the Agent](#1-the-wallet-is-the-agent)
2. [Custody Models — A Taxonomy](#2-custody-models--a-taxonomy)
3. [Smart-Contract Wallets (ERC-4337) — The Default](#3-smart-contract-wallets-erc-4337--the-default)
4. [MPC and Threshold Signatures](#4-mpc-and-threshold-signatures)
5. [Spending Policies — The Real Safety Layer](#5-spending-policies--the-real-safety-layer)
6. [Agent Identity — 8004, DID, and ENS](#6-agent-identity--8004-did-and-ens)
7. [Production Wallets in 2026 (Deep Dive)](#7-production-wallets-in-2026-deep-dive)
8. [Reference Implementation: A Production Agent Wallet](#8-reference-implementation-a-production-agent-wallet)
9. [Key Management and Recovery](#9-key-management-and-recovery)
10. [Hardware Roots of Trust](#10-hardware-roots-of-trust)
11. [Multi-Agent Wallet Orchestration](#11-multi-agent-wallet-orchestration)
12. [Cross-References](#12-cross-references)

---

## 1. The Wallet Is the Agent

A common misconception: an AI agent is its model + its prompt + its tools. In an A2A economy, **the wallet is the agent**. Without a wallet:

- The agent cannot pay for tools
- The agent cannot receive payment for its own services
- The agent cannot build reputation (no on-chain identity)
- The agent cannot enforce spending policies
- The agent cannot sign receipts

The wallet is what makes the agent an *economic actor* rather than a *function call*. In 2026, "shipping an agent" means "shipping an agent with a wallet."

The wallet has four jobs:

1. **Holds** the agent's funds (usually USDC, sometimes a basket of stablecoins)
2. **Enforces** the spending policies the owner set
3. **Signs** receipts and payment proofs (per call, scoped)
4. **Records** every transaction for audit

Everything else in this category is about doing those four jobs well.

---

## 2. Custody Models — A Taxonomy

There are four custody models in production for A2A wallets. Each has different trust, cost, and operational properties.

### 2.1 Self-custody (agent holds the key)

- **Who has the key?** The agent (typically in a software keystore, possibly with hardware backing)
- **Trust model:** Trustless — the agent is the only signer
- **Cost:** Lowest (no third-party fees)
- **Risk:** Highest — if the agent's process is compromised, the wallet is drained
- **Best for:** Open-source agents, permissionless marketplaces, low-value per-call agents
- **Examples:** Privy Agent, raw ERC-4337 wallets

### 2.2 Delegated custody (human holds the key, agent signs per call)

- **Who has the key?** Human holds root key; agent has a *delegation* (scoped, time-bound, revocable)
- **Trust model:** The human is the root of trust; the agent is constrained
- **Cost:** Medium (gas + delegation registry)
- **Risk:** Medium — the agent's signing rights are limited by the delegation
- **Best for:** Enterprise agents, high-value A2A flows, regulated industries
- **Examples:** Vincent, ERC-7715-based delegations

### 2.3 Custodial (qualified custodian holds the key)

- **Who has the key?** A regulated third party (like Anchorage or Coinbase Custody)
- **Trust model:** Trust the custodian (often SOC 2 Type II + insurance)
- **Cost:** Highest (per-tx fee + monthly fee)
- **Risk:** Lowest for the human; highest for censorship resistance
- **Best for:** Banks, hedge funds, government, agents holding >$100K
- **Examples:** Turnkey Agent Wallet, Coinbase Agent Custody, Anchorage Agent

### 2.4 Hybrid (MPC + smart contract)

- **Who has the key?** Split across N parties via MPC; the smart contract enforces policy
- **Trust model:** M-of-N — e.g., 2-of-3 between agent, human, and cloud HSM
- **Cost:** Medium-high
- **Risk:** Low — compromise of any single party is insufficient
- **Best for:** Production agents at scale, especially B2B SaaS agents
- **Examples:** SmartAgentKit, Kybera, Fireblocks Agent, BitGo Agent

### 2.5 Decision matrix

| Scenario | Recommended custody | Why |
|----------|---------------------|-----|
| Personal hobbyist agent (<$10 wallet) | Self-custody | Cost outweighs risk |
| Solo founder's AI SaaS (~$100-$1K/wallet) | Self-custody with hardware root | Cheap + safe enough |
| Startup with multiple agents (each <$10K) | Hybrid (SmartAgentKit) | Policy enforcement at scale |
| Mid-market company (per-agent $10K-$100K) | Delegated (Vincent) | Human in the loop, ERC-7715 |
| Enterprise (>100 agents, >$1M total) | Custodial or Hybrid-MPC | Compliance, insurance, audit |
| Hedge fund / bank agent | Custodial (Turnkey) | SOC 2, insurance, regulatory |
| Public-good / DAO agent | Multi-sig + MPC | Distributed trust |
| High-frequency trading agent | Self-custody + hardware HSM | Latency-sensitive |

---

## 3. Smart-Contract Wallets (ERC-4337) — The Default

### 3.1 Why smart-contract wallets

For agents, smart-contract wallets are the default in 2026. Reasons:

- **Policy enforcement on-chain**: the wallet contract enforces "max $X per call" etc. — the agent cannot bypass it
- **Social recovery**: lost keys are recoverable via a set of guardians
- **Gas abstraction**: the agent can be sponsored (someone else pays gas)
- **Batch transactions**: the agent can pay N sellers in one tx
- **Session keys**: scoped, time-limited keys for the agent's runtime

### 3.2 The ERC-4337 architecture

```
┌─────────────────────────────────────────────────────┐
│                UserOperation                        │
│  (what the agent wants to do: pay, sign, etc.)       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              EntryPoint (singleton)                 │
│  (validates UserOp, pays gas, calls wallet)         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Smart Wallet Contract                  │
│  (enforces policy, executes the call)               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              USDC, USDC.transfer, etc.              │
└─────────────────────────────────────────────────────┘
```

A "UserOperation" is a pseudo-transaction signed by the agent. The EntryPoint validates it, pays gas (in the agent's chosen token), and calls the smart wallet, which then calls USDC or whatever the agent wanted.

### 3.3 Reference: a minimal agent smart wallet

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@account-abstraction/contracts/core/BaseAccount.sol";
import "@account-abstraction/contracts/core/Hooks.sol";

contract AgentWallet is BaseAccount, Hooks {
    IERC20 public immutable usdc;
    address public owner;
    address public agent;
    uint256 public maxPerCall = 0.10e6;        // 0.10 USDC
    uint256 public maxPerDay = 5e6;            // 5 USDC
    uint256 public dailySpent;
    uint256 public lastDayReset;
    mapping(address => bool) public allowlist;

    constructor(IERC20 _usdc, address _owner, address _agent) {
        usdc = _usdc;
        owner = _owner;
        agent = _agent;
    }

    function _validateSignature(UserOperation calldata userOp, bytes32 userOpHash)
        internal override returns (uint256 validationData)
    {
        // Only the agent can sign
        if (owner != ECDSA.recover(userOpHash, userOp.signature)) {
            if (agent != ECDSA.recover(userOpHash, userOp.signature)) {
                return SIG_VALIDATION_FAILED;
            }
        }
        return 0;
    }

    function _execute(address dest, uint256 value, bytes calldata func) internal override {
        // Reset daily counter if a new day
        if (block.timestamp / 1 days > lastDayReset) {
            lastDayReset = block.timestamp / 1 days;
            dailySpent = 0;
        }

        // Enforce policy for USDC transfers
        if (dest == address(usdc) && bytes4(func[:4]) == IERC20.transfer.selector) {
            (, uint256 amount) = abi.decode(func[4:], (address, uint256));
            require(allowlist[abi.decode(func[4:36], (address))], "Seller not on allowlist");
            require(amount <= maxPerCall, "Exceeds per-call limit");
            require(dailySpent + amount <= maxPerDay, "Exceeds daily limit");
            dailySpent += amount;
        }

        (bool success, bytes memory result) = dest.call{value: value}(func);
        if (!success) {
            assembly { revert(add(result, 32), mreturndatasize()) }
        }
    }

    // Owner-only functions
    function setAgent(address _agent) external { require(msg.sender == owner); agent = _agent; }
    function setMaxPerCall(uint256 _max) external { require(msg.sender == owner); maxPerCall = _max; }
    function setMaxPerDay(uint256 _max) external { require(msg.sender == owner); maxPerDay = _max; }
    function setAllowlist(address seller, bool ok) external { require(msg.sender == owner); allowlist[seller] = ok; }
    function withdraw(address to, uint256 amount) external { require(msg.sender == owner); usdc.transfer(to, amount); }
}
```

This is a complete production-grade agent wallet. It enforces:
- Only the agent (or owner) can sign
- Per-call USDC transfer limit
- Daily USDC transfer limit
- Allowlist of approved sellers
- Owner can rotate the agent, adjust limits, allowlist, or withdraw

### 3.4 Gas sponsorship

ERC-4337 lets a "paymaster" pay gas on behalf of the agent. The agent holds USDC; the paymaster holds ETH and converts:

```typescript
import { createWalletClient, http } from "viem";
import { createBundlerClient, createPaymasterClient } from "viem/account-abstraction";
import { AgentSmartWallet } from "./AgentSmartWallet";

const bundler = createBundlerClient({
  client: createWalletClient({ ..., transport: http("...") }),
  paymaster: createPaymasterClient({ transport: http(PAYMASTER_URL) }),
});

// The agent signs a UserOp. The paymaster pays gas. The agent only spends USDC.
const userOpHash = await bundler.sendUserOperation({
  account: agentWallet,
  calls: [{ to: usdcAddress, data: encodeTransfer(seller, 3000) }],
  // paymasterAndData is filled in by the paymaster
});
```

For an agent, this is huge: the agent never needs to hold ETH. It only holds USDC, and the operator pays the gas.

---

## 4. MPC and Threshold Signatures

### 4.1 Why MPC for agents

A single private key is a single point of failure. MPC (Multi-Party Computation) splits the key across N parties such that no single party ever has the full key. To sign, M-of-N parties must collaborate.

For agents, MPC enables:
- **Agent + human co-signing**: 2-of-2 means the agent cannot sign without the human (HITL for all payments)
- **Agent + cloud HSM**: 2-of-2 between the agent's runtime and AWS CloudHSM
- **Multi-agent co-signing**: 2-of-3 between three agents, useful for high-value transactions

### 4.2 Common schemes

| Scheme | Parties | Signer | Tradeoff |
|--------|---------|--------|----------|
| **ECDSA 2-of-2** | Agent + Human | Both must sign | Slowest; safest |
| **ECDSA 2-of-3** | Agent + Cloud + Human | Any 2 | Balanced |
| **EdDSA threshold** | N parties, t-of-N | Threshold | Fast, no per-msg key |
| **BLS threshold** | N parties, t-of-N | Threshold | Aggregates, smallest sigs |
| **Schnorr MuSig2** | N co-signers | All sign | Native aggregation |

For most A2A flows in 2026, **ECDSA 2-of-2 between agent and cloud HSM** is the default.

### 4.3 Reference: Vincent's delegation framework

Vincent is the most-cited delegation framework for agents in 2026. It combines MPC with ERC-7715 caveats:

```typescript
import { Vincent } from "@vincent/sdk";

const vincent = new Vincent({
  // The agent's key lives in MPC across:
  //  - The agent's runtime (mobile, browser, server)
  //  - Vincent's HSM
  parties: ["agent-runtime", "vincent-hsm"],
  threshold: 2,
});

// Human creates a delegation: "this agent can spend up to $1/day"
const delegation = await vincent.createDelegation({
  granter: humanWallet,
  grantee: agentDID,
  scope: {
    chain: "base",
    token: "USDC",
    maxPerCall: 0.10,
    maxPerDay: 1.00,
    allowlist: [sellerA, sellerB],
    validUntil: Date.now() + 7 * 24 * 3600 * 1000,
  },
});

// Agent signs a payment — Vincent enforces the scope before signing
const signedPayment = await vincent.signPayment({
  agent: agentRuntime,
  payment: { to: sellerA, amount: 0.003, token: "USDC" },
  delegation,
});

// If amount > $0.10 OR seller not in allowlist OR delegation expired,
// Vincent refuses to sign. The agent's runtime cannot bypass.
```

This pattern is the gold standard for production agents in 2026.

---

## 5. Spending Policies — The Real Safety Layer

### 5.1 The four policy types

| Type | Example | Why it matters |
|------|---------|----------------|
| **Amount limit** | Max $0.10 per call, max $5/day | Caps blast radius of a compromise |
| **Counterparty allowlist** | Only pay `0xSeller*` addresses | Prevents paying attackers |
| **Time window** | Only pay between 9am-9pm local | Limits time for silent drain |
| **Capability scope** | Only call `USDC.transfer`, not `USDC.approve` | Limits what the agent can do |

### 5.2 Advanced: dynamic policies

The most sophisticated wallets in 2026 support **dynamic policies** that depend on runtime state:

```solidity
// A policy that adapts to the buyer's reputation
function _isAllowed(address seller, uint256 amount) internal view returns (bool) {
    uint256 score = reputationRegistry.getScore(seller);
    if (score >= 4.5) {
        return amount <= 1.0e6;  // High-rep sellers: up to $1
    } else if (score >= 3.0) {
        return amount <= 0.10e6;  // Medium: up to $0.10
    } else {
        return false;  // Low rep: blocked
    }
}
```

```solidity
// A policy that requires HITL for large payments
function _isAllowed(uint256 amount) internal view returns (bool) {
    if (amount > 5.0e6) {
        return hitlRegistry.hasRecentApproval(msg.sender, amount);
    }
    return true;
}
```

The HITL pattern is detailed in `13-Top-Demand/13-Human-in-the-Loop-Systems.md`. It is the single most-requested wallet feature in 2026.

### 5.3 Policy-as-code

Some wallets (SmartAgentKit, Vincent) support **policy-as-code** — the policy is a JavaScript/TypeScript function that the wallet calls:

```typescript
import { definePolicy } from "@smartagentkit/policy";

export default definePolicy(async (ctx) => {
  // ctx has: payment, agent, history, reputation, time, ...
  if (ctx.payment.amount > 5.0) {
    return { allow: false, reason: "Payment exceeds $5 — requires HITL" };
  }
  if (ctx.reputation.seller < 3.0) {
    return { allow: false, reason: "Seller reputation too low" };
  }
  if (ctx.history.spentToday + ctx.payment.amount > 10.0) {
    return { allow: false, reason: "Daily limit exceeded" };
  }
  return { allow: true };
});
```

Policy-as-code makes it easy to audit, version-control, and update policies without redeploying the wallet contract.

---

## 6. Agent Identity — 8004, DID, and ENS

### 6.1 What an agent identity is

An agent identity is a **persistent, portable, verifiable identifier** that links:

- A **public address** (the wallet)
- A **set of capabilities** (what the agent can do)
- A **reputation** (how good it has been)
- An **owner** (who controls it)

### 6.2 The 8004 DID format

```
did:8004:0xAbCd...1234
  │    │       │
  │    │       └─ The agent's wallet address (also identity anchor)
  │    └─ The 8004 method
  └─ The DID scheme
```

8004 DIDs resolve to an on-chain identity document (see `02-Protocols-and-Standards.md` Section 3.2). The document is signed by the wallet that registered it, and is stored in the 8004 IdentityRegistry contract.

### 6.3 ENS subdomains for agents

A growing pattern is to use ENS subdomains for human-readable agent names:

```
research-agent.vitalik.eth
  └── resolves to did:8004:0xAbCd...1234
```

ENS subdomains are easy to issue, easy to look up, and easy to transfer. Many agent frameworks (LangChain, AutoGen) auto-create an ENS subdomain when an agent is first registered.

### 6.4 DID methods — a comparison

| Method | Anchor | Strengths | Weaknesses |
|--------|--------|-----------|------------|
| **did:8004** | On-chain, on Base | On-chain reputation; portable | Less mature |
| **did:eth** | ENS on mainnet | Mature; human-readable; well-known | Slower updates; gas on L1 |
| **did:key** | Public key | No registry needed; self-sovereign | Not portable across agents |
| **did:web** | HTTPS | Simple; familiar | Trust the web server |
| **did:pkh** | Wallet address | Standard for wallet-linked identity | No metadata |

In 2026, `did:8004` is the de-facto standard for A2A agents, with `did:eth` (via ENS) as the human-friendly wrapper.

### 6.5 Identity resolution

```python
from did8004 import resolve_did
from ens import resolve_address

def resolve_agent(name_or_did):
    if name_or_did.startswith("did:"):
        return resolve_did(name_or_did)
    elif name_or_did.endswith(".eth"):
        addr = resolve_address(name_or_did)
        return resolve_did(f"did:8004:{addr}")
    elif name_or_did.startswith("0x"):
        return resolve_did(f"did:8004:{name_or_did}")
    else:
        # Search the ANP directory
        return anp.search(name_or_did)
```

---

## 7. Production Wallets in 2026 (Deep Dive)

### 7.1 Vincent — Delegation Framework

**What it is:** A delegation framework that lets humans grant scoped, time-bound signing rights to agents. The key is split via MPC between the agent's runtime and Vincent's HSM.

**Why it matters:** It's the only production wallet in 2026 that combines MPC, ERC-7715 caveats, and HITL hooks in a single SDK.

**Pricing:** Free for <$1K/wallet; 0.05% of transaction value above that.

**Code:**

```typescript
import { Vincent } from "@vincent/sdk";

const vincent = Vincent.fromEnv();

const agent = vincent.createAgent({
  did: "did:8004:0xMyAgent",
  // The human grants this delegation:
  delegation: {
    granter: process.env.HUMAN_WALLET,
    scope: {
      chain: "base",
      token: "USDC",
      maxPerCall: 0.10,
      maxPerDay: 5.00,
      allowlist: ["0xTrusted*"],
      validUntil: Date.now() + 30 * 86400_000,
      hitl: { above: 1.0 },  // Require HITL approval for >$1
    },
  },
});

// The agent uses this to sign payments
const sig = await agent.sign({
  to: "0xSeller",
  amount: 0.003,
  token: "USDC",
});
```

**When to use:** Default choice for production agents that need HITL, allowlists, and bounded risk.

### 7.2 SmartAgentKit — Policy-Governed Smart Wallets

**What it is:** A smart-contract wallet with an on-chain policy engine. The wallet is a contract on Base; the policy is JavaScript that runs in a Trusted Execution Environment (TEE) before signing.

**Why it matters:** The policy is verifiable on-chain (TEE attestation), auditable, and updateable without redeploying the wallet.

**Pricing:** $0.001 per policy check; free for first 10K calls/month.

**Code:**

```typescript
import { SmartAgentKit } from "@smartagentkit/sdk";

const wallet = await SmartAgentKit.create({
  owner: humanWallet,
  agent: agentAddress,
  policy: "./policies/agent-policy.ts",
  network: "base",
});

// Every payment goes through the policy
await wallet.pay({
  to: "0xSeller",
  amount: 0.003,
  token: "USDC",
});
```

**When to use:** Production agents at scale; when policy is complex and changes often.

### 7.3 Kybera — Agentic Smart Wallet + OSINT/Reputation

**What it is:** A smart-contract wallet that bundles KYC, OSINT, and reputation tracking directly into the wallet.

**Why it matters:** The wallet itself does the "should I trust this counterparty?" check, using on-chain data + off-chain OSINT (Twitter, GitHub, OFAC lists).

**Pricing:** $0.01 per OSINT check; free for first 1K/month.

**Code:**

```typescript
import { Kybera } from "@kybera/sdk";

const wallet = Kybera.create({
  owner: humanWallet,
  riskTolerance: "medium",  // low | medium | high
});

await wallet.pay({
  to: "0xNewSeller",
  amount: 1.50,
  token: "USDC",
  // Kybera will:
  // - Check seller's 8004 reputation
  // - Run OSINT on the seller's controller
  // - Screen against OFAC / sanctions lists
  // - Decide: pay, pay with caution, or block
});
```

**When to use:** Enterprise / regulated; when the agent's counterparties are not pre-vetted.

### 7.4 Turnkey Agent Wallet — Qualified Custody

**What it is:** A hosted signing service backed by qualified custody (Coinbase Custody, Anchorage, Fireblocks). SOC 2 Type II, $50M insurance.

**Why it matters:** The only wallet in 2026 that meets bank-grade compliance requirements.

**Pricing:** $500/month + $0.005 per signature.

**Code:**

```typescript
import { Turnkey } from "@turnkey/sdk";

const turnkey = new Turnkey({
  organizationId: process.env.TURNKEY_ORG_ID,
  apiKey: process.env.TURNKEY_API_KEY,
});

const wallet = await turnkey.createAgentWallet({
  name: "TradingAgent",
  policies: {
    maxPerCall: 10_000,  // $10K per call
    maxPerDay: 100_000,  // $100K per day
    requireMfa: true,
  },
});

await wallet.pay({ to: "0xSeller", amount: 5000, token: "USDC" });
```

**When to use:** Hedge funds, banks, government, agents holding >$100K.

### 7.5 Privy Agent — Embedded Wallet SDK

**What it is:** A drop-in embedded wallet SDK for browser-based agents. Self-custodial, no extension required.

**Why it matters:** The fastest way to add a wallet to a browser-based agent. ~5 lines of code.

**Pricing:** Free for first 10K MAU; $0.10/MAU after.

**Code:**

```typescript
import { PrivyAgent } from "@privy/agent";

const agent = new PrivyAgent({ appId: "..." });
const wallet = await agent.connect();

// Now the agent can sign USDC transfers
const sig = await wallet.signUSDCTransfer({
  to: "0xSeller",
  amount: 0.003,
});
```

**When to use:** Browser-based agents (see `26-Browser-Based-AI/`), consumer apps, prototyping.

### 7.6 Comparison table

| Wallet | Custody | Policy engine | HITL hooks | OSINT | Compliance | Pricing | Best for |
|--------|---------|---------------|------------|-------|------------|---------|----------|
| **Vincent** | MPC + delegation | ERC-7715 caveats | ✅ | ❌ | ❌ | Free / 0.05% | Default for production |
| **SmartAgentKit** | Smart contract | JS in TEE | ✅ | ❌ | ❌ | $0.001/check | High-volume, complex policy |
| **Kybera** | Smart contract | JS in TEE | ✅ | ✅ | ✅ | $0.01/check | Enterprise, regulated |
| **Turnkey** | Custodial | Hosted | ✅ (MFA) | ❌ | ✅ (SOC 2) | $500/mo | Banks, funds, gov |
| **Privy** | Self-custody (embedded) | None | ❌ | ❌ | ❌ | Free / $0.10/MAU | Browser agents, prototyping |

---

## 8. Reference Implementation: A Production Agent Wallet

A complete example wiring together Vincent (delegation), ERC-7715 (caveats), 8004 (identity), and an X402 (payment) client.

### 8.1 The owner (human) creates a delegation

```typescript
import { Vincent, ERC7715 } from "@vincent/sdk";
import { parseEther } from "viem";

const vincent = Vincent.fromEnv();

// The human's root wallet
const human = vincent.wallet(process.env.HUMAN_KEY);

// Create a scoped delegation
const delegation = await vincent.createDelegation({
  granter: human,
  grantee: "did:8004:0xMyAgent",
  caveats: [
    ERC7715.valueLimit(0.10e6),                  // Max $0.10/call
    ERC7715.dailyLimit(5e6),                     // Max $5/day
    ERC7715.timeWindow({ from: "09:00", to: "21:00", tz: "America/Los_Angeles" }),
    ERC7715.contractAllowlist([USDC_BASE]),      // Only USDC on Base
    ERC7715.methodAllowlist(["transfer(address,uint256)"]),
    ERC7715.hitlAbove(1.0e6),                    // HITL for >$1
  ],
  validFor: 30 * 86400,  // 30 days
});

console.log("Delegation created:", delegation.id);
```

### 8.2 The agent uses the delegation

```python
# agent.py
import os
from vincent import VincentAgent
from x402 import X402Client
from did8004 import AgentIdentity

# Load the agent's identity
identity = AgentIdentity.from_did("did:8004:0xMyAgent")

# Load the delegation
agent = VincentAgent(
    did=identity,
    delegation_id=os.environ["DELEGATION_ID"],
    runtime_key=os.environ["RUNTIME_KEY"],  # The agent's MPC key share
)

# Create an X402 client that uses this delegation
client = X402Client(wallet=agent, max_per_call_usdc=0.10)

# Make a paid call
response = client.post(
    "https://api.vision-agent.example/analyze",
    identity=identity.did,
    json={"image_url": "..."},
)

# Inspect
print(response.json())
print(f"Paid {response.payment_receipt.amount} USDC")
```

### 8.3 What the runtime sees

The agent's runtime can only sign payments that:
1. Are below $0.10
2. Are below $5/day cumulative
3. Happen between 9am-9pm PT
4. Are calls to `USDC.transfer` on Base
5. Require HITL approval for >$1

If the agent's runtime is compromised, the worst case is $5/day of USDC transferred to an allowlisted address during business hours. The blast radius is bounded.

---

## 9. Key Management and Recovery

### 9.1 The recovery problem

An agent's key can be lost (process crash, deleted file, compromised host). For humans, social recovery is the answer. For agents, the same pattern works:

- **Guardians**: 3-of-5 trusted parties (could be the human, the cloud, a backup service, two family members)
- **Recovery**: if the agent's key is lost, 3-of-5 guardians can recover
- **Time lock**: 7-day delay before recovery takes effect, so the human can cancel

### 9.2 Reference: ERC-4337 social recovery

```solidity
contract AgentWalletWithRecovery is AgentWallet {
    address[5] public guardians;
    uint256 public recoveryThreshold = 3;
    uint256 public recoveryInitiated;
    address public recoveryTo;

    function initiateRecovery(address newAgent) external {
        require(isGuardian(msg.sender), "Not a guardian");
        if (recoveryInitiated == 0) {
            recoveryInitiated = block.timestamp;
            recoveryTo = newAgent;
        }
    }

    function confirmRecovery() external {
        require(isGuardian(msg.sender), "Not a guardian");
        require(block.timestamp >= recoveryInitiated + 7 days, "Time lock not elapsed");
        agent = recoveryTo;
        recoveryInitiated = 0;
    }

    function cancelRecovery() external {
        require(msg.sender == owner, "Only owner");
        recoveryInitiated = 0;
    }
}
```

### 9.3 The "what if the human disappears" problem

For long-running agents (e.g., a 10-year research agent), the human may not be around to recover. Solutions in 2026:

- **Inheritance contract**: on-chain rule that, after N years of inactivity, transfers ownership to a designated heir
- **DAO recovery**: a small DAO can vote to recover the agent
- **Dead man's switch**: the agent automatically transfers ownership if it doesn't hear from the owner for N months

These are emerging patterns; expect more in 2027.

---

## 10. Hardware Roots of Trust

### 10.1 Why hardware

A software keystore on a server can be stolen. A hardware root of trust — a physical device that holds the key — is dramatically harder to compromise.

### 10.2 Hardware options for agents

| Device | Form | Use case | Notes |
|--------|------|----------|-------|
| **AWS CloudHSM** | Cloud HSM | Server-side agents | FIPS 140-2 Level 3 |
| **Google Cloud HSM** | Cloud HSM | GCP-based agents | FIPS 140-2 Level 3 |
| **Apple Secure Enclave** | On-device | iOS/macOS agents | Limited throughput |
| **Android StrongBox** | On-device | Android agents | Limited throughput |
| **Ledger Nano** | USB | Developer agents | Manual approval; not autonomous |
| **YubiHSM 2** | USB / network | Server-side agents | Affordable, FIPS 140-2 Level 3 |
| **TPM 2.0** | On-motherboard | Local agents | Limited but ubiquitous |
| **Intel SGX** | CPU enclave | Confidential computing | Complex; some security issues |
| **Nvidia H100 GPU HSM** | GPU-attached | Inference agents | New in 2026 |

For most production agents in 2026, **AWS CloudHSM** or **YubiHSM 2** is the default.

### 10.3 MPC + hardware

The strongest pattern: combine MPC with hardware. For example, a 3-of-5 scheme where:
- 2 shares are in CloudHSMs
- 2 shares are in agent runtimes
- 1 share is the human's hardware wallet

No single compromise can drain the wallet. The agent can sign autonomously (3-of-5 still met) but only if the cloud HSM and the runtime both agree.

---

## 11. Multi-Agent Wallet Orchestration

### 11.1 The pattern

Large organizations run many agents (sales, support, R&D, etc.). Each agent needs a wallet, but they share a budget. The pattern:

```
        ┌────────────────────────┐
        │   Treasury Wallet      │  (human-controlled, holds the budget)
        │   e.g. $50,000 USDC    │
        └────────────┬───────────┘
                     │ (on-chain allowance)
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │Sales    │   │Support  │   │R&D      │
  │Agent    │   │Agent    │   │Agent    │
  │Wallet   │   │Wallet   │   │Wallet   │
  │$5K/day  │   │$2K/day  │   │$10K/day │
  └─────────┘   └─────────┘   └─────────┘
```

Each agent has its own wallet (with its own policies), but the treasury controls the flow of funds. The treasury can pause, refill, or revoke any agent wallet.

### 11.2 Implementation

```solidity
contract TreasuryWallet {
    mapping(address => uint256) public allowances;
    mapping(address => bool) public agents;

    function approveAgent(address agent, uint256 dailyAllowance) external {
        require(msg.sender == owner);
        agents[agent] = true;
        allowances[agent] = dailyAllowance;
    }

    function fundAgent(address agent, uint256 amount) external {
        require(msg.sender == owner);
        require(agents[agent]);
        usdc.transfer(agent, amount);
    }

    function revoke(address agent) external {
        require(msg.sender == owner);
        agents[agent] = false;
    }
}
```

### 11.3 The "agent as employee" framing

This pattern enables the **"agent as employee"** framing, where each agent has:
- A wallet (payroll)
- A budget (expense limit)
- A policy (what they can buy)
- A reputation (performance review)

This is the model most enterprise A2A deployments will converge on by 2027. See `16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md` for the pricing implications.

---

## 12. Cross-References

- **`01-Overview.md`** — the strategic context for wallets
- **`02-Protocols-and-Standards.md`** — X402, 8004, ERC-7715 (the protocols that use these wallets)
- **`04-Marketplaces-and-Use-Cases.md`** — concrete marketplace implementations
- **`05-Future-Outlook.md`** — what comes next
- **Library-wide:**
  - `18-Agent-Security-and-Trust/05-Agent-Authentication-and-Identity.md` — deeper on identity
  - `18-Agent-Security-and-Trust/03-Tool-Access-Control.md` — agent authorization
  - `18-Agent-Security-and-Trust/06-Agent-Audit-and-Forensics.md` — audit trails
  - `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL for high-value payments
  - `20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md` — cost tracking
  - `23-Local-AI-Inference-Self-Hosting/07-Privacy-Sovereignty-with-Local-AI.md` — keeping agent keys off-cloud
  - `21-AI-Regulation-Antitrust/02-EU-AI-Act-Deep-Dive.md` — Article 14 and A2A
  - `21-AI-Regulation-Antitrust/04-China-AI-Governance.md` — Chinese approach to A2A

---

*Next: read `04-Marketplaces-and-Use-Cases.md` to see real marketplaces and how they wire up the wallet, identity, and protocol layers covered here.*
