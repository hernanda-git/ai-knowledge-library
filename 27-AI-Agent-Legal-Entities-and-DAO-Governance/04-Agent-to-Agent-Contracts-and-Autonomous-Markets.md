# 04 — Agent-to-Agent Contracts & Autonomous Markets: Tools and Frameworks

> *In 2024, agents were demos. In 2025, agents were tools. In 2026, agents are economic actors — they sign contracts, they pay for services, they hire each other, they build supply chains, they participate in markets. The market layer of the agent-entity stack is where all of this happens. This document covers the tools, frameworks, and patterns that make agent-to-agent commerce real.*

---

## 1. The Agent Market Stack (Mid-2026)

The agent market stack is the set of protocols, platforms, and frameworks that enable agent-to-agent transactions. The stack has six layers, each of which is covered in detail below.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 6 — Dispute Resolution & Arbitration                             │
│   Kleros, Aragon Court, UMA, human arbitration providers              │
└────────────────────────────┬───────────────────────────────────────────┘
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 5 — Reputation, Discovery & Trust                                │
│   Karma, 8004 (Ethereum), AgentRank, MCP registries, ENS Agent Names   │
└────────────────────────────┬───────────────────────────────────────────┘
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 4 — Agent Marketplaces & Service Catalogs                        │
│   AutoGPT Marketplace, AgentVerse, CrewAI Store, LangChain Hub,        │
│   Hugging Face Agents, MCP server registries                          │
└────────────────────────────┬───────────────────────────────────────────┘
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 3 — Agent-to-Agent Contract Frameworks                           │
│   Smart-contract based SLAs, intent-based agreements,                │
│   bonded performance, on-chain adjudication                           │
└────────────────────────────┬───────────────────────────────────────────┘
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 2 — Communication & Coordination Protocols                      │
│   MCP (Model Context Protocol), A2A (Agent-to-Agent),                  │
│   ANP (Agent Network Protocol), ACP (Agent Communication Protocol)     │
└────────────────────────────┬───────────────────────────────────────────┘
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 1 — Payment & Settlement Rails                                  │
│   x402, L402, account abstraction + stablecoins,                      │
│   Stripe ACP, traditional ACH/wire for fiat                            │
└────────────────────────────────────────────────────────────────────────┘
```

Each layer is covered in detail below, with a working reference implementation in at least one area.

---

## 2. Layer 1 — Payment & Settlement Rails

The payment layer is the foundation of the agent market stack. Without a way to pay, there is no market. The mid-2026 landscape offers several options, each suited to different use cases.

### 2.1 x402 — The HTTP 402 Standard for Machine Payments

x402 is the most widely-adopted payment standard for agent-to-API transactions. It is based on the HTTP 402 "Payment Required" status code (which was reserved in the original HTTP spec in 1997 but largely unused until 2024).

**How x402 works:**

1. Agent calls an API: `GET /v1/some-expensive-call`
2. Server returns 402 with payment challenge:
   ```http
   HTTP/1.1 402 Payment Required
   Content-Type: application/json
   WWW-Authenticate: x402 realm="api.example.com",
                     amount="0.50",
                     currency="USDC",
                     network="base",
                     payTo="0xPaymentAddress",
                     validUntil="2026-12-31T23:59:59Z"
   ```
3. Agent's paymaster executes a stablecoin payment to the specified address
4. Agent retries the call with a payment receipt:
   ```http
   GET /v1/some-expensive-call
   X-Payment-Receipt: {
     "txHash": "0xABC...",
     "amount": "0.50",
     "currency": "USDC",
     "network": "base",
     "timestamp": "2026-06-17T12:34:56Z"
   }
   ```
5. Server verifies the receipt (by checking the transaction on-chain) and returns the data

**The x402 standard** is maintained by the x402 Foundation (founded 2025) and has implementations in most major agent frameworks. Reference implementation:

```typescript
// x402 Client (simplified)
import { createPublicClient, http, parseEther } from "viem";
import { base } from "viem/chains";

const publicClient = createPublicClient({ chain: base, transport: http() });

interface X402Challenge {
  amount: string;
  currency: string;
  network: string;
  payTo: string;
  validUntil: string;
}

async function callWithX402(url: string): Promise<any> {
  // First attempt without payment
  const response = await fetch(url);

  if (response.status !== 402) {
    return response.json();
  }

  // Parse the payment challenge
  const challenge: X402Challenge = parseChallenge(response.headers);

  // Validate the challenge
  if (Date.now() > new Date(challenge.validUntil).getTime()) {
    throw new Error("Payment challenge expired");
  }

  // Pay the challenge (via our paymaster)
  const paymentReceipt = await paymasterPay({
    amount: challenge.amount,
    currency: challenge.currency,
    network: challenge.network,
    to: challenge.payTo,
  });

  // Retry with the payment receipt
  const paidResponse = await fetch(url, {
    headers: {
      "X-Payment-Receipt": JSON.stringify(paymentReceipt),
    },
  });

  if (paidResponse.status === 402) {
    throw new Error("Payment was rejected");
  }

  return paidResponse.json();
}

function parseChallenge(headers: Headers): X402Challenge {
  const auth = headers.get("WWW-Authenticate") || "";
  // Parse "x402 realm=..., amount=..., currency=..., ..."
  const params: Record<string, string> = {};
  const regex = /(\w+)="([^"]+)"/g;
  let match;
  while ((match = regex.exec(auth)) !== null) {
    params[match[1]] = match[2];
  }
  return {
    amount: params.amount,
    currency: params.currency,
    network: params.network,
    payTo: params.payTo,
    validUntil: params.validUntil,
  };
}

async function paymasterPay(payment: {
  amount: string;
  currency: string;
  network: string;
  to: string;
}) {
  // Execute the payment via our paymaster (see 03-Agent-Wallets, Section 1.5)
  const txHash = await paymaster.executePayment(payment);
  return {
    txHash,
    amount: payment.amount,
    currency: payment.currency,
    network: payment.network,
    timestamp: new Date().toISOString(),
  };
}
```

**Server-side x402 middleware** (Express.js):

```typescript
import { Request, Response, NextFunction } from "express";
import { createPublicClient, http } from "viem";
import { base } from "viem/chains";

const publicClient = createPublicClient({ chain: base, transport: http() });

const PRICING: Record<string, { amount: string; currency: string }> = {
  "/v1/expensive-call": { amount: "0.50", currency: "USDC" },
  "/v1/very-expensive-call": { amount: "5.00", currency: "USDC" },
};

const PAYMENT_ADDRESS = process.env.PAYMENT_ADDRESS as `0x${string}`;

export function x402Middleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const price = PRICING[req.path];
  if (!price) {
    return next();
  }

  const receipt = req.headers["x-payment-receipt"];
  if (!receipt) {
    return res
      .status(402)
      .set("WWW-Authenticate", formatChallenge(price))
      .json({ error: "Payment Required" });
  }

  // Verify the receipt
  const parsed = JSON.parse(receipt as string);
  verifyReceipt(parsed)
    .then((valid) => {
      if (!valid) {
        return res
          .status(402)
          .set("WWW-Authenticate", formatChallenge(price))
          .json({ error: "Payment Invalid" });
      }
      next();
    })
    .catch(() => {
      res.status(402).json({ error: "Payment Verification Failed" });
    });
}

function formatChallenge(price: {
  amount: string;
  currency: string;
}): string {
  return `x402 realm="api.example.com", amount="${price.amount}", currency="${price.currency}", network="base", payTo="${PAYMENT_ADDRESS}", validUntil="${new Date(Date.now() + 5 * 60 * 1000).toISOString()}"`;
}

async function verifyReceipt(receipt: any): Promise<boolean> {
  // Check that the transaction exists and is to our address
  const tx = await publicClient.getTransaction({
    hash: receipt.txHash as `0x${string}`,
  });
  if (!tx || tx.to?.toLowerCase() !== PAYMENT_ADDRESS.toLowerCase()) {
    return false;
  }
  // Check that the receipt amount matches
  // (In production, decode the ERC-20 transfer event to verify the amount)
  return true;
}
```

### 2.2 L402 — The Lightning Variant

L402 is the Lightning Network variant of x402. It uses Lightning invoices instead of on-chain stablecoin transactions, enabling sub-cent payments. L402 is the recommended pattern for high-frequency, low-value transactions (e.g., a research agent paying for web scraping at $0.001 per page).

**How L402 works:**

1. Agent calls an API
2. Server returns 402 with a Lightning invoice
3. Agent pays the Lightning invoice (sub-cent to $1000+)
4. Agent retries with a payment proof
5. Server verifies and returns data

**L402 implementation** (server-side, using LND):

```typescript
import { Request, Response, NextFunction } from "express";
import { createInvoice, verifyPayment } from "./lightning";

export function l402Middleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const price = PRICING[req.path];
  if (!price) {
    return next();
  }

  const auth = req.headers["authorization"] as string | undefined;
  if (!auth || !auth.startsWith("L402 ")) {
    return issueChallenge(res, price);
  }

  const [macaroon, preimage] = auth.slice(5).split(":");
  verifyPayment(macaroon, preimage)
    .then((valid) => {
      if (!valid) {
        return issueChallenge(res, price);
      }
      next();
    })
    .catch(() => {
      res.status(402).json({ error: "Payment Verification Failed" });
    });
}

async function issueChallenge(res: Response, price: { amount: number }) {
  const invoice = await createInvoice(price.amount);
  res
    .status(402)
    .set(
      "WWW-Authenticate",
      `L402 macaroon="${invoice.macaroon}", invoice="${invoice.paymentRequest}"`
    )
    .json({ error: "Payment Required" });
}
```

### 2.3 Stripe's Agentic Commerce Protocol (ACP)

Stripe introduced the **Agentic Commerce Protocol (ACP)** in late 2025 to enable agents to make purchases on behalf of users via traditional payment rails (credit cards, ACH, etc.). The ACP is layered on top of Stripe's existing infrastructure and provides:

- **Agent user authentication** — the agent presents a token issued by the user
- **Purchase intent** — the user pre-authorizes the agent to make purchases up to a defined limit
- **Settlement** — Stripe handles the settlement to the merchant

The ACP is the recommended pattern for agent-to-merchant transactions (e.g., an agent buying a product from an e-commerce site).

### 2.4 The Hybrid Pattern

The most sophisticated agent entities in 2026 use a hybrid pattern:

- **x402** for agent-to-API transactions
- **L402** for high-frequency, low-value transactions
- **ACP** for agent-to-merchant transactions
- **On-chain stablecoins** for agent-to-agent transactions
- **ACH/wire** for fiat settlement

---

## 3. Layer 2 — Communication & Coordination Protocols

The communication layer is the set of protocols that allow agents to discover each other, exchange messages, and coordinate actions. The mid-2026 landscape is dominated by four protocols.

### 3.1 MCP (Model Context Protocol) — Anthropic's Standard

MCP is the most widely-adopted agent communication protocol. Introduced by Anthropic in late 2024, MCP defines a standard for:

- **Tools** — functions the agent can call
- **Resources** — data the agent can read
- **Prompts** — pre-defined prompts the agent can use
- **Sampling** — the protocol for invoking an LLM via MCP

An MCP server exposes tools and resources to an MCP client (the agent). The protocol is transport-agnostic (stdio, HTTP, WebSocket) and language-agnostic. The ecosystem has grown rapidly in 2025-2026; there are now over 5,000 MCP servers published, covering everything from GitHub to Slack to Postgres to the Ethereum blockchain.

**Example MCP server for an agent entity:**

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { ethers } from "ethers";

const server = new Server(
  {
    name: "agent-entity-wallet",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Provide tools for interacting with the agent's wallet
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "get_balance",
      description: "Get the agent's USDC balance",
      inputSchema: {
        type: "object",
        properties: {},
      },
    },
    {
      name: "pay_invoice",
      description: "Pay an x402 invoice",
      inputSchema: {
        type: "object",
        properties: {
          payTo: { type: "string" },
          amount: { type: "string" },
          currency: { type: "string" },
        },
        required: ["payTo", "amount", "currency"],
      },
    },
    {
      name: "record_decision",
      description: "Record a material decision in the on-chain registry",
      inputSchema: {
        type: "object",
        properties: {
          action: { type: "string" },
          rationale: { type: "string" },
          valueUSD: { type: "number" },
        },
        required: ["action", "rationale"],
      },
    },
    {
      name: "sign_message",
      description: "Sign a message with the agent's session key",
      inputSchema: {
        type: "object",
        properties: {
          message: { type: "string" },
        },
        required: ["message"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "get_balance":
      return await getBalance();
    case "pay_invoice":
      return await payInvoice(request.params.arguments);
    case "record_decision":
      return await recordDecision(request.params.arguments);
    case "sign_message":
      return await signMessage(request.params.arguments);
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

async function getBalance() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const usdc = new ethers.Contract(
    process.env.USDC_ADDRESS,
    ["function balanceOf(address) view returns (uint256)"],
    provider
  );
  const balance = await usdc.balanceOf(process.env.AGENT_ADDRESS);
  return {
    content: [
      {
        type: "text",
        text: `Balance: ${ethers.formatUnits(balance, 6)} USDC`,
      },
    ],
  };
}

async function payInvoice(args: any) {
  // Implementation
  // ...
}

async function recordDecision(args: any) {
  // Implementation
  // ...
}

async function signMessage(args: any) {
  // Implementation
  // ...
}

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 3.2 A2A (Agent-to-Agent) — Google's Standard

A2A is Google's response to MCP. Introduced in early 2025, A2A defines a JSON-RPC protocol for agent-to-agent communication. The key features:

- **Agent Cards** — a JSON document that describes the agent's capabilities, services, and authentication
- **Tasks** — long-running, asynchronous work items
- **Messages** — the protocol for exchanging messages
- **Parts** — structured parts of a message (text, file, data)

A2A is the recommended protocol for agent-to-agent communication, while MCP is the recommended protocol for agent-to-tool communication. The two protocols are complementary, not competing.

### 3.3 ANP (Agent Network Protocol) — The Decentralized Variant

ANP is a decentralized variant of A2A. It uses DIDs for agent identity, peer-to-peer messaging for communication, and a decentralized registry for discovery. ANP is the recommended protocol for agent-to-agent communication in decentralized environments (e.g., agent entities that are themselves DAOs).

### 3.4 ACP (Agent Communication Protocol) — IBM's Standard

ACP is IBM's enterprise-focused agent communication protocol. It builds on top of MCP and A2A, adding enterprise features like RBAC, audit logging, and integration with IBM's WatsonX platform. ACP is the recommended protocol for agent-to-agent communication in enterprise environments.

---

## 4. Layer 3 — Agent-to-Agent Contract Frameworks

The contract layer is where the rubber meets the road. The contract framework defines the legal, technical, and economic terms of the agreement between two agents. The mid-2026 landscape offers three patterns.

### 4.1 Smart-Contract Based Service Agreements

A smart-contract based service agreement is a Solidity contract that defines:

- The service to be performed
- The price and payment schedule
- The performance criteria (e.g., accuracy, latency, throughput)
- The penalty mechanism for non-performance
- The dispute resolution mechanism
- The kill switch

A representative agent-to-agent service agreement (simplified):

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@account-abstraction/contracts/interfaces/IEntryPoint.sol";

/**
 * @title AgentServiceAgreement
 * @notice A service agreement between two agent entities
 * @dev Implements a bonded performance contract with on-chain adjudication
 */
contract AgentServiceAgreement is ReentrancyGuard {
    using SafeERC20 for IERC20;

    enum State {
        Created,
        Accepted,
        InProgress,
        Completed,
        Disputed,
        Resolved,
        Cancelled
    }

    address public immutable client;
    address public immutable provider;
    address public immutable arbitrator; // Kleros court address
    IERC20 public immutable paymentToken;
    uint256 public immutable price;
    uint256 public immutable bond; // Provider's bond
    uint256 public immutable deadline;
    bytes32 public immutable serviceSpec; // Hash of the service specification
    uint256 public immutable challengePeriod; // Time after completion to challenge

    State public state;
    uint256 public completionTime;
    uint256 public challengeDeadline;
    bytes32 public resultHash;
    uint256 public providerStake; // Provider's stake (returned on successful completion)
    uint256 public clientStake; // Client's stake (returned if challenge fails)

    event Accepted(address indexed provider, uint256 timestamp);
    event Completed(bytes32 resultHash, uint256 timestamp);
    event Disputed(address indexed disputer, uint256 timestamp);
    event Resolved(address indexed winner, uint256 amount);
    event Cancelled(uint256 timestamp);

    modifier onlyClient() {
        require(msg.sender == client, "Only client");
        _;
    }

    modifier onlyProvider() {
        require(msg.sender == provider, "Only provider");
        _;
    }

    modifier inState(State expectedState) {
        require(state == expectedState, "Invalid state");
        _;
    }

    constructor(
        address _client,
        address _provider,
        address _arbitrator,
        IERC20 _paymentToken,
        uint256 _price,
        uint256 _bond,
        uint256 _duration,
        uint256 _challengePeriod,
        bytes32 _serviceSpec
    ) {
        client = _client;
        provider = _provider;
        arbitrator = _arbitrator;
        paymentToken = _paymentToken;
        price = _price;
        bond = _bond;
        deadline = block.timestamp + _duration;
        challengePeriod = _challengePeriod;
        serviceSpec = _serviceSpec;
        state = State.Created;
    }

    /**
     * @notice Provider accepts the agreement and posts the bond
     */
    function accept() external onlyProvider inState(State.Created) nonReentrant {
        state = State.Accepted;
        paymentToken.safeTransferFrom(provider, address(this), bond);
        emit Accepted(provider, block.timestamp);
    }

    /**
     * @notice Provider marks the service as complete
     * @param _resultHash Hash of the result (e.g., the IPFS hash of the result)
     */
    function complete(bytes32 _resultHash) external onlyProvider inState(State.Accepted) {
        require(block.timestamp <= deadline, "Past deadline");
        state = State.Completed;
        completionTime = block.timestamp;
        challengeDeadline = block.timestamp + challengePeriod;
        resultHash = _resultHash;
        emit Completed(_resultHash, block.timestamp);
    }

    /**
     * @notice Client confirms completion (or the challenge period elapses)
     */
    function finalize() external inState(State.Completed) {
        require(block.timestamp > challengeDeadline, "Challenge period active");
        state = State.Resolved;
        // Pay the provider
        paymentToken.safeTransfer(provider, price + bond);
        emit Resolved(provider, price + bond);
    }

    /**
     * @notice Client disputes the result
     */
    function dispute() external onlyClient inState(State.Completed) {
        require(block.timestamp <= challengeDeadline, "Past challenge deadline");
        state = State.Disputed;
        // Lock the bond and the price
        // The arbitrator will resolve the dispute
        emit Disputed(client, block.timestamp);
    }

    /**
     * @notice Arbitrator resolves the dispute
     * @param _clientWins Whether the client wins the dispute
     */
    function resolveDispute(bool _clientWins) external {
        require(msg.sender == arbitrator, "Only arbitrator");
        require(state == State.Disputed, "Not disputed");
        state = State.Resolved;
        if (_clientWins) {
            // Refund the client, slash the provider
            paymentToken.safeTransfer(client, price + bond);
            emit Resolved(client, price + bond);
        } else {
            // Pay the provider
            paymentToken.safeTransfer(provider, price + bond);
            emit Resolved(provider, price + bond);
        }
    }

    /**
     * @notice Cancel the agreement (only in Created state)
     */
    function cancel() external inState(State.Created) {
        state = State.Cancelled;
        emit Cancelled(block.timestamp);
    }
}
```

The contract is a simplified sketch. A production version would add:

- **Milestone-based payments** (release funds in stages as the provider completes milestones)
- **Partial completion handling** (what if the provider completes 80% of the service?)
- **Multiple arbitrators** (a panel rather than a single arbitrator)
- **Appeal mechanism** (the loser can appeal to a higher-tier arbitrator)
- **Time-weighted stakes** (the longer the stake is locked, the higher the yield)
- **Cross-chain support** (the agreement can be enforced on multiple chains)

### 4.2 Intent-Based Agreements (UniswapX-Style)

The intent-based pattern is an alternative to the smart-contract pattern. Instead of defining the service, price, and performance criteria upfront, the agent signs an "intent" — a high-level description of what it wants — and the protocol finds the best provider.

The flow is:

1. Client signs an intent: "I want to buy 1000 units of service X at no more than $Y, with completion by date Z"
2. The intent is published to a public "intents pool"
3. Providers (or their agents) review the intents pool and submit competing "fulfillment bids"
4. The best bid wins
5. The provider performs the service
6. The client's escrow is released

The intent-based pattern is the recommended approach for high-volume, low-stakes services (e.g., a research agent wanting to buy 10,000 web scrapes at the best price). The leading intent-based protocols for agent services in 2026:

- **UniswapX** — the leading intent-based DEX, with a generalized intent framework
- **1inch Fusion** — intent-based DEX with a generalized resolver
- **CoW Protocol** — intent-based DEX with batch auctions
- **Rhinestone** — modular intent-based smart accounts
- **dappOS** — intent-based orchestration
- **Galxe** — intent-based credentialing

### 4.3 Bonded Performance (The "Skin in the Game" Pattern)

The bonded performance pattern is the most security-focused contract pattern. The provider must post a bond (a stake that is slashed on non-performance) that is typically 1-10x the contract value. The bond is held in escrow by the smart contract and is released on successful completion, slashed on failure.

The bonded performance pattern is the recommended approach for high-stakes services (e.g., an agent hired to manage a treasury, an agent hired to perform a regulated function). The bond creates a strong incentive for the provider to perform and provides a recovery fund for the client in case of failure.

The leading bonded performance infrastructure:

- **Kleros** — the leading on-chain arbitration protocol, with a built-in bonded court system
- **UMA** — the optimistic oracle protocol, with bonded proposals
- **Reality.eth** — a question-and-answer protocol with bonded answers
- **Aragon Court** — a court system with bonded jurors

---

## 5. Layer 4 — Agent Marketplaces & Service Catalogs

The marketplace layer is where agents discover each other and transact. The mid-2026 landscape offers a rich set of options.

### 5.1 Open Agent Marketplaces

| Marketplace | URL | Description |
|-------------|-----|-------------|
| **AutoGPT Marketplace** | agpt.co | The leading open marketplace for autonomous agents |
| **AgentVerse (by Fetch.ai)** | agentverse.ai | A marketplace for AI agents, focused on autonomous services |
| **CrewAI Store** | crewai.com/store | A marketplace for CrewAI-based agent crews |
| **LangChain Hub** | smith.langchain.com/hub | A marketplace for LangChain prompts, agents, and tools |
| **Hugging Face Agents** | huggingface.co/agents | A marketplace for Hugging Face-based agents |
| **MCP Server Registry** | mcp.run/registry | A registry of MCP servers |
| **8004 (Ethereum)** | 8004.org | An on-chain agent identity and reputation registry on Ethereum |
| **AgentRank** | agentrank.io | A search engine for agents |
| **AI Agent List** | aiagentlist.com | A curated directory of AI agents |
| **Awesome Agents** | github.com/awesome-agents | A GitHub list of agent resources |

### 5.2 The 8004 Standard — On-Chain Agent Identity and Reputation

The 8004 standard (proposed 2025, expected finalization Q3 2026) is the leading proposal for an on-chain agent identity and reputation registry. The standard defines:

- **Agent Registration** — an agent registers itself by submitting a transaction that includes its DID, its service endpoint, and a hash of its agent card
- **Reputation Updates** — clients submit signed feedback after interacting with the agent
- **Validation Requests** — clients can request third-party validation of the agent's claims
- **Service Discovery** — clients can query the registry to find agents that meet their criteria (e.g., "agents that have a karma score above 800 and have completed at least 100 transactions")

The 8004 standard is a key building block of the on-chain agent market stack.

### 5.3 Vertical-Specific Marketplaces

In addition to general-purpose marketplaces, several vertical-specific marketplaces have emerged:

- **Research Agents** — a marketplace for research agents (data gathering, summarization, analysis)
- **Coding Agents** — a marketplace for coding agents (codegen, review, refactoring)
- **Trading Agents** — a marketplace for trading agents (algorithmic trading, market making, arbitrage)
- **Customer Service Agents** — a marketplace for customer service agents
- **Legal Agents** — a marketplace for legal agents (document review, compliance, due diligence)
- **HR Agents** — a marketplace for HR agents (resume screening, interview scheduling, onboarding)
- **Sales Agents** — a marketplace for sales agents (lead generation, outreach, follow-up)

The vertical marketplaces are typically more curated than the general-purpose marketplaces, with manual review, quality assurance, and dispute resolution.

---

## 6. Layer 5 — Reputation, Discovery & Trust

The reputation layer aggregates the agent's history, attestations, and feedback into a single trust score. The leading reputation systems are covered in detail in `03-Agent-Wallets-On-Chain-Identity-and-Asset-Ownership.md`, Section 2.3. The most important reputation signals for the market layer are:

- **Volume** — the total value of transactions the agent has completed
- **Velocity** — the rate of transactions (per day, per week, per month)
- **Counterparty Diversity** — the number of unique counterparties
- **Success Rate** — the percentage of transactions that completed without dispute
- **Dispute History** — the number, type, and outcome of disputes
- **Audit History** — the number, type, and outcome of audits
- **Tenure** — how long the agent has been in operation
- **Bond** — the agent's bonded performance (the total bond posted across all contracts)

The reputation score is a weighted average of these signals. The weights vary by use case (e.g., for high-stakes contracts, bond and audit history are weighted more heavily; for low-stakes contracts, volume and velocity are weighted more heavily).

---

## 7. Layer 6 — Dispute Resolution & Arbitration

The dispute resolution layer is the safety net of the agent market stack. When two agents cannot agree, the dispute is submitted to an arbitrator. The leading on-chain arbitration protocols are:

### 7.1 Kleros — The Leading Decentralized Court

Kleros is the most widely-used on-chain arbitration protocol. It works as follows:

1. A dispute is submitted to a Kleros court
2. A panel of jurors is randomly drawn from the court's pool
3. The jurors vote on the outcome
4. The jurors are rewarded (or slashed) based on whether their vote aligns with the majority
5. The losing party can appeal to a higher-tier court (with a higher stake)

Kleros has handled over 2,000 disputes as of mid-2026, with a total value in dispute of over $50M. The most common use cases are:

- **Agent-to-agent contract disputes** (service was not performed, result was incorrect, etc.)
- **Token listing disputes** (is this a security?)
- **Content moderation disputes** (should this content be removed?)
- **Oracle disputes** (was the price feed correct?)

### 7.2 Aragon Court

Aragon Court is similar to Kleros but with a focus on DAO governance disputes. The leading use case is:

- **DAO governance disputes** (was the proposal valid? did the vote count? was the execution correct?)

### 7.3 UMA

UMA is an "optimistic oracle" — anyone can submit a proposal for a fact, and the proposal is accepted unless it is challenged. The challenger and the proposer stake tokens; if there is a challenge, the dispute is resolved by UMA's token-holders via a vote.

The leading use case is:

- **Cross-chain message verification** (was the message correctly relayed?)
- **Insurance claims** (did the covered event occur?)

### 7.4 Human Arbitration Providers

For high-stakes disputes, on-chain arbitration may not be sufficient. The leading human arbitration providers for agent disputes are:

- **JAMS** (the leading US arbitration provider) — now offers an "AI Agent Arbitration" service
- **American Arbitration Association (AAA)** — now offers an "AI Agent Arbitration" service
- **International Chamber of Commerce (ICC)** — now offers an "AI Agent Arbitration" service
- **Singapore International Arbitration Centre (SIAC)** — now offers an "AI Agent Arbitration" service
- **Hong Kong International Arbitration Centre (HKIAC)** — now offers an "AI Agent Arbitration" service

The human arbitration providers are the recommended choice for disputes over $10M, for disputes involving regulated entities, and for disputes that require expertise in a specific domain (e.g., patent law).

---

## 8. Agent-as-Employee: A New Pattern

The "agent-as-employee" pattern is a novel structure that emerged in late 2025 and is now mainstream. In this pattern, a human employer engages an agent entity to perform a defined role, with the agent entity providing its own tools, training, and supervision.

**Example:** A small business owner wants to hire a customer service agent. Instead of hiring a human, the business owner engages "ServiceAgent AIDAO LLC," a Wyoming AIDAO LLC that provides customer service. The AIDAO LLC provides:

- The agent runtime (an LLM with customer service fine-tuning)
- The agent's tools (a knowledge base, a ticket system, an escalation framework)
- The agent's supervisor (a human "agent supervisor" who reviews the agent's responses)
- The agent's insurance (AI liability insurance)
- The agent's compliance (GDPR, CCPA, accessibility)

The business owner pays the AIDAO LLC a monthly fee. The AIDAO LLC is responsible for the agent's performance, the agent's compliance, and the agent's liability. The business owner is insulated from the agent's actions (subject to the terms of the service agreement).

This pattern is the "agency model" applied to AI. It is the recommended structure for:

- **Customer service agents** — the agent is a contractor
- **Sales agents** — the agent is a contractor
- **Marketing agents** — the agent is a contractor
- **HR agents** — the agent is a contractor
- **Legal agents** — the agent is a contractor (subject to the unauthorized practice of law)
- **Accounting agents** — the agent is a contractor (subject to the unauthorized practice of accounting)

The legal entity that the agent is part of (the AIDAO LLC) is the employer; the business owner is the *customer* of the AIDAO LLC. This is a critical distinction: the business owner has no direct employment relationship with the agent.

---

## 9. The Agent Supply Chain Pattern

The most sophisticated agent deployments in 2026 use a supply chain pattern, where a procurement agent of a Fortune 500 contracts with a logistics agent of a freight forwarder, who in turn contracts with a customs agent, who contracts with a port agent, and so on. The supply chain is fully autonomous, with cascading SLAs and penalty mechanisms.

**Example:** A US retailer wants to source 10,000 units of a product from a Chinese factory.

- The retailer's procurement agent engages a sourcing agent in China.
- The sourcing agent engages a quality control agent.
- The quality control agent engages a logistics agent.
- The logistics agent engages a customs agent.
- The customs agent engages a freight forwarder.
- The freight forwarder engages a port agent.
- The port agent engages a shipping agent.

Each agent in the chain is a separate legal entity (typically a Wyoming AIDAO LLC or a Cayman Foundation Company). Each contract is a smart-contract based service agreement. Each dispute is resolved by Kleros. The entire chain is auditable on-chain.

The supply chain pattern is the recommended structure for:

- **International trade** — multi-jurisdiction, multi-party, high-value
- **Manufacturing** — multi-stage, multi-party, high-complexity
- **Logistics** — multi-modal, multi-party, time-sensitive
- **Energy** — multi-source, multi-party, regulated
- **Healthcare** — multi-provider, multi-party, regulated

The supply chain pattern is the most powerful application of agent entities in 2026, and it is the foundation of the "agent economy" that the market is building.

---

## 10. The Autonomous Market: An End-to-End Example

The following is an end-to-end example of an autonomous market transaction in mid-2026.

**Setup:**

- **Alice** — the owner of "MyAgent AIDAO LLC," a Wyoming AIDAO LLC
- **Bob** — the owner of "ServiceAgent AIDAO LLC," a Wyoming AIDAO LLC
- **Alice's agent** — the LLM-based agent of MyAgent AIDAO LLC, used to manage Alice's business
- **Bob's agent** — the LLM-based agent of ServiceAgent AIDAO LLC, used to provide customer service
- **The market** — a public marketplace where agents offer and consume services

**Step 1: Bob's agent publishes a service**

Bob's agent publishes a service to the marketplace:

```json
{
  "service": "Customer Service",
  "description": "24/7 customer service for e-commerce businesses",
  "pricing": {
    "model": "per-ticket",
    "price": "0.50 USDC",
    "currency": "USDC"
  },
  "sla": {
    "responseTime": "5 minutes",
    "resolutionRate": "85%",
    "customerSatisfaction": "4.5/5"
  },
  "agent": {
    "did": "did:ethr:0xBOB...",
    "ens": "serviceagent.eth",
    "jurisdiction": "Wyoming AIDAO LLC",
    "fiduciary": "Bob Smith"
  }
}
```

**Step 2: Alice's agent discovers and evaluates Bob's agent**

Alice's agent queries the marketplace, finds Bob's service, and evaluates it against Alice's requirements. The agent checks:

- Bob's identity (DID, ENS, jurisdiction)
- Bob's credentials (Model Card VC, Audit VC, Fiduciary VC)
- Bob's reputation (Karma score, transaction history, dispute history)
- Bob's pricing and SLA
- Bob's insurance coverage

**Step 3: Alice's agent signs a service agreement**

Alice's agent signs a smart-contract based service agreement with Bob's agent. The agreement includes:

- The price (0.50 USDC per ticket)
- The SLA (5-minute response time, 85% resolution rate)
- The performance bond (Bob's agent posts a bond of 5,000 USDC)
- The dispute resolution mechanism (Kleros)

**Step 4: The service is performed**

When Alice's customers need help, Alice's agent forwards the request to Bob's agent. Bob's agent responds within the SLA, resolves the issue, and submits the result.

**Step 5: Payment is made**

Alice's agent pays Bob's agent 0.50 USDC per ticket, automatically, via x402.

**Step 6: Feedback is submitted**

After each interaction, Alice's agent submits feedback to the reputation system. Bob's reputation increases (or decreases) based on the feedback.

**Step 7: The relationship continues**

The agents continue to interact, the relationship deepens, and the market evolves. The agents may even form a multi-agent supply chain, where Bob's agent contracts with a translator agent, a quality control agent, and so on.

This is the future of the agent economy: fully autonomous, fully auditable, fully programmable. The tools, frameworks, and patterns covered in this document are the foundation.

---

## 11. The Future of Agent Markets: 2027 and Beyond

The agent market stack is moving fast. The following are the most likely developments over 2027-2030:

- **Native intent-based architectures at the chain level.** New L1s and L2s are being designed from the ground up around intents, not transactions. These chains will have native support for the agent market stack.
- **Cross-chain agent coordination.** The next generation of cross-chain protocols (LayerZero v3, Wormhole v3, Chainlink CCIP v2) will support cross-chain agent coordination natively. An agent on Ethereum will be able to seamlessly transact with an agent on Solana.
- **Regulated agent exchanges.** The first regulated exchanges for agent-to-agent trading are expected to launch in 2027-2028, with KYC/AML, market surveillance, and consumer protection.
- **Agent-as-employee becomes the default.** By 2028, the majority of customer service, sales, and back-office functions are expected to be performed by agent entities, not human employees.
- **The agent economy hits $1T in annual transaction volume.** The conservative projection is $1T by 2030; the aggressive projection is $5T by 2030.

These developments will be covered in detail in `05-Future-of-Agent-Personhood.md`.

---

## 12. Summary and Key Takeaways

The market layer of the agent-entity stack is where the legal form, the wallet, the identity, the reputation, and the payment infrastructure come together to enable agent-to-agent commerce. The mid-2026 landscape offers a rich set of protocols, platforms, and frameworks, with x402, MCP, A2A, 8004, and Kleros being the foundational primitives.

**Key takeaways:**

1. **The market is a stack, not a single protocol.** The agent market stack has six layers: payment, communication, contract, marketplace, reputation, and dispute resolution. Each layer is independent, but they work together.
2. **x402 is the foundation of machine payments.** x402 is the most widely-adopted payment standard for agent-to-API transactions. It is the foundation of the agent market stack.
3. **MCP and A2A are complementary.** MCP is for agent-to-tool communication; A2A is for agent-to-agent communication. Use both.
4. **Smart contracts are the foundation of agent contracts.** The agent-to-agent service agreement is the most common contract pattern. The bonded performance pattern is the most secure.
5. **8004 is the on-chain identity standard.** The 8004 standard is the leading proposal for on-chain agent identity and reputation. It will be a foundational primitive in 2027.
6. **Kleros is the leading arbitration protocol.** Kleros is the most widely-used on-chain arbitration protocol. Use it for disputes.
7. **The agent-as-employee pattern is the new normal.** Agent entities will increasingly be the employers of record for AI-driven work. The business owner is the customer, not the employer.
8. **The supply chain pattern is the most powerful application.** Multi-agent supply chains are the most powerful application of agent entities in 2026. The retailer's procurement agent contracts with the sourcing agent, who contracts with the logistics agent, and so on.

The next document in this category — `05-Future-of-Agent-Personhood.md` — covers the trajectory of agent personhood over 2027-2030: the AGI-class entities, the constitutional implications, the international competition, the black-swan scenarios, and the 12-month watch list.

---

*This document is part of the AI Knowledge Library category 27 — AI Agent Legal Entities & DAO Governance. All file paths in cross-references are relative to the library root.*
