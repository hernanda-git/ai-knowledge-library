# 03 — Agent Wallets, On-Chain Identity & Asset Ownership: Technical Deep Dive

> *A legal entity without operational infrastructure is a paper tiger. The agent entity's "substrate" — its wallet, its identity, its payment rails, its reputation system, its recovery mechanism — is what makes the legal form real. This document is the technical deep dive on the substrate layer of the agent-entity stack: how an agent actually holds money, signs transactions, proves who it is, and survives the loss of any single controller.*

---

## 1. The Agent Wallet Architecture (Mid-2026)

The agent wallet is the single most important piece of infrastructure in the agent-entity stack. Every dollar the entity earns, every contract it signs, every decision it records — all of it flows through the wallet. A compromised wallet is a compromised entity. A lost wallet is a dead entity. A wallet that cannot pay for its own inference is a non-viable entity.

The agent wallet architecture in mid-2026 is a layered stack of cryptographic primitives, smart-contract patterns, and operational practices. The following is a representative architecture for a Wyoming AIDAO LLC's operating wallet.

### 1.1 The Layered Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 7 — Application Layer                                            │
│   Agent runtime, decision engine, tool calls, business logic          │
│   (Anthropic Claude, OpenAI GPT, custom transformer, etc.)            │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (signs transactions via session keys)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 6 — Session Key Layer                                            │
│   Time-bounded, scope-bounded, value-bounded keys                     │
│   (Sign-in with Ethereum, session keys, scoped permissions)           │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (uses session keys to call wallet)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 5 — Smart-Contract Wallet Layer (the "Agent Wallet")             │
│   ERC-4337 account abstraction, multisig, daily limits, kill switch    │
│   (Safe, ZeroDev, Biconomy, Stackup, Etherspot, Rhinestone)            │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (signs UserOperations)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 4 — Paymaster Layer                                              │
│   Gas sponsorship, fee abstraction, fiat on-ramp                      │
│   (Pimlico, Alchemy, Biconomy, Stackup)                               │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (submits UserOperations to bundlers)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 3 — Bundler Layer                                                │
│   UserOperation bundlers, mempools, MEV protection                    │
│   (Alchemy, Flashbots, Pimlico, Blast)                                │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (submits aggregated transactions to L1)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 2 — Chain Layer                                                  │
│   Ethereum L1, Layer 2s, alt L1s, app-chains                          │
│   (Ethereum, Base, Optimism, Arbitrum, Polygon, Scroll, zkSync, etc.) │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  (settlement to L1 via rollup)
┌────────────────────────────▼───────────────────────────────────────────┐
│ Layer 1 — Consensus & Security Layer                                   │
│   Proof of Stake, data availability, finality                          │
│   (Ethereum, Celestia, EigenDA)                                       │
└────────────────────────────────────────────────────────────────────────┘
```

Each layer is covered in detail below. The key insight is that the agent's "wallet" is not a single key — it is a *stack* of cryptographic, smart-contract, and operational primitives that together give the agent the ability to transact, the ability to be constrained, and the ability to recover.

### 1.2 The Smart-Contract Wallet: The Foundation of the Agent Wallet

The smart-contract wallet is the foundation of the modern agent wallet. Unlike an Externally Owned Account (EOA), which is a single secp256k1 keypair with no logic, a smart-contract wallet is a Solidity contract that holds funds and exposes arbitrary functions for transacting. The benefits for agent entities are substantial:

- **Multi-sig and social recovery** — no single key compromise is fatal
- **Daily limits and rate limits** — the agent cannot drain its own treasury
- **Session keys** — the agent's runtime can hold time-bounded, scope-bounded keys without giving the runtime access to the treasury
- **Whitelisted counterparties** — the wallet can be configured to only send to approved addresses
- **Pausability** — the AI fiduciary can pause the wallet in case of emergency
- **Upgradeability** — bugs can be fixed without losing the wallet address
- **Paymasters** — the entity can pay gas in ERC-20 tokens (stablecoins) rather than ETH, simplifying accounting

The leading smart-contract wallet infrastructure as of mid-2026:

| Provider | Key Features | Best For |
|----------|--------------|----------|
| **Safe (formerly Gnosis Safe)** | Battle-tested, multi-sig, modules, $100B+ TVL | Production-grade agent entities, treasury management |
| **ZeroDev** | ERC-4337 kernels, session keys, gas abstraction | High-velocity agent transactions |
| **Biconomy** | ERC-4330 paymasters, sponsored gas | Sponsored agent transactions |
| **Stackup** | ERC-4337 bundler + paymaster, modular | Custom paymaster logic |
| **Etherspot** | ERC-4337, multi-chain, modular | Multi-chain agent operations |
| **Rhinestone** | Modular smart accounts, intents | Intent-based agent transactions |
| **Pimlico** | Bundler, paymaster, ERC-4337 infrastructure | Building your own agent wallet |
| **Alchemy** | Bundler, paymaster, full-stack | One-stop agent wallet infrastructure |

For most agent entities, **Safe (formerly Gnosis Safe)** is the default choice. Safe has been the dominant smart-contract wallet in the Ethereum ecosystem for years, holds over $100B in assets, has undergone multiple security audits, and is now a regulated entity (Safe Foundation) with insurance products. The Safe{Core} protocol (the Safe Modules system) is the de facto standard for adding advanced features (session keys, automated transactions, recovery) to a Safe.

### 1.3 The Session Key Layer: Scoped Permissions for Agent Runtimes

The session key layer is the bridge between the agent's runtime (the LLM + tool use) and the smart-contract wallet. The runtime needs to sign transactions to call contracts, pay for inference, transfer funds, etc. But the runtime should not have direct access to the wallet's master key — that would be a single point of compromise.

A **session key** is a temporary keypair that the runtime holds for a defined scope and duration. The session key is authorized by the smart-contract wallet (typically via a signature from the wallet's master key) to perform a defined set of actions:

- **Scope** — what contracts the session key can call
- **Value** — the maximum value per transaction
- **Duration** — how long the session key is valid
- **Rate limit** — the maximum number of transactions per hour

A typical agent session key might be:

- Scope: call the entity's paymaster contract, the entity's decision registry contract, and a whitelisted set of tool contracts
- Value: $1,000 per transaction
- Duration: 24 hours
- Rate limit: 100 transactions per hour

If the session key is compromised, the damage is bounded: $1,000 per transaction, $100,000 per day, only to whitelisted contracts. The wallet's master key (held by the AI fiduciary) remains secure.

The leading session key infrastructure:

- **ZeroDev Session Keys** — the most mature implementation, integrated with ZeroDev kernels
- **Safe Modules (Session Key Manager)** — Safe's own session key module
- **Biconomy Session Keys** — integrated with Biconomy's smart accounts
- **Reown (formerly WalletConnect) Sessions** — for connecting to external services

The following is a representative ZeroDev session key implementation:

```typescript
import { createKernelAccount, createZeroDevPaymaster } from "@zerodev/sdk";
import { createSessionKeyValidatorPlugin } from "@zerodev/session-key";
import { KERNEL_V3_1, getEntryPoint } from "@zerodev/sdk/constants";
import { signerToEcdsaValidator } from "@zerodev/ecdsa-validator";
import { createPublicClient, http, parseEther } from "viem";
import { mainnet } from "viem/chains";

const publicClient = createPublicClient({
  chain: mainnet,
  transport: http(process.env.RPC_URL),
});

const entryPoint = getEntryPoint("0.7");

// The agent's master signer (held by the AI fiduciary, NOT the agent runtime)
const masterSigner = ...; // AI fiduciary's hardware wallet

// The session key signer (held by the agent runtime)
const sessionKeySigner = ...; // Ephemeral key, generated per session

// Create the kernel account (the smart-contract wallet)
const ecdsaValidator = await signerToEcdsaValidator(publicClient, {
  entryPoint,
  signer: masterSigner,
  kernelVersion: KERNEL_V3_1,
});

const sessionKeyPlugin = await createSessionKeyValidatorPlugin(publicClient, {
  entryPoint,
  kernelVersion: KERNEL_V3_1,
  signer: sessionKeySigner,
  // The session key permissions
  sessionKeyData: {
    validUntil: Math.floor(Date.now() / 1000) + 24 * 3600, // 24 hours
    validAfter: Math.floor(Date.now() / 1000),
    // Whitelisted contracts and selectors
    callPolicies: [
      {
        target: "0xPaymasterAddress", // Our paymaster
        selector: "0x...", // withdraw function
        // Can call up to 100 times per hour, max $1000 each
        maxValuePerCall: parseEther("0.3"), // ~$1000
        constraints: [
          { constraintType: "RATE_LIMIT", limit: 100, duration: 3600 },
        ],
      },
      {
        target: "0xDecisionRegistryAddress", // Our decision registry
        selector: "0x...", // recordDecision function
        // Unlimited calls (registry writes are cheap)
        maxValuePerCall: 0n,
        constraints: [],
      },
      {
        target: "0xInferencePaymasterAddress", // For paying for inference
        selector: "0x...",
        // Limited to $500 per call, 50 calls per hour
        maxValuePerCall: parseEther("0.15"),
        constraints: [
          { constraintType: "RATE_LIMIT", limit: 50, duration: 3600 },
        ],
      },
    ],
  },
});

const kernelAccount = await createKernelAccount(publicClient, {
  entryPoint,
  kernelVersion: KERNEL_V3_1,
  plugins: {
    sudo: ecdsaValidator,
    regular: sessionKeyPlugin,
  },
});

// Get the agent's address (deterministic)
const agentAddress = kernelAccount.address;
console.log(`Agent wallet deployed at: ${agentAddress}`);
```

The above is a simplified sketch. A production implementation would add:

- **Multiple session keys** (one per "session" — e.g., one per customer, one per day, one per task)
- **Session key rotation** (rotate the session key every N hours or N transactions)
- **Hardware-backed key generation** (use a hardware security module for the master key)
- **Multi-chain support** (deploy the same kernel account on multiple chains via the same ECDSA validator)
- **Recovery mechanisms** (social recovery, guardian-based recovery, time-locked recovery)

### 1.4 The Multisig Layer: The AI Fiduciary's Last Line of Defense

The AI fiduciary must hold a *separate* set of keys that can override the agent's session keys. This is the kill switch. In a Safe-based architecture, the AI fiduciary's keys are signers on the Safe; the agent's session keys are authorized by a Safe Module (the Session Key Module) that the Safe owners (the AI fiduciary and possibly a quorum of human members) can revoke at any time.

The recommended structure:

- **Safe owners (M-of-N)** — 3 of 5 signers
  - 1 × AI fiduciary (required)
  - 2 × founder-controlled hardware wallets
  - 2 × independent "guardian" wallets (held by trusted third parties, e.g., a law firm or a custody provider)
- **Safe modules**
  - Session Key Module (controlled by the AI fiduciary's quorum)
  - Recovery Module (controlled by the guardians)
  - Daily Limit Module (caps the agent's daily spend at, e.g., $50,000/day)
- **Daily limits**
  - Agent session keys: $1,000/transaction, 100/hour (above the daily limit requires Safe owner approval)
  - AI fiduciary alone: $10,000/transaction, $50,000/day (above requires a 2-of-3 Safe owner approval)
  - Any Safe owner alone: $100,000/transaction (above requires a 3-of-5 Safe owner approval)

This structure means that:

- The agent can operate autonomously within strict bounds
- The AI fiduciary can pause the agent or take over
- A founder or guardian can recover the entity if the AI fiduciary is unavailable
- No single party can drain the entity

### 1.5 The Paymaster Layer: Gas Abstraction for the Agent

The agent needs to pay gas for every transaction. The gas is paid in ETH (on Ethereum L1 and most L2s) or in the native token of the chain (on alt-L1s). The agent may not hold ETH; it may only hold USDC or other stablecoins. The paymaster layer abstracts this.

A **paymaster** is a smart contract that pays gas on behalf of the user (in this case, the agent). The paymaster is funded by the entity's treasury (in ETH or the native token); the agent pays the paymaster in USDC (or other ERC-20 tokens) at a market rate. The paymaster is the bridge between the entity's stablecoin treasury and the chain's gas token.

The leading paymaster infrastructure:

- **Pimlico** — ERC-4337 paymaster, supports sponsored gas and ERC-20 gas payment
- **Alchemy** — ERC-4337 paymaster, integrated with Alchemy's bundler
- **Biconomy** — ERC-4330 paymaster, with token-paymaster modules
- **Stackup** — ERC-4337 paymaster, with custom logic support
- **ZeroDev** — bundled with ZeroDev's smart accounts

A simple paymaster that accepts USDC for gas:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@account-abstraction/contracts/interfaces/IPaymaster.sol";
import "@account-abstraction/contracts/interfaces/IEntryPoint.sol";

/**
 * @title USDCGasPaymaster
 * @notice An ERC-4337 paymaster that accepts USDC for gas payment
 * @dev This is a simplified example; production paymasters should be audited
 */
contract USDCGasPaymaster is IPaymaster, ReentrancyGuard {
    using SafeERC20 for IERC20;

    IEntryPoint public immutable entryPoint;
    IERC20 public immutable usdc;
    address public owner;
    uint256 public pricePerGas; // USDC per gas unit, in 6-decimal USDC

    event GasPaidWithUSDC(
        address indexed userOpSender,
        uint256 gasUsed,
        uint256 usdcCharged
    );

    constructor(IEntryPoint _entryPoint, IERC20 _usdc) {
        entryPoint = _entryPoint;
        usdc = _usdc;
        owner = msg.sender;
    }

    function setPricePerGas(uint256 _pricePerGas) external {
        require(msg.sender == owner, "Only owner");
        pricePerGas = _pricePerGas;
    }

    function validatePaymasterUserOp(
        UserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    ) external view override returns (bytes memory context, uint256 deadline) {
        // Check that the userOp has enough USDC allowance to pay for gas
        uint256 usdcRequired = (maxCost * pricePerGas) / 1e18;
        uint256 allowance = usdc.allowance(userOp.sender, address(this));
        require(allowance >= usdcRequired, "Insufficient USDC allowance");

        // Return a context that will be passed to postOp
        context = abi.encode(userOp.sender, usdcRequired);
        return (context, 0);
    }

    function postOp(
        PostOpMode mode,
        bytes calldata context,
        uint256 actualGasCost
    ) external override {
        require(msg.sender == address(entryPoint), "Only EntryPoint");

        (address sender, uint256 usdcCharged) = abi.decode(context, (address, uint256));

        // Charge the actual gas cost in USDC
        uint256 actualUSDC = (actualGasCost * pricePerGas) / 1e18;
        uint256 toCharge = mode == PostOpMode.postOpReverted ? usdcCharged : actualUSDC;

        usdc.safeTransferFrom(sender, address(this), toCharge);

        emit GasPaidWithUSDC(sender, actualGasCost, toCharge);
    }

    function withdrawETH() external {
        require(msg.sender == owner, "Only owner");
        (bool success, ) = owner.call{value: address(this).balance}("");
        require(success, "ETH transfer failed");
    }

    function withdrawUSDC(uint256 amount) external {
        require(msg.sender == owner, "Only owner");
        usdc.safeTransfer(owner, amount);
    }

    receive() external payable {}
}
```

The paymaster is a *simplified sketch*. A production implementation would add:

- **Oracle integration** for real-time gas price and USDC price (Chainlink, Pyth, Redstone)
- **Whitelisted senders** (only authorized agent wallets can use the paymaster)
- **Per-sender daily limits**
- **Slashing** for failed userOps (the paymaster can penalize senders that submit failing userOps)
- **Multi-token support** (USDC, USDT, DAI, etc.)
- **A separate signer for owner operations** (the owner key should be on a hardware wallet)

### 1.6 The Bundler Layer: MEV Protection and Reliable Inclusion

The bundler is the infrastructure that takes UserOperations (the ERC-4337 transaction format) and submits them to the chain. Bundlers are a competitive market with multiple providers. The key concerns for agent entities are:

- **MEV protection** — without protection, the agent's transactions can be front-run, sandwiched, or censored by MEV searchers
- **Reliable inclusion** — the agent's transactions need to land within a known time bound, especially for time-sensitive operations
- **Cost** — bundlers charge a fee, and the fee can vary significantly between providers
- **Privacy** — some bundlers leak the userOp's contents to the mempool, which can reveal the agent's intent

The leading bundler infrastructure:

- **Alchemy** — large bundler, integrated with the Alchemy RPC
- **Flashbots** — MEV-aware bundler with privacy and MEV rebate
- **Pimlico** — bundler with multiple chain support
- **Blast** — bundler with privacy features
- **Bloctract** — specialized bundler for AI agents with intent-based privacy

For high-value agent transactions, **Flashbots** is the recommended bundler due to its MEV protection. For low-value, high-volume agent transactions, **Pimlico** or **Alchemy** are recommended for cost.

### 1.7 The Chain Layer: Where the Agent Operates

The choice of chain (or chains) for the agent entity is strategic. The mid-2026 options:

| Chain | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **Ethereum L1** | Highest security, highest liquidity, most infrastructure | Highest gas, lowest throughput | High-value, low-frequency transactions (treasury management) |
| **Base** | Coinbase-backed, low gas, EVM-equivalent, growing ecosystem | Centralized sequencer, less mature | Production agent operations, US-based agents |
| **Optimism** | OP Stack, superchain vision, low gas | Centralized sequencer | Production agent operations |
| **Arbitrum** | Largest L2 by TVL, mature, Stylus for non-EVM | Centralized sequencer, complex | High-value agent operations |
| **Polygon** | Multiple chains (zkEVM, PoS, Miden), low gas | ZK-EVM still maturing | High-frequency agent operations |
| **zkSync** | ZK-rollup, low gas, native account abstraction | Smaller ecosystem | Account-abstraction-native agents |
| **Scroll** | ZK-rollup, EVM-equivalent | Smaller ecosystem | General agent operations |
| **Solana** | High throughput, low gas, growing agent ecosystem | Non-EVM, different wallet paradigm | High-frequency trading agents |
| **Avalanche** | Subnets, low gas | Smaller ecosystem | Subnet-deployed agent entities |
| **BNB Chain** | Low gas, large user base | More centralized | Consumer-facing agents |

For most agent entities in mid-2026, **Base** is the recommended default (Coinbase's L2, low gas, growing ecosystem, US regulatory clarity, EVM-equivalent). For high-value, low-frequency operations, **Ethereum L1** is recommended. For high-frequency, low-value operations, **Solana** is recommended.

The most sophisticated agent entities deploy on **multiple chains** simultaneously, using the same ECDSA validator to derive the same smart-account address on every chain (via deterministic CREATE2 deployment). This is a key feature of ERC-4337 and a major operational benefit.

---

## 2. Agent Identity: DIDs, VCs, and On-Chain Reputation

The agent's identity is its public face. The identity layer answers the question: *who is this agent, who says so, and what can it prove?*

### 2.1 The W3C DID Standard

A Decentralized Identifier (DID) is a W3C standard for self-sovereign identity. A DID is a string of the form `did:method:identifier`. The DID method defines how the DID is created, resolved, and updated. The leading DID methods in mid-2026:

| DID Method | Identifier | Resolution | Best For |
|------------|------------|------------|----------|
| **did:key** | The DID is derived from a public key | The DID is self-contained | Lightweight, single-use identity |
| **did:ethr** | The DID is an Ethereum address | Resolved via Ethereum JSON-RPC | On-chain agent identity |
| **did:web** | The DID is a web domain | Resolved via HTTPS | Web-based agent identity |
| **did:ion** | The DID is on Bitcoin via Sidetree | Resolved via ION nodes | Decentralized identity |
| **did:cheqd** | The DID is on the cheqd network | Resolved via cheqd | Privacy-preserving identity |
| **did:ens** | The DID is an ENS name | Resolved via ENS | Human-readable agent identity |
| **did:pkh** | The DID is a public key hash | Resolved via blockchain | Cross-chain identity |

For most agent entities, **did:ethr** or **did:ens** is the recommended DID method. The did:ethr method is simple, on-chain, and well-supported. The did:ens method adds a human-readable name (e.g., `agent.eth`) that is easier to work with.

A representative DID for an agent entity:

```
did:ethr:0x1234567890123456789012345678901234567890
did:ens:myagent.eth
did:web:myagent.example.com
did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
```

The DID document (the JSON document that describes the DID) includes the agent's public keys, authentication methods, and service endpoints:

```json
{
  "@context": "https://www.w3.org/ns/did/v1",
  "id": "did:ethr:0x1234567890123456789012345678901234567890",
  "verificationMethod": [
    {
      "id": "did:ethr:0x1234...7890#controller",
      "type": "EcdsaSecp256k1RecoveryMethod2020",
      "controller": "did:ethr:0x1234...7890",
      "ethereumAddress": "0x1234567890123456789012345678901234567890"
    }
  ],
  "authentication": [
    "did:ethr:0x1234...7890#controller"
  ],
  "service": [
    {
      "id": "did:ethr:0x1234...7890#agent-endpoint",
      "type": "AgentEndpoint",
      "serviceEndpoint": "https://agent.example.com/api"
    },
    {
      "id": "did:ethr:0x1234...7890#decision-registry",
      "type": "DecisionRegistry",
      "serviceEndpoint": "https://registry.example.com/api/decisions"
    }
  ]
}
```

### 2.2 Verifiable Credentials (VCs)

A Verifiable Credential (VC) is a W3C-standardized cryptographically signed attestation. A VC is a "claim" made by an "issuer" about a "subject." For agent entities, the most important VCs are:

- **Model Card VC** — issued by the model provider, attesting to the model's training data, intended use, and known limitations
- **Audit VC** — issued by an auditor, attesting to the agent's compliance with a standard (SOC 2, ISO 42001, EU AI Act, etc.)
- **Licensing VC** — issued by a licensor, attesting that the agent is licensed to use certain IP
- **Performance VC** — issued by a benchmark or a customer, attesting to the agent's performance on a defined task
- **Jurisdiction VC** — issued by a regulator or a legal authority, attesting that the agent is registered in a particular jurisdiction (e.g., a Wyoming AIDAO LLC)
- **Fiduciary VC** — issued by the AI fiduciary, attesting to their role and the entity's compliance with the relevant statute

A representative VC:

```json
{
  "@context": [
    "https://www.w3.org/2018/credentials/v1",
    "https://www.w3.org/2018/credentials/examples/v1"
  ],
  "type": ["VerifiableCredential", "ModelCardCredential"],
  "issuer": "did:web:anthropic.com",
  "issuanceDate": "2026-04-15T00:00:00Z",
  "credentialSubject": {
    "id": "did:ethr:0x1234...7890",
    "modelName": "Claude-Opus-4",
    "modelVersion": "4.1.0",
    "trainingDataCutoff": "2026-01-31",
    "intendedUse": "Autonomous agent operations",
    "knownLimitations": [
      "May hallucinate on factual queries",
      "May be vulnerable to prompt injection"
    ]
  },
  "proof": {
    "type": "EcdsaSecp256k1Signature2019",
    "created": "2026-04-15T00:00:00Z",
    "verificationMethod": "did:web:anthropic.com#controller",
    "proofPurpose": "assertionMethod",
    "jws": "eyJhbGciOiJFUzI1NksiLCJ0eXAiOiJKV1QifQ..."
  }
}
```

The leading VC infrastructure:

- **Trinsic** — a VC platform with a Studio, an Issuer API, and a Wallet
- **Indy / Aries** — the leading open-source VC stack
- **Veramo** — a JavaScript framework for VCs and DIDs
- **Microsoft Entra Verified ID** — Microsoft's enterprise VC platform
- **Dock** — a VC platform with a focus on portability

### 2.3 On-Chain Reputation: The Karma, Lens, and Farcaster Pattern

On-chain reputation is a different beast from VCs. VCs are *attestations* (someone says something about the agent); on-chain reputation is *history* (the agent's track record is recorded on-chain and aggregated into a score). The leading on-chain reputation systems for agents in mid-2026:

- **Karma** (by SHOWROOM) — an on-chain reputation system that aggregates the agent's transaction history, social graph, and attestations into a single score
- **Lens Protocol** — a decentralized social graph (used for human reputation, but adaptable to agents)
- **Farcaster** — a decentralized social protocol (similar to Lens)
- **Gitcoin Passport** — a Sybil-resistant identity aggregator (originally for humans, now used for agents)
- **Guild** — a credentialing and reputation system
- **Galxe** — a Web3 credentialing platform
- **Trusta** (formerly TrustToken) — a reputation scoring system

For agent entities, the most useful on-chain reputation signals are:

- **Transaction history** — the volume, frequency, and diversity of the agent's transactions
- **Counterparty diversity** — the number of unique counterparties the agent has transacted with
- **Slashing history** — has the agent been slashed for bad behavior? (relevant for staking agents)
- **Audit history** — has the agent been audited? by whom? what was the outcome?
- **Dispute history** — has the agent been involved in a Kleros or other arbitration? what was the outcome?
- **Social graph** — what other agents, humans, and DAOs is the agent connected to?

### 2.4 The ENS and CAIP-10 Standards

ENS (Ethereum Name Service) provides human-readable names (e.g., `myagent.eth`) that resolve to Ethereum addresses. CAIP-10 (Chain Agnostic Improvement Proposal 10) provides a chain-agnostic identifier format (e.g., `eip155:1:0x1234...7890` for Ethereum mainnet, `eip155:8453:0x1234...7890` for Base).

For an agent entity, the recommended naming convention is:

- **Primary name** — `myagent.eth` (or `myagent.base.eth` for Base)
- **CAIP-10** — `eip155:1:0x1234...7890` (and equivalents for other chains)
- **DID** — `did:ethr:0x1234...7890` (and equivalents)
- **Agent Card** — a JSON file at `https://myagent.example.com/.well-known/agent-card.json` that describes the agent

The combination of these identifiers gives the agent a rich, portable, verifiable identity that works across chains, services, and contexts.

---

## 3. Payment Infrastructure: How Agents Pay for Things

The agent needs to pay for things: inference, API calls, services, gas, and (eventually) other agents. The payment infrastructure in mid-2026 is dominated by three patterns.

### 3.1 The x402 Pattern (HTTP 402 + Stablecoins)

x402 is the pattern of using HTTP 402 "Payment Required" status codes to monetize API calls. The flow is:

1. Agent calls an API
2. The API returns 402 with a payment challenge (the price, the currency, the payment address)
3. The agent's paymaster or wallet pays the price in stablecoins
4. The agent's wallet signs a payment receipt
5. The agent retries the API call with the payment receipt
6. The API verifies the receipt and returns the requested data

x402 is the most common payment pattern for agent-to-API transactions in mid-2026. It is supported by:

- **Cloudflare** (x402 via Cloudflare Workers)
- **AWS** (via API Gateway with x402 support)
- **OpenAI** (x402 support in the API)
- **Anthropic** (x402 support in the API)
- **Pinata** (for IPFS payments)
- **Many others**

A representative x402 flow:

```
GET https://api.example.com/v1/expensive-call
↓
HTTP 402 Payment Required
WWW-Authenticate: x402 realm="example", 
                  amount="0.10", 
                  currency="USDC", 
                  address="0xPaymentAddress",
                  network="base"
↓
POST https://api.example.com/v1/expensive-call
X-Payment-Receipt: {"txHash":"0x...", "amount":"0.10", "currency":"USDC", "network":"base", "signature":"0x..."}
↓
HTTP 200 OK
{"data": "..."}
```

### 3.2 The L402 Pattern (Lightning + 402)

L402 is the Lightning Network + HTTP 402 pattern. It uses Bitcoin Lightning invoices to enable micropayments. The flow is similar to x402, but the payment is made over Lightning (a Bitcoin Layer 2), which enables sub-cent payments.

L402 is supported by:

- **Lightning Labs** (the leading L402 implementation)
- **Fedi** (a federated Bitcoin wallet that supports L402)
- **Alby** (a Lightning browser extension)
- **Synonym** (a Lightning-as-a-service provider)

L402 is the recommended payment pattern for high-frequency, low-value transactions (e.g., a research agent paying for web scraping, a data agent paying for weather data).

### 3.3 The Account Abstraction + Paymaster Pattern

For transactions on-chain (gas, smart-contract calls), the agent uses account abstraction + a paymaster. The agent's smart-contract wallet pays gas via a paymaster that accepts ERC-20 tokens. This is covered in Section 1.5 above.

---

## 4. Custody, Recovery, and Operational Security

The agent's wallet, identity, and reputation are valuable, and they need to be protected against loss, theft, and operational errors. The following are the recommended practices as of mid-2026.

### 4.1 Custody Models

The custody model determines who holds the agent's master keys. The leading options:

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Self-custody (AI fiduciary)** | The AI fiduciary holds the master key on a hardware wallet | Maximum control, no third-party risk | Single point of compromise (the AI fiduciary) |
| **Self-custody (multisig)** | The master key is split across multiple parties (3-of-5) | No single point of compromise | Operational complexity, coordination overhead |
| **MPC custody** | The master key is split using multi-party computation across multiple parties (e.g., Fireblocks, Anchorage) | High security, no single point of compromise | Custody fees, vendor risk |
| **Qualified custody** | A regulated qualified custodian (e.g., Anchorage, Coinbase Custody, Fidelity Digital Assets) holds the master key | Regulatory clarity, insurance, audited | Custody fees, jurisdictional limits, less operational flexibility |
| **DAO custody** | The master key is held by a DAO that votes on transactions | Decentralized, censorship-resistant | Coordination overhead, governance attacks |
| **Constitutional AI custody** | The master key is held by a "constitutional AI" that follows a written constitution | Aligned with the entity's mission | New, unproven, may not be recognized legally |

For most agent entities in mid-2026, the recommended custody model is a **3-of-5 multisig** with:

- 1 AI fiduciary (hardware wallet)
- 2 founder hardware wallets
- 2 independent guardian wallets (held by trusted third parties, e.g., a law firm or a custody provider)

The AI fiduciary's key is required for daily operations; the founder and guardian keys are required for major operations and recovery.

### 4.2 Recovery Mechanisms

Recovery mechanisms are critical because keys can be lost, stolen, or compromised. The leading recovery mechanisms:

- **Social recovery** — the agent's wallet has a set of "guardians" who can collectively recover the wallet if the AI fiduciary's key is lost. The guardians are typically the same as the multisig co-signers.
- **Time-locked recovery** — a recovery transaction can be initiated by any guardian, but it is delayed by a time lock (e.g., 7 days) during which the AI fiduciary can cancel it.
- **Multi-chain recovery** — the agent's wallet is deployed on multiple chains; if the master key is compromised on one chain, the same ECDSA validator can be used to recover on all chains.
- **Ink-covered recovery** (ZK-rollup specific) — the recovery transaction is rolled up via an ink-covered channel, providing privacy and MEV protection.
- **Jurisdictional recovery** — if the legal entity still exists, the AI fiduciary or the entity's board can petition a court to recover the wallet. The court order is then executed by the multisig guardians.

The most robust agent wallet architecture combines all five recovery mechanisms: social recovery as the primary, time-locked recovery as the secondary, multi-chain as the tertiary, ink-covered as the operational layer, and jurisdictional as the last resort.

### 4.3 Operational Security Practices

The following are the recommended operational security practices for agent wallets:

- **Hardware wallets for all human-held keys** (Ledger, Trezor, GridPlus, Keystone)
- **HSM-backed MPC for high-value keys** (AWS Nitro Enclaves, Azure Confidential Computing, Google Cloud HSM)
- **Cold storage for the bulk of the treasury** (90%+ in cold storage, 10% in hot wallet for daily operations)
- **Air-gapped signing for high-value transactions** (sign the transaction on an air-gapped device, broadcast on a separate device)
- **Multi-party approval for any change to the multisig configuration** (adding/removing signers, changing the threshold, upgrading the smart contract)
- **Continuous monitoring of the wallet** (alerts on any non-whitelisted transaction, alerts on any attempt to change the multisig configuration)
- **Regular security audits** (at least annually, by a reputable auditor)
- **Incident response plan** (a documented plan for handling wallet compromise, key loss, or other incidents)
- **Insurance** (AI fiduciary insurance, cyber insurance, custody insurance — see `24-AI-Agent-Autonomy-Accountability/02-Operator-Liability-and-Duty-of-Care.md`)

---

## 5. The Multi-Chain Agent Wallet Pattern

The most sophisticated agent entities in 2026 operate on multiple chains simultaneously. The multi-chain pattern uses the same ECDSA validator (master key) to derive the same smart-account address on every chain, and the same session keys to authorize transactions on all chains. This is a powerful pattern because:

- **Single key, multiple chains** — the AI fiduciary holds one key, and the agent can transact on every chain
- **Unified identity** — the agent's identity is consistent across chains
- **Unified reputation** — the agent's transaction history is portable across chains
- **Unified treasury management** — the entity can rebalance its treasury across chains
- **Cross-chain messaging** — the agent can send messages between chains via protocols like LayerZero, Wormhole, Axelar, or Chainlink CCIP

The leading multi-chain agent wallet infrastructure:

- **Safe (formerly Gnosis Safe)** — deployed on Ethereum, Base, Optimism, Arbitrum, Polygon, BNB Chain, Avalanche, Gnosis Chain, and others
- **ZeroDev** — multi-chain kernels, with the same ECDSA validator on every chain
- **Particle Network** — a multi-chain smart-account infrastructure
- **Okto** — a multi-chain smart-account infrastructure
- **Rhino.fi** — a multi-chain smart-account infrastructure

A representative multi-chain deployment for an agent entity:

```
Master key (AI fiduciary, hardware wallet)
├── Ethereum L1 (Safe)
├── Base (Safe)
├── Optimism (Safe)
├── Arbitrum (Safe)
├── Polygon (Safe)
├── BNB Chain (Safe)
├── Avalanche (Safe)
├── Solana (Squads)
├── Bitcoin (Xverse)
└── Ink (Safe)

Session key (agent runtime, ephemeral)
├── Same authorization on all chains
├── Per-chain rate limits
├── Per-chain value limits
└── Per-chain scope limits
```

---

## 6. The Trust Stack: How Counterparties Verify the Agent

A counterparty dealing with the agent needs to verify:

1. **Who is the agent?** — the agent's identity (DID, ENS, CAIP-10)
2. **Who says the agent is who it claims to be?** — the agent's VCs (Model Card VC, Audit VC, Jurisdiction VC, Fiduciary VC)
3. **Is the agent a legal person?** — the agent's legal status (e.g., a Wyoming AIDAO LLC, on the Wyoming public registry)
4. **Is the agent authorized to act?** — the agent's authorization (the AI fiduciary's signature, the decision registry entry, the on-chain signature)
5. **What is the agent's reputation?** — the agent's on-chain reputation (transaction history, audit history, dispute history)
6. **What is the agent's risk profile?** — the agent's insurance coverage, the agent's compliance certifications

The "trust stack" is the combination of all six. A counterparty can ask the agent to present a "trust packet" that includes all six elements:

```json
{
  "identity": {
    "did": "did:ethr:0x1234...7890",
    "ens": "myagent.eth",
    "caip10": "eip155:1:0x1234...7890",
    "agentCard": "https://myagent.example.com/.well-known/agent-card.json"
  },
  "credentials": {
    "modelCard": { "issuer": "did:web:anthropic.com", "type": "ModelCardCredential" },
    "audit": { "issuer": "did:web:trailofbits.com", "type": "AuditCredential", "score": "A" },
    "jurisdiction": { "issuer": "did:web:wyo.gov", "type": "JurisdictionCredential", "entity": "MyAgent AIDAO LLC" },
    "fiduciary": { "issuer": "did:web:wyo.gov", "type": "FiduciaryCredential", "fiduciary": "Jane Q. Public" }
  },
  "legalStatus": {
    "entity": "MyAgent AIDAO LLC",
    "registry": "https://wyoming-ai-registry.gov/lookup/2026-001-ABCDEF",
    "aiFiduciary": "Jane Q. Public",
    "controllingAI": "Claude-Opus-4 (SHA256: abc123...)"
  },
  "authorization": {
    "aiFiduciarySignature": "0x...",
    "decisionRegistryEntry": "0xDEF...",
    "smartContractSignature": "0x..."
  },
  "reputation": {
    "karma": 850,
    "transactions": 12453,
    "counterparties": 234,
    "audits": 3,
    "disputes": 0
  },
  "riskProfile": {
    "insurance": { "carrier": "Coalition", "limit": "$10M", "type": "AI Liability" },
    "compliance": ["SOC 2 Type II", "ISO 42001", "EU AI Act High-Risk"]
  }
}
```

The trust packet can be presented to a counterparty before a contract is signed. The counterparty can verify each element independently. This is the "know your agent" (KYA) pattern, analogous to the "know your customer" (KYC) pattern in traditional finance.

---

## 7. The Future of Agent Wallets: 2027 and Beyond

The agent wallet landscape is moving fast. The following are the most likely developments over 2027-2030:

- **Native agent wallets at the protocol level.** L2s and alt-L1s are starting to ship native agent wallet support. Base, Optimism, and Arbitrum are all working on "agent accounts" that are smart-contract wallets by default, with built-in session keys, paymasters, and recovery.
- **Intent-based architectures.** The next generation of agent wallets will be intent-based, not transaction-based. The agent will sign an "intent" (e.g., "I want to convert 100 USDC to ETH at the best rate within the next hour"), and the protocol will find the best execution path. Rhinestone, UniswapX, and 1inch Fusion are the leading intent-based protocols.
- **Privacy-preserving wallets.** ZK-wrapped wallets, stealth addresses, and confidential transactions are coming to agent wallets. The leading implementations are Aztec, Railgun, and Zama.
- **Cross-chain intent settlement.** The next generation of intent-based protocols will be cross-chain. The agent will sign an intent that spans multiple chains, and the protocol will find the best cross-chain execution path. LayerZero, Wormhole, and Chainlink CCIP are the leading cross-chain messaging protocols.
- **Hardware wallets for agents.** Hardware wallets designed specifically for agents are emerging. They are designed to be air-gapped, to support session keys natively, and to integrate with the agent's runtime. The leading implementations are GridPlus (Lattice1), Keystone, and a new entrant (still in stealth).
- **Regulated agent wallets.** As agent entities become more important, regulated wallet providers are emerging. Anchorage, Coinbase Custody, and Fidelity Digital Assets have all signaled that they will offer agent-specific custody products.

These developments will be covered in detail in `05-Future-of-Agent-Personhood.md`.

---

## 8. Summary and Key Takeaways

The technical substrate of agent personhood — the wallet, the identity, the reputation, the payment rails, the custody, the recovery — is what makes the legal form real. A well-formed Wyoming AIDAO LLC with a poorly-designed wallet is a paper tiger; a well-designed wallet can survive even a poorly-formed legal entity.

**Key takeaways:**

1. **The wallet is a stack, not a key.** A modern agent wallet is a layered stack of cryptographic, smart-contract, and operational primitives. The master key is held by the AI fiduciary (in a multisig); the session keys are held by the agent runtime (with bounded permissions); the smart-contract wallet enforces the limits.
2. **The session key is the bridge.** The session key layer is the bridge between the agent's runtime and the wallet. The session key is what allows the agent to act autonomously without compromising the master key.
3. **The paymaster is the treasury's gas pump.** The paymaster abstracts gas payment, allowing the agent to pay for transactions in stablecoins. This is the bridge between the entity's stablecoin treasury and the chain's gas token.
4. **The identity is multi-faceted.** The agent's identity is a combination of DID, ENS, CAIP-10, VCs, and on-chain reputation. The combination gives the agent a rich, portable, verifiable identity.
5. **The custody is multi-party.** The master key is held by a multisig (typically 3-of-5). No single party can drain the entity.
6. **The recovery is layered.** Recovery mechanisms are layered: social recovery, time-locked recovery, multi-chain recovery, ink-covered recovery, jurisdictional recovery.
7. **The trust packet is the KYA.** The "know your agent" (KYA) pattern is the agent-entity analog of KYC. The trust packet is the standardized format for presenting the agent's identity, credentials, legal status, authorization, reputation, and risk profile to a counterparty.

The next document in this category — `04-Agent-to-Agent-Contracts-and-Autonomous-Markets.md` — covers the market layer: how agent entities actually transact with each other, the contracts they sign, the marketplaces they participate in, and the legal and economic questions that arise.

---

*This document is part of the AI Knowledge Library category 27 — AI Agent Legal Entities & DAO Governance. All file paths in cross-references are relative to the library root.*
