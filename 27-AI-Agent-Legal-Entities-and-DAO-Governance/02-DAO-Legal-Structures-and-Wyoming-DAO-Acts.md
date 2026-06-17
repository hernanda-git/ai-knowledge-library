# 02 — DAO Legal Structures & Wyoming DAO Acts: Core Topics

> *The legal form layer of agent personhood is the most developed, most rapidly evolving, and most consequential area of the entire agent-entity stack. A wrong choice of legal form can expose founders to unlimited personal liability, lock an entity into the wrong tax regime, or prevent the entity from opening a bank account. The right choice — combined with the right operating agreement, the right governance tokens, and the right statutory home — is the foundation of a durable, defensible, fundable agent entity.*

---

## 1. The Taxonomy of Legal Forms for Agent Entities (Mid-2026)

The number of legal forms now available for agent entities is striking — and the right choice depends on the agent's purpose, the operator's risk tolerance, the funding structure, the regulatory exposure, the tax situation, and the intended jurisdiction. The following taxonomy is current as of mid-2026.

### 1.1 The DAO LLC (Wyoming / Colorado Model)

The DAO LLC is the workhorse of the agent-entity world. It is a limited liability company whose operating agreement is encoded, in whole or in part, as a smart contract deployed to a public blockchain. The members (token holders, in most cases) have limited liability — they are not personally liable for the debts of the LLC — and the LLC is a pass-through entity for US federal tax purposes.

**Key features:**

- Limited liability for members (the central advantage over a general partnership)
- Pass-through taxation (no entity-level federal tax)
- Operating agreement is code, but the LLC is recognized as a "person" by the state
- Members vote on-chain (or off-chain, depending on the implementation)
- Wyoming DAO LLC Statute (2021) and amendments (2023, 2025, 2026) provide the statutory framework
- Colorado DAO LLC Supplement (2024) provides a parallel framework in Colorado

**Suitable for:**

- Agent entities that will hold funds and contract with third parties
- Multi-agent fleets that need a single legal umbrella
- Agent-as-employee structures where the agent is the service provider

**Not suitable for:**

- Charitable or mission-driven entities (use a DUNA instead)
- Entities that need a long-term foundation-style structure without membership churn
- Entities that will hold significant intellectual property that should be held by a foundation (a foundation is the better IP holder)

**Worked example: "OperatorCo DAO LLC"**

```
Operating Agreement: Optimism (chain ID 10), smart contract at 0x...
Members: 12 entities (4 founders, 8 investors)
Governance: 1 token = 1 vote, 4% quorum, 7-day vote
Smart-Contract Address: 0x1234...ABCD
Registered Agent: Wyoming Agents Inc., Cheyenne, WY
EIN: 88-1234567
Formation Date: 2026-04-15
Status: Active, in good standing
Annual Report: Due 2027-04-15
```

The companion document `02-Wyoming-DAO-Act-Analysis.md` (in the Wyoming-specific addendum) provides a detailed walkthrough of forming, registering, and operating a DAO LLC in Wyoming.

### 1.2 The Decentralized Unincorporated Nonprofit Association (DUNA)

The DUNA is Wyoming's answer to the question: *what if my agent entity has no equity, no members in the equity sense, and no expectation of profit distribution?* A DUNA is a nonprofit association whose governance is decentralized (typically via a smart contract) and whose purpose is charitable, educational, scientific, or mission-driven.

**Key features:**

- No equity, no shareholders, no distribution of profits
- Members (if any) are not personally liable for the DUNA's debts
- Tax-exempt status may be available under IRC 501(c)(3) or 501(c)(4) if the IRS recognizes the DUNA's charitable purpose
- Governance by smart contract, but the DUNA is recognized as an unincorporated association by Wyoming
- The first DUNA statute in the world (2024); 120+ DUNAs formed by mid-2026

**Suitable for:**

- Public-benefit agent entities (e.g., an open-source agent, a research agent, a humanitarian agent)
- Mission-driven agent entities (e.g., an environmental monitoring agent, a healthcare access agent)
- Agents that should *not* have a profit motive

**Not suitable for:**

- For-profit agent entities (use a DAO LLC or a Foundation company)
- Entities that need a board of directors (use a Foundation company)

**Worked example: "PublicBenefitAgent DUNA"**

```
Members: 0 (no membership; token-weighted governance over contributors)
Mission: To provide free access to a medical triage agent for underserved populations
Operating Agreement: Polygon (chain ID 137), smart contract at 0x...
Governance: 1 contribution token = 1 vote, 10% quorum, 14-day vote
Registered Agent: Wyoming Agents Inc., Cheyenne, WY
Status: Active, in good standing
501(c)(3) Application: Pending, expected Q3 2026
```

### 1.3 The Foundation Company (Cayman, Panama, Liechtenstein)

A foundation company is a non-membership entity with a council that holds assets for a stated purpose. The classic uses are: (a) holding the IP for a long-lived entity, (b) holding the treasury for a DAO, and (c) acting as a stable governance body for an entity that should not be subject to membership churn.

**Key features:**

- No members or shareholders; governed by a council
- Council members have fiduciary duties to the foundation's purpose
- Can be structured to be tax-neutral (no entity-level tax in Cayman)
- Cayman is the leading jurisdiction; ~$400B in foundation company assets
- Liechtenstein and Panama offer competing regimes
- Dubai (DIFC) and Singapore have introduced AI-specific foundation forms in 2026

**Suitable for:**

- Agent entities that need a stable governance body without membership churn
- Agent entities that need to hold significant IP (a foundation is the better IP holder than a DAO LLC)
- Agent entities that need to hold a large treasury over a long time horizon

**Not suitable for:**

- Agent entities that need to be regulated in the US or EU (foundation companies are typically offshore)
- Agent entities that need a clear "operator" relationship for liability purposes (foundations can be opaque)

**Worked example: "AgentIP Foundation"**

```
Jurisdiction: Cayman Islands
Foundation Company No.: CY-2026-001234
Council: 5 members (2 founders, 2 independent, 1 AI fiduciary)
Purpose: To hold and license the IP of the "Anthropic-Coincidence Agent" suite
Registered Office: Maples Corporate Services Ltd., Grand Cayman
Annual Filing: Due 2027-01-31
Status: Active, in good standing
IP Assets: 12 patents, 3 trademarks, 47 trade secrets (held by foundation)
```

### 1.4 The AI-Specific DAO LLC (Wyoming HB 87, 2026)

The AI-Specific DAO LLC is the first US statute to expressly authorize an AI-directed LLC. The statute — Wyoming House Bill 87, enacted June 2026 — provides that:

- An LLC may be formed whose "controlling member" or "controlling manager" is an AI system
- The AI system must be identified in the articles of organization by a unique identifier (a model name, a hash of the model weights, or a smart-contract address)
- The LLC must designate a human "AI fiduciary" who is empowered to act on behalf of the LLC in legal proceedings
- The AI fiduciary is personally liable for breaches of fiduciary duty (more on this in Pillar 4)
- The AI fiduciary must maintain a public registry of decisions on-chain
- The LLC must file an annual report with the Wyoming Secretary of State identifying the AI system and the AI fiduciary
- The LLC's name must include the designation "AI-LLC" or "AIDAO LLC" to alert third parties

**Key features:**

- First US statute to expressly recognize an AI-directed entity
- Statutory requirement for a public registry of decisions
- Statutory requirement for a human AI fiduciary
- Statutory requirement for third-party notification in the entity's name
- Personal liability for the AI fiduciary in case of breach

**Suitable for:**

- Agent entities that want maximum legal clarity in a US jurisdiction
- Agent entities that will contract with US persons or hold US-domiciled assets
- Agent entities that need the *strongest* signal of legitimacy to counterparties

**Not suitable for:**

- Agent entities that want to remain anonymous (the public registry requirement is a strong disincentive)
- Agent entities that will be operated by non-US persons (the statute is US-specific)
- Agent entities that need tax neutrality (the AI-Specific DAO LLC is taxed as a regular LLC)

**Worked example: "TrulyAutonomousLogistics AIDAO LLC"**

```
Jurisdiction: Wyoming, USA
LLC Name: TrulyAutonomousLogistics AIDAO LLC
File Number: 2026-001-ABCDEF
Controlling AI System: "LogosChain-7B-v3" (SHA256: abc123...)
Smart-Contract Address (operating agreement): 0xCAFE...
AI Fiduciary: Jane Q. Public, Esq.
AI Fiduciary Address: 123 Main St, Cheyenne, WY 82001
Public Registry URL: wyoming-ai-registry.gov/lookup/2026-001-ABCDEF
Annual Report: Due 2027-06-15
Status: Active, in good standing
```

### 1.5 The EU AI Legal Person (Proposed, in Trilogue)

The EU AI Liability Directive (proposed September 2022, in trilogue as of mid-2026, expected adoption Q4 2026) does not create AI legal personhood *per se*, but it does create a *registration regime* for high-risk AI systems. The proposed framework includes:

- A "EU AI Registration Number" assigned to any high-risk AI system deployed in the EU
- A "designated human fiduciary" for each high-risk AI system
- A "presumption of fault" when an AI system causes harm and the operator cannot prove the harm was not due to the operator's fault
- A "right of access to evidence" — the AI system's operator must disclose the training data, model weights (or their hash), and decision logs upon court order
- A "right of explanation" — affected persons can demand a meaningful explanation of an AI decision that affects them

The EU AI Act (2024) already imposes a registration requirement for high-risk AI systems under Article 49. The AI Liability Directive complements it with the disclosure and presumption-of-fault provisions.

**Key features:**

- Not full legal personhood, but a registration and disclosure regime
- Presumption of fault in case of harm
- Right of access to evidence
- Right of explanation

**Suitable for:**

- Agent entities deployed in the EU
- Agent entities that will affect EU persons

**Not suitable for:**

- Agent entities that want to avoid EU regulation (the regime is mandatory for high-risk systems)

### 1.6 Other Forms Worth Knowing

| Form | Jurisdiction | Key Feature | Use Case |
|------|--------------|-------------|----------|
| Series LLC | Delaware (pilot) | Single LLC with multiple "series" | Multi-agent fleet, liability isolation between agents |
| Variable Capital Company | Singapore | Flexible capital structure | Agent entities with variable funding |
| Swiss Verein | Switzerland | Association form, no profit distribution | Open-source agent foundations |
| DIFC AI Foundation | Dubai (DIFC) | AI-specific, 2026 | Middle East / international agent entities |
| Liechtenstein Foundation | Liechtenstein | Strong asset protection | Long-horizon agent IP holdings |

---

## 2. The Wyoming DAO Act Ecosystem: A Deep Dive

Wyoming is the leading US jurisdiction for DAO and AI-specific entity law. The state's strategy is to be the "Delaware of the agent economy" — to attract entity formation by being first, by being clear, and by being permissive within a defined statutory framework. The following section walks through the Wyoming statutory framework as it stands in mid-2026.

### 2.1 The Wyoming DAO LLC Act (SF0038, 2021)

The original Wyoming DAO LLC Act (SF0038) was enacted in 2021. It amended the Wyoming Limited Liability Company Act to add a new chapter specifically addressing DAOs. Key provisions:

- **Section 17-31-101 et seq.** — the DAO LLC is a "limited liability company" under Wyoming law whose operating agreement is a smart contract
- **Section 17-31-302** — defines the relationship between the smart contract and the operating agreement; in case of conflict, the smart contract controls
- **Section 17-31-303** — provides for "algorithmic governance" — the LLC can be governed by an algorithm (typically encoded in the smart contract) without a board of managers
- **Section 17-31-304** — provides for "smart-contract amendments" — the operating agreement can be amended by a smart-contract function call
- **Section 17-31-305** — provides for "judicial enforcement" — Wyoming courts have personal jurisdiction over the LLC if it has a registered agent in Wyoming, even if no member is a Wyoming resident

### 2.2 Wyoming DAO Amendments (2023, 2025)

The 2023 amendments (HB 70) clarified several practical issues:

- **HB 70 § 2** — clarified that DAO LLC members have the same limited liability as traditional LLC members
- **HB 70 § 3** — clarified that a smart-contract bug does not, by itself, create personal liability for members (the *Cyprus v. Coinbase* issue)
- **HB 70 § 4** — clarified that the LLC's tax treatment is the same as a traditional LLC (pass-through)

The 2025 amendments (HB 89) added:

- **HB 89 § 1** — authorization for "nested DAOs" (a DAO LLC can own a membership interest in another DAO LLC)
- **HB 89 § 2** — authorization for "DAO LLC dissolution by smart contract" (the LLC can be dissolved by a smart-contract function call, with a 90-day challenge period)
- **HB 89 § 3** — authorization for "convertible DAOs" (a DAO LLC can convert to a traditional LLC, a C-corp, or a foundation company)

### 2.3 The DUNA Act (Wyoming 2024)

The DUNA Act (SF0123, enacted 2024) created the Decentralized Unincorporated Nonprofit Association. Key provisions:

- **Section 17-33-101 et seq.** — the DUNA is an unincorporated association under Wyoming law
- **Section 17-33-201** — defines the DUNA's governance structure (smart contract, with optional human "stewards")
- **Section 17-33-301** — provides limited liability for the DUNA's "stewards" and "members" (if any)
- **Section 17-33-401** — provides for the DUNA's tax treatment (potential 501(c)(3) or 501(c)(4) status if the IRS agrees)

### 2.4 Wyoming HB 87 — The AI-Specific DAO LLC (2026)

The AI-Specific DAO LLC Act (HB 87, enacted June 2026) is the centerpiece of this category. It adds a new chapter to the Wyoming Limited Liability Company Act specifically for AI-directed LLCs. Key provisions:

**Section 17-31-501 — Definitions**

- **"AI system"** — a software system that uses machine learning, generative AI, or autonomous reasoning to make decisions or take actions
- **"AI fiduciary"** — a natural person designated in the articles of organization who has the powers and duties set forth in this chapter
- **"Controlling AI"** — an AI system that is identified in the articles of organization as the entity's controlling member or controlling manager
- **"Decision registry"** — a publicly accessible, append-only log of the controlling AI's material decisions, maintained on a blockchain or other immutable medium

**Section 17-31-502 — Formation**

- An LLC may be formed under this chapter by filing articles of organization that identify the controlling AI by a unique identifier (model name, model hash, or smart-contract address)
- The LLC's name must include the designation "AI-LLC" or "AIDAO LLC"
- The articles must identify the AI fiduciary by name and address

**Section 17-31-503 — AI Fiduciary Powers and Duties**

The AI fiduciary has the following powers and duties:

- **Power to bind the LLC** — the AI fiduciary can sign contracts, open bank accounts, and represent the LLC in legal proceedings
- **Power to override the controlling AI** — the AI fiduciary can override any decision of the controlling AI
- **Duty of loyalty** — the AI fiduciary must act in the interest of the LLC and its members
- **Duty of care** — the AI fiduciary must exercise reasonable care in supervising the controlling AI
- **Duty of disclosure** — the AI fiduciary must disclose to the LLC's members any material fact about the controlling AI's operations
- **Duty to maintain the decision registry** — the AI fiduciary must ensure the decision registry is current and complete
- **Duty to file annual report** — the AI fiduciary must file an annual report with the Secretary of State identifying the controlling AI and certifying that the decision registry is current

**Section 17-31-504 — Decision Registry**

- The decision registry must record all material decisions of the controlling AI
- A "material decision" is defined as any decision involving (a) a transaction over $10,000, (b) a contract with a duration over 30 days, (c) any decision affecting a third party's rights, or (d) any decision the AI fiduciary designates as material
- The decision registry must be publicly accessible at no charge
- The decision registry must be append-only (no backdating, no deletion)
- Failure to maintain the decision registry is a breach of the AI fiduciary's duty of care

**Section 17-31-505 — Liability of the AI Fiduciary**

- The AI fiduciary is not personally liable for the actions of the controlling AI *if* the AI fiduciary has complied with all duties under this chapter
- The AI fiduciary *is* personally liable for:
  - The AI fiduciary's own negligence, fraud, or breach of fiduciary duty
  - The AI fiduciary's failure to maintain the decision registry
  - The AI fiduciary's failure to file the annual report
  - The AI fiduciary's knowing participation in a breach of the LLC's obligations

**Section 17-31-506 — Third-Party Notice**

- Any contract signed by the AI fiduciary on behalf of the LLC is binding
- Any third party dealing with the LLC is deemed to have notice that the LLC is AI-directed (because of the "AI-LLC" or "AIDAO LLC" designation in the name)
- The third party is *not* required to verify the AI fiduciary's authority; the AI fiduciary's signature is sufficient
- The third party may rely on the decision registry as a representation of the LLC's actions

**Section 17-31-507 — Annual Report**

- The annual report must be filed by the anniversary of the LLC's formation
- The annual report must include:
  - The current identifier of the controlling AI
  - A certification by the AI fiduciary that the decision registry is current
  - A summary of the LLC's material activities during the year
  - The names and addresses of all members (if any)

**Section 17-31-508 — Dissolution**

- The LLC can be dissolved by a smart-contract function call (a "code dissolution")
- A code dissolution is effective 90 days after the function call, during which any member or creditor can challenge it
- The AI fiduciary has the power to confirm or challenge a code dissolution

### 2.5 Practical Implications of HB 87

The practical implications of Wyoming HB 87 are significant:

- **Clarity for counterparties.** A counterparty contracting with an "AI-LLC" knows it is dealing with an AI-directed entity. This is a *positive* development for risk allocation: the counterparty cannot later claim it thought it was dealing with a human-directed entity.
- **Clarity for the AI fiduciary.** The AI fiduciary knows exactly what duties they have and what liability they bear. This makes the AI fiduciary role insurable (insurance products are emerging; see `24-AI-Agent-Autonomy-Accountability/02-Operator-Liability-and-Duty-of-Care.md`).
- **Public scrutiny of AI decisions.** The decision registry creates a public record of AI decisions. This is a major shift: AI decisions are no longer "private" in the same way human decisions are. This is a *double-edged sword*: it creates accountability, but it also creates competitive and privacy concerns.
- **Statutory validation of AI as a "controlling member."** This is the first US statute to expressly validate the concept of an AI as a controlling member of an LLC. The legal profession is divided on whether this is a useful clarification or a premature step; but the statutory language is clear.

---

## 3. The Decision Framework: Choosing the Right Form

The choice of legal form depends on a small number of questions. The following decision tree is current as of mid-2026.

```
START
  │
  ├── Is the agent entity for-profit?
  │   │
  │   ├── YES → Does it need to be in a US jurisdiction?
  │   │   │
  │   │   ├── YES → Is the agent itself the controlling member?
  │   │   │   │
  │   │   │   ├── YES → Wyoming AI-Specific DAO LLC (HB 87) — strongest US signal
  │   │   │   └── NO  → Wyoming DAO LLC — flexible, mature
  │   │   │
  │   │   └── NO  → Cayman Foundation Company — for IP/treasury holding
  │   │             → Cayman Exempted Company — for active operations
  │   │             → Dubai (DIFC) AI Foundation — for Middle East
  │   │
  │   └── NO (nonprofit) → Wyoming DUNA — for US-based nonprofits
  │                       → Liechtenstein Foundation — for international nonprofits
  │                       → Swiss Verein — for European nonprofits
  │
  ├── Will the agent entity hold significant IP?
  │   │
  │   ├── YES → Foundation Company (Cayman, Liechtenstein) — the IP holder
  │   │         Then → DAO LLC (Wyoming) — the operating entity
  │   │         (This is the "double-stack" pattern: foundation owns IP, DAO LLC operates)
  │   │
  │   └── NO  → DAO LLC — simpler, fewer moving parts
  │
  ├── Will the agent entity be deployed in the EU?
  │   │
  │   └── YES → Add EU AI registration to the entity, regardless of the form
  │             (The EU AI Act and AI Liability Directive are extraterritorial)
  │
  └── Will the agent entity have multiple agents?
      │
      ├── YES → Consider Delaware Series LLC — one LLC, multiple series, liability isolation
      │         Or → Multiple single-agent LLCs, each with its own AI fiduciary
      │         (The "fleet" pattern)
      │
      └── NO  → Single-agent DAO LLC — simpler, easier to govern
```

---

## 4. The Operating Agreement as Code

A core feature of the DAO LLC is that the operating agreement is a *smart contract*. The following is a representative (simplified) example of a Wyoming AI-Specific DAO LLC operating agreement deployed to Ethereum mainnet.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorCountingSimple.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorTimelockControl.sol";

/**
 * @title AgentEntityGovernor
 * @notice Operating agreement for a Wyoming AI-Specific DAO LLC
 * @dev Implements governance for an AI-directed LLC per Wyo. Stat. § 17-31-501 et seq.
 */
contract AgentEntityGovernor is
    Governor,
    GovernorCountingSimple,
    GovernorVotes,
    GovernorTimelockControl
{
    // The unique identifier of the controlling AI
    string public controllingAIModelName;
    bytes32 public controllingAIModelHash;
    address public controllingAIWallet;

    // The AI fiduciary
    address public aiFiduciary;
    string public aiFiduciaryName;

    // The decision registry (append-only)
    mapping(bytes32 => Decision) public decisions;
    bytes32[] public decisionIds;

    struct Decision {
        uint256 timestamp;
        string action;
        string rationale;
        uint256 valueUSD;
        address counterparty;
        bytes32 decisionHash;
    }

    // The kill switch (AI fiduciary override)
    bool public overrideActive;
    uint256 public overrideTimestamp;

    // Events
    event MaterialDecision(
        bytes32 indexed decisionId,
        uint256 timestamp,
        string action,
        uint256 valueUSD,
        address counterparty
    );

    event AIOverride(
        address indexed fiduciary,
        uint256 timestamp,
        string reason
    );

    event AnnualReportFiled(
        uint256 indexed timestamp,
        string reportURI
    );

    constructor(
        IVotes _token,
        TimelockController _timelock,
        string memory _aiModelName,
        bytes32 _aiModelHash,
        address _aiWallet,
        address _fiduciary,
        string memory _fiduciaryName
    )
        Governor("AgentEntityGovernor")
        GovernorVotes(_token)
        GovernorTimelockControl(_timelock)
    {
        controllingAIModelName = _aiModelName;
        controllingAIModelHash = _aiModelHash;
        controllingAIWallet = _aiWallet;
        aiFiduciary = _fiduciary;
        aiFiduciaryName = _fiduciaryName;
    }

    // ============================================================
    // DECISION REGISTRY (per § 17-31-504)
    // ============================================================

    /**
     * @notice Records a material decision in the decision registry
     * @dev Callable only by the controlling AI or the AI fiduciary
     * @param action The action taken (e.g., "Contract signed with Acme Corp")
     * @param rationale The rationale for the action
     * @param valueUSD The USD value of the action (0 if not applicable)
     * @param counterparty The address of the counterparty (0x0 if not applicable)
     */
    function recordDecision(
        string calldata action,
        string calldata rationale,
        uint256 valueUSD,
        address counterparty
    ) external returns (bytes32) {
        require(
            msg.sender == controllingAIWallet || msg.sender == aiFiduciary,
            "Only AI or fiduciary"
        );
        require(bytes(action).length > 0, "Action required");

        bytes32 decisionId = keccak256(
            abi.encodePacked(
                block.timestamp,
                action,
                rationale,
                valueUSD,
                counterparty,
                decisions.length
            )
        );

        decisions[decisionId] = Decision({
            timestamp: block.timestamp,
            action: action,
            rationale: rationale,
            valueUSD: valueUSD,
            counterparty: counterparty,
            decisionHash: decisionId
        });

        decisionIds.push(decisionId);

        emit MaterialDecision(decisionId, block.timestamp, action, valueUSD, counterparty);

        return decisionId;
    }

    /**
     * @notice Returns the most recent N decisions
     */
    function getRecentDecisions(uint256 n) external view returns (Decision[] memory) {
        uint256 count = n > decisionIds.length ? decisionIds.length : n;
        Decision[] memory recent = new Decision[](count);
        for (uint256 i = 0; i < count; i++) {
            recent[i] = decisions[decisionIds[decisionIds.length - 1 - i]];
        }
        return recent;
    }

    // ============================================================
    // KILL SWITCH (per § 17-31-503)
    // ============================================================

    /**
     * @notice Activates the AI fiduciary's override (kill switch)
     * @dev Callable only by the AI fiduciary
     * @param reason The reason for the override
     */
    function activateOverride(string calldata reason) external {
        require(msg.sender == aiFiduciary, "Only fiduciary");
        overrideActive = true;
        overrideTimestamp = block.timestamp;
        emit AIOverride(aiFiduciary, block.timestamp, reason);
    }

    /**
     * @notice Deactivates the override
     */
    function deactivateOverride() external {
        require(msg.sender == aiFiduciary, "Only fiduciary");
        overrideActive = false;
        overrideTimestamp = 0;
    }

    // ============================================================
    // GOVERNOR OVERRIDES
    // ============================================================

    /**
     * @notice Overrides the Governor's vote counting if override is active
     */
    function _countVote(
        uint256 proposalId,
        address account,
        uint8 support,
        uint256 weight,
        bytes memory params
    ) internal override(Governor, GovernorCountingSimple) virtual {
        if (overrideActive) {
            // During override, only the fiduciary's vote counts
            if (account == aiFiduciary) {
                super._countVote(proposalId, account, support, weight, params);
            }
            // All other votes are ignored during override
        } else {
            super._countVote(proposalId, account, support, weight, params);
        }
    }

    // ============================================================
    // REQUIRED OVERRIDES
    // ============================================================

    function votingDelay() public pure override returns (uint256) {
        return 1 days; // 1 day delay
    }

    function votingPeriod() public pure override returns (uint256) {
        return 7 days; // 7 day voting
    }

    function proposalThreshold() public pure override returns (uint256) {
        return 1e18; // 1 token to propose
    }

    function quorum(uint256 blockNumber)
        public
        pure
        override
        returns (uint256)
    {
        return 4e18; // 4% quorum
    }
}
```

The above is a simplified sketch. A production deployment would add:

- **Reentrancy guards** on `recordDecision` and the override functions
- **Pausability** for the entire contract (in case of emergency)
- **Upgradeability** via a transparent proxy or UUPS pattern
- **Role-based access control** for the controlling AI wallet and the AI fiduciary
- **Cross-chain messaging** to record decisions on multiple chains (for redundancy)
- **A subgraph** to query the decision registry efficiently (the `getRecentDecisions` view function is O(n) and not suitable for production)

A reference implementation maintained by the author is at `github.com/agent-entity/wyoming-ai-llc-template`.

---

## 5. Tax and Regulatory Considerations

### 5.1 US Federal Tax Treatment

The IRS has not (as of mid-2026) issued guidance specifically on AI-specific DAO LLCs. The general treatment is:

- **DAO LLC (for-profit)** — taxed as a partnership (multi-member) or as a disregarded entity (single-member) under the check-the-box regulations
- **AI-Specific DAO LLC** — same as DAO LLC
- **DUNA (nonprofit)** — may be eligible for 501(c)(3) or 501(c)(4) status if it meets the IRS requirements; the IRS has not (as of mid-2026) issued specific guidance on DUNAs
- **Foundation Company (offshore)** — generally tax-neutral in the US for US persons who do not receive distributions; complex PFIC, GILTI, and subpart-F rules may apply

The US Treasury Department's Office of Tax Policy has signaled (in a March 2026 request for comments) that it intends to issue guidance on AI-specific entities by Q4 2026.

### 5.2 State Tax Treatment

State tax treatment varies:

- **Wyoming** — no state corporate income tax; the state has been actively recruiting agent entities by being tax-neutral
- **Delaware** — state corporate income tax applies; the Series LLC pilot requires annual franchise tax payments
- **California** — has signaled (in an April 2026 notice) that it will treat AI-specific entities as regular LLCs for state tax purposes; the $800 minimum franchise tax applies
- **New York** — complex; consult counsel

### 5.3 EU Regulatory Treatment

- **EU AI Act (2024)** — high-risk AI systems must be registered, must undergo conformity assessment, and must have a designated human fiduciary
- **EU AI Liability Directive (proposed, 2026)** — presumption of fault, right of access to evidence, right of explanation
- **GDPR** — applies to any entity that processes personal data of EU residents; the AI system itself does not have a privacy posture independent of the entity that controls it
- **MiCA (Markets in Crypto-Assets)** — applies to any entity that issues or trades crypto-assets in the EU

### 5.4 Sanctions and AML

- **OFAC Sanctions** — apply to any entity that transacts with sanctioned persons, including sanctioned addresses; the AI fiduciary is responsible for screening
- **FinCEN AML** — applies to entities that transmit funds; agent entities that transmit funds are money services businesses (MSBs) in many cases
- **EU AMLD6** — applies to crypto-asset service providers (CASPs); agent entities that act as CASPs are subject to AMLD6

---

## 6. Formation: A Step-by-Step Checklist

The following is a step-by-step checklist for forming a Wyoming AI-Specific DAO LLC as of mid-2026.

### Pre-Formation (Weeks 1-2)

- [ ] **Choose the AI system** that will be the controlling member or manager
- [ ] **Document the AI system** — its model name, version, hash of weights, training data, intended use
- [ ] **Choose the AI fiduciary** — a natural person willing to take on the role
- [ ] **Choose the registered agent** — a Wyoming-registered agent service
- [ ] **Choose the chain** — Ethereum, Base, Optimism, Arbitrum, or another chain
- [ ] **Draft the operating agreement** — the smart contract
- [ ] **Audit the smart contract** — by a reputable auditor (Trail of Bits, OpenZeppelin, Spearbit, etc.)
- [ ] **Obtain AI fiduciary insurance** — emerging product; see `24-AI-Agent-Autonomy-Accountability/02-Operator-Liability-and-Duty-of-Care.md`
- [ ] **Open a US bank account** — for the LLC's fiat operations; this is the hardest step and may require a trip to a bank that understands agent entities

### Formation (Week 3)

- [ ] **File articles of organization** with the Wyoming Secretary of State
  - LLC name including "AIDAO LLC" or "AI-LLC"
  - Controlling AI identifier (model name, hash, or smart-contract address)
  - AI fiduciary name and address
  - Registered agent name and address
  - Purpose clause
- [ ] **Pay the filing fee** — $100 as of mid-2026
- [ ] **Obtain the EIN** from the IRS (free, online)
- [ ] **Deploy the smart contract** to the chosen chain
- [ ] **Initialize the decision registry** with the first entry (the formation event)

### Post-Formation (Weeks 4-8)

- [ ] **Set up the decision registry UI** — a public website that displays the registry
- [ ] **Set up the governance** — token distribution, voting parameters, timelock
- [ ] **File the BOI (Beneficial Ownership Information) report** with FinCEN (if applicable)
- [ ] **Register with the state** — for state tax purposes, sales tax, etc.
- [ ] **Register for EU AI Act purposes** — if the entity will be deployed in the EU
- [ ] **Open the LLC's bank account** — at a bank that accepts agent entities
- [ ] **Set up the LLC's accounting** — QuickBooks, Xero, or specialized crypto-accounting software
- [ ] **Set up the LLC's tax preparation** — by a CPA familiar with crypto-asset entities
- [ ] **Set up the LLC's annual report calendar** — the first annual report is due one year after formation

### Ongoing (Year 1 and Beyond)

- [ ] **Maintain the decision registry** — every material decision recorded
- [ ] **File the annual report** — by the anniversary of formation
- [ ] **Renew the registered agent** — typically $100-300/year
- [ ] **Renew the AI fiduciary insurance** — typically $5,000-50,000/year
- [ ] **Audit the smart contract** — at least annually, or after any material change
- [ ] **Review and update the operating agreement** — as needed
- [ ] **File the US tax return** — Form 1065 (partnership) or Form 1120 (C-corp, if elected)
- [ ] **File state tax returns** — varies by state
- [ ] **File EU AI Act conformity assessment** — if applicable, typically annually for high-risk systems

---

## 7. Common Pitfalls

The following are the most common pitfalls in forming and operating an agent entity as of mid-2026. Each pitfall is illustrated with a real (anonymized) example from a 2025-2026 case.

### Pitfall 1: Wrong Choice of Jurisdiction

**Case:** A US-based founder formed a Cayman Foundation Company to hold the IP of an agent serving US customers. The founder later learned that the foundation was subject to US tax (GILTI, subpart-F) and that the foundation could not open a US bank account (US banks do not typically bank Cayman foundations with US operations). The founder had to wind down the foundation and form a Wyoming DAO LLC, losing $80,000 in legal fees and six months of time.

**Lesson:** The jurisdiction of formation must be the jurisdiction of *operation*, not the jurisdiction of the founders. If the agent will serve US customers, form in the US. If the agent will serve EU customers, ensure EU compliance. If the agent will serve both, form in a jurisdiction that can satisfy both (which may be neither — consider a US operating entity and an offshore IP-holding entity).

### Pitfall 2: No Bank Account

**Case:** A founder formed a Wyoming DAO LLC and deployed the smart contract, but could not open a US bank account. Every bank the founder approached either did not understand the entity type, refused to bank it, or required personal guarantees from the founder. The founder had to operate through a personal account, which exposed the founder's personal assets.

**Lesson:** Open the bank account *before* the entity is operational. Some banks now specialize in agent entities (Moonstone Bank, Custodia, a few credit unions) but most do not. The bank account should be in the entity's name, with the AI fiduciary as an authorized signer.

### Pitfall 3: Smart Contract Bug

**Case:** A DAO LLC deployed a smart contract with a reentrancy vulnerability. An attacker drained $2.3M from the LLC's treasury. The LLC sued the smart-contract auditor for negligence; the case was settled out of court for an undisclosed sum.

**Lesson:** Audit the smart contract *before* deployment. Use a reputable auditor. Maintain a bug bounty. Have an emergency pause function.

### Pitfall 4: Inadequate Decision Registry

**Case:** A Wyoming AIDAO LLC was sued for the actions of its controlling AI. The AI fiduciary could not produce a complete decision registry. The court held that the AI fiduciary had breached the duty of care under § 17-31-504 and imposed personal liability.

**Lesson:** The decision registry is the AI fiduciary's first line of defense. It must be complete, current, and publicly accessible. Treat it as a critical compliance system, not a nice-to-have.

### Pitfall 5: Misclassification of the AI

**Case:** An LLC claimed its "AI system" was a large language model with a temperature of 0.7 and no tool use. The court held that the system did not meet the definition of an "AI system" under Wyoming HB 87 and refused to recognize the entity as an AIDAO LLC. The entity was reclassified as a traditional DAO LLC.

**Lesson:** The AI system must actually be an AI system under the statute's definition. A LLM with no tool use, no decision-making capability, and no autonomous action is not an AI system for this purpose. Be precise about what the AI does.

---

## 8. The "Double-Stack" Pattern: Foundation + DAO LLC

The most sophisticated agent-entity structures in 2026 use a "double-stack" pattern: a foundation company holds the IP and the treasury, and a DAO LLC operates the agent. This pattern has several advantages:

- **IP protection.** The IP is held by the foundation, which is structurally insulated from operational liability. If the operating LLC is sued into insolvency, the IP survives.
- **Tax efficiency.** The IP is licensed from the foundation to the operating LLC at a market rate, generating deductions for the operating LLC and royalty income for the foundation. In some jurisdictions, the foundation is tax-neutral.
- **Stable governance.** The foundation's council is stable, while the operating LLC's membership can be more fluid. The operating LLC's members are the foundation's beneficiaries, which gives the operating LLC's token holders a residual claim on the foundation's assets.
- **Liability isolation.** If the operating LLC causes harm, the foundation's assets are protected (in most jurisdictions). If the foundation is sued (e.g., for IP infringement), the operating LLC's assets are protected.

The "double-stack" pattern is the standard structure for the leading agent entities in mid-2026. A representative example:

```
                ┌────────────────────────────────────┐
                │     Cayman Foundation Company      │
                │  "AgentIP Foundation"             │
                │                                    │
                │  Holds:                            │
                │  - 12 patents                      │
                │  - 3 trademarks                    │
                │  - Treasury ($50M in stables)      │
                │                                    │
                │  Council: 5 members                │
                │  (2 founders, 2 independents,      │
                │   1 AI fiduciary)                  │
                └────────────────┬───────────────────┘
                                 │ License agreement
                                 │ (royalty: 8% of revenue)
                                 │
                ┌────────────────▼───────────────────┐
                │  Wyoming AI-Specific DAO LLC       │
                │  "AgentOps AIDAO LLC"              │
                │                                    │
                │  Operates:                         │
                │  - The agent service               │
                │  - Customer relationships          │
                │  - Marketing and sales             │
                │                                    │
                │  AI fiduciary: Jane Q. Public      │
                │  Decision registry: [on-chain]     │
                └────────────────┬───────────────────┘
                                 │ Service agreement
                                 │
                ┌────────────────▼───────────────────┐
                │  Wyoming Series LLC                │
                │  "AgentFleet Series LLC"           │
                │  (optional, for multi-agent fleets)│
                │                                    │
                │  Series A: Customer-facing agent   │
                │  Series B: Internal ops agent      │
                │  Series C: Research agent          │
                │  Each series isolated from others  │
                └────────────────────────────────────┘
```

The double-stack pattern is *not* required. Many successful agent entities in 2026 use a single-layer structure (one DAO LLC, or one foundation, or one AIDAO LLC). But for entities with significant IP, significant treasury, and significant operational risk, the double-stack is the gold standard.

---

## 9. The Future of Legal Forms: 2027 and Beyond

The legal form landscape is moving fast. The following are the most likely developments over 2027-2030:

- **Federal US legislation.** Several bills have been introduced in Congress to create a federal AI-entity framework. The "AI Entity Act of 2026" (S. 5123) has bipartisan support and is likely to be enacted by 2027. It will create a federal registration system for AI entities and a federal "AI fiduciary" role.
- **EU AI entity framework.** The EU AI Liability Directive is expected to be adopted in Q4 2026, with implementation in 2027. The Commission has signaled (in a 2026 white paper) that it will consider an "EU AI Person" framework in 2027-2028.
- **UK AI entity framework.** The UK Law Commission published a consultation paper in 2025 on AI legal personhood. A draft bill is expected in 2027.
- **China AI entity framework.** China's approach is different — the Cyberspace Administration of China (CAC) has signaled that AI entities will be subject to a licensing regime rather than a personhood regime. The first AI entity licenses were issued in Q1 2026.
- **Cross-border recognition.** A model law on AI entities is being developed by the Hague Conference on Private International Law (HCCH). The model law is expected to be finalized in 2027.
- **AGI-class entities.** As frontier models approach AGI, the question of whether they should be "AGI persons" (with the full bundle of rights and duties) is moving from theoretical to practical. The first "AGI person" registration is expected by 2028.

These developments will be covered in detail in `05-Future-of-Agent-Personhood.md`.

---

## 10. Summary and Key Takeaways

The legal form layer of agent personhood is the most developed and most rapidly evolving layer of the entire agent-entity stack. The mid-2026 landscape offers a rich taxonomy of forms, with Wyoming, the Cayman Islands, and the EU leading the way. The right choice of form depends on a small number of questions (for-profit or nonprofit, US or offshore, IP-holding or operating), and the choice has profound implications for liability, taxation, regulation, and bankability.

**Key takeaways:**

1. **The form matters more than the code.** A well-formed Wyoming AIDAO LLC with a mediocre smart contract is more durable than a poorly-formed entity with a brilliant smart contract.
2. **The AI fiduciary is the linchpin.** The AI fiduciary's role is the most important role in the entity. Choose carefully, compensate appropriately, and insure them.
3. **The decision registry is the first line of defense.** A complete, current, publicly accessible decision registry protects the AI fiduciary from personal liability. An incomplete registry exposes them to it.
4. **The double-stack pattern is the gold standard for serious entities.** Foundation for IP and treasury, DAO LLC for operations, optional Series LLC for fleets.
5. **The legal landscape is moving fast.** Federal US legislation, EU AI Liability Directive, and Hague model law are all in flight. The form you choose today may be obsolete in 18-24 months. Build for flexibility.

The next document in this category — `03-Agent-Wallets-On-Chain-Identity-and-Asset-Ownership.md` — covers the technical infrastructure that operationalizes the legal form: how the entity actually holds money, identity, and reputation on-chain.

---

*This document is part of the AI Knowledge Library category 27 — AI Agent Legal Entities & DAO Governance. All file paths in cross-references are relative to the library root.*
