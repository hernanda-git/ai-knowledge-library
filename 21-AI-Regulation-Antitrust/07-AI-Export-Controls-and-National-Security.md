# 07 — AI Export Controls and National Security

## 1. Introduction: AI as a National Security Asset

Artificial intelligence has emerged as the defining dual-use technology of the 21st century, with implications for national security that rival — and arguably exceed — those of nuclear technology, cryptography, and aerospace. Unlike prior dual-use technologies, AI is uniquely scalable, rapidly improvable, and deeply embedded across civilian, commercial, and military domains. This convergence has triggered an unprecedented wave of export controls, investment screening mechanisms, and technology protection regimes.

### 1.1 The Strategic Importance of AI Capabilities

| Dimension | Civilian Impact | Military Impact |
|-----------|----------------|-----------------|
| Compute | Cloud services, scientific computing | Weapons systems, simulation, C2 |
| Algorithms | LLMs, recommendation systems | Autonomous targeting, signals analysis |
| Data | Enterprise analytics, personalization | Intelligence fusion, reconnaissance |
| Talent | Research breakthroughs, startups | Defense labs, classified projects |

### 1.2 The Three Layers of AI National Security Control

1. **Hardware controls**: Semiconductor fabrication equipment, advanced chips, GPUs, networking hardware
2. **Software controls**: Model weights, training algorithms, inference code, AI frameworks
3. **Knowledge controls**: Research publication restrictions, talent mobility limits, technical data transfers

### 1.3 Key Actors and Institutions

| Actor | Role | Key Instruments |
|-------|------|-----------------|
| BIS (Bureau of Industry and Security) | US export control enforcement | EAR, Entity List, FDPR |
| CFIUS (Committee on Foreign Investment in the US) | Foreign investment review | Section 721 Defense Production Act |
| DoD (Department of Defense) | National security classification | Security classification guidance |
| DOC (Department of Commerce) | Dual-use export licensing | Commerce Control List (CCL) |
| Wassenaar Arrangement | Multilateral export coordination | Munitions List + Dual-Use List |

### 1.4 The Geopolitical Context: US-China Technology Competition

The fundamental driver of AI export controls is the strategic competition between the United States and the People's Republic of China. The US views AI as a domain where it currently holds a technological edge, and export controls are the primary mechanism to preserve that advantage. China views AI as essential to its military modernization (as articulated in its "military-civil fusion" strategy) and economic ambitions.

| Year | US Action | Chinese Response |
|------|-----------|------------------|
| 2022 | First BIS semiconductor rules (Oct 7) | Accelerated domestic chip development |
| 2023 | Expanded BIS rules (Oct 17) | Export controls on critical minerals |
| 2024 | Model weight controls, cloud computing rules | Anti-espionage law expansion |
| 2025 | Additional semiconductor restrictions | Domestic GPU ecosystem push |
| 2026 | CFIUS digital technology expansion | Cross-border data controls tightened |

---

## 2. The Bureau of Industry and Security (BIS) Regulatory Framework

### 2.1 BIS Structure and Authority

The Bureau of Industry and Security, part of the US Department of Commerce, administers the Export Administration Regulations (EAR). Under the International Emergency Economic Powers Act (IEEPA), the President has delegated broad authority to BIS to control exports of dual-use items — goods, software, and technology that have both civilian and military applications.

**Key statutory authorities:**
- Export Control Reform Act of 2018 (ECRA) — permanent authorization
- IEEPA — emergency authority (used for rapid rulemakings)
- Arms Export Control Act — governs defense articles (via DDTC)
- Trading with the Enemy Act — limited application

### 2.2 The Commerce Control List (CCL) and AI-Related Categories

The CCL categorizes controlled items by Export Control Classification Number (ECCN). AI-related controls span multiple categories:

| ECCN | Description | Control Parameters |
|------|-------------|-------------------|
| 3A090 | High-performance integrated circuits | TPU performance thresholds, interconnect bandwidth |
| 4A090 | Computing systems incorporating 3A090 chips | System-level performance aggregation |
| 4D090 | Software for AI training that exceeds thresholds | Floating-point operations, parameter counts |
| 4E090 | Technology for developing/producing 4D090 software | Technical data, know-how, blueprints |

### 2.3 Entity List and Unverified List

BIS maintains several lists of restricted parties:

**Entity List:**
- Chinese AI chip companies (Huawei, SMIC, YMTC)
- AI research institutes linked to military-civil fusion
- Companies supporting surveillance/repression technologies
- Quantum computing entities

**Unverified List:**
- Entities where BIS cannot verify end-use compliance
- Lower restriction level than Entity List
- Triggers additional licensing requirements

**Military End-User (MEU) List:**
- Chinese and Russian military entities
- Expansive definition covering logistics, medical, and procurement units

### 2.4 Foreign Direct Product Rule (FDPR)

The FDPR is arguably BIS's most powerful tool. It extends US jurisdiction to foreign-produced items that are:
1. Direct products of US-origin technology/software, OR
2. Produced by plants that are direct products of US-origin technology/software

**Application to AI:**
- Foreign-made chips using US EDA software fall under EAR
- Foreign AI chips incorporating US-designed architecture
- Foreign AI training clusters built with US equipment

**Entity List FDPR (2024 expansion):**
- Applies FDPR to all transactions involving Entity List parties
- Covers foreign-produced chips, servers, and AI accelerators
- Strictest application focused on Chinese semiconductor entities

---

## 3. Semiconductor Export Controls: The Chip Wars

### 3.1 The October 2022 Rules (The First Shot)

On October 7, 2022, BIS published interim final rules that fundamentally reshaped the global semiconductor landscape:

**Key restrictions:**
- Controls on advanced computing ICs (ECCN 3A090): Chips with ≥48 GB/s interconnect bandwidth AND ≥600 GB/s aggregate bidirectional bandwidth OR ≥300 GB/s of aggregate memory bandwidth AND ≥24 GB of memory bandwidth-density AND ≥1.6 TFLOPs (FP32) or ≥5.3 TFLOPs (INT8)
- Supercomputer controls: Systems with ≥100 FP64/FP32 TFLOPs AND ≥200 GB/s interconnect bandwidth
- US persons restrictions: Denying US citizens/permanent residents from supporting Chinese semiconductor fabrication without a license
- Semiconductor manufacturing equipment controls: Tools capable of ≤28nm logic, 128-layer NAND, 18nm DRAM

### 3.2 The October 2023 Expansion (Closing Loopholes)

**Major changes:**
- **Performance density threshold**: Added a new metric (performance per die area) to prevent circumvention via chip disaggregation
- **Interconnect expansion**: Extended controls to cover chips packaged with advanced packaging technologies
- **Cloud licensing requirement**: Companies providing cloud computing above certain thresholds must report non-US entities accessing those resources
- **Notification requirements**: Expansion of entity list to include additional Chinese entities

### 3.3 The March 2024 Rules (The Broadest Controls Yet)

**New control parameters:**
- **Total Processing Performance (TPP)**: Replaced older metrics with TPP = 2 × (FP8 throughput + FP16 throughput) × bit length
- **Performance Density (PD)**: Controls on chips exceeding 5.92 TFLOPs/mm² (FP16)
- **"No workaround" provisions**: Explicitly closed disaggregation and multi-die package workarounds

**Impact on commercially available chips:**
- NVIDIA H100, H200, B100, B200 — all controlled
- AMD MI250, MI300 — all controlled
- Intel Gaudi 2, Gaudi 3 — all controlled
- Lower-performance variants (H100 80GB reduced NVLink, A800, H800) — specifically designed for Chinese market, also controlled

### 3.4 Industry Impact and Circumvention

**NVIDIA's response:**
- Designed H800 specifically for China (slower interconnect)
- Subsequent restrictions closed this workaround
- Latest offerings fully excluded from Chinese market

**Chinese countermeasures:**
- Huawei Ascend 910B, 920 development
- Domestic EDA tool development
- Accelerated RISC-V adoption
- Chiplet-based architectures to combine lower-performance dies
- Underground supply chains via Hong Kong, Singapore, Vietnam, Malaysia

**Estimated impact on Chinese AI development (2026):**
- 3-5 year delay in reaching frontier AI capabilities
- Increased costs for training runs (2-3x higher than US/European equivalents)
- Reduced access to cutting-edge hardware for research
- Accelerated domestic innovation in alternative architectures

### 3.5 Semiconductor Fabrication Equipment Controls

Controls extend beyond chips themselves to the equipment used to manufacture them:

**Controlled equipment categories:**
- Lithography systems (especially EUV and DUV immersion)
- Etch and deposition tools
- Metrology and inspection equipment
- Chemical mechanical planarization (CMP) tools
- Wafer testing and packaging equipment

**Key companies affected:**
- ASML (Netherlands) — EUV lithography, high-end DUV lithography
- Tokyo Electron (Japan) — etch, deposition, cleaning
- Applied Materials (US) — deposition, etch, CMP
- Lam Research (US) — etch, deposition
- KLA (US) — metrology, inspection

**Multilateral coordination:**
- Netherlands: ASML export licenses required for advanced DUV systems
- Japan: Tokyo Electron and Nikon export controls aligned with US
- South Korea: Samsung and SK Hynix face restrictions on Chinese fabs
- Non-cooperating jurisdictions become transshipment hubs

---

## 4. AI Model Weight Controls

### 4.1 The Emergence of Weight-Level Controls

In 2024-2025, the US government expanded export controls to encompass AI model weights — the trained parameters that determine model behavior. This represents a significant escalation from hardware-focused controls to software/knowledge controls.

**Why model weights matter:**
- The trained model weights embody the value of massive compute investments ($100M-$1B+ per model)
- Weights can be copied at near-zero marginal cost
- A frontier model downloaded in China represents billions in compute value transferred
- Weights can be fine-tuned for military applications
- Dual-use capability is embedded in the weights themselves

### 4.2 2025 Model Weight Rule Details

**Control parameters:**
- Models exceeding 10²⁶ FLOPs (training compute threshold)
- Models achieving benchmark performance above certain thresholds
- Models with specific capabilities (cybersecurity, advanced reasoning, biological design)

**Licensing requirements:**
- Export of controlled model weights requires a BIS license
- License applications evaluated on a case-by-case basis
- Presumption of denial for Chinese entities
- Licenses to other destinations evaluated on end-use and end-user

**Open-source carveout:**
- Models with publicly available weights that are not subject to access restrictions
- "Publicly available" defined as accessible to any person without restriction
- Negotiated API access does not qualify
- Time-limited embargoes on model release for security review

### 4.3 The Open-Source Tension

The open-source AI community has been significantly affected:

**Arguments against weight controls:**
- Open-source allows global research community to audit and improve models
- Controls fragment the global AI community along geopolitical lines
- Enforcement is practically impossible (weights can be transmitted via encrypted channels)
- Open-source models improve safety through transparency

**Arguments for weight controls:**
- Dual-use capabilities in frontier models pose unacceptable national security risks
- China will not reciprocate transparency while receiving open-source benefits
- Export controls can establish norms for responsible AI stewardship
- Partial controls (threshold-based) are better than no controls

**Practical impact:**
- Meta's Llama 4, Mistral Large, and other open-weight models face distribution restrictions
- Weight release delayed for post-training safety evaluation
- Differential release strategies (less capable versions to certain jurisdictions)
- Increased use of API-only access for frontier capabilities

### 4.4 Enforcement Challenges

Model weight control enforcement faces unique difficulties:

**Detection problems:**
- Encrypted model weights indistinguishable from random data
- Steganographic embedding in benign files
- Incremental transfer across distributed systems
- Quantization reduces detectability

**Attribution problems:**
- Determining whether a given model was trained using controlled compute
- Tracing weights to their original training run
- Proving violation of non-proliferation terms

**Jurisdictional problems:**
- Cloud providers hosting training in multiple jurisdictions
- International research collaborations spanning controlled and uncontrolled jurisdictions
- Training hardware distributed across multiple countries

---

## 5. Cloud Computing and AI Infrastructure Controls

### 5.1 Cloud-Based Training as Regulatory Frontier

As chip controls tightened, Chinese AI companies increasingly turned to cloud computing services hosted outside of China to access high-performance compute. This prompted a new regulatory frontier.

**Cloud service restrictions (2024-2026):**

| Provider | Restriction | Effective Date |
|----------|-------------|---------------|
| AWS | Chinese entities cannot access GPU instances above threshold | Q2 2024 |
| Azure | Geofencing of AI training infrastructure | Q2 2024 |
| GCP | Enhanced due diligence for Chinese accounts | Q3 2024 |
| Oracle | Screening of cloud resellers | Q4 2024 |

### 5.2 Cloud Infrastructure Provider Reporting Requirements

**Know-Your-Customer obligations:**
- Cloud providers must verify identity of entities accessing large GPU clusters
- Reporting of unusual usage patterns (training vs. inference)
- Periodic audits of customer lists
- Flagging of resold/white-label cloud services

**Threshold-based reporting:**
- Any entity deploying >10,000 GPU-hours in a rolling 30-day period
- Any training run estimated to exceed 10²⁵ FLOPs
- Deployment of inter-datacenter high-bandwidth networking
- Access to H100/B200 or equivalent hardware

### 5.3 The GPU-as-a-Service Loophole

Chinese entities attempted to work around restrictions through:
1. Purchasing GPU time through non-Chinese intermediaries
2. Using virtual private servers configured with NVIDIA GPUs
3. Academic collaborations providing compute access
4. Subsidiaries in third countries (Singapore, Malaysia, UAE)

**Regulatory responses:**
- Expanded "deemed export" rules to cover cloud-based technology transfers
- Cloud providers required to implement geolocation-based access restrictions
- Enhanced screening of university collaborations
- Third-country reseller monitoring programs

---

## 6. The Wassenaar Arrangement and Multilateral Export Controls

### 6.1 Wassenaar Overview

The Wassenaar Arrangement on Export Controls for Conventional Arms and Dual-Use Goods and Technologies is a 42-member multilateral export control regime. Established in 1996, it succeeded the Cold War-era COCOM (Coordinating Committee for Multilateral Export Controls).

**Membership (key players):**
- All EU member states
- United States, United Kingdom, Canada, Australia, New Zealand (Five Eyes)
- Japan, South Korea, Singapore
- Russia (suspended), Ukraine
- Argentina, Brazil, Mexico, South Africa, Turkey

**Notable non-members:**
- China
- India
- Israel (observer status)
- Saudi Arabia, UAE, Qatar

### 6.2 AI-Relevant Wassenaar Controls

**Dual-Use List Category 4 (Computers):**
- "Neural network computers" and "neural network modules"
- Digital computers rated for fault tolerance (for space/nuclear)
- Computers with ECCN 4A003 and 4A004 characteristics

**Dual-Use List Category 5 (Telecommunications and Information Security):**
- Cryptography controls relevant to AI model protection
- Network infrastructure components

**Munitions List:**
- Items specifically designed for military AI applications
- Autonomous weapons system components
- Electronic warfare AI subsystems

### 6.3 Limitations of Wassenaar

**Consensus-based decision-making:**
- All controls require unanimous consensus
- China's absence limits effectiveness
- Members can and do interpret controls differently

**Enforcement variability:**
- Some members have robust enforcement (US, UK, Netherlands)
- Others have limited enforcement capacity
- Re-export controls depend on transshipment country cooperation

**Technology lag:**
- Wassenaar lists are updated every 1-3 years
- AI technology evolves much faster
- Controls often outdated by the time they are adopted

### 6.4 Beyond Wassenaar: Minilateral Approaches

Given Wassenaar's limitations, several minilateral arrangements have emerged:

**Five Eyes AI Export Control Working Group:**
- Australia, Canada, New Zealand, UK, US
- Shared threat assessments and enforcement intelligence
- Harmonized licensing approaches

**US-Japan-Netherlands Semiconductor Agreement:**
- Coordinated lithography controls (ASML)
- Shared semiconductor supply chain intelligence
- Joint enforcement actions

**Chip 4 Alliance (US-Japan-South Korea-Taiwan):**
- Semiconductor supply chain resilience
- Technology protection standards
- Investment screening coordination

---

## 7. CFIUS and Foreign Investment Screening

### 7.1 CFIUS Overview

The Committee on Foreign Investment in the United States (CFIUS) reviews foreign acquisitions of US businesses that could affect national security. AI-related investments have become a primary focus.

**CFIUS membership:**
- Treasury (chair)
- Justice, Homeland Security, Commerce, Defense, State, Energy
- Office of the US Trade Representative
- Office of Science and Technology Policy

### 7.2 AI-Related CFIUS Jurisdiction

**Mandatory filing triggers (2024-2026 expansion):**
1. Foreign investment in US AI companies developing dual-use technology
2. Investments involving entities with Chinese state-affiliated ownership
3. Joint ventures transferring AI technology to foreign entities
4. Acquisition of US AI startups by foreign strategic investors

**Critical technology identification:**
- BIS has identified AI as a critical technology under Section 1758 of ECRA
- This triggers mandatory CFIUS filings for certain transactions
- Covers companies producing, designing, or testing AI technologies

### 7.3 Notable CFIUS AI Cases

| Year | Transaction | Outcome |
|------|-------------|---------|
| 2023 | US AI startup acquired by Chinese parent | Blocked via divestment order |
| 2024 | Chinese investment in US autonomous vehicle company | Mitigation agreement with national security oversight |
| 2024 | UAE investment in US AI infrastructure provider | Cleared with governance conditions |
| 2025 | Chinese venture capital investment in AI training platform | Divestiture required |
| 2025 | UK semiconductor design acquisition by China-linked entity | Blocked |
| 2026 | Japanese investment in US AI safety startup | Cleared |

### 7.4 Outbound Investment Controls

In 2025-2026, the US expanded controls to cover not just inbound foreign investment (CFIUS), but also outbound US investment into China:

**Outbound Investment Security Program (OISP):**
- Requires notification of US investments in Chinese AI, semiconductors, quantum computing
- Prohibition of certain transactions (private equity, venture capital in sensitive areas)
- Reporting requirements for portfolio investments
- Applies to both direct investments and fund investments

**Covered activities:**
- AI systems designed for military/intelligence applications
- AI systems for surveillance or social credit implementation
- Advanced semiconductor design and fabrication
- Quantum computing hardware and software

---

## 8. Geopolitics of AI: The Great Power Competition

### 8.1 The Bifurcated AI Ecosystem

Export controls have created two largely separate AI ecosystems:

| Dimension | US-Aligned Block | China-Aligned Block |
|-----------|-----------------|---------------------|
| Hardware | NVIDIA, AMD, Intel | Huawei Ascend, Cambricon |
| Cloud | AWS, Azure, GCP, Oracle | Alibaba Cloud, Baidu AI Cloud |
| Foundation Models | OpenAI, Anthropic, Google, Meta | Baidu (ERNIE), Alibaba (Qwen), ByteDance |
| Talent | US/European universities | Domestic universities + returnees |
| Standards | OECD, ISO, NIST | Chinese national standards |
| Data | Diverse, regulated | State-controlled, centralized |

### 8.2 The "Small AI" vs. "Big AI" Dynamic

**Small AI** — models and systems that can operate within controlled environments:
- Open-source models running on consumer hardware
- Specialized domain models requiring limited compute
- On-device inference for privacy-preserving applications
- Quantized/distilled models

**Big AI** — frontier systems requiring massive compute:
- Requires advanced GPU clusters (H100/B200+)
- Demands high-bandwidth interconnects
- Needs significant power and cooling infrastructure
- Concentrated in US-aligned block

### 8.3 The Race for Sovereign AI Capabilities

Many nations are pursuing sovereign AI capabilities:

| Country | Strategy | Status (2026) |
|---------|----------|---------------|
| India | BharatGPT, domestic GPU clusters, AI mission | Emerging |
| Saudi Arabia | KAUST compute, Vision 2030 AI pillar | Growing |
| UAE | Falcon models, MBZUAI, Technology Innovation Institute | Significant |
| Singapore | National AI compute, talent hub | Strong |
| France | Mistral, Jean Zay supercomputer, Gaia-X | Strong |
| Japan | Fugaku extension, strategic AI plan | Developing |
| South Korea | Exa-scale computing, national AI research lab | Developing |
| Brazil | National AI strategy, Portuguese-language models | Emerging |

### 8.4 Technology Transfer and Espionage

**State-sponsored technology acquisition:**
- Directed investment in AI startups
- Academic research partnerships
- Talent recruitment programs (Thousand Talents Plan successor)
- Reverse engineering of controlled chips
- Joint venture arrangements

**Counter-espionage measures:**
- Enhanced visa screening for AI researchers
- University disclosure requirements for foreign funding
- FBI counterintelligence briefings for AI startups
- Trade secret protection enhancements
- Increased classification of AI research at national labs

### 8.5 The Semiconductor Supply Chain

**Concentration risks:**
- 90% of advanced logic chips fabricated in Taiwan (TSMC)
- 80% of advanced packaging in Taiwan
- 70% of EUV lithography from Netherlands (ASML)
- 90% of high-bandwidth memory from South Korea (Samsung, SK Hynix)
- 85% of EDA tools from US companies (Synopsys, Cadence, Siemens EDA)

**Strategic vulnerabilities:**
- Taiwan strait contingency planning (worst-case semiconductor disruption)
- TSMC Arizona, Kumamoto, and Dresden fabs as geographic diversification
- Japan's Rapidus initiative for advanced logic fabrication
- US CHIPS Act domestic fab construction
- European Chips Act capacity building

---

## 9. Talent and Knowledge Controls

### 9.1 US Persons Restrictions

BIS rules prohibit "US persons" (citizens, permanent residents, green card holders, US entities) from supporting certain activities involving Chinese semiconductor fabrication:

**Prohibited activities without license:**
- Installing, maintaining, repairing controlled equipment at Chinese fabs
- Providing technical guidance on advanced process nodes
- Training Chinese engineers on controlled technology
- Developing customized semiconductor designs for Chinese entities

**Impact on industry:**
- US engineers reassigned from Chinese operations
- EDA tool support restricted for Chinese accounts
- Foundry services limited without extensive licensing
- Professional networking constrained

### 9.2 Research Publication and Collaboration

**National Security Presidential Memorandum (NSPM-33):**
- Disclosure requirements for foreign government-sponsored talent programs
- Research security training requirements
- Export control compliance in academic settings
- Restrictions on certain international collaborations

**Journal and conference controls:**
- Some AI conferences require disclosure of affiliations
- Dual-use research of concern (DURC) review for papers
- Pre-publication national security review for sensitive topics
- Virtual participation restrictions for sanctioned entities

### 9.3 Academic Talent Pipeline

**The talent competition:**
- Chinese nationals representing ~60% of international AI PhDs in US
- 25-30% of top-tier AI conference papers authored by Chinese researchers
- US-returning PhD graduates declining (from ~85% to ~65% over 5 years)
- Increased domestic retention in China

**Policy responses:**
- Visa processing times extended for AI-related fields
- Enhanced background checks for sensitive domain researchers
- STEM OPT program modifications
- "Reverse brain drain" countermeasures

---

## 10. Legal and Compliance Challenges

### 10.1 Compliance Burdens

**For US companies:**
- Determining whether a new market is restricted
- Classifying products under ECCN codes
- Screening end users and end uses
- Maintaining technology control plans
- Record-keeping requirements (5 years)

**For non-US companies:**
- Understanding extraterritorial application of US law
- Managing re-export risks from US-origin components
- Navigating conflicting requirements (US vs. country of operation)
- FCPA and sanctions implications of circumvention

### 10.2 Due Diligence Requirements

**Customer screening:**
- BIS Entity List, Denied Persons List, Unverified List
- OFAC sanctions lists (SDN List)
- Consolidated Screening List (CSL) maintained by US government
- State Department AECA debarred parties

**Internal compliance programs:**
- Written export management system (EMS)
- Employee training programs
- Technology control plans for physical and digital assets
- Regular internal audits
- Designated export control officer

### 10.3 Enforcement and Penalties

**Types of violations:**
- Unlicensed export of controlled items
- Re-export of US-origin items without authorization
- Dealing with sanctioned entities
- False statements on license applications
- Circumvention through third-country transshipment

**Penalty ranges:**
- Civil penalties: Up to $368,136 per violation (adjusted for inflation)
- Criminal penalties: Up to $1,000,000 and 20 years imprisonment per violation
- Denial of export privileges: Suspension or revocation
- Debarment from government contracts
- Seizure of exported items

**Notable enforcement actions:**
- Semiconductor equipment company fined $50M for unauthorized Chinese shipments
- Cloud provider fined $15M for inadequate KYC on AI training accounts
- Individual engineer sentenced to 3 years for sharing GPU architecture with Chinese entity
- Trading company debarred for re-exporting controlled chips via Hong Kong

### 10.4 Managing Compliance Across Jurisdictions

**Conflict of laws challenges:**
- EU blocking statute prohibits compliance with some US extraterritorial sanctions
- Chinese Anti-Foreign Sanctions Law prohibits compliance with foreign sanctions
- Companies faced with impossible compliance matrices

**Safe harbor strategies:**
- Jurisdictional carve-outs in corporate structure
- China-only subsidiaries with limited IP access
- Technology walling-off between operating units
- Diversified supply chains

---

## 11. Future Trajectory: 2026-2030

### 11.1 Likely Control Expansions

**1. Compute-level tracking:**
- Universal compute registry for frontier training runs
- Hardware-level serialization of AI accelerators
- Blockchain-based chain of custody for model training

**2. Inference controls:**
- Geofencing of high-capability inference endpoints
- Tiered access based on jurisdiction and use case
- API-level restrictions for frontier models

**3. Biotechnology AI convergence:**
- DNA synthesis screening for AI-designed biological sequences
- Controls on AI models with advanced bio-design capabilities
- Dual-use research review expansion

**4. Expanded multilateral coordination:**
- Wassenaar modernization and acceleration
- New technology-specific control regimes
- Intelligence-sharing for enforcement

### 11.2 Likely Circumvention Trajectory

**Continued circumvention methods:**
- Virtual-private-network-based cloud access
- Shipment through uninspected ports
- False destinations and transshipment
- Journal/paper-based technology transfer
- Third-party talent recruitment

**Enforcement countermeasures:**
- AI-powered customs screening
- Supply chain tracing technologies
- Predictive analytics for suspicious transactions
- International cooperation on border enforcement

### 11.3 Impact on Global AI Development

**Projected outcomes:**
- Continued but slowed Chinese AI advancement (3-5 year gap)
- Emergence of a robust non-Chinese, non-US AI ecosystem (EU, India, Middle East)
- Increased dual-use risk as AI diffuses across adversarial states
- Long-term erosion of US technological leadership if controls stifle domestic innovation
- Greater emphasis on AI safety and governance as capabilities proliferate

### 11.4 The Strategic Stability Question

Analogies to nuclear non-proliferation are increasingly drawn:
- **Like nuclear weapons**: AI at frontier capabilities poses existential-scale risks
- **Unlike nuclear weapons**: AI capabilities are software-based, widely distributed, and rapidly improvable
- **Challenge**: Export controls may delay but cannot prevent proliferation
- **Goal**: Buy time for governance frameworks to mature and for defensive technologies to advance

---

## 12. Conclusion

AI export controls have become a central pillar of national security policy in the 2020s, far more expansive and consequential than any prior technology control regime. The combination of semiconductor restrictions, model weight controls, cloud computing oversight, investment screening, and talent controls represents a comprehensive effort to manage the strategic implications of a transformative technology.

The effectiveness of these controls depends on:
1. **Multilateral cooperation**: Unilateral controls are leaky; coordinated enforcement is essential
2. **Domestic innovation policy**: Export controls must be paired with robust domestic R&D investment
3. **Enforcement capacity**: New technologies for supply chain tracing and model provenance tracking
4. **Adaptive governance**: Control parameters must evolve as fast as the technology they regulate
5. **Strategic clarity**: Clear articulation of which capabilities are truly national security-critical

The trajectory of AI export controls will be one of the defining features of global technology governance through the end of this decade, with profound implications for the pace, direction, and distribution of AI capabilities worldwide.

---

## References and Further Reading

- BIS Export Administration Regulations (15 CFR 730-774)
- CFIUS Regulations (31 CFR 800-802)
- Wassenaar Arrangement Dual-Use List (2025 edition)
- National Security Commission on AI (NSCAI) Final Report
- CSIS: "The Geopolitics of AI Chip Supply Chains"
- CRS Report: "US Semiconductor Export Controls on China"
- OECD: "Export Controls and Artificial Intelligence"
- RAND: "Strategic Competition in AI Semiconductor Supply Chains"
- CNAS: "The FDPR and AI Technology Protection"
- Georgetown Journal of International Law: "Model Weights as Munitions"
