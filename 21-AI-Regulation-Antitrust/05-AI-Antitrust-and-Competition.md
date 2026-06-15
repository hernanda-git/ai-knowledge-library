# 05 — AI Antitrust and Competition: Market Concentration, Investigations, and Regulatory Remedies

## 1. Introduction: The Emerging Field of AI Competition Law

Artificial intelligence markets present novel challenges for competition law. The traditional tools of antitrust analysis — market definition, market power assessment, barriers to entry — must adapt to an industry characterized by: winner-take-most dynamics, vertically integrated tech giants, strategic investments that blur the line between partnership and control, a critical input (compute) dominated by a single supplier, and rapid technological change that traditional competition analysis struggles to capture.

As of mid-2026, competition authorities in the US (DOJ, FTC), UK (CMA), EU (DG COMP), and China (SAMR) have launched multiple investigations into AI market structure. This document provides comprehensive analysis of AI antitrust dynamics, ongoing investigations, legal theories, and potential remedies.

### 1.1 Why AI Markets Are Different

| Traditional Industry | AI Industry |
|---------------------|-------------|
| Tangible goods | Intangible: data, models, compute |
| Defined product markets | Fluid: foundation models serve many markets |
| Observable pricing | Complex: API pricing, credits, bundling |
| Stable market shares | Rapidly evolving capability leadership |
| Clear competitor set | Ambiguous: partners, investors, customers |
| Measurable output | Quality hard to measure (benchmarks, safety) |
| Established entry barriers | Compute + data + talent = unprecedented barriers |

### 1.2 The AI Value Chain

Understanding AI antitrust requires mapping the value chain:

```
Layer 1: Compute/Hardware
├── GPU/Accelerator Design: NVIDIA (~85% market share), AMD (~10%), Intel (~5%)
├── Cloud Infrastructure: AWS (~32%), Azure (~23%), GCP (~11%), others
├── Chip Manufacturing: TSMC, Samsung, SMIC
└── Memory/HBM: SK Hynix, Samsung, Micron

Layer 2: Foundation Models
├── Proprietary Frontier: OpenAI (GPT-4o/o3), Google (Gemini 2.5), Anthropic (Claude 4)
├── Open Weight: Meta (Llama 4), Mistral, DeepSeek, Alibaba (Qwen)
├── Closed Non-Frontier: Cohere, AI21, Writer, others
└── Vertical Models: Bloom (genomics), Med-PaLM (medical), Codex (code)

Layer 3: Infrastructure/Platform
├── Model Hosting: Hugging Face, Replicate, Fireworks, Together AI
├── ML Platforms: LangChain, LlamaIndex, Weights & Biases
├── Training Platforms: MosaicML (Databricks), Anyscale
└── AI App Platforms: Vercel, Streamlit

Layer 4: Applications
├── Enterprise AI: Microsoft Copilot, Salesforce Einstein, ServiceNow
├── Consumer AI: ChatGPT, Gemini, Claude, Perplexity
├── Vertical AI: Harvey (legal), GitHub Copilot (code), Jasper (marketing)
└── Autonomous Systems: Waymo, Tesla FSD, Ghost Autonomy
```

## 2. Market Concentration: The Facts

### 2.1 Foundation Model Market

**Concentration metrics (2026):**

| Company | Frontier Models | Estimated Market Share (API Revenue) |
|---------|----------------|--------------------------------------|
| OpenAI | GPT-4o, o3, o3-mini | ~40-45% |
| Google DeepMind | Gemini 2.5 Ultra/Pro/Flash | ~20-25% |
| Anthropic | Claude 4, Claude 3.5 | ~15-20% |
| Meta | Llama 4 (open weight) | ~5-8% (API), ~30%+ (open weight downloads) |
| Others | DeepSeek, Mistral, Cohere, AI21 | ~5-10% |

**HHI (Herfindahl-Hirschman Index)** for foundation model API market: ~2,500-3,000 (highly concentrated by DOJ/FTC guidelines — threshold >2,500)

### 2.2 GPU Market

**NVIDIA's dominance:**
- ~85% share of AI training GPU market (2025 data)
- ~95% share of large-scale AI training (models with >10^25 FLOPs)
- B200 "Blackwell" architecture: $30,000-$70,000 per unit
- Backlog: 6-18 months for large orders
- CUDA ecosystem lock-in: Millions of developers trained on CUDA, massive software library

**Competitors:**
- AMD MI300X/MI400: ~8-10% market share; competitive hardware but software ecosystem gap
- Intel Gaudi 3: ~2-3%; limited adoption
- Google TPU v5/v6: Not sold externally; used only within Google Cloud
- AWS Trainium 2/Inferentia: AWS internal; limited external availability
- Chinese alternatives (Huawei Ascend, Cambricon): ~0-2% outside China; constrained by foundry access

### 2.3 Cloud AI Market

| Cloud Provider | AI Services | Estimated Market Share |
|---------------|-------------|----------------------|
| AWS | Amazon Bedrock, SageMaker, Trainium | ~30-35% |
| Azure | OpenAI Service, Azure AI Studio | ~25-30% |
| Google Cloud | Vertex AI, Gemini API, TPU | ~10-15% |
| Others | Oracle, IBM, CoreWeave, Lambda | ~20-25% |

### 2.4 Talent Market

- Top AI researchers: ~90% employed by big tech (Google, Meta, Microsoft, Amazon, Apple, OpenAI, Anthropic)
- Median AI PhD salary (2025): $300,000-$1,000,000+
- Acquisition-for-hire ("acqui-hire") common: Companies acquired primarily for AI talent
- Non-compete restrictions (FTC ban effective September 2024, partially stayed) affect mobility

## 3. Strategic Investments: The Control Question

The most contentious antitrust issue in AI is the web of strategic investments by big tech in AI startups.

### 3.1 Microsoft-OpenAI

**Investment structure:**
- Total investment: ~$13B+ (various rounds, 2019-2024)
- Structure: Mix of cash and Azure compute credits
- Azure is OpenAI's exclusive cloud provider
- Microsoft has significant revenue share from OpenAI (reported ~20-75% depending on revenue tier)
- Board observer seat (relinquished November 2023 after governance crisis, regained via observer-less arrangements)
- Microsoft provides infrastructure, distribution (Copilot integration), and enterprise channel

**Regulatory scrutiny:**
- **UK CMA (December 2023)**: Initial merger assessment found partnership does not qualify as a relevant merger situation under Section 33 Enterprise Act 2002. However, CMA continues to monitor for changes
- **FTC (January 2024)**: Issued compulsory orders to Microsoft, OpenAI, and others seeking information on AI investments
- **DOJ (2025-present)**: Joint investigation with FTC on whether Microsoft's investment constitutes de facto acquisition
- **EU DG COMP**: Reviewing whether Microsoft-OpenAI falls under EU Merger Regulation jurisdiction or Article 102 TFEU

**Key legal theory:**
- Microsoft's investment may be an "executive merger" or "acquisition of control" even without formal shareholding
- The contractual arrangements (exclusivity, revenue share, board access) give Microsoft de facto control over a key competitor
- If not a merger, may constitute anticompetitive conduct through exclusive dealing or foreclosure

### 3.2 Amazon-Anthropic (The 716-Part Investigation)

**Investment structure:**
- Total investment: ~$8B (announced various tranches, 2023-2025)
- $4B initial investment (September 2023); $2.75B additional (March 2024); remainder through convertible notes
- AWS is Anthropic's primary cloud provider
- Anthropic models made available through Amazon Bedrock (AWS AI platform)
- Amazon obtains rights to use Anthropic models for internal operations and to offer as AWS services
- AWS credits structure: Part of investment is in compute credits, creating lock-in
- Amazon has investment-linked warrant rights to acquire equity at a future date

**The 716-Part Inquiry:**
- **Lead regulator**: UK Competition and Markets Authority (CMA)
- **Cooperation**: CMA cooperating with EU DG COMP, FTC, DOJ
- **Scope**: 716 questions across 8 workstreams:
  1. Corporate governance and board relationships
  2. Financial arrangements and equity structure
  3. Cloud service agreements and exclusivity
  4. Intellectual property and model distribution rights
  5. Competitive overlap between Amazon and Anthropic
  6. Impact on foundation model market competition
  7. Impact on cloud AI services competition
  8. Remedies and commitments

**Key concerns identified:**

1. **Vertical foreclosure**: Amazon may favor Anthropic models on AWS Bedrock over competing models, disadvantaging Google, OpenAI, and others in the cloud AI market
2. **Compute lock-in**: AWS credits tie Anthropic's spending to AWS, reducing incentive to multi-cloud or switch providers
3. **Information asymmetry**: Amazon gains insight into Anthropic's roadmap, model capabilities, and customer data through its cloud provider role — potentially using this for competitive advantage in its own AI development
4. **Strategic direction influence**: Through investment terms, Amazon may influence Anthropic's strategic decisions (model safety philosophy, pricing, partnership strategy)
5. **Acquisition embedded**: The investment structure may be designed as a gradual acquisition where Amazon increases control over time without triggering traditional merger review thresholds

**Anthropic S-1 Analysis:**

Anthropic's confidential S-1 registration statement for IPO (filed late 2025, made partially public through regulatory filings) reveals:

- **Governance structure**: Long-term benefit trust (similar to purpose corporation) — board committed to "responsible AI" mission
- **Amazon relationship**: Detailed in S-1 as material risk factor — "Our agreements with Amazon provide them with significant commercial rights that may conflict with our mission"
- **Revenue concentration**: Significant revenue derived from AWS Bedrock distribution channel
- **Compute dependency**: "We rely on Amazon Web Services for substantially all of our compute requirements. Transitioning to alternative providers would be costly, time-consuming, and disruptive"
- **Warrants and dilution**: Amazon's warrants represent significant potential equity stake; conversion terms create dilution risk for other shareholders
- **Governance constraints**: Certain strategic decisions require Amazon consent under investment agreements

S-1 disclosure highlights the tension: Anthropic's stated mission of "safe, responsible AI" vs. commercial dependencies that may compromise strategic independence.

### 3.3 Google-Anthropic

- Google invested ~$2B in Anthropic (2022-2024)
- Google was Anthropic's first major cloud provider (before Amazon)
- Google also has cloud credit arrangements
- TPU access provided to Anthropic
- Regulatory concern: Google has investments in both Anthropic and its own AI (Gemini), creating conflict of interest

### 3.4 Other Strategic Investments

| Investor | Investee | Amount | Regulatory Status |
|----------|----------|--------|-------------------|
| Microsoft | Mistral AI | €2B | Under CMA monitoring |
| Salesforce | Hugging Face | Reported $500M+ | Not known to be investigated |
| Databricks | Multiple | Various | Under review for competitive effects |
| SoftBank | OpenAI | Reported $500M (via Vision Fund) | Monitored |
| NVIDIA | CoreWeave, Cohere, Multiple | Various | Under investigation for investment reciprocity |

## 4. NVIDIA GPU Market Dominance

### 4.1 Market Position

NVIDIA controls the market for AI training accelerators with specific features that create lock-in and barriers to entry:

**Technology lock-in:**
- **CUDA ecosystem**: 5M+ developers, 4M+ CUDA-accelerated applications
- **cuDNN, TensorRT, NeMo**: NVIDIA software stack optimized for AI, not easily ported to competitors
- **NVIDIA AI Enterprise**: Enterprise support and security for CUDA
- **DGX systems**: Integrated hardware-software AI infrastructure

**Supply allocation:**
- NVIDIA controls GPU allocation through direct sales and partner channels
- Select customers (big tech) receive priority allocation
- Startups and researchers face long wait times or cannot access GPUs at scale
- Allocation may favor customers that do not compete with NVIDIA's AI software or hardware roadmap

### 4.2 Antitrust Investigations

**DOJ Investigation (2024-Present):**
- Allegations of discriminatory GPU allocation favoring certain customers
- Examination of CUDA bundling practices (tying CUDA licenses to GPU purchases)
- Investigation into NVIDIA's acquisition of Mellanox (networking) — whether it foreclosed competitors
- NVIDIA's investment strategy: Does NVIDIA condition GPU access on startups accepting investment?

**French Autorité de la Concurrence (2024):**
- Raided NVIDIA offices (September 2024)
- Investigation into anticompetitive practices in AI accelerator market
- Focus on pricing practices, client dependency, and market foreclosure
- Preliminary findings expected late 2026

**EU DG COMP:**
- Informal investigation into NVIDIA's supply allocation and tying practices
- Requests for information sent to cloud providers, AI developers, and NVIDIA (2024-2025)
- Potential formal investigation if evidence supports

### 4.3 Bundling and Tying Analysis

NVIDIA's CUDA-GPU bundling raises classic antitrust tying concerns:

- **Tying product**: AI training GPU (H100, B200) — dominant market position
- **Tied product**: CUDA software platform and associated tools
- **Theory**: Customers cannot use NVIDIA GPUs effectively without CUDA, even though competitors (OpenCL, SYCL, oneAPI) could provide alternatives with sufficient investment
- **Market power in tying**: NVIDIA's >85% GPU market share satisfies market power requirement
- **Anti-competitive effect**: Courts competitors' ability to develop competing software ecosystems; raises competitors' costs by requiring CUDA compatibility

**Potential defense**: NVIDIA argues CUDA-GPU is a single integrated product, not separate products. Apple-style argument: performance optimization requires deep hardware-software integration.

## 5. Barriers to Entry in AI

Understanding barriers to entry is crucial for antitrust analysis. In AI markets, barriers are uniquely high:

### 5.1 Compute Cost

| Model Category | Training Compute (FLOPs) | Estimated Cost |
|----------------|------------------------|----------------|
| Small (7B params) | ~10^21-10^22 | $100K-500K |
| Medium (70B) | ~10^23-10^24 | $2M-10M |
| Large (405B) | ~10^24-10^25 | $20M-100M |
| Frontier (next-gen) | ~10^25-10^26 | $100M-1B+ |
| Super Frontier (2026+) | >10^26 | $1B-10B |

**Capital requirements**: Training frontier models requires capital that exceeds most startup funding rounds. Only companies with $10B+ market cap or substantial VC backing can compete.

### 5.2 Data Access

- **Scale**: Frontier models need trillions of tokens of training data
- **Quality**: High-quality, diverse, filtered datasets are critical
- **Proprietary data**: Companies with access to unique datasets (user behavior, enterprise data, scientific data) have competitive advantages
- **Copyright uncertainty**: Legal challenges to training data use create uncertainty for training data acquisition

### 5.3 Talent

- <100 researchers worldwide with experience training large-scale frontier models
- Top researchers compensated at executive levels ($5M-$20M+ total comp)
- Talent acquisition costs are a barrier to entry for startups

### 5.4 Distribution/Moat

- **Incumbent advantage**: Established AI products (ChatGPT, Gemini, Claude) have user base, brand, and feedback data
- **API dependencies**: Companies building on top of OpenAI/Google/Anthropic APIs face switching costs and vendor lock-in
- **Enterprise relationships**: Existing relationships with IT departments, procurement teams, and security review processes favor incumbents

## 6. Competition Law Theories Applied to AI

### 6.1 Monopolization (Section 2 Sherman Act / Article 102 TFEU)

**Relevant markets** (disputed in ongoing cases):
- **Market for AI training chips** (NVIDIA dominance)
- **Market for foundation model APIs** (OpenAI/Google/Anthropic)
- **Market for cloud AI services** (AWS/Azure/GCP)
- **Market for AI training compute (cloud)** (AWS/Azure/GCP)
- **Market for AI researchers** (big tech concentration)

**Conduct theories:**
1. **Exclusive dealing**: Cloud providers requiring model developers to use exclusively their cloud; NVIDIA requiring CUDA exclusivity
2. **Tying/bundling**: NVIDIA CUDA-GPU bundle; cloud AI services bundled with other cloud products
3. **Predatory pricing**: AI API pricing below cost to exclude competitors (subsidized by cloud revenue)
4. **Refusal to deal**: NVIDIA refusing to supply GPUs to competitors' AI developers
5. **Acquisition of nascent competitors**: Big tech acquiring AI startups before they become competitors

### 6.2 Merger Control

**Key questions:**
- Do strategic investments in AI startups constitute "acquisition of control" triggering merger notification?
- What is the proper counterfactual in AI markets? (Would the startup have remained an independent competitor?)
- How to assess vertical effects in AI? (Cloud provider + AI model developer = vertical integration?)

### 6.3 Digital Markets Regulation

**EU Digital Markets Act (DMA)** :
- Designated gatekeepers (Microsoft, Google, Amazon, Apple, Meta, ByteDance)
- AI services may be considered "core platform services" covered by DMA obligations
- Interoperability requirements could apply to AI platform services
- Self-preferencing prohibitions: Google cannot prefer Gemini over competitors in search; Amazon cannot prefer Anthropic over competitors in AWS Bedrock
- Data access provisions: Gatekeepers must share data with business users

**UK Digital Markets, Competition and Consumers Act (2024):**
- Enhanced antitrust enforcement for digital markets
- CMA can designate firms with "Strategic Market Status" (SMS) in digital activities
- Designation enables tailored conduct requirements and pro-competitive interventions (PCIs)
- AI likely to be within scope

### 6.4 Abuse of Economic Dependence

**Emerging theory**: AI startups may be economically dependent on big tech (for compute, distribution, funding). Exploitation of this dependence (unfair contract terms, excessive pricing, discriminatory access) may constitute abuse even without market dominance.

## 7. Pending Cases and Investigations (2026)

| Case/Investigation | Regulator | Target(s) | Status |
|-------------------|-----------|-----------|--------|
| Microsoft-OpenAI | FTC, DOJ, CMA, EU | Microsoft, OpenAI | Ongoing investigation |
| Amazon-Anthropic | CMA (lead), FTC, EU | Amazon, Anthropic | 716-part inquiry underway |
| NVIDIA GPU practices | DOJ, French Autorité, EU | NVIDIA | DOJ investigation; French raid completed |
| Google AI search integration | DOJ (remedies phase) | Google | Part of DOJ v. Google search monopolization |
| Google advertising AI | EU, CMA | Google | Ongoing investigations |
| OpenAI non-profit conversion | FTC | OpenAI | Investigation into for-profit transition |
| Microsoft-Mistral | CMA | Microsoft, Mistral | Monitoring; possible investigation |
| Big Tech AI hiring | DOJ | Apple, Google, others | No-poach agreements investigation |
| OpenAI exclusive deals | EU, FTC | OpenAI, Apple | Distribution agreement review |
| Meta AI open-source | EU, FTC | Meta | Llama open-weight competition effects |

## 8. AI and Collusion: Algorithmic Coordination

### 8.1 Algorithmic Pricing

AI systems that recommend or set prices can facilitate coordination:

- **Hub-and-spoke conspiracy**: AI price-setting software as the "hub" coordinating competing firms' pricing
- **Algorithmic tacit collusion**: Reinforcement learning models that learn to collude without explicit communication
- **Information exchange through AI**: AI systems monitoring and reacting to competitor pricing in real time

**Notable cases:**
- **RealPage (DOJ, 2024)**: Antitrust lawsuit against real estate software company accused of using AI algorithm to coordinate rent prices among competing landlords. Landmark case on algorithmic collusion
- **Softeon/Amazon (EU)**: Investigation into algorithmic pricing synchronization on e-commerce platforms
- **Various hotel/airline pricing**: Investigations into AI-based revenue management systems

### 8.2 AI in Bid-Rigging

- AI systems in procurement auctions learning to allocate markets (bid rotation)
- Automated bidding systems on platforms (AdWords, e-commerce) that learn to coordinate
- Challenges of proving mens rea (intent) when AI independently learns collusive strategies

## 9. Remedies and Proposed Interventions

Economic literature and regulatory practice suggest several classes of remedies for AI market concentration:

### 9.1 Structural Remedies

1. **Divestiture of AI investments**: Order companies to divest stakes in AI startups (e.g., Amazon from Anthropic, Microsoft from OpenAI)
2. **Spin-off requirements**: Separating cloud AI services from model development (functional separation like UK banking ring-fencing)
3. **GPU capacity divestiture**: Requiring NVIDIA to sell GPU foundry capacity or make capacity available to competitors (structural remedy akin to essential facilities)

### 9.2 Behavioral/Conduct Remedies

1. **Open access requirements**: AI model providers must offer API access on non-discriminatory terms to all cloud providers (preventing vertical foreclosure)
2. **Interoperability mandates**: Cloud AI services must interoperate; model portability requirements
3. **Data portability and access**: Users must be able to transfer their AI training data and fine-tuned models between providers
4. **Compute access mandates**: GPU providers must allocate supply on fair, reasonable, non-discriminatory (FRAND) terms
5. **Transparency requirements**: Disclosure of GPU allocation methodology, AI system capabilities, and investment terms
6. **Chinese wall requirements**: Separating AI investment teams from cloud business teams within big tech
7. **Standard essential AI patents** (FRAND licensing): If AI model architecture becomes essential, FRAND licensing required

### 9.3 Compute Access as a Remedy

A novel approach gaining traction: requiring dominant firms to provide compute access to competitors:

- **GPU-as-essential-facility**: If GPU access is indispensable for AI competition and not duplicable, NVIDIA may have duty to deal
- **Compute sharing pool**: Dominant cloud providers contribute compute capacity to shared pool accessible to AI startups
- **Government-subsidized compute**: NAIRR (US), CalCompute (California), EuroHPC (EU) — public compute for AI research and competition

### 9.4 Regulatory Remedies Under Consideration

| Remedy | Target | Proponents | Status |
|--------|--------|------------|--------|
| Cloud AI functional separation | AWS, Azure, GCP | UK CMA, EU DG COMP | Under study |
| GPU FRAND licensing | NVIDIA | DOJ, French Autorité | Investigation stage |
| Model weight sharing mandates | Frontier model developers | Academic, civil society | Controversial, early stage |
| API non-discrimination | OpenAI, Google, Anthropic | UK CMA | Proposed in Amazon-Anthropic inquiry |
| Data sharing for training | Big tech | EU DMA | DMA provisions applicable |
| Open-weight requirements for publicly funded models | Government grantees | NSF, DARPA | Already policy in some US grants |

## 10. The Regulatory Debate: Intervention vs. Restraint

### 10.1 Arguments for Intervention

1. **Market failure**: AI has winner-take-most dynamics that will lead to monopoly without intervention
2. **Innovation harm**: Concentration reduces innovation by suppressing alternative approaches (open models, decentralized AI, specialized AI)
3. **Safety concentration**: Single points of failure if all AI capability resides in few firms
4. **Democracy concerns**: AI power concentrating in few corporations with private governance
5. **Race to the bottom**: Without antitrust enforcement, competitive pressure may compromise AI safety

### 10.2 Arguments Against Intervention

1. **Dynamic efficiency**: AI is still young; intervention may freeze market structure before optimal equilibrium emerges
2. **Innovation benefits of concentration**: Large investments needed for frontier progress may require concentration
3. **Jurisdictional mismatch**: National antitrust enforcement can't regulate a global market; firms will relocate
4. **Risk of regulatory capture**: Antitrust intervention may protect incumbents from innovative entrants
5. **Difficulty defining markets**: AI capabilities evolve too quickly for traditional market definition

### 10.3 The Emerging Consensus

Most competition authorities are converging on a **"monitor and intervene selectively"** approach:
- Investigate, don't block, strategic investments (unless control clearly transferred)
- Focus on conduct remedies (non-discrimination, transparency) over structural remedies
- Prioritize compute access and interoperability remedies
- Coordinate internationally to avoid gaps and inconsistent remedies
- Use digital market regulation (DMA, DMCC Act) as supplement to traditional antitrust

## 11. International Coordination

AI antitrust enforcement is increasingly coordinated across jurisdictions:

**Coordination mechanisms:**
- **ICN (International Competition Network)**: AI working group (established 2024)
- **EU-US Joint Technology Competition Policy Dialogue**: Regular meetings on AI competition
- **CMA-FTC-DG COMP trilateral**: Case-specific cooperation on Microsoft-OpenAI and Amazon-Anthropic
- **OECD Competition Committee**: AI competition policy papers and roundtables
- **BRICS Competition Authorities**: Coordination on AI market concentration

**Challenges:**
- Differing legal standards (US: consumer welfare standard; EU: fairness/competitive process standard)
- Data sharing restrictions between jurisdictions
- Different remedial approaches (US favors remedies that don't disrupt markets; EU more interventionist)
- Chinese approach (state-directed AI competition policy) not aligned with Western antitrust frameworks

## 12. Predictions for AI Antitrust (2026-2028)

1. **First major structural remedy**: Either Amazon-Anthropic divestiture or functional separation of cloud AI services in one jurisdiction within 18 months
2. **NVIDIA remedy**: Behavioral remedy (non-discriminatory GPU allocation) likely before structural remedy
3. **Merger reform**: AI investments will trigger revised merger thresholds (lower notification thresholds for tech-AI investments)
4. **Private litigation**: First major private antitrust case in AI markets (likely related to GPU access or AI API exclusion)
5. **Legislation**: US Congress likely to consider antitrust-specific AI legislation (disclosure of AI investments, enhanced merger review)
6. **Interoperability standard**: Mandated model-API interoperability will be major regulatory advance
7. **Compute governance**: International framework for compute allocation monitoring and transparency will emerge
8. **Open-source AI policy**: Tension between open-weight AI as competitive force vs. open-weight as safety concern will shape regulatory approach

---

**Document metadata**: Created June 2026. Part of the AI Regulation & Antitrust knowledge base. For sectoral regulation context, see Document 06. For export controls context, see Document 07.
