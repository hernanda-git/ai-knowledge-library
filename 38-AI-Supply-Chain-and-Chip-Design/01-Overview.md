# 01 — AI Supply Chain & Chip Design: The Strategic Overview

> **Why this document exists.** As of mid-2026, the AI revolution is fundamentally a hardware supply chain story. The world's most advanced AI systems depend on a razor-thin manufacturing pipeline: TSMC fabricates ~90% of cutting-edge AI chips in Taiwan, ASML holds a monopoly on EUV lithography machines, and the US-China tech war has turned semiconductor exports into the primary geopolitical battleground. A single earthquake in Hsinchu, a single export control escalation, or a single packaging bottleneck at CoWoS can cascade into billions of dollars of AI capacity disruption. This document maps the entire AI supply chain — from rare earth minerals to deployed inference clusters — and explains why supply chain resilience is now a first-class strategic concern for every AI company, government, and investor.

---

## Table of Contents

1. [The AI Supply Chain: A Primer](#1-the-ai-supply-chain-a-primer)
2. [The Critical Bottlenecks](#2-the-critical-bottlenecks)
3. [Geopolitical Risk Map](#3-geopolitical-risk-map)
4. [The CHIPS Act and Global Reshoring Race](#4-the-chips-act-and-global-reshoring-race)
5. [US-China Export Controls & Tech Decoupling](#5-us-china-export-controls--tech-decoupling)
6. [The TSMC Dependency Problem](#6-the-tsmc-dependency-problem)
7. [ASML and the EUV Monopoly](#7-asml-and-the-euv-monopoly)
8. [Packaging Bottlenecks: CoWoS and Beyond](#8-packaging-bottlenecks-cowos-and-beyond)
9. [Rare Earth Minerals and Critical Materials](#9-rare-earth-minerals-and-critical-materials)
10. [The Reshoring and Friend-Shoring Movement](#10-the-reshoring-and-friend-shoring-movement)
11. [Supply Chain Risk Quantification Framework](#11-supply-chain-risk-quantification-framework)
12. [Implications for AI Builders](#12-implications-for-ai-builders)
13. [Cross-References](#13-cross-references)
14. [Glossary](#14-glossary)

---

## 1. The AI Supply Chain: A Primer

### 1.1 What the AI Supply Chain Actually Looks Like

Most discussions of AI focus on algorithms, models, and software. But every AI system ultimately runs on physical hardware that must be designed, fabricated, packaged, tested, shipped, installed, and powered. The AI supply chain spans:

```
Raw Materials → Chip Design → Fabrication → Packaging → Testing → Assembly → Deployment → Operation
```

Each step involves specialized suppliers, concentrated in specific geographies, with lead times measured in months to years.

### 1.2 The End-to-End AI Hardware Stack

| Layer | What It Is | Key Players | Concentration Risk |
|-------|-----------|-------------|-------------------|
| **Raw Materials** | Silicon wafers, rare earth elements, neon gas, photoresists | Shin-Etsu (Japan), SUMCO (Japan), Lynas (Australia) | HIGH — Japan controls 50%+ of silicon wafer supply |
| **EUV Lithography** | Machines that print nanometer-scale circuit patterns | ASML (Netherlands) | CRITICAL — sole supplier globally |
| **Chip Fabrication** | Manufacturing transistors on silicon wafers | TSMC (Taiwan), Samsung (South Korea), Intel (US) | CRITICAL — TSMC handles ~90% of <5nm production |
| **Advanced Packaging** | 3D stacking, chiplet integration, HBM assembly | TSMC (CoWoS), Samsung (I-Cube), ASE (Taiwan) | HIGH — CoWoS capacity is the #1 bottleneck in 2026 |
| **HBM Memory** | High-bandwidth memory for AI accelerators | SK Hynix (South Korea), Samsung (South Korea), Micron (US) | HIGH — SK Hynix has ~50% HBM market share |
| **Chip Design** | Architecture, IP, EDA tools | NVIDIA, AMD, Broadcom, Synopsys, Cadence | MODERATE — design is distributed but EDA tools concentrated |
| **Testing & Assembly** | Final test, packaging, burn-in | ASE Group (Taiwan), Amkor (South Korea/US) | MODERATE |
| **Cloud Deployment** | Data center build-out, power, cooling | AWS, Azure, GCP, Meta, xAI | LOW — distributed but power-constrained |
| **Inference at Edge** | On-device AI chips | Qualcomm, Apple, MediaTek, Google | LOW — diversified |

### 1.3 Why 2026 Is the Inflection Year

Three forces converge in 2026 to make supply chain the #1 risk for AI:

1. **Demand explosion**: Global AI chip demand grew 3x in 2025 and is projected to grow 2x again in 2026. Every hyperscaler is spending $50B–$100B+ on AI infrastructure.

2. **Capacity constraints**: TSMC's advanced node capacity is booked through 2027. CoWoS packaging capacity is the binding constraint — not transistor fabrication.

3. **Geopolitical escalation**: US export controls on AI chips to China (October 2022, October 2023, January 2025, December 2025 updates) have created parallel supply chains and black-market diversion channels.

### 1.4 The Supply Chain as Competitive Advantage

Companies that solve supply chain problems gain structural advantages:

| Company | Supply Chain Move | Advantage Gained |
|---------|------------------|-----------------|
| **NVIDIA** | Long-term CoWoS capacity reservation at TSMC | 12-month lead time vs. 18+ for competitors |
| **Google** | In-house TPU design + TSMC fabrication relationship | $0.60/1M tokens vs. $3.00 on H100 |
| **Microsoft** | Maia chip + Azure data center vertical integration | 40% lower inference cost vs. third-party GPUs |
| **Meta** | MTIA v3 custom silicon + Open Compute Project | $0.18/1M tokens for Llama inference |
| **xAI** | Direct TSMC engagement + Memphis data center | 200K GPU cluster in 6 months (vs. industry 12–18 months) |

---

## 2. The Critical Bottlenecks

### 2.1 The Seven Chokepoints

The AI supply chain has seven critical chokepoints where a single point of failure can cascade globally:

**Chokepoint 1: EUV Lithography (ASML)**
- ASML is the sole manufacturer of Extreme Ultraviolet (EUV) lithography machines
- Each High-NA EUV machine costs ~$380M and weighs 150 tons
- Lead time: 18–24 months
- Without EUV, sub-5nm chips cannot be manufactured
- ASML's 2025 revenue: €28B; 2026 projected: €32B

**Chokepoint 2: TSMC Advanced Node Fabrication**
- TSMC produces ~90% of sub-5nm chips globally
- Apple, NVIDIA, AMD, Qualcomm, Broadcom, and most AI chip companies are TSMC customers
- A single fab costs $20B+ and takes 3–4 years to build
- TSMC's N3 (3nm) and N2 (2nm) nodes are the most advanced in production

**Chokepoint 3: CoWoS Packaging**
- Chip-on-Wafer-on-Substrate (CoWoS) is required for GPU+HBM integration
- TSMC controls ~95% of advanced CoWoS capacity
- 2025 capacity: ~35K wafers/month; 2026 target: ~55K wafers/month
- Every AI chip (H100, B200, TPU v6, MI300X) requires CoWoS
- This is the #1 binding constraint on AI chip supply in 2026

**Chokepoint 4: HBM (High-Bandwidth Memory)**
- SK Hynix holds ~50% market share; Samsung ~35%; Micron ~15%
- HBM3e and HBM4 are required for AI accelerators
- SK Hynix HBM4 production ramp: Q3 2026
- A single B200 GPU requires 8 HBM3e stacks ($3,000+ per stack)

**Chokepoint 5: EUV Photoresists and Photomasks**
- JSR (Japan) and TOK (Japan) dominate EUV photoresist supply
- Photomasks for EUV require specialized pellicles
- A single contamination event can delay production by months

**Chokepoint 6: Neon and Specialty Gases**
- Ukraine historically supplied 50%+ of semiconductor-grade neon
- Russia supplied ~30% of palladium for electronics
- Post-2022 supply disruption forced industry diversification
- Current neon supply: primarily from US, Estonia, and China

**Chokepoint 7: Silicon Wafers**
- Shin-Etsu (Japan) and SUMCO (Japan) control ~55% of global silicon wafer supply
- 300mm wafers are the standard for advanced chip manufacturing
- Wafer quality directly impacts yield and chip performance

### 2.2 Bottleneck Severity Matrix

| Bottleneck | Sole Supplier? | Geographic Concentration | Time to Build Alternative | Severity |
|-----------|---------------|-------------------------|--------------------------|----------|
| EUV lithography | Yes (ASML) | Netherlands | 10+ years | 🔴 CRITICAL |
| Advanced fabrication | Near-monopoly (TSMC) | Taiwan | 3–5 years | 🔴 CRITICAL |
| CoWoS packaging | Near-monopoly (TSMC) | Taiwan | 2–3 years | 🔴 CRITICAL |
| HBM memory | Oligopoly (3 players) | South Korea | 1–2 years | 🟠 HIGH |
| EUV photoresists | Oligopoly (2–3 players) | Japan | 1–2 years | 🟠 HIGH |
| Neon gas | Diversified post-2022 | Multiple | 6–12 months | 🟡 MODERATE |
| Silicon wafers | Oligopoly (2–3 players) | Japan | 2–3 years | 🟡 MODERATE |

### 2.3 Single-Point-of-Failure Analysis

The most dangerous single point of failure in the entire AI supply chain is **TSMC's Fab 18 in Tainan, Taiwan**. This facility:
- Produces the majority of NVIDIA's B200/GB200 chips
- Handles most of Apple's A-series and M-series production
- Has no equivalent alternative anywhere in the world
- Is located in a seismically active zone (1999 earthquake damaged fabs)
- Faces geopolitical risk from People's Republic of China military exercises

A 6-month disruption at Fab 18 would:
- Halt ~40% of global AI chip production
- Cost the global economy an estimated $500B+
- Delay AI deployments across every industry by 12–18 months

---

## 3. Geopolitical Risk Map

### 3.1 The Three Geopolitical Axes

The AI chip supply chain is shaped by three intersecting geopolitical dynamics:

**Axis 1: US-China Tech Decoupling**
- October 2022: US imposes sweeping export controls on advanced AI chips to China
- October 2023: Controls expanded to cover more chip types and thresholds
- January 2025: Further tightening on "foundational model" training chips
- December 2025: Controls extended to certain inference chips and packaging equipment
- China's response: domestic chip development (Huawei Ascend, SMIC 7nm), stockpiling, black-market diversion

**Axis 2: Taiwan Strait Tension**
- TSMC's Taiwan concentration creates existential risk for global AI
- China views Taiwan as a breakaway province; military exercises near Taiwan have increased
- US has committed to Taiwan's defense (Taiwan Relations Act)
- "Silicon Shield" theory: Taiwan's chip dominance deters invasion because China also depends on TSMC
- Counter-argument: China is building domestic capacity to reduce this dependency

**Axis 3: EU/US Industrial Policy Competition**
- US CHIPS Act: $52.7B in subsidies for domestic semiconductor manufacturing
- EU Chips Act: €43B in public and private investment
- Japan: ¥2T ($13B) in semiconductor subsidies
- South Korea: $19B in chip industry support
- India: $10B semiconductor incentive program

### 3.2 Country Risk Assessment

| Country/Region | Role in AI Supply Chain | Key Risk | Risk Level |
|---------------|------------------------|----------|------------|
| **Taiwan** | TSMC fabrication, ASE packaging | China military action, earthquake | 🔴 CRITICAL |
| **Netherlands** | ASML EUV lithography | Export control pressure from US | 🟠 HIGH |
| **Japan** | Silicon wafers, photoresists, materials | Natural disasters, limited diversification | 🟠 HIGH |
| **South Korea** | HBM memory, Samsung fabrication | North Korea, China competition | 🟠 HIGH |
| **United States** | Chip design, EDA tools, some fabrication | CHIPS Act delays, power constraints | 🟡 MODERATE |
| **China** | Largest AI chip market, growing domestic capacity | Export controls, overcapacity risk | 🟡 MODERATE |
| **Germany** | ASML components, chemical supplies | Energy costs, political instability | 🟢 LOW |
| **India** | Growing design talent, new fabs planned | Infrastructure gaps, slow buildout | 🟢 LOW |

### 3.3 Scenario Planning: What If Taiwan Is Disrupted?

| Scenario | Probability (5yr) | Impact on AI Supply Chain | Recovery Time |
|----------|-------------------|--------------------------|---------------|
| Major earthquake (>7.0) hitting Tainan/Hsinchu | 15–20% | 3–6 month production halt; $200B+ impact | 6–12 months |
| China naval blockade (non-invasion) | 5–10% | 6–12 month supply disruption; $500B+ impact | 12–24 months |
| China military invasion | 2–5% | Total TSMC disruption; $1T+ global impact | 3–5 years |
| Prolonged drought affecting fab operations | 10–15% | 1–3 month partial disruption; $50B impact | 3–6 months |
| Political crisis causing TSMC talent exodus | 5–8% | Gradual capacity decline; $100B+ annual impact | 2–4 years |

---

## 4. The CHIPS Act and Global Reshoring Race

### 4.1 US CHIPS and Science Act

The CHIPS Act, signed August 2022, represents the largest US industrial policy investment in decades:

| Provision | Amount | Status (June 2026) |
|-----------|--------|-------------------|
| Manufacturing incentives | $39B | $28B disbursed; 5 major fabs under construction |
| R&D investment | $13.2B | National Semiconductor Technology Center (NSTC) operational |
| Investment tax credit | 25% | Active; estimated $24B in credits over 10 years |
| Workforce development | $200M | CHIPS Workforce Education programs in 36 states |

**Key CHIPS Act Projects:**

| Company | Location | Investment | Node | Expected Production |
|---------|----------|-----------|------|-------------------|
| TSMC | Phoenix, AZ | $40B (expanded from $12B) | 4nm, 3nm, 2nm | Fab 1: 2025; Fab 2: 2026; Fab 3: 2028 |
| Intel | Columbus, OH | $28B | Intel 18A (1.8nm) | 2025–2027 |
| Samsung | Taylor, TX | $17B (expanded from $10B) | 4nm, 3nm | 2026 |
| Micron | Boise, ID + Syracuse, NY | $20B | DRAM + HBM | 2026–2028 |
| GlobalFoundries | Malta, NY | $12B | 12nm–65nm | 2025–2027 |

### 4.2 EU Chips Act

The EU Chips Act (€43B) focuses on:

| Objective | Investment | Key Projects |
|-----------|-----------|--------------|
| Manufacturing capacity doubling | €18B public | Intel Magdeburg (Germany), TSMC Dresden (Germany) |
| R&D and first industrial deployment | €3.3B | European Chips Joint Undertaking |
| Monitoring and crisis response | €5B+ | Supply chain early warning system |
| Design capabilities | €2B | Support for European chip design companies |

### 4.3 Asia-Pacific Reshoring

| Country | Investment | Focus | Timeline |
|---------|-----------|-------|----------|
| **Japan** | ¥2T ($13B) | TSMC Kumamoto (JASM), Rapidus (2nm) | 2024–2027 |
| **South Korea** | $19B | Samsung Pyeongtaek expansion, SK Hynix Icheon HBM | 2025–2028 |
| **India** | $10B | Tata-PSMC Gujarat fab, ATMP facilities | 2026–2029 |
| **Singapore** | $5B | GlobalFoundries expansion | 2025–2027 |

### 4.4 Reshoring Reality Check

Despite massive investments, reshoring faces fundamental challenges:

1. **Cost differential**: US fab construction costs are 30–50% higher than Taiwan
2. **Talent shortage**: The US needs 50,000+ additional semiconductor workers; training takes 2–4 years
3. **Supply chain gaps**: Building a fab requires 500+ suppliers, most of which are in Asia
4. **Timeline mismatch**: fabs take 3–4 years to build; AI demand is growing NOW
5. **Yield ramp**: New fabs take 12–18 months to reach competitive yields

**Bottom line**: By 2028, US domestic chip production may reach 10–15% of global advanced node capacity (up from ~5% in 2025). Taiwan will remain dominant through at least 2030.

---

## 5. US-China Export Controls & Tech Decoupling

### 5.1 Timeline of Export Controls

| Date | Action | Impact |
|------|--------|--------|
| **Oct 2022** | BIS sweeping controls on advanced AI chips (A100, H100 banned) | NVIDIA creates A800/H800 variants for China market |
| **Oct 2023** | Controls expanded: A800/H800 now banned; "performance density" threshold lowered | NVIDIA creates H20/L20 variants; still legal for China |
| **Jan 2025** | Further tightening: inference chips covered; model weights potentially restricted | H20 sales now require license; black market grows |
| **Dec 2025** | Extended to advanced packaging equipment and certain EDA tools | TSMC restricted from producing chips for certain Chinese entities |

### 5.2 China's Domestic Chip Response

China has invested heavily in domestic semiconductor capabilities:

| Company/Initiative | Focus | Progress (June 2026) |
|-------------------|-------|---------------------|
| **Huawei Ascend 910C** | AI training chip | 7nm (SMIC DUV); 60–70% of H100 performance |
| **SMIC** | Fabrication | 7nm DUV (limited yield); 5nm in development |
| **Biren Technology** | GPU design | BR100 series; limited by SMIC capacity |
| **Moore Threads** | GPU design | MTT S80 series; consumer + data center |
| **Enflame** | AI accelerator | CloudBlazer series; competitive for inference |
| **Cambricon** | AI processor | Siyuan series; government procurement |
| **Huawei HiSilicon** | Kirin + Ascend | Design capability ahead of manufacturing capability |

### 5.3 The Black Market and Diversion Problem

Despite export controls, AI chips continue to reach China through:

1. **Third-country transshipment**: Chips shipped to Singapore, Malaysia, or UAE, then forwarded to China
2. **"Gray market" brokers**: Intermediaries who purchase legal chips and resell to restricted entities
3. **Model access workarounds**: Chinese companies access restricted models via API (e.g., through non-US providers)
4. **Stockpiling**: Chinese companies purchased massive quantities of H100/H800 before controls took effect

**Estimated diverted chip volume**: $5B–$10B annually (industry estimates vary widely)

### 5.4 Two-Chip-World Scenario

The most likely long-term outcome is a bifurcated global chip supply chain:

| Dimension | US-Aligned Stack | China-Aligned Stack |
|-----------|-----------------|-------------------|
| **Leading-edge fabrication** | TSMC (Taiwan), Samsung (Korea), Intel (US) | SMIC (China), limited to 7nm+ |
| **AI chip design** | NVIDIA, AMD, Google, Microsoft, Apple | Huawei, Biren, Moore Threads, Cambricon |
| **EUV lithography** | ASML (Netherlands) | No domestic alternative; attempting DUV multi-patterning |
| **HBM memory** | SK Hynix, Samsung, Micron | CXMT (China); 2+ years behind |
| **EDA tools** | Synopsys, Cadence, Siemens EDA | Empyrean, X-EPIC; limited to mature nodes |
| **AI models** | GPT-5, Claude 4, Gemini 3 | DeepSeek, Qwen, GLM, Baichuan |

---

## 6. The TSMC Dependency Problem

### 6.1 TSMC's Dominance in Numbers

| Metric | TSMC Share | Next Largest |
|--------|-----------|-------------|
| <5nm fabrication | ~90% | Samsung (~8%) |
| Advanced packaging (CoWoS) | ~95% | ASE (~3%) |
| AI GPU fabrication | ~95% | Samsung (~5%) |
| Apple chip fabrication | 100% | None |
| Global foundry revenue | ~57% | Samsung (~12%) |

### 6.2 TSMC's Geographic Expansion

TSMC is diversifying geographically, but slowly:

| Location | Node | Capacity (WPM) | Status | % of Global Advanced |
|----------|------|----------------|--------|---------------------|
| **Hsinchu, Taiwan** | N3, N2 | 100K+ | Production | ~40% |
| **Tainan, Taiwan** | N5, N3 | 150K+ | Production | ~35% |
| **Taichung, Taiwan** | N2, A16 | Planned | Under construction | Future |
| **Phoenix, AZ** | N4, N3 | 20K (target) | Fab 1 ramping | ~5% |
| **Kumamoto, Japan** | N12, N16 | 15K | Production (JASM) | ~3% |
| **Dresden, Germany** | N28, N16 | 30K (planned) | Under construction | ~2% |

### 6.3 The "Taiwan Risk Premium"

Financial markets have begun pricing in Taiwan risk:

- TSMC ADR trades at a 10–15% discount to Taiwan-listed shares (partly reflects geopolitical risk)
- Insurance premiums for TSMC-dependent companies increased 200%+ since 2022
- Some enterprises are maintaining 6–12 months of chip inventory (vs. traditional 3 months)
- "Taiwan discount" on AI company valuations: estimated 5–10% for companies with >80% TSMC dependency

---

## 7. ASML and the EUV Monopoly

### 7.1 Why ASML Is the Most Important Company in AI

ASML's EUV lithography machines are the only way to manufacture chips at nodes below 7nm. Without ASML:

- No NVIDIA B200/GB200
- No Apple M4/M5
- No Google TPU v6+
- No AMD MI400
- No advanced AI chips of any kind

### 7.2 ASML's Product Line

| Product | Type | Resolution | Throughput | Price | Customers |
|---------|------|-----------|-----------|-------|-----------|
| **TWINSCAN NXE:3800E** | EUV | 13nm | 185 WPH | ~$200M | TSMC, Samsung, Intel |
| **TWINSCAN EXE:5000** | High-NA EUV | 8nm | 185 WPH | ~$380M | TSMC, Intel (2026–2027) |
| **TWINSCAN NXT:2100i** | DUV immersion | 38nm | 300+ WPH | ~$80M | Legacy nodes, multi-patterning |

### 7.3 ASML's Supply Chain

ASML itself depends on a concentrated supply chain:

| Component | Supplier | Location | Dependency |
|-----------|---------|----------|------------|
| EUV light source | Trumpf | Germany | Sole supplier |
| Mirrors/lenses | Carl Zeiss SMT | Germany | Sole supplier |
| Wafer stages | ASML (internal) | Netherlands | — |
| Software | ASML (internal) | Netherlands | — |
| Key optics coating | Carl Zeiss | Germany | Sole supplier |

### 7.4 Export Control Implications

ASML is subject to Dutch export controls (aligned with US restrictions):

- Cannot sell EUV machines to China (since 2019)
- DUV immersion machines restricted for certain Chinese customers (since 2023)
- Potential future restriction on all advanced DUV sales to China
- ASML's China revenue declined from ~15% to ~8% of total (2023–2026)

---

## 8. Packaging Bottlenecks: CoWoS and Beyond

### 8.1 Why Packaging Is the New Bottleneck

The shift to advanced packaging (chiplets, 3D stacking, HBM integration) has made packaging the binding constraint on AI chip supply:

- A single B200 GPU requires: 1 GPU die + 8 HBM3e stacks + silicon interposer + substrate
- CoWoS (Chip-on-Wafer-on-Substrate) integrates all these components
- TSMC's CoWoS capacity was designed for HPC, not the AI explosion

### 8.2 CoWoS Capacity vs. Demand

| Year | CoWoS Capacity (wafers/month) | AI Chip Demand (equivalent) | Gap |
|------|-------------------------------|---------------------------|-----|
| 2024 | ~35K | ~45K | -10K (severe shortage) |
| 2025 | ~45K | ~55K | -10K (continued shortage) |
| 2026 | ~60K | ~65K | -5K (improving but tight) |
| 2027 | ~85K (projected) | ~80K | +5K (near balance) |

### 8.3 Alternative Packaging Technologies

| Technology | Developer | Throughput | Cost | Readiness |
|-----------|----------|-----------|------|-----------|
| **CoWoS-S** | TSMC | High | High | Production |
| **CoWoS-R** | TSMC | Higher | Lower | 2026 |
| **I-Cube** | Samsung | Medium | Medium | Production (limited) |
| **Foveros** | Intel | Medium | Medium | Production (Logic only) |
| **Hybrid Bonding** | Multiple | Low | Very High | R&D/limited production |

### 8.4 The HBM Packaging Challenge

HBM (High-Bandwidth Memory) packaging is itself a bottleneck:

- Each HBM stack requires TSV (Through-Silicon Via) fabrication
- SK Hynix is the leader in HBM packaging yield
- Samsung is ramping HBM3e production but lagging in yield
- HBM4 (expected 2027) will require even more advanced packaging

---

## 9. Rare Earth Minerals and Critical Materials

### 9.1 Materials Required for AI Chips

| Material | Use in AI Chips | Primary Source | Risk |
|----------|----------------|---------------|------|
| **Silicon** | Wafer substrate | Japan (50%+), China (15%) | Low-Moderate |
| **Neon** | EUV lithography gas | Ukraine (was 50%), now diversified | Moderate |
| **Palladium** | Capacitors, connectors | Russia (40%), South Africa (35%) | High |
| **Gallium** | Compound semiconductors | China (95%) | High |
| **Germanium** | Fiber optics, IR optics | China (60%) | High |
| **Tantalum** | Capacitors | DRC (40%), Rwanda (20%) | High (conflict minerals) |
| **Cobalt** | Plating, batteries | DRC (70%) | High (conflict minerals) |
| **Lithium** | Data center batteries | Australia (45%), Chile (25%) | Moderate |
| **Copper** | Interconnects, wiring | Chile (25%), Peru (10%) | Moderate |
| **Rare earths** | Magnets in cooling systems | China (60%), Myanmar (15%) | High |

### 9.2 China's Critical Mineral Leverage

China's dominance in critical minerals creates additional supply chain risk:

- In 2023, China imposed export controls on gallium and germanium
- In 2024, China imposed export controls on rare earth processing technology
- In 2025, China restricted graphite exports (used in battery anodes)
- These controls are widely viewed as retaliation for US chip export controls

### 9.3 Diversification Efforts

| Initiative | Focus | Investment | Timeline |
|-----------|-------|-----------|----------|
| US Critical Minerals Strategy | Gallium, germanium production | $1.5B | 2025–2030 |
| Australian Critical Minerals | Lithium, rare earths | A$2B | 2025–2028 |
| EU Critical Raw Materials Act | Reduce China dependency to <65% by 2030 | €2B | 2025–2030 |
| African mining partnerships | Cobalt, tantalum, manganese | Varies | Ongoing |

---

## 10. The Reshoring and Friend-Shoring Movement

### 10.1 Definitions

- **Reshoring**: Bringing manufacturing back to the home country (e.g., US fabs in Arizona)
- **Friend-shoring**: Moving supply chains to allied/like-minded countries (e.g., TSMC in Japan)
- **Near-shoring**: Moving supply chains closer to the home market (e.g., Mexico for US)
- **China+1**: Maintaining China operations while adding capacity elsewhere

### 10.2 Who's Friend-Shoring Where

| Company | From | To | Strategy |
|---------|------|-----|----------|
| TSMC | Taiwan | Japan, US, Germany | Diversify geopolitical risk |
| Samsung | South Korea | US, Vietnam | US CHIPS Act incentives |
| Intel | US | Ireland, Germany, Israel | Already global; expanding |
| Apple | China | India, Vietnam | Manufacturing diversification |
| NVIDIA | (design in US) | — | Fabrication stays at TSMC; design diversifying |

### 10.3 The Cost of Reshoring

Reshoring semiconductor manufacturing is expensive:

| Factor | Taiwan | US | Delta |
|--------|--------|-----|-------|
| Fab construction cost | $15B | $22B | +47% |
| Operating cost (per wafer) | $8,000 | $12,000 | +50% |
| Worker wages (annual) | $45K | $85K | +89% |
| Power cost (per kWh) | $0.10 | $0.08 | -20% (US cheaper) |
| Water cost | Higher | Lower | US advantage |
| Regulatory burden | Lower | Higher | US disadvantage |
| Yield maturity | 12 months | 18–24 months | US slower |

---

## 11. Supply Chain Risk Quantification Framework

### 11.1 The SCRAM Framework

**Supply Chain Risk Assessment for AI Manufacturers (SCRAM)**:

| Risk Category | Metrics | Weight | Scoring |
|--------------|---------|--------|---------|
| **S**ingle-source dependency | % of supply from sole source | 25% | 1 (diversified) to 5 (sole source) |
| **C**oncentration | Geographic concentration (HHI) | 20% | 1 (global) to 5 (single country) |
| **R**ecoverability | Time to restore capacity | 20% | 1 (<1 month) to 5 (>2 years) |
| **A**lternatives | Available substitutes | 20% | 1 (many) to 5 (none) |
| **M**arket power | Supplier's market dominance | 15% | 1 (competitive) to 5 (monopoly) |

### 11.2 Example Risk Assessment: NVIDIA B200 GPU

| Component | S | C | R | A | M | Weighted Score | Risk Level |
|-----------|---|---|---|---|---|---------------|------------|
| GPU die (TSMC 4N) | 5 | 5 | 4 | 1 | 5 | 4.2 | 🔴 CRITICAL |
| HBM3e (SK Hynix) | 4 | 4 | 3 | 2 | 4 | 3.4 | 🟠 HIGH |
| CoWoS packaging (TSMC) | 5 | 5 | 3 | 1 | 5 | 4.0 | 🔴 CRITICAL |
| Substrate (Ibiden/Shinko) | 3 | 3 | 2 | 3 | 3 | 2.8 | 🟡 MODERATE |
| Thermal solution (Cooler Master) | 1 | 1 | 1 | 5 | 1 | 1.6 | 🟢 LOW |
| **Overall B200 Supply Chain** | | | | | | **3.2** | 🟠 HIGH |

### 11.3 Risk Mitigation Strategies for AI Companies

| Strategy | Description | Cost | Effectiveness |
|----------|-------------|------|--------------|
| **Multi-source qualification** | Qualify 2+ suppliers for critical components | Medium | High |
| **Strategic inventory** | Maintain 6–12 months of critical chip inventory | High (capital) | High |
| **Design for alternative silicon** | Architect software to run on multiple chip vendors | High (engineering) | Very High |
| **Long-term supply agreements** | Lock in capacity with TSMC/SK Hynix 2+ years ahead | Medium | High |
| **Geographic diversification** | Deploy inference across multiple regions/clouds | Medium | Medium |
| **Vertical integration** | Design custom silicon (like Google TPU, Microsoft Maia) | Very High | Very High |
| **Secondary market sourcing** | Monitor broker market for available capacity | Low | Low-Medium |

---

## 12. Implications for AI Builders

### 12.1 What Every AI Company Should Do Now

1. **Map your supply chain**: Know exactly which chips, memory, and packaging your AI systems depend on
2. **Quantify your TSMC exposure**: If >50% of your compute depends on TSMC advanced nodes, you have concentration risk
3. **Build multi-silicon capability**: Ensure your inference stack can run on at least 2 different silicon vendors
4. **Secure capacity commitments**: Lock in CoWoS and HBM capacity 12–24 months ahead
5. **Diversify geopolitically**: Deploy inference across multiple countries/regions
6. **Monitor export control changes**: Subscribe to BIS Federal Register notices
7. **Stockpile strategically**: 6 months of critical chips is the new normal

### 12.2 The Strategic Moat of Supply Chain Control

In 2026, supply chain access is becoming a competitive moat:

- Companies with TSMC capacity commitments can deploy AI faster
- Companies with custom silicon (Google, Microsoft, Meta) are insulated from market shortages
- Companies with multi-silicon inference can optimize costs across vendors
- Companies that solved CoWoS access have 6–12 month deployment advantages

### 12.3 2027–2030 Outlook

| Trend | Timeline | Impact |
|-------|----------|--------|
| TSMC Arizona reaches volume production | 2027 | US domestic advanced chip supply increases to ~10% |
| Intel 18A node ramps | 2026–2027 | First credible TSMC alternative for leading-edge |
| Samsung GAA (Gate-All-Around) at 2nm | 2027 | Foundry competition intensifies |
| CoWoS capacity doubles | 2027 | Packaging bottleneck eases |
| China achieves 5nm DUV multi-patterning | 2028–2029 | Parallel chip supply chain matures |
| EUV High-NA in volume production | 2028 | Next-generation chip capabilities |
| On-shore US capacity reaches 15%+ | 2028–2029 | Reduced Taiwan dependency |

---

## 13. Cross-References

| Document | Category | Relevance |
|----------|----------|-----------|
| `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` | LLMs | Deep-dive on chip architectures, software stacks, and cost modeling |
| `25-Multi-Cloud-AI-Strategy/` | Multi-Cloud | Cloud procurement and multi-cloud inference deployment |
| `35-AI-Energy-and-Sustainability/` | Energy | Data center power constraints and energy costs |
| `23-Local-AI-Inference-Self-Hosting/` | Local AI | On-premise inference as supply chain hedge |
| `30-Small-Language-Models/` | SLMs | Efficiency and smaller models as supply chain optimization |
| `21-AI-Regulation-Antitrust/` | Regulation | Export controls, antitrust, and industrial policy |
| `05-Enterprise/` | Enterprise | Enterprise AI deployment and procurement |
| `13-Top-Demand/` | Top Demand | Market demand signals and compute economics |

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **CoWoS** | Chip-on-Wafer-on-Substrate — TSMC's advanced packaging technology for integrating GPU dies with HBM memory |
| **DUV** | Deep Ultraviolet lithography — older lithography technology (193nm) still used for mature nodes |
| **EUV** | Extreme Ultraviolet lithography — 13.5nm wavelength technology required for sub-5nm chips |
| **HBM** | High-Bandwidth Memory — 3D-stacked DRAM used in AI accelerators for high-throughput data access |
| **HHI** | Herfindahl-Hirschman Index — measure of market concentration (higher = more concentrated) |
| **Multi-patterning** | Using multiple lithography exposures to achieve finer features than a single pass allows |
| **Node** | A generation of chip manufacturing technology (e.g., 3nm, 5nm); smaller = more transistors per area |
| **Reshoring** | Moving manufacturing back to the home country from overseas |
| **Silicon Shield** | Theory that Taiwan's critical role in chip manufacturing deters military aggression |
| **Supply chain** | The network of organizations and processes involved in producing and delivering a product |
| **TSV** | Through-Silicon Via — vertical electrical connections through a silicon wafer, used in HBM stacking |
| **Wafer** | A thin disc of silicon used as the substrate for manufacturing integrated circuits |

---

*Last updated: June 29, 2026*
*See also: The AI Supply Chain Dashboard (real-time monitoring of fab capacity, export controls, and geopolitical risk)*
