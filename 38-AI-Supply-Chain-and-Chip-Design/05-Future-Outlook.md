# 05 — Future Outlook: AI Supply Chain and Chip Design 2026–2035

> **Why this document exists.** The AI chip supply chain is undergoing its most dramatic transformation since the invention of the integrated circuit. This document projects the next decade of developments: which technologies will mature, which bottlenecks will ease, which new risks will emerge, and what strategic moves will define winners and losers. We cover the semiconductor roadmap, geopolitical trajectories, supply chain evolution, and the long-term implications for AI builders.

---

## Table of Contents

1. [Semiconductor Technology Roadmap (2026–2035)](#1-semiconductor-technology-roadmap-2026-2035)
2. [Supply Chain Evolution](#2-supply-chain-evolution)
3. [Geopolitical Trajectories](#3-geopolitical-trajectories)
4. [Emerging Technologies](#4-emerging-technologies)
5. [Market Projections](#5-market-projections)
6. [Strategic Implications for AI Builders](#6-strategic-implications-for-ai-builders)
7. [Risk Scenarios](#7-risk-scenarios)
8. [Cross-References](#8-cross-references)

---

## 1. Semiconductor Technology Roadmap (2026–2035)

### 1.1 Node Scaling Trajectory

| Year | TSMC | Samsung | Intel | Key Feature |
|------|------|---------|-------|-------------|
| 2026 | N2 (2nm) | GAA 2nm | Intel 18A (1.8nm) | GAA transistors in production |
| 2027 | N2P | GAA 2nm+ | Intel 14A (1.4nm) | Backside power delivery (N2P) |
| 2028 | A16 (1.6nm) | CFET 1.4nm | Intel 10A (1nm) | CFET (Complementary FET) R&D |
| 2029 | A14 (1.4nm) | CFET 1.2nm | Intel 7A (0.7nm) | 2D materials research |
| 2030 | A10 (1nm) | — | — | Physical limits approach |
| 2031–2035 | Below 1nm (research) | — | — | 2D channels, carbon nanotubes |

### 1.2 The End of Classical Scaling

Classical transistor scaling (Dennard scaling) ended around 2006. Since then, performance gains come from:

| Technique | Contribution (2026) | Contribution (2030 est.) |
|-----------|--------------------|-----------------------|
| **Transistor density** | 40% | 25% (slowing) |
| **Architectural innovation** | 25% | 35% (increasing) |
| **Packaging** | 20% | 25% |
| **Software optimization** | 15% | 15% |

**The implication**: By 2030, more AI performance improvement will come from packaging and architecture than from transistor scaling alone.

### 1.3 Packaging as the New Scaling Vector

| Year | Packaging Technology | Capability |
|------|---------------------|-----------|
| 2026 | CoWoS-S/R, I-Cube | 2.5D integration, 3–5 chiplets |
| 2027 | CoWoS-L, hybrid bonding | Higher density, 5–10 chiplets |
| 2028 | 3D stacking (Foveros, SoIC) | True 3D, 10+ chiplets |
| 2029 | Monolithic 3D | Transistors on multiple layers |
| 2030+ | Wafer-scale integration | Cerebras-style for more vendors |

### 1.4 The Chiplet Ecosystem Maturity

| Year | Chiplet Ecosystem | Impact |
|------|------------------|--------|
| 2026 | UCIe 2.0 in production | First multi-vendor chiplet products |
| 2027 | UCIe 3.0 specification | Higher bandwidth, lower power |
| 2028 | Chiplet marketplace emerges | Buy chiplets from different vendors |
| 2029 | Standardized chiplet interfaces | Mix GPU, memory, I/O from any vendor |
| 2030 | Chiplet-as-a-Service | Custom AI chips assembled from catalog |

---

## 2. Supply Chain Evolution

### 2.1 Geographic Diversification Trajectory

| Region | 2026 Advanced Node Share | 2030 Projected Share | Key Driver |
|--------|------------------------|---------------------|-----------|
| **Taiwan** | ~65% | ~45% | TSMC expansion elsewhere |
| **South Korea** | ~15% | ~18% | Samsung GAA ramp |
| **United States** | ~10% | ~18% | CHIPS Act fabs |
| **Japan** | ~3% | ~8% | Rapidus, JASM |
| **Europe** | ~2% | ~5% | EU Chips Act |
| **China** | ~5% (mature) | ~8% (mature) | Domestic development |

### 2.2 The 2026–2030 Reshoring Timeline

| Milestone | Date | Impact |
|-----------|------|--------|
| TSMC Arizona Fab 1 (N4) volume production | 2026 | First US advanced node production |
| TSMC Arizona Fab 2 (N3) start of production | 2027 | US catches up on 3nm |
| Intel 18A volume production | 2027 | First competitive US alternative to TSMC |
| Rapidus (Japan) 2nm pilot | 2027 | Japan returns to leading edge |
| TSMC Arizona Fab 3 (N2) start | 2028 | US reaches 2nm |
| Samsung Texas 2nm start | 2028 | Korean diversification |
| Intel 14A volume | 2028 | Intel catches up |
| US advanced node share reaches 15% | 2029 | Meaningful diversification |
| TSMC Japan 3nm start | 2030 | Japan becomes significant player |

### 2.3 The Supply Chain Structure of 2030

**2026 structure (today):**
```
Design (US) → Fabrication (Taiwan) → Packaging (Taiwan) → Assembly (Asia) → Deploy (Global)
```

**2030 structure (projected):**
```
Design (US, EU, Asia) → Fabrication (Taiwan, US, Korea, Japan) → Packaging (Taiwan, US, Korea) → Assembly (Global) → Deploy (Global)
```

### 2.4 The Multi-Source Future

By 2030, AI companies will have genuine multi-source options:

| Component | 2026 Source Options | 2030 Source Options |
|-----------|-------------------|-------------------|
| **Leading-edge fabrication** | TSMC (primary), Samsung (limited) | TSMC, Samsung, Intel, Rapidus |
| **Advanced packaging** | TSMC (dominant) | TSMC, Samsung, Intel, ASE |
| **HBM memory** | SK Hynix, Samsung, Micron | SK Hynix, Samsung, Micron, CXMT |
| **EUV lithography** | ASML (sole) | ASML (still sole, but High-NA) |
| **Chiplet interconnect** | Proprietary (NVLink) | UCIe standard (multi-vendor) |

---

## 3. Geopolitical Trajectories

### 3.1 Three Geopolitical Scenarios (2026–2035)

**Scenario 1: Managed Competition (40% probability)**
- US and China maintain export controls but avoid escalation
- Parallel supply chains develop but don't fully decouple
- Taiwan remains autonomous; TSMC diversifies geographically
- AI development continues on both sides with different chip stacks
- **Impact on AI**: Slower global progress, but both ecosystems advance

**Scenario 2: Full Decoupling (25% probability)**
- US tightens controls; China retaliates with rare earth restrictions
- Complete separation of chip supply chains
- China achieves domestic 5nm by 2029; builds parallel ecosystem
- Two competing AI standards emerge
- **Impact on AI**: Significant fragmentation, higher costs, slower innovation

**Scenario 3: Taiwan Crisis (15% probability)**
- Military conflict or blockade disrupts TSMC
- Global chip supply drops 40–60% for 6–18 months
- Emergency reshoring accelerated by 5+ years
- AI deployment delayed globally by 12–24 months
- **Impact on AI**: Severe short-term disruption, permanent reshoring acceleration

**Scenario 4: Gradual Diversification (20% probability)**
- No major crisis; reshoring proceeds slowly over 10+ years
- Taiwan remains dominant through 2030+; diversification is gradual
- Geopolitical tension remains but doesn't disrupt supply
- **Impact on AI**: Minimal near-term disruption; long-term resilience improves

### 3.2 The "Silicon Shield" Reassessment

The traditional "Silicon Shield" theory (Taiwan's chip dominance deters invasion) is being reassessed:

| Factor | Supports Shield | Undermines Shield |
|--------|----------------|------------------|
| **China's TSMC dependency** | China needs TSMC for own chips | China is building domestic capacity |
| **Global economic impact** | Invasion would cost China economically | Economic coercion may precede military action |
| **US commitment** | Taiwan Relations Act | US domestic political shifts |
| **Diversification** | TSMC building outside Taiwan | Reduces Taiwan's leverage over time |
| **Time horizon** | Shield strengthens short-term | Shield weakens as alternatives emerge |

### 3.3 Export Control Evolution

| Year | Expected Development | Impact |
|------|---------------------|--------|
| 2026 | Controls extended to inference chips | Broader impact on Chinese AI |
| 2027 | Controls on advanced packaging equipment | Slows Chinese chip advancement |
| 2028 | Model weight export restrictions debated | Potential impact on AI model access |
| 2029 | Multilateral controls (EU, Japan align) | Stronger enforcement |
| 2030+ | New framework for AI-specific controls | Comprehensive AI governance |

---

## 4. Emerging Technologies

### 4.1 Photonic Computing

Photonic (optical) computing could revolutionize AI chips:

| Aspect | Current (Electronic) | Photonic (2030+) |
|--------|---------------------|-----------------|
| Energy per operation | ~1 pJ | ~0.01 pJ (100x improvement) |
| Bandwidth | TB/s | PB/s (optical interconnect) |
| Speed of light | Limited by wire delay | Speed of light in glass |
| Maturity | Production | R&D / early prototype |

**Key players**: Lightmatter, Ayar Labs, Celestial AI, Intel (silicon photonics)

### 4.2 In-Memory Computing

Processing data where it's stored, eliminating the memory wall:

| Approach | Technology | Speedup | Maturity |
|----------|-----------|---------|----------|
| **Analog in-memory** | RRAM, PCM | 10–100x for matrix multiply | Prototype |
| **Digital in-memory** | SRAM-based | 5–10x | Early production |
| **Processing-in-memory** | HBM with compute | 3–5x | R&D |

### 4.3 Quantum-AI Hybrid Systems

Quantum computing may complement AI for specific workloads:

| Application | Quantum Advantage | Timeline |
|------------|------------------|----------|
| **Optimization** (supply chain) | Exponential speedup for certain problems | 2028–2030 |
| **Sampling** (generative models) | Faster MCMC sampling | 2029–2032 |
| **Linear algebra** (training) | Potential speedup for specific operations | 2032+ |

### 4.4 Neuromorphic Computing

Brain-inspired computing for ultra-low-power AI:

| System | Neurons | Synapses | Power | Use Case |
|--------|---------|----------|-------|----------|
| Intel Loihi 3 | 1B | 120B | 1W | Edge AI, robotics |
| IBM NorthPole | 256 cores | — | 75W | Inference |
| BrainChip Akida | 1.2M | 10B | 0.5W | Edge inference |

### 4.5 Superconducting Computing

Ultra-fast, ultra-low-power computing using superconducting electronics:

| Aspect | CMOS | Superconducting |
|--------|------|----------------|
| Switching speed | ~10 GHz | ~100+ GHz |
| Power per switch | ~1 fJ | ~1 aJ (1000x less) |
| Operating temp | Room temperature | 4K (cryogenic) |
| Maturity | Production | Research |
| Timeline to AI relevance | — | 2035+ |

---

## 5. Market Projections

### 5.1 AI Chip Market Size

| Year | Market Size | Growth | Key Driver |
|------|-----------|--------|-----------|
| 2024 | $80B | — | H100/B200 demand |
| 2025 | $140B | +75% | Training + inference expansion |
| 2026 | $220B | +57% | Enterprise AI deployment |
| 2027 | $320B | +45% | Edge AI + automotive |
| 2028 | $420B | +31% | Inference-dominant market |
| 2030 | $600B | — | Full AI integration |

### 5.2 Foundry Market Share Projections

| Company | 2026 Share | 2030 Projected | Trend |
|---------|-----------|---------------|-------|
| **TSMC** | 57% | 48% | Declining (diversification) |
| **Samsung** | 12% | 15% | Growing (GAA ramp) |
| **Intel Foundry** | 3% | 10% | Growing (CHIPS Act, 18A) |
| **GlobalFoundries** | 5% | 5% | Stable (mature nodes) |
| **SMIC** | 6% | 8% | Growing (China domestic) |
| **Rapidus** | 0% | 3% | New entrant (Japan 2nm) |
| **Others** | 17% | 11% | Consolidating |

### 5.3 HBM Market Projections

| Year | HBM Revenue | Capacity (GB/month) | Key Development |
|------|-----------|-------------------|----------------|
| 2026 | $25B | 500M GB | HBM3E dominant |
| 2027 | $35B | 800M GB | HBM4 introduction |
| 2028 | $45B | 1.2B GB | HBM4 volume |
| 2029 | $55B | 1.8B GB | HBM4E |
| 2030 | $65B | 2.5B GB | HBM5 |

---

## 6. Strategic Implications for AI Builders

### 6.1 Near-Term Strategy (2026–2027)

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Secure TSMC capacity | Capacity still tight through 2027 |
| 2 | Qualify 2+ silicon vendors | Reduce single-source dependency |
| 3 | Build CoWoS buffer stock | Packaging is the binding constraint |
| 4 | Diversify inference deployment | Multi-region reduces geopolitical risk |
| 5 | Monitor export control changes | Regulatory landscape is volatile |

### 6.2 Medium-Term Strategy (2028–2030)

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Adopt chiplet-based designs | UCIe ecosystem matures |
| 2 | Evaluate Intel Foundry | 18A/14A becomes credible alternative |
| 3 | Consider custom silicon | If scale justifies $1B+ investment |
| 4 | Build supply chain digital twin | Predictive risk management |
| 5 | Invest in photonics partnerships | Prepare for 2030+ transition |

### 6.3 Long-Term Strategy (2031–2035)

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Plan for post-CMOS | Photonic/neuromorphic options mature |
| 2 | Build multi-geography manufacturing | True supply chain resilience |
| 3 | Develop quantum-AI hybrid capabilities | Optimization and sampling advantages |
| 4 | Invest in 2D material research | Graphene, MoS2 channels |
| 5 | Prepare for AI-specific regulations | Comprehensive governance frameworks |

### 6.4 The Winner's Playbook

The companies that will dominate AI through 2035 share these characteristics:

1. **Supply chain control**: Long-term capacity commitments, custom silicon, multi-source qualification
2. **Software-hardware co-design**: Compiler and kernel optimization for their specific silicon
3. **Vertical integration**: Design → fabrication → deployment in-house (or closely partnered)
4. **Geographic diversification**: Inference and training distributed across multiple regions
5. **Risk management**: Digital twins, scenario planning, proactive disruption response

---

## 7. Risk Scenarios

### 7.1 The "What If" Matrix

| Scenario | Probability (10yr) | Impact | Recovery | Mitigation |
|----------|-------------------|--------|----------|-----------|
| **TSMC disruption (earthquake)** | 30–40% | Severe (6–12 months) | 6–12 months | Safety stock, multi-source |
| **TSMC disruption (geopolitical)** | 15–25% | Catastrophic (12+ months) | 2–5 years | Reshoring acceleration |
| **Export control escalation** | 40–50% | Moderate (slows China AI) | Ongoing | Diversified customer base |
| **Rare earth restriction** | 25–35% | Moderate (material shortage) | 1–2 years | Material substitution R&D |
| **ASML technology restriction** | 10–15% | Severe (no new fabs) | 5–10 years | Multi-patterning alternatives |
| **Power grid failure** | 20–30% | Moderate (data center outage) | Days–weeks | Redundant power, backup |
| **Workforce shortage** | 60–70% | Moderate (slower buildout) | Ongoing | Automation, training programs |
| **New physics discovery** | 5–10% | Variable (could disrupt or help) | Unknown | Research partnerships |

### 7.2 The Black Swan Scenarios

| Scenario | Impact | Why It Matters |
|----------|--------|---------------|
| **Successful Chinese EUV** | Reshuffles entire supply chain | Would break ASML monopoly |
| **Room-temperature superconductor** | Revolutionary power efficiency | Would transform chip design |
| **AGI breakthrough** | Accelerates all chip design | AI designs its own chips |
| **Global chip embargo** | Complete supply chain halt | Would force massive reshoring |
| **New computing paradigm** | Makes current chips obsolete | Photonic, quantum, or neuromorphic |

---

## 8. Cross-References

| Document | Category | Relevance |
|----------|----------|-----------|
| `38-AI-Supply-Chain-and-Chip-Design/01-Overview.md` | Supply Chain | Strategic overview |
| `38-AI-Supply-Chain-and-Chip-Design/02-Core-Topics.md` | Supply Chain | Core technical topics |
| `38-AI-Supply-Chain-and-Chip-Design/03-Technical-Deep-Dive.md` | Supply Chain | Engineering details |
| `38-AI-Supply-Chain-and-Chip-Design/04-Tools-and-Frameworks.md` | Supply Chain | Tools and platforms |
| `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` | LLMs | Detailed chip cost tables |
| `17-Research-Frontiers-2026/` | Research | Emerging technologies |
| `35-AI-Energy-and-Sustainability/` | Energy | Power constraints |
| `21-AI-Regulation-Antitrust/` | Regulation | Export controls and policy |

---

*Last updated: June 29, 2026*
