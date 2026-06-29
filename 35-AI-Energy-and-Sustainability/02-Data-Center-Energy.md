# 02 — Data Center Energy: Power, Cooling, and the Grid

> **Category:** 35-AI-Energy-and-Sustainability
> **Last updated:** June 29, 2026
> **Cross-references:** `05-Enterprise/04-AI-Infrastructure.md`, `01-Overview.md`, `25-Multi-Cloud-AI-Strategy/01-Overview.md`

---

## Table of Contents

1. [Data Center Power Fundamentals](#1-data-center-power-fundamentals)
2. [The Grid Connection Problem](#2-the-grid-connection-problem)
3. [Power Distribution Architecture](#3-power-distribution-architecture)
4. [Cooling Systems Deep Dive](#4-cooling-systems-deep-dive)
5. [Power Usage Effectiveness (PUE)](#5-power-usage-effectiveness-pue)
6. [Water Usage Effectiveness (WUE)](#6-water-usage-effectiveness-wue)
7. [Energy Storage and Backup](#7-energy-storage-and-backup)
8. [The Nuclear Option](#8-the-nuclear-option)
9. [Renewable Energy Integration](#9-renewable-energy-integration)
10. [Case Studies: Powering AI at Scale](#10-case-studies-powering-ai-at-scale)
11. [Cross-References](#11-cross-references)

---

## 1. Data Center Power Fundamentals

### Power Requirements by AI Workload

Different AI workloads have vastly different power profiles:

| Workload | Power Density (kW/rack) | Duration | Total Energy |
|----------|------------------------|----------|--------------|
| Model training (GPT-5 scale) | 40-100 kW | 3-6 months | 50-200 GWh |
| Fine-tuning (7B model) | 5-15 kW | Hours-days | 1-50 MWh |
| Inference (high-throughput) | 10-30 kW | Continuous | 50-200 kWh/day |
| Inference (low-latency) | 15-40 kW | Continuous | 100-400 kWh/day |
| RAG pipeline | 5-20 kW | Continuous | 30-100 kWh/day |

### The Rack Power Problem

Modern AI racks are power-dense:

```
Traditional IT rack:     5-10 kW/rack
Cloud-optimized rack:    15-25 kW/rack
AI training rack:        40-100 kW/rack
Next-gen AI rack (2027): 100-200 kW/rack
```

This density creates cascading challenges:
- **Electrical infrastructure**: Standard data center wiring can't handle 100kW/rack
- **Cooling**: Air cooling maxes out at ~25kW/rack; liquid cooling needed above that
- **Floor loading**: AI racks weigh 2-3x more than standard IT racks
- **Fire suppression**: Higher power density requires upgraded suppression systems

### Power Delivery Efficiency

From grid to GPU, power is lost at every conversion step:

| Stage | Typical Efficiency | Loss |
|-------|-------------------|------|
| Grid transmission | 95% | 5% |
| Substation transformation | 98% | 2% |
| UPS (uninterruptible power supply) | 96% | 4% |
| PDU (power distribution unit) | 97% | 3% |
| Server PSU (power supply unit) | 92-96% | 4-8% |
| Voltage regulator (on motherboard) | 95% | 5% |
| **Total (grid to chip)** | **~78%** | **~22%** |

This means for every 100W of grid power, only ~78W reaches the GPU. The remaining 22W becomes heat that must be removed by cooling systems.

---

## 2. The Grid Connection Problem

### The Queue

In 2026, connecting a new data center to the power grid is one of the biggest bottlenecks:

**US Market:**
- Northern Virginia: 3-5 year wait for utility connections
- Central Oregon: 2-4 year wait
- Dallas-Fort Worth: 2-3 year wait
- Chicago suburbs: 3-5 year wait

**European Market:**
- Dublin, Ireland: Moratorium on new connections (partially lifted)
- Amsterdam: Strict limits on new data center power
- Frankfurt: Power grid at capacity; new connections require grid upgrades
- London: National Grid queues growing to 5+ years

**Asia-Pacific:**
- Singapore: Temporary construction ban (2019-2023)
- Tokyo: Grid constraints limiting new builds
- Sydney: 3-5 year connection queues

### Grid Capacity by Region

| Region | Available Grid Capacity (GW) | DC Demand (GW) | Deficit |
|--------|------------------------------|----------------|---------|
| Northern Virginia | 3.5 | 5.0+ | -1.5 GW |
| Dublin | 0.8 | 1.5+ | -0.7 GW |
| Amsterdam | 1.2 | 2.0+ | -0.8 GW |
| Singapore | 0.5 | 1.0+ | -0.5 GW |
| Phoenix, AZ | 2.0 | 2.5+ | -0.5 GW |

### Utility Response Strategies

Utilities are responding with:

1. **Grid upgrades**: Multi-billion dollar transmission line projects
2. **Time-of-use pricing**: Higher rates during peak hours
3. **Demand response programs**: Paying data centers to reduce load during grid stress
4. **Behind-the-meter generation**: On-site power generation (gas, nuclear, solar)
5. **Grid-scale energy storage**: Battery systems to buffer demand spikes

---

## 3. Power Distribution Architecture

### Traditional vs. AI-Optimized

**Traditional Data Center Power Chain:**
```
Grid → Substation → Medium Voltage Switchgear → Transformer → UPS → PDU → Rack → Server
```

**AI-Optimized Power Chain:**
```
Grid → Substation → Medium Voltage Switchgear → Busway → Direct-to-Rack UPS → Liquid Cooled Rack
```

Key differences:
- **Busway distribution**: Eliminates traditional PDUs for higher efficiency
- **Direct-to-rack UPS**: Smaller, more efficient UPS per rack
- **48V DC distribution**: Emerging standard for AI racks (vs. traditional 12V)
- **Busbar systems**: Higher current capacity for dense deployments

### Power Architecture Comparison

| Architecture | Efficiency | Cost ($/kW) | Density Support | Reliability |
|-------------|-----------|-------------|-----------------|-------------|
| Traditional (2N) | 85-88% | $8,000-12,000 | ≤25 kW/rack | 99.999% |
| Simplified (N+1) | 88-91% | $6,000-9,000 | ≤50 kW/rack | 99.99% |
| Busway | 91-93% | $7,000-10,000 | ≤100 kW/rack | 99.99% |
| Direct liquid | 93-95% | $9,000-14,000 | ≤200 kW/rack | 99.99% |

### Electrical Standards for AI Data Centers

- **IEEE 1613**: Environmental testing for substation equipment
- **IEC 62040**: Uninterruptible power systems
- **NEC Article 645**: Information technology equipment rooms
- **NFPA 75**: Standard for protection of IT equipment
- **ASHRAE TC 9.9**: Thermal guidelines for data processing environments

---

## 4. Cooling Systems Deep Dive

### Why Cooling Matters for AI

Cooling is the second-largest energy consumer in data centers (after IT equipment):

- **Traditional IT**: 30-40% of total energy goes to cooling
- **AI workloads**: 20-35% of total energy (higher IT load means cooling is a smaller fraction, but absolute cooling demand is higher)

### Cooling Technology Spectrum

| Technology | Max Density | PUE Impact | Water Use | Capital Cost | Maturity |
|-----------|-------------|-----------|-----------|--------------|----------|
| Raised floor air | 10 kW/rack | 1.4-1.8 | High | Low | Mature |
| In-row cooling | 25 kW/rack | 1.3-1.5 | Medium | Medium | Mature |
| Rear-door heat exchanger | 35 kW/rack | 1.2-1.4 | Low | Medium | Mature |
| Direct liquid cooling (DLC) | 100 kW/rack | 1.05-1.15 | Low | High | Growing |
| Immersion cooling | 200+ kW/rack | 1.02-1.08 | Very low | Very high | Emerging |
| Phase-change cooling | 300+ kW/rack | 1.01-1.05 | Zero | Very high | Experimental |

### Direct Liquid Cooling (DLC)

DLC is becoming the standard for AI data centers:

**How it works:**
1. Coolant (water or dielectric fluid) circulates through cold plates
2. Cold plates are mounted directly on GPUs and CPUs
3. Heat is transferred from chip → cold plate → coolant → heat exchanger → cooling tower
4. Return coolant temperature: 35-45°C (vs. 15-25°C for air cooling)

**Benefits:**
- 3-5x better heat removal than air cooling
- Enables 100kW+ rack densities
- Reduces cooling energy by 40-60%
- Eliminates hot spots and thermal throttling
- Reduces data center footprint by 40%

**Challenges:**
- Higher capital cost ($5-15M per MW vs. $3-8M for air-cooled)
- Requires specialized plumbing and leak detection
- Maintenance complexity increases
- Not all equipment supports liquid cooling yet

### Immersion Cooling

The most aggressive cooling approach:

**Single-phase immersion:**
- Servers submerged in dielectric fluid (mineral oil or synthetic)
- Fluid absorbs heat and is circulated to heat exchangers
- PUE: 1.02-1.05
- Enables 200+ kW/rack

**Two-phase immersion:**
- Servers submerged in engineered fluid that boils at low temperature
- Vapor rises, condenses, and returns as liquid
- PUE: 1.01-1.03
- Enables 300+ kW/rack
- Fluid cost: $500-2,000/gallon

### Waste Heat Recovery

Data center waste heat is increasingly being reused:

| Use Case | Temperature Required | Examples |
|----------|---------------------|----------|
| District heating | 60-90°C | Stockholm, Helsinki data centers |
| Greenhouse agriculture | 30-40°C | Netherlands data center farms |
| Aquaculture | 25-35°C | Salmon farming in Norway |
| Industrial processes | 100-200°C | Limited applications |
| Desalination | 80-120°C | Middle East pilot projects |

---

## 5. Power Usage Effectiveness (PUE)

### Definition and Calculation

```
PUE = Total Facility Energy / IT Equipment Energy
```

- **PUE = 1.0**: All energy goes to IT equipment (theoretical ideal)
- **PUE = 1.2**: 20% overhead for cooling, lighting, etc. (excellent)
- **PUE = 1.5**: 50% overhead (average for 2024)
- **PUE = 2.0**: 100% overhead (old, inefficient facility)

### Industry Trends

| Year | Average PUE (Industry) | Best-in-Class PUE | Google PUE |
|------|----------------------|-------------------|------------|
| 2015 | 1.70 | 1.10 | 1.10 |
| 2018 | 1.58 | 1.07 | 1.07 |
| 2020 | 1.50 | 1.06 | 1.06 |
| 2022 | 1.45 | 1.05 | 1.05 |
| 2024 | 1.38 | 1.03 | 1.03 |
| 2026 (est.) | 1.30 | 1.02 | 1.02 |

### PUE by Cooling Technology

| Cooling Method | Typical PUE | Best PUE |
|---------------|-------------|----------|
| Air-cooled (traditional) | 1.4-1.6 | 1.2 |
| Air-cooled (optimized) | 1.2-1.4 | 1.1 |
| Liquid-cooled (direct) | 1.05-1.15 | 1.03 |
| Immersion-cooled | 1.02-1.08 | 1.01 |

### PUE Limitations

PUE has significant limitations as a metric:

1. **Doesn't account for IT efficiency**: A PUE of 1.1 with inefficient GPUs may be worse than PUE of 1.3 with efficient GPUs
2. **Seasonal variation**: PUE can vary 20-30% between summer and winter
3. **Measurement methodology**: Different companies measure PUE differently
4. **Doesn't capture water usage**: A low-PUE facility using evaporative cooling may waste more water
5. **Doesn't capture carbon**: A low-PUE facility running on coal is worse than a higher-PUE facility running on renewables

### Emerging Metrics

- **CUE (Carbon Usage Effectiveness)**: CO₂ emissions per IT energy consumed
- **WUE (Water Usage Effectiveness)**: Liters of water per kWh of IT energy
- **ERE (Energy Reuse Effectiveness)**: Accounts for energy recovered and reused
- **GUE (Grid Usage Effectiveness)**: Measures grid impact and demand response capability

---

## 6. Water Usage Effectiveness (WUE)

### The Hidden Cost

Water is often overlooked in AI sustainability discussions, but it's becoming critical:

- Data centers use water for **cooling** (evaporative cooling towers)
- Data centers use water for **humidification** (maintaining air quality)
- Semiconductor fabrication uses **massive** amounts of ultra-pure water

### WUE Calculation

```
WUE = Annual Water Usage (liters) / IT Equipment Energy (kWh)
```

| Cooling Method | Typical WUE (L/kWh) | Best WUE |
|---------------|---------------------|----------|
| Air-cooled (no water) | 0 | 0 |
| Evaporative cooling | 1.5-2.5 | 1.0 |
| Hybrid (air + water) | 0.5-1.5 | 0.3 |
| Liquid cooling (closed loop) | 0.1-0.3 | 0.05 |
| Immersion cooling | 0.05-0.15 | 0.02 |

### Water Stress Regions

Data center water consumption is most controversial in:

| Region | Water Stress Level | DC Growth Rate | Risk |
|--------|-------------------|----------------|------|
| Phoenix, AZ | Extreme | +40%/year | High |
| Las Vegas, NV | Extreme | +25%/year | High |
| Dublin, Ireland | High | +30%/year | Medium |
| Parts of India | Extreme | +50%/year | High |
| Singapore | High | +20%/year | Medium |

---

## 7. Energy Storage and Backup

### Battery Energy Storage Systems (BESS)

BESS is becoming standard for AI data centers:

| Technology | Energy Density | Cycle Life | Response Time | Cost ($/kWh) |
|-----------|---------------|-----------|---------------|-------------|
| Lithium-ion (NMC) | 150-250 Wh/kg | 1,000-5,000 | Milliseconds | $200-300 |
| Lithium iron phosphate (LFP) | 90-160 Wh/kg | 3,000-10,000 | Milliseconds | $150-250 |
| Sodium-ion | 100-160 Wh/kg | 2,000-5,000 | Milliseconds | $100-200 |
| Flow batteries (vanadium) | 20-35 Wh/kg | 10,000+ | Seconds | $300-500 |
| Solid-state (emerging) | 300-500 Wh/kg | 5,000+ | Milliseconds | $400-800 |

### Backup Power Architecture

AI data centers typically use:

1. **UPS (Uninterruptible Power Supply)**: Battery backup for 5-15 minutes
2. **Backup generators**: Diesel or natural gas for extended outages
3. **Grid-scale storage**: Battery systems for demand response and peak shaving
4. **Behind-the-meter generation**: On-site solar, wind, or nuclear

### The UPS Evolution

Traditional UPS systems are being replaced by more efficient designs:

| UPS Type | Efficiency | Footprint | Cost | Use Case |
|----------|-----------|-----------|------|----------|
| Double-conversion online | 92-96% | Large | High | Mission-critical |
| Delta conversion online | 96-98% | Medium | Medium | High-efficiency |
| Flywheel UPS | 97-99% | Small | Medium | Short-duration |
| Lithium-ion UPS | 97-99% | Small | Medium | Modern facilities |
| DC-DC direct | 98-99% | Smallest | Low | Next-gen designs |

---

## 8. The Nuclear Option

### Why Nuclear for AI?

Nuclear power offers unique advantages for AI data centers:

1. **Baseload power**: 24/7 availability (unlike solar/wind)
2. **High energy density**: Small land footprint per MW
3. **Zero carbon emissions**: No direct greenhouse gas emissions
4. **Long operational life**: 40-80 year plant lifetime
5. **Predictable costs**: Fuel costs are a small fraction of total

### Nuclear Projects for AI

| Project | Company | Technology | Capacity | Timeline |
|---------|---------|-----------|----------|----------|
| Three Mile Island Unit 1 | Constellation + Microsoft | PWR (existing) | 835 MW | 2028 |
| Susquehanna | Amazon + Talen | ABWR (existing) | 2.5 GW | 2026 |
| Kairos Power | Google | FHR (SMR) | 500 MW | 2030 |
| Oklo | Sam Altman | FBR (SMR) | 75 MW | 2028 |
| TerraPower | Bill Gates | MSR (SMR) | 345 MW | 2030 |
| Helion | Sam Altman | Fusion | TBD | 2030+ |

### Small Modular Reactors (SMRs)

SMRs are particularly attractive for data centers:

**Advantages over traditional nuclear:**
- Factory-built, shipped to site (lower construction cost)
- Smaller footprint (can be co-located with data centers)
- Lower upfront capital ($1-3B vs. $10-20B for traditional)
- Scalable (add modules as demand grows)
- Enhanced safety features (passive cooling, underground installation)

**Challenges:**
- Regulatory approval timelines (5-10 years)
- Public perception and NIMBY issues
- Waste management still unresolved
- Unproven economics at scale
- Licensing costs can exceed construction costs

### Nuclear Data Center Architecture

```
Grid Connection (backup) ←→ Nuclear Plant (primary) ←→ Data Center Campus
                                    ↓
                            Direct Power Bus
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Training Hall    Inference Hall    Storage Hall
```

Key design principles:
- **Behind-the-meter**: Data center directly connected to nuclear plant, bypassing grid
- **Direct current**: Nuclear plant generates DC, data center uses DC (eliminating AC/DC conversion losses)
- **Co-location**: Nuclear plant and data center on same campus
- **Shared cooling**: Nuclear plant waste heat used for district heating

---

## 9. Renewable Energy Integration

### Solar + AI Data Centers

Solar is increasingly used for data center power:

| Configuration | Capacity Factor | Land Use | Cost ($/MWh) |
|--------------|----------------|---------|-------------|
| Rooftop solar | 15-20% | Minimal | $50-80 |
| Ground-mount solar | 20-25% | High | $30-50 |
| Solar + storage (4hr) | 20-25% + dispatchable | High | $60-100 |
| Solar + storage (8hr) | 20-25% + extended dispatch | High | $80-120 |

### Wind + AI Data Centers

Wind power complements solar (often blows more at night):

| Configuration | Capacity Factor | Land Use | Cost ($/MWh) |
|--------------|----------------|---------|-------------|
| Onshore wind | 25-45% | Medium | $25-50 |
| Offshore wind | 40-55% | Offshore | $50-80 |
| Wind + storage | 25-45% + dispatchable | Medium | $50-90 |

### Geothermal for AI

Enhanced geothermal systems (EGS) are emerging:

- **Fervo Energy**: Partnered with Google for 24/7 carbon-free energy
- **Borehole technology**: Drilling techniques from oil & gas adapted for geothermal
- **Capacity factor**: 90%+ (baseload, like nuclear)
- **Cost**: $50-100/MWh (competitive with other clean energy)
- **Availability**: Can be built almost anywhere with deep drilling

### Hybrid Energy Systems

The optimal approach is usually a hybrid:

```
Nuclear (baseload, 24/7)  ─┐
                           ├→  AI Data Center
Solar (daytime)           ─┤
                           ├→  AI Data Center
Wind (variable)           ─┤
                           ├→  AI Data Center
Battery (buffering)       ─┘
```

---

## 10. Case Studies: Powering AI at Scale

### Case Study 1: Microsoft's Nuclear Revival

**Challenge:** Microsoft's AI workloads (Azure OpenAI Service) are growing faster than renewable energy supply.

**Solution:**
- Signed 20-year PPA with Constellation Energy for Three Mile Island Unit 1
- Investing in Kairos Power SMRs
- Committed to 100% carbon-free energy by 2030
- Developed AI-optimized data center designs with 1.12 PUE

**Results:**
- 835 MW of clean, baseload power secured
- Estimated 30% reduction in carbon intensity for Azure AI workloads
- Set precedent for nuclear-AI partnerships

### Case Study 2: Google's 24/7 Carbon-Free Energy

**Challenge:** Google committed to running on 24/7 carbon-free energy by 2030 but was falling behind.

**Solution:**
- Partnered with Fervo Energy for enhanced geothermal
- Investing in Kairos Power SMRs
- Using AI to optimize data center cooling (DeepMind collaboration)
- Developing carbon-aware computing (routing workloads to clean grids)

**Results:**
- 64% carbon-free energy achieved (2026)
- 40% reduction in cooling energy through AI optimization
- Pioneering "carbon-intelligent computing" platform

### Case Study 3: Amazon's Vertically Integrated Approach

**Challenge:** AWS needs massive power for AI training clusters (Trainium, Inferentia).

**Solution:**
- Acquired Talen Energy (nuclear plant operator)
- Building SMR-powered data centers in Pennsylvania
- Largest corporate buyer of renewable energy
- Developing custom AI chips optimized for energy efficiency

**Results:**
- 2.5 GW of nuclear power secured
- Custom chips achieving 3-5x better performance-per-watt
- Carbon-neutral cloud operations by 2030 (renewables + offsets)

### Case Study 4: Meta's Open Source Efficiency

**Challenge:** Meta's AI training clusters (600K+ GPUs) consume enormous power.

**Solution:**
- Open-sourcing LLaMA models (enabling community efficiency improvements)
- Investing in liquid cooling for training clusters
- Building massive solar installations at data center campuses
- Publishing energy consumption data for transparency

**Results:**
- LLaMA models deployed on efficient hardware globally
- 30% reduction in inference energy through model optimization
- Open-source community driving efficiency innovations

---

## 11. Cross-References

| Document | Relevance |
|----------|-----------|
| `05-Enterprise/04-AI-Infrastructure.md` | GPU architecture, cluster design, networking |
| `01-Overview.md` | AI energy crisis overview, key statistics |
| `25-Multi-Cloud-AI-Strategy/01-Overview.md` | Geographic workload distribution for energy optimization |
| `23-Local-AI-Inference-Self-Hosting/01-Overview.md` | Local inference reducing cloud dependency |
| `30-Small-Language-Models/01-Overview-and-Efficiency.md` | Model efficiency reducing energy per query |
| `07-Emerging/02-AI-Safety.md` | Safety considerations including resource constraints |

---

*Previous: [01-Overview.md](01-Overview.md) | Next: [03-Green-AI.md](03-Green-AI.md)*
