# 04 — Nuclear and Renewable Power for AI: The Clean Energy Race

> **Category:** 35-AI-Energy-and-Sustainability
> **Last updated:** June 29, 2026
> **Cross-references:** `02-Data-Center-Energy.md`, `05-Enterprise/04-AI-Infrastructure.md`, `25-Multi-Cloud-AI-Strategy/01-Overview.md`

---

## Table of Contents

1. [Why Clean Energy for AI Is a Strategic Imperative](#1-why-clean-energy-for-ai-is-a-strategic-imperative)
2. [Nuclear Power: The AI Industry's Silver Bullet?](#2-nuclear-power-the-ai-industrys-silver-bullet)
3. [Small Modular Reactors (SMRs)](#3-small-modular-reactors-smrs)
4. [Enhanced Geothermal Systems](#4-enhanced-geothermal-systems)
5. [Solar at Scale](#5-solar-at-scale)
6. [Wind Power for Data Centers](#6-wind-power-for-data-centers)
7. [Energy Storage Technologies](#7-energy-storage-technologies)
8. [Power Purchase Agreements (PPAs)](#8-power-purchase-agreements-ppas)
9. [Behind-the-Meter Generation](#9-behind-the-meter-generation)
10. [Case Studies: Clean Energy for AI at Scale](#10-case-studies-clean-energy-for-ai-at-scale)
11. [The Geopolitics of AI Energy](#11-the-geopolitics-of-ai-energy)
12. [Cross-References](#12-cross-references)

---

## 1. Why Clean Energy for AI Is a Strategic Imperative

### The Triple Challenge

The AI industry faces three interconnected energy challenges:

1. **Scale**: AI compute demand is growing 10x every 2 years
2. **Reliability**: AI workloads need 24/7 power (unlike many energy uses)
3. **Sustainability**: Public and regulatory pressure for clean energy

### Why Renewables Alone Aren't Enough

Solar and wind have a fundamental problem for AI: **intermittency**.

```
AI needs:        ████████████████████  (24/7 power)
Solar provides:  ░░░░████████░░░░░░░░  (daytime only)
Wind provides:   ░██░░░██░░░░██░░░██░  (variable)
```

**The math:**
- AI data centers run 24/7/365
- Solar capacity factor: 20-25% (produces power ~5-6 hours/day at full capacity)
- Wind capacity factor: 25-45% (variable, often lower at night when AI demand is high)
- To get 1 GW of 24/7 power from solar alone, you need ~4-5 GW of solar + massive storage
- Storage for 12-16 hours of backup is expensive ($200-500M per GWh)

This is why nuclear is so attractive: **capacity factor of 90%+** (runs 24/7 with minimal downtime).

### The Carbon Budget

For AI companies to meet net-zero commitments:

| Company | Current Carbon Intensity (g CO₂/kWh) | 2030 Target | Gap |
|---------|--------------------------------------|-------------|-----|
| Microsoft | 250 | 0 | Large |
| Google | 180 | 0 | Large |
| Amazon | 220 | 0 | Large |
| Meta | 200 | 0 | Large |
| Apple | 150 | 0 | Medium |

To close this gap, companies need **clean baseload power** — which means nuclear, geothermal, or massive storage.

---

## 2. Nuclear Power: The AI Industry's Silver Bullet?

### The Nuclear Renaissance

Nuclear power is experiencing a revival driven by AI:

**Historical context:**
- 1970s-1980s: Nuclear buildout (peak 30 GW/year)
- 1990s-2010s: Stagnation (Fukushima, cost overruns)
- 2020s: Revival (climate urgency, AI demand)

**2024-2026 nuclear investments for AI:**

| Company | Investment | Technology | Capacity | Timeline |
|---------|-----------|-----------|----------|----------|
| Microsoft + Constellation | $16B | Three Mile Island Unit 1 | 835 MW | 2028 |
| Amazon + Talen | $6.5B | Susquehanna nuclear plant | 2.5 GW | 2026 |
| Google + Kairos | $5B | Small modular reactors | 500 MW | 2030 |
| Oracle | $3B | Three SMR data centers | 1 GW | 2027-2029 |
| Sam Altman / Helion | $500M | Nuclear fusion | TBD | 2030+ |

### Why Nuclear for AI?

| Advantage | Details |
|-----------|---------|
| **Baseload power** | 24/7 availability (capacity factor 90%+) |
| **High energy density** | 1 GW on ~1 km² (vs. 4-5 km² for solar) |
| **Zero direct CO₂** | No greenhouse gas emissions during operation |
| **Long lifetime** | 40-80 year plant life |
| **Predictable costs** | Fuel is ~10% of total cost (low fuel price volatility) |
| **Grid stability** | Provides inertia and frequency regulation |

### Nuclear Challenges

| Challenge | Details |
|-----------|---------|
| **Construction time** | 7-15 years for traditional reactors |
| **Capital cost** | $10-20B for traditional reactors |
| **Regulatory approval** | 5-10 years for new designs |
| **Public perception** | NIMBY, safety concerns |
| **Waste management** | Long-term storage unresolved |
| **Workforce** | Specialized skills in short supply |

### Nuclear Economics for AI

| Metric | Traditional Nuclear | SMR | Solar + Storage |
|--------|-------------------|-----|-----------------|
| Capital cost ($/kW) | $6,000-10,000 | $4,000-8,000 | $3,000-6,000 |
| Operating cost ($/MWh) | $30-50 | $40-70 | $20-40 |
| Levelized cost ($/MWh) | $60-100 | $80-140 | $50-90 |
| Capacity factor | 90%+ | 90%+ | 25% (solar) + storage |
| 24/7 reliability | Excellent | Excellent | Good (with storage) |
| Construction time | 7-15 years | 3-7 years | 1-3 years |

---

## 3. Small Modular Reactors (SMRs)

### What Are SMRs?

SMRs are nuclear reactors with output less than 300 MWe, designed for:
- Factory fabrication (reduced construction cost and time)
- Modular deployment (add capacity as needed)
- Enhanced safety (passive cooling, underground installation)
- Co-location with data centers

### SMR Technologies for AI

| Company | Technology | Output | Status | Key Feature |
|---------|-----------|--------|--------|-------------|
| Kairos Power | Fluoride salt-cooled | 140 MW | Construction | Low-pressure, walk-away safe |
| TerraPower | Molten sodium | 345 MW | Construction | Bill Gates backed, Natrium |
| Oklo | Fast reactor | 75 MW | Licensing | Sam Altman backed |
| NuScale | Light water | 77 MW | Licensed | First SMR design approved by NRC |
| X-energy | High-temp gas | 80 MW | Licensing | Process heat capable |
| GE Hitachi | BWRX-300 | 300 MW | Licensed | Simplified BWR design |

### SMR Installation at Data Centers

**Architecture for co-located SMR + data center:**

```
┌─────────────────────────────────────────────┐
│                 Data Center Campus           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Training  │  │Inference │  │ Storage  │  │
│  │  Hall     │  │  Hall    │  │  Hall    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┴────┐  │
│  │          Power Distribution           │  │
│  │         (48V DC Bus)                  │  │
│  └──────────────────┬────────────────────┘  │
│                     │                        │
│  ┌──────────────────┴────────────────────┐  │
│  │     SMR Unit(s) + Heat Recovery       │  │
│  │  ┌─────────┐  ┌─────────┐            │  │
│  │  │ SMR-1   │  │ SMR-2   │  (Phase 2) │  │
│  │  │ 75 MW   │  │ 75 MW   │            │  │
│  │  └─────────┘  └─────────┘            │  │
│  └──────────────────┬────────────────────┘  │
│                     │                        │
│  ┌──────────────────┴────────────────────┐  │
│  │      Grid Connection (backup)         │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Design principles:**
- **Behind-the-meter**: Direct connection, bypassing grid losses
- **48V DC distribution**: Eliminates AC/DC conversion inefficiency
- **Waste heat recovery**: SMR heat used for building heating or industrial processes
- **Passive safety**: No operator action needed for safe shutdown
- **Modular expansion**: Start with one SMR, add more as demand grows

### SMR Regulatory Pathway

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Pre-application | 1-2 years | Engage with regulator, define design |
| Design certification | 3-5 years | Technical review, safety analysis |
| Construction permit | 2-3 years | Site review, environmental assessment |
| Operating license | 1-2 years | Final safety review, testing |
| **Total** | **7-12 years** | First power generation |

---

## 4. Enhanced Geothermal Systems

### How EGS Works

Enhanced Geothermal Systems (EGS) create artificial geothermal reservoirs:

1. **Drill deep wells** (3-5 km) into hot rock formations
2. **Hydraulic stimulation** creates fractures in the rock
3. **Water is pumped down** one well, heated by the rock
4. **Hot water/steam returns** up the second well
5. **Turbine generates electricity** from the steam
6. **Cooled water is recycled** back down

### EGS for AI Data Centers

| Advantage | Details |
|-----------|---------|
| **Baseload power** | 90%+ capacity factor |
| **Small footprint** | ~1 acre per 10 MW |
| **Scalable** | Can add wells incrementally |
| **Location flexible** | Can be built almost anywhere with deep drilling |
| **Low carbon** | Near-zero emissions |
| **Long lifetime** | 30-50 year reservoir life |

### Key Players

| Company | Technology | Status | Backing |
|---------|-----------|--------|---------|
| Fervo Energy | Horizontal drilling, fiber-optic monitoring | Pilot operational | Google, Chevron |
| Eavor Technologies | Closed-loop (no stimulation) | Pilot operational | Chevron, BP |
| Eavor-Lite | Closed-loop in sedimentary basins | Testing | Multiple |
| Quaise Energy | Millimeter-wave drilling (ultra-deep) | R&D | Breakthrough Energy |

### EGS Economics

| Metric | Value |
|--------|-------|
| Capital cost | $5-10M per MW |
| Operating cost | $30-60/MWh |
| Levelized cost | $50-100/MWh |
| Capacity factor | 90%+ |
| Construction time | 2-4 years |
| Carbon intensity | <10 g CO₂/kWh |

### Fervo Energy + Google Partnership

Fervo Energy's partnership with Google demonstrates EGS potential:

- **Project**: Cape Station, Utah
- **Capacity**: 400 MW (expandable to 2 GW)
- **Innovation**: Horizontal drilling from oil & gas, fiber-optic distributed sensing
- **Timeline**: First power by 2026, full capacity by 2028
- **Goal**: 24/7 carbon-free energy for Google data centers

---

## 5. Solar at Scale

### Solar for AI Data Centers

Solar is increasingly used for data center power, but with significant caveats:

**Advantages:**
- Lowest cost of new electricity generation ($20-40/MWh)
- Fast deployment (6-18 months)
- Modular (can add capacity incrementally)
- Low maintenance costs

**Challenges:**
- Intermittent (capacity factor 20-25%)
- Requires storage for 24/7 operation
- Large land footprint (~5 acres per MW)
- Seasonal variation (less in winter)

### Solar + Storage Configuration

```
Daytime:
Solar Panels (4 GW) → Direct to Data Center (1 GW) + Charge Batteries (2 GW)

Nighttime:
Batteries (8 GWh) → Data Center (1 GW) for 8 hours

Result: 24/7 power from 4 GW solar + 8 GWh storage
Cost: ~$4B (solar) + ~$2B (storage) = ~$6B total
```

### Solar Economics for AI

| Configuration | Cost ($/MWh) | Capacity Factor | 24/7? |
|--------------|-------------|-----------------|-------|
| Solar only | $20-40 | 20-25% | No |
| Solar + 4hr battery | $40-70 | 20-25% + dispatchable | Partial |
| Solar + 8hr battery | $60-90 | 20-25% + extended dispatch | Mostly |
| Solar + 12hr battery | $80-120 | 20-25% + full overnight | Yes |
| Solar + nuclear | $50-80 | 20-25% + 90%+ | Yes |

### Floating Solar

An emerging option for water-constrained regions:

- Panels installed on reservoirs or water bodies
- Reduces water evaporation (beneficial for cooling)
- Higher efficiency (water cools panels)
- No land use competition
- Cost: $30-50/MWh (slightly higher than ground-mount)

---

## 6. Wind Power for Data Centers

### Onshore Wind

| Metric | Value |
|--------|-------|
| Capacity factor | 25-45% |
| Cost | $25-50/MWh |
| Land footprint | ~3 acres per MW (but most land usable for agriculture) |
| Construction time | 1-3 years |
| Lifetime | 20-30 years |

### Offshore Wind

| Metric | Value |
|--------|-------|
| Capacity factor | 40-55% |
| Cost | $50-80/MWh |
| Land footprint | Offshore (no land use) |
| Construction time | 3-5 years |
| Lifetime | 25-35 years |

### Wind + AI Data Centers

Wind is particularly complementary to AI workloads:

- **Nighttime generation**: Wind often blows more at night (when AI demand is high)
- **Winter generation**: Wind is stronger in winter (when solar is weakest)
- **Coastal locations**: Many data centers are near coasts (for subsea cables)

**Challenge:** Wind is still intermittent. A data center running on 100% wind needs:
- 2-3x nameplate capacity for capacity factor
- 8-12 hours of battery storage
- Geographic diversity (wind farms in multiple locations)

---

## 7. Energy Storage Technologies

### Battery Technologies for Data Centers

| Technology | Energy Density | Cycle Life | Response Time | Cost ($/kWh) | Best For |
|-----------|---------------|-----------|---------------|-------------|----------|
| Lithium-ion (NMC) | 150-250 Wh/kg | 1,000-5,000 | Milliseconds | $200-300 | Short-duration |
| LFP | 90-160 Wh/kg | 3,000-10,000 | Milliseconds | $150-250 | Medium-duration |
| Sodium-ion | 100-160 Wh/kg | 2,000-5,000 | Milliseconds | $100-200 | Cost-sensitive |
| Iron-air | 50-80 Wh/kg | 10,000+ | Minutes | $50-100 | Long-duration |
| Flow batteries | 20-35 Wh/kg | 10,000+ | Seconds | $300-500 | Long-duration |

### Long-Duration Energy Storage (LDES)

For 24/7 renewable power, data centers need 12-16 hours of storage:

| Technology | Duration | Cost ($/kWh) | Status |
|-----------|----------|-------------|--------|
| Pumped hydro | 8-24 hours | $150-250 | Mature |
| Compressed air | 8-24 hours | $100-200 | Mature |
| Iron-air batteries | 100+ hours | $20-50 | Emerging |
| Gravity storage | 8-16 hours | $100-200 | Pilot |
| Hydrogen storage | Days-weeks | $200-400 | Emerging |

### Hydrogen as Energy Storage

Hydrogen is emerging as a long-duration storage option:

```
Excess Renewable Energy → Electrolysis → H₂ Storage → Fuel Cell → Electricity

Round-trip efficiency: 30-40% (low, but enables seasonal storage)
Cost: $200-400/kWh stored
Best for: Multi-day/seasonal storage
```

### Virtual Power Plants (VPPs)

Aggregating distributed storage for data center use:

```python
class VirtualPowerPlant:
    def __init__(self):
        self.batteries = []  # Distributed battery systems
        self.solar_panels = []  # Distributed solar
        self.ev_chargers = []  # EV batteries as storage
        self.data_center_load = 0
    
    async def dispatch(self, target_load_mw):
        """Dispatch clean energy to data center."""
        available = []
        
        # Check all distributed resources
        for battery in self.batteries:
            if battery.state_of_charge > 0.2:  # 20% minimum
                available.append({
                    "type": "battery",
                    "capacity": battery.available_mw,
                    "cost": battery.marginal_cost,
                    "carbon": 0,
                })
        
        for solar in self.solar_panels:
            if solar.current_output > 0:
                available.append({
                    "type": "solar",
                    "capacity": solar.current_output,
                    "cost": 0,  # Zero marginal cost
                    "carbon": 0,
                })
        
        # Sort by cost, dispatch cheapest first
        available.sort(key=lambda x: x["cost"])
        
        total_dispatched = 0
        for resource in available:
            if total_dispatched >= target_load_mw:
                break
            dispatch_mw = min(resource["capacity"], target_load_mw - total_dispatched)
            await self.dispatch_resource(resource, dispatch_mw)
            total_dispatched += dispatch_mw
        
        return total_dispatched
```

---

## 8. Power Purchase Agreements (PPAs)

### What Are PPAs?

Power Purchase Agreements are long-term contracts between energy buyers and generators:

| PPA Type | Duration | Risk Level | Best For |
|----------|---------|-----------|----------|
| Fixed-price PPA | 10-20 years | Low (price certainty) | Budget planning |
| Index-linked PPA | 10-20 years | Medium (tracks market) | Market-aligned |
| Shape PPA | 10-20 years | Medium (custom shape) | Matching load profile |
| Pay-as-produced | 10-20 years | High (variable output) | Direct renewable match |

### Nuclear PPAs for AI

The Microsoft-Constellation deal is the template:

```
Deal structure:
- Microsoft commits to buy 835 MW of power for 20 years
- Constellation restarts Three Mile Island Unit 1
- Price: Estimated $80-100/MWh (above market, but guarantees supply)
- Carbon: Zero direct emissions
- Timeline: Power available by 2028
```

### Solar/Wind PPAs for AI

```
Typical solar PPA:
- Duration: 15-20 years
- Price: $20-40/MWh (declining)
- Structure: Fixed price, no escalation
- Carbon: Zero direct emissions
- Risk: Intermittency (need storage or backup)
```

### PPA Portfolio Strategy

Most AI companies use a portfolio approach:

```
Portfolio for 1 GW data center:
- Nuclear PPA: 300 MW (baseload, 24/7)
- Solar PPA: 400 MW (daytime, cheap)
- Wind PPA: 200 MW (nighttime, complementary)
- Battery storage: 200 MW / 800 MWh (buffering)
- Grid backup: 100 MW (emergency)

Total clean energy: 900 MW (90%)
Grid dependence: 10% (backup only)
```

---

## 9. Behind-the-Meter Generation

### What Is Behind-the-Meter?

Behind-the-meter (BTM) generation is power produced on-site or directly connected to a data center, bypassing the grid:

```
Grid (backup) ←→ Substation ←→ Data Center ←→ On-site Generation (solar, nuclear, etc.)
```

### BTM Generation Options

| Technology | Capacity | Cost ($/kW) | Timeline | Grid Independence |
|-----------|----------|-------------|----------|-------------------|
| Rooftop solar | 1-5 MW | $1,000-1,500 | 6-12 months | Partial |
| Ground-mount solar | 5-50 MW | $800-1,200 | 12-24 months | Partial |
| Battery storage | 10-100 MW | $1,500-3,000 | 6-18 months | Partial |
| SMR nuclear | 50-300 MW | $4,000-8,000 | 3-7 years | Full |
| Fuel cells | 1-20 MW | $3,000-5,000 | 6-18 months | Partial |

### BTM Economic Analysis

```python
class BTMEconomics:
    def __init__(self, data_center_mw, grid_rate, carbon_tax):
        self.dc_mw = data_center_mw
        self.grid_rate = grid_rate  # $/kWh
        self.carbon_tax = carbon_tax  # $/tonne CO₂
    
    def analyze_solar_btm(self, capacity_mw, installation_cost):
        annual_generation_mwh = capacity_mw * 8760 * 0.22  # 22% capacity factor
        annual_savings = annual_generation_mwh * self.grid_rate
        annual_carbon_avoided = annual_generation_mwh * 0.4  # 400g CO₂/kWh grid average
        annual_carbon_savings = annual_carbon_avoided * self.carbon_tax / 1000
        
        total_annual_benefit = annual_savings + annual_carbon_savings
        payback_years = installation_cost / total_annual_benefit
        
        return {
            "annual_generation_mwh": annual_generation_mwh,
            "annual_savings": annual_savings,
            "annual_carbon_savings": annual_carbon_savings,
            "payback_years": payback_years,
            "roi_20yr": (total_annual_benefit * 20 - installation_cost) / installation_cost,
        }
    
    def analyze_nuclear_btm(self, capacity_mw, installation_cost):
        annual_generation_mwh = capacity_mw * 8760 * 0.90  # 90% capacity factor
        annual_savings = annual_generation_mwh * self.grid_rate
        annual_carbon_avoided = annual_generation_mwh * 0.4
        annual_carbon_savings = annual_carbon_avoided * self.carbon_tax / 1000
        
        total_annual_benefit = annual_savings + annual_carbon_savings
        payback_years = installation_cost / total_annual_benefit
        
        return {
            "annual_generation_mwh": annual_generation_mwh,
            "annual_savings": annual_savings,
            "annual_carbon_savings": annual_carbon_savings,
            "payback_years": payback_years,
            "roi_20yr": (total_annual_benefit * 20 - installation_cost) / installation_cost,
        }

# Example: 500 MW data center
analyzer = BTMEconomics(
    data_center_mw=500,
    grid_rate=0.08,  # $0.08/kWh
    carbon_tax=50,   # $50/tonne CO₂
)

solar = analyzer.analyze_solar_btm(
    capacity_mw=200,
    installation_cost=240_000_000,  # $240M
)

nuclear = analyzer.analyze_nuclear_btm(
    capacity_mw=300,
    installation_cost=2_400_000_000,  # $2.4B
)

print(f"Solar payback: {solar['payback_years']:.1f} years")
print(f"Nuclear payback: {nuclear['payback_years']:.1f} years")
print(f"Solar 20yr ROI: {solar['roi_20yr']:.0%}")
print(f"Nuclear 20yr ROI: {nuclear['roi_20yr']:.0%}")
```

---

## 10. Case Studies: Clean Energy for AI at Scale

### Case Study 1: Microsoft + Constellation (Nuclear)

**The Deal:**
- Microsoft committed to 20-year PPA for Three Mile Island Unit 1
- 835 MW of clean, baseload power
- Estimated cost: $80-100/MWh
- Power available by 2028

**Why it matters:**
- First major corporate nuclear deal for AI
- Sets precedent for nuclear-AI partnerships
- Demonstrates nuclear's value for 24/7 AI workloads
- Created political momentum for nuclear energy

**Challenges:**
- Public perception (Three Mile Island name)
- Regulatory approval required
- Long timeline (2+ years to restart)
- Community opposition possible

### Case Study 2: Google + Fervo (Geothermal)

**The Project:**
- Cape Station, Utah
- 400 MW (expandable to 2 GW)
- Enhanced geothermal using oil & gas drilling techniques
- Fiber-optic distributed sensing for reservoir monitoring

**Why it matters:**
- Proves EGS is commercially viable
- Can be built almost anywhere (not just volcanic regions)
- 24/7 baseload power (like nuclear, but faster to build)
- Google gets clean energy for data centers

**Challenges:**
- Still in pilot phase
- Higher cost than mature nuclear
- Induced seismicity concerns (though low risk)
- Scaling requires drilling expertise

### Case Study 3: Amazon + Talen (Nuclear Acquisition)

**The Deal:**
- Amazon acquired Talen Energy (nuclear plant operator)
- Susquehanna nuclear plant: 2.5 GW
- Direct power to AWS data centers in Pennsylvania
- Largest single nuclear acquisition for AI

**Why it matters:**
- Vertical integration of power and compute
- Guarantees clean power supply for decades
- Sets precedent for tech companies owning power generation
- Potentially enables nuclear-powered AI cloud

**Challenges:**
- Regulatory scrutiny (tech companies owning nuclear)
- Integration complexity
- Community relations
- Long-term maintenance responsibility

### Case Study 4: Microsoft + Nuclear Waste Solutions

**The Innovation:**
- Microsoft exploring nuclear fuel recycling
- Partnering with companies developing advanced fuel cycles
- Aiming to reduce nuclear waste by 90%
- Potential to extend fuel supply by 100x

**Why it matters:**
- Addresses nuclear waste concern (biggest public objection)
- Could make nuclear more sustainable
- Reduces long-term storage requirements
- Demonstrates corporate commitment to solving nuclear challenges

---

## 11. The Geopolitics of AI Energy

### Energy as Competitive Advantage

Countries are positioning themselves as AI energy hubs:

| Country | Strategy | Advantage |
|---------|----------|-----------|
| US | Nuclear + solar + gas | Abundant resources, tech companies |
| China | Nuclear + solar + coal | Manufacturing scale, state support |
| UAE | Nuclear + solar + gas | Wealth, stability, desert solar |
| France | Nuclear (existing) | 70% nuclear grid, clean baseload |
| Norway | Hydro + wind | Clean, cheap, but limited scale |
| Iceland | Geothermal + hydro | Clean, but remote |
| Saudi Arabia | Solar + nuclear | Desert solar, wealth |

### Chip-Nation Energy Nexus

AI chip manufacturing and energy are interconnected:

```
AI Chips ←→ Semiconductor Fab ←→ Clean Energy ←→ Data Centers ←→ AI Workloads
    ↑              ↑                  ↑                ↑
    └──────────────┴──────────────────┴────────────────┘
              All require clean, reliable power
```

**Semiconductor fabs** are massive energy consumers:
- A single fab uses 100-200 MW (enough for a small city)
- TSMC consumes 6% of Taiwan's electricity
- Intel's Ohio fab will use 1 GW (nuclear-powered planned)

### Energy Diplomacy

AI is reshaping energy diplomacy:

- **US-EU**: Joint clean energy standards for AI
- **US-Japan**: Nuclear technology sharing for AI
- **US-India**: Solar + nuclear partnerships
- **China-BRI**: Exporting nuclear technology globally
- **OPEC+**: Adapting to AI energy demand (some pivoting to nuclear)

---

## 12. Cross-References

| Document | Relevance |
|----------|-----------|
| `01-Overview.md` | AI energy crisis overview |
| `02-Data-Center-Energy.md` | Data center power architecture |
| `03-Green-AI.md` | Model efficiency and carbon-aware computing |
| `05-Enterprise/04-AI-Infrastructure.md` | Technical infrastructure design |
| `25-Multi-Cloud-AI-Strategy/01-Overview.md` | Geographic workload distribution |
| `21-AI-Regulation-Antitrust/01-Overview.md` | Energy regulation landscape |
| `12-Business-Prospects/02-AI-Market-Overview.md` | Market context and investment trends |

---

*Previous: [03-Green-AI.md](03-Green-AI.md) | Next: [05-Future-Outlook.md](05-Future-Outlook.md)*
