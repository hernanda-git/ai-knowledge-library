# 04 — Tools, Frameworks, and Platforms for AI Supply Chain Management

> **Why this document exists.** Managing the AI chip supply chain requires a sophisticated toolkit spanning EDA (Electronic Design Automation), supply chain visibility, risk modeling, procurement optimization, and simulation. This document catalogs the essential tools, frameworks, and platforms that AI companies, chip designers, and supply chain professionals use to design, manufacture, procure, and deploy AI hardware. It also covers open-source tools, industry standards, and emerging platforms.

---

## Table of Contents

1. [EDA Tools for AI Chip Design](#1-eda-tools-for-ai-chip-design)
2. [Supply Chain Visibility Platforms](#2-supply-chain-visibility-platforms)
3. [Risk Modeling and Simulation Tools](#3-risk-modeling-and-simulation-tools)
4. [Procurement and Sourcing Platforms](#4-procurement-and-sourcing-platforms)
5. [Industry Standards and Consortia](#5-industry-standards-and-consortia)
6. [Open-Source Tools and Frameworks](#6-open-source-tools-and-frameworks)
7. [Cloud-Based Chip Design Platforms](#7-cloud-based-chip-design-platforms)
8. [Monitoring and Observability](#8-monitoring-and-observability)
9. [Implementation Guide](#9-implementation-guide)
10. [Cross-References](#10-cross-references)

---

## 1. EDA Tools for AI Chip Design

### 1.1 The Big Three EDA Vendors

| Vendor | Revenue (2025) | Key AI Chip Products | Strengths |
|--------|---------------|---------------------|-----------|
| **Synopsys** | $6.1B | DSO.ai, Fusion Compiler, PrimeTime | AI-driven design optimization, timing closure |
| **Cadence** | $4.6B | Innovus, Cerebrus, Voltus | Place-and-route, power analysis, ML-driven optimization |
| **Siemens EDA** | $2.1B | Calibre, Catapult HLS | Physical verification, high-level synthesis |

### 1.2 Synopsys AI-Driven Design Flow

```
Architecture → RTL → Synthesis → Place & Route → Signoff → Tapeout
     ↓            ↓         ↓              ↓            ↓         ↓
  DS Arch     Design     Fusion        ICC2         PrimeTime  Manufacturing
  Explorer    Compiler   Compiler                   StarRC     Preparation
     ↓            ↓         ↓              ↓            ↓
  ┌──────────────────────────────────────────────────┐
  │              Synopsys DSO.ai                       │
  │  (AI-driven design space exploration)              │
  │  - Optimizes PPA across entire flow                │
  │  - Explores 1000x more design points               │
  │  - Reduces design closure time by 30-50%           │
  └──────────────────────────────────────────────────┘
```

### 1.3 Cadence Cerebrus

Cadence's AI-driven design optimization platform:

| Feature | Description | Impact |
|---------|-------------|--------|
| **ML-driven P&R** | Reinforcement learning for placement | 15–25% better QoR |
| **Timing prediction** | ML models predict timing before full implementation | 5x faster exploration |
| **Power optimization** | AI-driven clock gating and power domains | 10–20% power reduction |
| **Multi-objective optimization** | Simultaneously optimize PPA | Pareto-optimal designs |

### 1.4 Open-Source EDA

| Tool | Purpose | Community | Maturity |
|------|---------|-----------|----------|
| **OpenROAD** | RTL-to-GDSII flow | Academic + industry | Production-ready for research |
| **OpenLane** | Automated digital design flow | SkyWater MPW | Used for tapeouts |
| **Magic** | Layout editor and DRC | Academic | Mature for analog |
| **KLayout** | Layout viewer and processing | Industry | Production-ready |
| **Yosys** | Logic synthesis | Academic | Production-ready |
| **nextpnr** | Place and route | Academic | Research-grade |

### 1.5 AI-Specific Design Tools

| Tool | Vendor | Purpose | Key Feature |
|------|--------|---------|-------------|
| **cuLitho** | NVIDIA | Computational lithography | GPU-accelerated ILT |
| **Synopsys AI** | Synopsys | Chip floorplanning | Google AlphaChip-inspired |
| **Ansys PowerArtist** | Ansys | RTL power analysis | Early power estimation |
| **Mentor Tessent** | Siemens | DFT (Design for Test) | Scan chain insertion |

---

## 2. Supply Chain Visibility Platforms

### 2.1 Platform Comparison

| Platform | Focus | Key Features | Pricing Model |
|----------|-------|-------------|---------------|
| **Resilinc** | Multi-tier risk monitoring | AI-powered disruption detection, 200+ risk factors | Enterprise SaaS ($50K–$500K/yr) |
| **Everstream Analytics** | Predictive risk intelligence | ML risk scoring, 100K+ suppliers tracked | Enterprise SaaS ($100K–$1M/yr) |
| **Interos** | Relationship mapping | 6+ tier supply chain mapping, real-time risk | Enterprise SaaS ($75K–$300K/yr) |
| **Coupa** | Procurement + risk | Supply chain finance, spend analytics | Enterprise SaaS ($200K–$2M/yr) |
| **o9 Solutions** | AI-driven planning | Demand sensing, supply planning, digital brain | Enterprise SaaS ($300K–$3M/yr) |
| **Blue Yonder** | Supply chain management | End-to-end visibility, AI-driven decisions | Enterprise SaaS ($500K–$5M/yr) |

### 2.2 Resilinc Deep Dive

Resilinc is the leading supply chain risk platform for hardware companies:

**Key capabilities:**
- Monitors 200+ risk factors across 200K+ suppliers
- AI-powered disruption detection (earthquakes, floods, sanctions, financial distress)
- Multi-tier mapping (tracks Tier 1 through Tier 6+ suppliers)
- Event-driven alerts with recommended actions
- Integration with ERP systems (SAP, Oracle)

**Implementation example:**
```python
# Conceptual Resilinc API integration
class ResilincMonitor:
    """Monitor supply chain risks using Resilinc-like platform."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.risk_threshold = 0.7  # Alert threshold
        
    def monitor_supplier(self, supplier_id):
        """Monitor a specific supplier for risks."""
        # In production, this would call Resilinc API
        risk_factors = {
            'geopolitical': self._check_geopolitical(supplier_id),
            'natural_disaster': self._check_natural_disaster(supplier_id),
            'financial': self._check_financial(supplier_id),
            'regulatory': self._check_regulatory(supplier_id),
            'operational': self._check_operational(supplier_id),
        }
        
        composite_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            'supplier_id': supplier_id,
            'risk_factors': risk_factors,
            'composite_risk': composite_risk,
            'alert': composite_risk > self.risk_threshold,
            'recommended_actions': self._generate_actions(risk_factors)
        }
    
    def _check_geopolitical(self, supplier_id):
        """Check geopolitical risk for supplier location."""
        # Simplified: real implementation uses Resilinc data
        return 0.3  # Example: moderate risk
    
    def _check_natural_disaster(self, supplier_id):
        """Check natural disaster risk."""
        return 0.2  # Example: low risk
    
    def _check_financial(self, supplier_id):
        """Check financial health."""
        return 0.1  # Example: low risk
    
    def _check_regulatory(self, supplier_id):
        """Check regulatory risk (export controls, sanctions)."""
        return 0.5  # Example: moderate risk
    
    def _check_operational(self, supplier_id):
        """Check operational risk (capacity, quality)."""
        return 0.4  # Example: moderate risk
    
    def _generate_actions(self, risk_factors):
        """Generate recommended mitigation actions."""
        actions = []
        if risk_factors['geopolitical'] > 0.5:
            actions.append("Consider alternative sourcing from different region")
        if risk_factors['natural_disaster'] > 0.5:
            actions.append("Increase safety stock for affected components")
        if risk_factors['regulatory'] > 0.5:
            actions.append("Review export control compliance")
        return actions
```

---

## 3. Risk Modeling and Simulation Tools

### 3.1 Monte Carlo Simulation for Supply Chain Risk

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SupplyChainNode:
    """Model a single node in the supply chain."""
    name: str
    base_capacity: int  # units/month
    lead_time_weeks: int
    location: str
    
    # Risk parameters
    disruption_prob: float  # probability of disruption per month
    disruption_duration_weeks: tuple  # (min, max) duration
    capacity_loss_range: tuple  # (min, max) fraction of capacity lost

class SupplyChainMonteCarlo:
    """Monte Carlo simulation for AI chip supply chain risk."""
    
    def __init__(self, nodes: List[SupplyChainNode], num_simulations: int = 10000):
        self.nodes = nodes
        self.num_simulations = num_simulations
        
    def simulate(self, months: int = 12) -> Dict:
        """Run Monte Carlo simulation."""
        results = []
        
        for _ in range(self.num_simulations):
            monthly_outputs = []
            
            for month in range(months):
                month_output = 0
                
                for node in self.nodes:
                    # Check if disruption occurs this month
                    if np.random.random() < node.disruption_prob:
                        # Disruption occurs
                        duration = np.random.randint(
                            node.disruption_duration_weeks[0],
                            node.disruption_duration_weeks[1] + 1
                        )
                        capacity_loss = np.random.uniform(
                            node.capacity_loss_range[0],
                            node.capacity_loss_range[1]
                        )
                        effective_capacity = node.base_capacity * (1 - capacity_loss)
                    else:
                        effective_capacity = node.base_capacity
                        duration = 0
                    
                    month_output += effective_capacity
                
                monthly_outputs.append(month_output)
            
            results.append({
                'monthly_outputs': monthly_outputs,
                'total_output': sum(monthly_outputs),
                'min_monthly': min(monthly_outputs),
                'max_monthly': max(monthly_outputs),
            })
        
        # Aggregate results
        total_outputs = [r['total_output'] for r in results]
        
        return {
            'mean_total_output': np.mean(total_outputs),
            'std_total_output': np.std(total_outputs),
            'percentile_5': np.percentile(total_outputs, 5),
            'percentile_50': np.percentile(total_outputs, 50),
            'percentile_95': np.percentile(total_outputs, 95),
            'prob_below_80pct': np.mean([1 if t < 0.8 * np.mean(total_outputs) else 0 for t in total_outputs]),
            'value_at_risk_5pct': np.percentile(total_outputs, 5),
        }

# Example: AI chip supply chain simulation
nodes = [
    SupplyChainNode('TSMC Fab 18', 100000, 12, 'Taiwan', 0.02, (4, 24), (0.3, 0.8)),
    SupplyChainNode('TSMC Phoenix', 20000, 12, 'US', 0.01, (2, 8), (0.2, 0.5)),
    SupplyChainNode('SK Hynix', 50000, 8, 'South Korea', 0.015, (2, 12), (0.2, 0.6)),
    SupplyChainNode('ASE Packaging', 80000, 6, 'Taiwan', 0.02, (2, 12), (0.3, 0.7)),
]

sim = SupplyChainMonteCarlo(nodes, num_simulations=10000)
results = sim.simulate(months=12)

print(f"Mean annual output: {results['mean_total_output']:,} units")
print(f"Std deviation: {results['std_total_output']:,} units")
print(f"5th percentile (VaR): {results['percentile_5']:,} units")
print(f"95th percentile: {results['percentile_95']:,} units")
print(f"Prob of <80% mean output: {results['prob_below_80pct']:.1%}")
```

### 3.2 Scenario Planning Tools

| Tool | Purpose | Methodology |
|------|---------|-------------|
| **Palantir Foundry** | Scenario planning + optimization | Graph-based modeling |
| **AnyLogic** | Multi-method simulation | Agent-based + discrete event + system dynamics |
| **Simio** | Simulation + scheduling | Object-based simulation |
| **FlexSim** | 3D discrete event simulation | Visual modeling |
| **Arena** | Process simulation | Discrete event |

### 3.3 Digital Twin Platforms for Manufacturing

| Platform | Vendor | Focus |
|----------|--------|-------|
| **NVIDIA Omniverse** | NVIDIA | 3D simulation, physics-based modeling |
| **Azure Digital Twins** | Microsoft | IoT-based digital twin |
| **AWS IoT TwinMaker** | Amazon | Industrial digital twin |
| **Siemens Xcelerator** | Siemens | Full product lifecycle digital twin |
| **PTC ThingWorx** | PTC | Industrial IoT + digital twin |

---

## 4. Procurement and Sourcing Platforms

### 4.1 Semiconductor Procurement Platforms

| Platform | Purpose | Key Features |
|----------|---------|-------------|
| **Octopart** | Component search and sourcing | Real-time pricing, availability from 30+ distributors |
| **FindChips** | Component search | Multi-source comparison |
| **NetComponents** | Electronic component marketplace | 300M+ parts |
| **Avnet** | Semiconductor distribution | Design-to-delivery services |
| **Arrow Electronics** | Semiconductor distribution | Engineering support, supply chain services |
| **WPG Holdings** | Semiconductor distribution | Asia-Pacific focused |

### 4.2 AI-Specific Procurement Strategies

**Strategy 1: Long-Term Agreement (LTA)**
```
Duration: 2–3 years
Commitment: Volume guarantee (e.g., 100K GPUs/year)
Benefit: Guaranteed capacity, locked pricing
Risk: Overcommitment if demand changes
Best for: Hyperscalers with predictable demand
```

**Strategy 2: Capacity Reservation**
```
Duration: 12–24 months
Commitment: Reserve manufacturing capacity (not volume)
Benefit: Flexibility to adjust volume
Risk: Reservation fees, underutilization charges
Best for: Growth-stage AI companies
```

**Strategy 3: Spot Market**
```
Duration: As-needed
Commitment: None
Benefit: Maximum flexibility
Risk: Unavailable during shortages, price volatility
Best for: Small companies, experimental projects
```

### 4.3 The Chip Broker Market

During shortages, the broker market becomes critical:

| Aspect | Authorized Distribution | Broker Market |
|--------|----------------------|---------------|
| **Price** | MSRP ±10% | 2–10x MSRP during shortage |
| **Availability** | Limited during shortage | Available (but verify authenticity) |
| **Authenticity** | Guaranteed | Risk of counterfeit |
| **Warranty** | Full manufacturer warranty | Limited or none |
| **Best for** | Production volumes | Emergency shortages, legacy parts |

---

## 5. Industry Standards and Consortia

### 5.1 Key Standards for AI Chips

| Standard | Organization | Purpose | Version |
|----------|-------------|---------|---------|
| **UCIe** | UCIe Consortium | Chiplet interconnect | 2.0 (2025) |
| **PCIe** | PCI-SIG | Host interface | 7.0 (2025) |
| **CXL** | CXL Consortium | Memory expansion | 3.0 (2024) |
| **NVLink** | NVIDIA | GPU-to-GPU interconnect | 6.0 (2026) |
| **NVSwitch** | NVIDIA | GPU fabric switch | 4th gen (2026) |
| **InfiniBand** | NVMexpress/OCP | Network fabric | XDR (2025) |
| **OCP** | Open Compute Project | Server design | Datacenter 3.0 |
| **ODSA** | OCP Die-to-Die | Open chiplet standard | 2024 |
| **CHIPS Alliance** | Linux Foundation | Open-source EDA + chip design | Active |

### 5.2 The UCIe Standard in Detail

UCIe (Universal Chiplet Interconnect Express) is the most important emerging standard:

**UCIe 2.0 specifications:**

| Parameter | UCIe 1.0 | UCIe 2.0 |
|-----------|---------|---------|
| Data rate | 8 GT/s per pin | 32 GT/s per pin |
| Width | 64–256 pins | 64–512 pins |
| Bandwidth | 32–256 GB/s | 128–1024 GB/s |
| Latency | <2 ns | <1.5 ns |
| Power efficiency | <0.5 pJ/bit | <0.3 pJ/bit |
| Bump pitch | 100–130 μm | 40–55 μm |

**UCIe ecosystem:**
- Intel, AMD, ARM, TSMC, Samsung, Qualcomm, Google, Meta
- Enables mix-and-match chiplets from different vendors
- Critical for reducing AI chip cost and time-to-market

### 5.3 Open Compute Project (OCP)

OCP standards relevant to AI:

| Standard | Focus | Impact |
|----------|-------|--------|
| **OCP Accelerator Module (OAM)** | Standard form factor for AI accelerators | Enables vendor-agnostic GPU trays |
| **OCP NIC 3.0** | Network interface card standard | Standardized 400GbE/800GbE NICs |
| **OCP Datacenter Specification** | Server rack and power | Standardized datacenter infrastructure |
| **MTCA (MicroTCA)** | AdvancedTCA for telecom | High-availability computing |

---

## 6. Open-Source Tools and Frameworks

### 6.1 Open-Source EDA Stack

The open-source EDA ecosystem has matured significantly:

```bash
# Complete open-source chip design flow (OpenLane 2.0)
# Step 1: RTL Design (Verilog/SystemVerilog)
# Step 2: Logic Synthesis (Yosys)
# Step 3: Floorplanning (OpenROAD)
# Step 4: Place and Route (OpenROAD)
# Step 5: Clock Tree Synthesis (OpenROAD)
# Step 6: Routing (OpenROAD)
# Step 7: Signoff (Magic, KLayout, Netgen)

# Example: Run OpenLane flow
git clone https://github.com/efabless/openlane2.git
cd openlane2
python -m openlane --design my_ai_chip --pdk sky130_fd_sc_hd
```

### 6.2 Supply Chain Analysis Tools

| Tool | Purpose | Language | License |
|------|---------|----------|---------|
| **Supply Chain Risk Dashboard** | Risk visualization | Python | MIT |
| **Open Supply Chain** | End-to-end modeling | Python | Apache 2.0 |
| **SCORML** | SCOR model implementation | Python | BSD |
| **NetworkX** | Supply chain graph analysis | Python | BSD |
| **SimPy** | Discrete event simulation | Python | BSD |

### 6.3 Hardware Security Tools

| Tool | Purpose | Developer |
|------|---------|----------|
| **OpenTitan** | Open-source silicon root of trust | Google + lowRISC |
| **CHIPS Alliance** | Open-source hardware IP | Linux Foundation |
| **RISC-V** | Open-source processor ISA | RISC-V International |
| **PULP Platform** | Open-source RISC-V SoC | ETH Zurich |
| **BOOM** | Out-of-order RISC-V core | UC Berkeley |

---

## 7. Cloud-Based Chip Design Platforms

### 7.1 Cloud EDA Services

| Platform | Vendor | Key Feature |
|----------|--------|-------------|
| **Synopsys Cloud** | Synopsys | Full Synopsys stack on cloud |
| **Cadence CloudBurst** | Cadence | Cadence tools on cloud |
| **Siemens Cloud** | Siemens EDA | Calibre on cloud |
| **Google Cloud for EDA** | Google | Scalable compute for EDA workloads |
| **AWS for EDA** | Amazon | HPC instances for simulation |

### 7.2 Benefits of Cloud EDA

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Scalability** | Burst to thousands of cores for simulation | 5–10x faster signoff |
| **Cost efficiency** | Pay only for compute used | 30–50% cost reduction |
| **Collaboration** | Global teams access same design data | Faster iteration |
| **Security** | Cloud provider security infrastructure | Often better than on-prem |

### 7.3 Cloud EDA Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|-----------|
| **IP security** | Design data in cloud | Encryption, secure enclaves |
| **Latency** | Large design files over network | Edge caching, WAN optimization |
| **Vendor lock-in** | Tools tied to specific cloud | Multi-cloud strategy |
| **Compliance** | Export control requirements | Region-specific deployments |

---

## 8. Monitoring and Observability

### 8.1 Fab Monitoring

| System | Purpose | Data Collected |
|--------|---------|---------------|
| **MES** (Manufacturing Execution System) | Production tracking | Wafer counts, yield, cycle time |
| **SPC** (Statistical Process Control) | Quality monitoring | Process parameters, defect rates |
| **FDC** (Fault Detection and Classification) | Real-time monitoring | Sensor data, equipment health |
| **R2R** (Run-to-Run) | Process control | Recipe adjustments |

### 8.2 Supply Chain Monitoring Dashboard

```python
class SupplyChainDashboard:
    """Real-time supply chain monitoring dashboard."""
    
    def __init__(self):
        self.metrics = {
            'inventory_levels': {},
            'supplier_health': {},
            'demand_forecast': {},
            'risk_alerts': [],
        }
    
    def update_metrics(self, data):
        """Update dashboard with latest data."""
        self.metrics.update(data)
    
    def get_status(self):
        """Get current supply chain status."""
        status = {
            'overall_health': self._compute_health(),
            'critical_alerts': self._get_critical_alerts(),
            'inventory_status': self._get_inventory_status(),
            'supplier_status': self._get_supplier_status(),
            'recommendations': self._generate_recommendations(),
        }
        return status
    
    def _compute_health(self):
        """Compute overall supply chain health score (0-100)."""
        # Simplified: real implementation uses weighted metrics
        return 75  # Example: 75/100
    
    def _get_critical_alerts(self):
        """Get critical alerts requiring immediate attention."""
        return [
            {'severity': 'HIGH', 'message': 'TSMC CoWoS capacity utilization at 95%'},
            {'severity': 'MEDIUM', 'message': 'SK Hynix HBM4 ramp delayed by 2 weeks'},
        ]
    
    def _get_inventory_status(self):
        """Get inventory status across supply chain."""
        return {
            'gpu_chips': {'on_hand': 50000, 'in_transit': 30000, 'on_order': 100000},
            'hbm_memory': {'on_hand': 200000, 'in_transit': 100000, 'on_order': 500000},
        }
    
    def _get_supplier_status(self):
        """Get health status of key suppliers."""
        return {
            'TSMC': {'health': 'GOOD', 'capacity_utilization': 0.95},
            'SK Hynix': {'health': 'WARNING', 'capacity_utilization': 0.92},
            'ASE': {'health': 'GOOD', 'capacity_utilization': 0.85},
        }
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        return [
            'Increase HBM safety stock from 4 weeks to 8 weeks',
            'Initiate secondary source qualification for CoWoS packaging',
            'Review TSMC capacity reservation for Q4 2026',
        ]

# Example usage
dashboard = SupplyChainDashboard()
status = dashboard.get_status()
print(f"Overall Health: {status['overall_health']}/100")
print(f"Critical Alerts: {len(status['critical_alerts'])}")
for alert in status['critical_alerts']:
    print(f"  [{alert['severity']}] {alert['message']}")
```

---

## 9. Implementation Guide

### 9.1 Building an AI Supply Chain Management System

**Phase 1: Foundation (Months 1–3)**
- Implement basic supplier tracking (names, locations, capabilities)
- Set up inventory monitoring for critical components
- Create manual risk assessment process

**Phase 2: Visibility (Months 3–6)**
- Integrate with supply chain visibility platform (Resilinc/Everstream)
- Implement automated disruption alerts
- Build multi-tier supplier mapping

**Phase 3: Analytics (Months 6–9)**
- Deploy Monte Carlo risk simulation
- Implement demand forecasting
- Build procurement optimization

**Phase 4: Automation (Months 9–12)**
- Automate reordering based on risk signals
- Implement AI-driven procurement recommendations
- Deploy digital twin simulation

### 9.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Collection** | APIs, EDI, IoT sensors | Supplier data, inventory, shipments |
| **Data Storage** | Snowflake, BigQuery, or PostgreSQL | Central data warehouse |
| **Analytics** | Python (pandas, numpy), Spark | Data processing and analysis |
| **Simulation** | AnyLogic, SimPy, custom Python | Risk simulation and scenario planning |
| **Visualization** | Tableau, Power BI, or custom | Dashboard and reporting |
| **Alerting** | PagerDuty, Slack, email | Disruption notifications |
| **Integration** | MuleSoft, custom APIs | ERP, procurement system integration |

### 9.3 Key Metrics to Track

| Metric | Definition | Target | Frequency |
|--------|-----------|--------|-----------|
| **Inventory Days of Supply** | Days of inventory on hand | 30–60 days | Daily |
| **Supplier Lead Time** | Average time from order to delivery | <12 weeks | Weekly |
| **Capacity Utilization** | % of reserved capacity being used | 80–95% | Weekly |
| **Disruption Response Time** | Time from alert to mitigation action | <48 hours | Per event |
| **Cost Variance** | Actual vs. budgeted procurement cost | <5% | Monthly |
| **Quality Rate** | % of components passing incoming inspection | >99.5% | Weekly |

---

## 10. Cross-References

| Document | Category | Relevance |
|----------|----------|-----------|
| `38-AI-Supply-Chain-and-Chip-Design/01-Overview.md` | Supply Chain | Strategic overview |
| `38-AI-Supply-Chain-and-Chip-Design/02-Core-Topics.md` | Supply Chain | Core technical topics |
| `38-AI-Supply-Chain-and-Chip-Design/03-Technical-Deep-Dive.md` | Supply Chain | Engineering details |
| `25-Multi-Cloud-AI-Strategy/` | Multi-Cloud | Cloud procurement |
| `05-Enterprise/` | Enterprise | Enterprise AI deployment |
| `20-Agent-Infrastructure-and-Observability/` | Infrastructure | Monitoring and observability |

---

*Last updated: June 29, 2026*
