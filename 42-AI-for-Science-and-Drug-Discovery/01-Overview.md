# 42-AI-for-Science-and-Drug-Discovery

# AI for Science & Drug Discovery: A Comprehensive Guide

> **Category**: 42-AI-for-Science-and-Drug-Discovery  
> **Last Updated**: June 2026  
> **Difficulty**: Intermediate to Advanced  
> **Cross-References**: [17-Research-Frontiers-2026/08-AI-for-Science.md](../17-Research-Frontiers-2026/08-AI-for-Science.md), [02-LLMs/02-Model-Families.md](../02-LLMs/02-Model-Families.md), [32-Agent-Memory-Systems/](../32-Agent-Memory-Systems/)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Market Landscape & Scale](#market-landscape--scale)
3. [The Nobel Catalyst: AI for Science Goes Mainstream](#the-nobel-catalyst-ai-for-science-goes-mainstream)
4. [Core Application Domains](#core-application-domains)
5. [Key Players & Ecosystem](#key-players--ecosystem)
6. [Technical Foundations](#technical-foundations)
7. [Impact Metrics & ROI](#impact-metrics--roi)
8. [Challenges & Limitations](#challenges--limitations)
9. [Regulatory Landscape](#regulatory-landscape)
10. [Future Outlook 2026–2030](#future-outlook-20262030)

---

## Executive Summary

AI for Science represents one of the most transformative applications of artificial intelligence, moving beyond language and image generation to fundamentally accelerate scientific discovery itself. In 2024, the Nobel Prize in Chemistry was awarded to Demis Hassabis, John Jumper (AlphaFold/DeepMind), and David Baker (protein design) — cementing AI as a legitimate scientific methodology rather than merely a tool.

By mid-2026, AI for Science has matured from proof-of-concept demonstrations to production-grade systems that pharmaceutical companies, materials scientists, climate researchers, and mathematicians rely on daily. The convergence of large language models, diffusion models, graph neural networks, and specialized scientific foundation models has created an unprecedented toolkit for discovery.

### Key Statistics (June 2026)

| Metric | Value | Source |
|--------|-------|--------|
| Global AI for Science market | $8.2B | Grand View Research |
| AI-discovered drug candidates in clinical trials | 67 | PharmaAI Database |
| Average drug discovery time reduction | 40–60% | McKinsey Analysis |
| AlphaFold database entries | 214M proteins | EMBL-EBI |
| AI materials discoveries (GNoME) | 2.2M stable crystals | Google DeepMind |
| Scientific AI papers published (2025) | 12,400+ | arXiv/CS.AI |

### What Makes This Category Unique

Unlike general-purpose AI applications, AI for Science requires:

1. **Domain Expertise Integration**: Models must encode physical laws, chemical constraints, and biological principles
2. **Extreme Accuracy Requirements**: A 1% error in molecular docking can mean a failed drug candidate
3. **Interpretability Needs**: Scientists need to understand *why* a model predicts something, not just *what*
4. **Data Scarcity**: Unlike internet-scale text data, scientific datasets are often small, expensive, and noisy
5. **Validation Loops**: Predictions must be experimentally verified, creating tight AI-lab feedback cycles

---

## Market Landscape & Scale

### The AI-Science Convergence Timeline

| Year | Milestone | Significance |
|------|-----------|--------------|
| 2018 | AlphaFold1 at CASP12 | First evidence deep learning could predict protein structure |
| 2020 | AlphaFold2 at CASP14 | Near-experimental accuracy achieved |
| 2022 | AlphaFold DB: 200M proteins | Structural biology democratized |
| 2023 | GNoME: 2.2M materials | AI materials discovery at scale |
| 2024 | Nobel Prize in Chemistry | AI for Science gains ultimate validation |
| 2024 | AlphaFold3: biomolecular complexes | Unified protein-small molecule prediction |
| 2025 | Isomorphic Labs drug trials | First AlphaFold-derived drugs enter Phase I |
| 2025 | GenCast: weather beating ECMWF | AI surpasses best physics-based weather models |
| 2026 | Scientific LLMs mature | Domain-specific foundation models proliferate |

### Investment Landscape

The AI for Science sector has attracted enormous capital:

- **Isomorphic Labs** (Alphabet): $3B+ invested, partnership with Eli Lilly and Novartis
- **Recursion Pharmaceuticals**: $800M+ raised, AI-driven drug discovery platform
- **Schrödinger Inc.**: $1.2B market cap, physics-based + ML molecular modeling
- **Insilico Medicine**: $400M+ raised, end-to-end AI drug discovery
- **BenevolentAI**: $290M raised, knowledge graph-driven drug discovery
- **Exscientia**: Acquired by Recursion for $650M, AI-designed drugs in trials
- **Citrine Information**: $300M+ for AI materials informatics
- **WeatherFlow**: $150M for AI weather forecasting

### Market Segmentation

```
AI for Science Market (2026)
├── Drug Discovery & Pharma: $4.1B (50%)
│   ├── Target Identification: $0.8B
│   ├── Lead Optimization: $1.2B
│   ├── Clinical Trial Design: $1.1B
│   └── Drug Repurposing: $1.0B
├── Materials Science: $1.8B (22%)
│   ├── Battery Materials: $0.7B
│   ├── Catalysts: $0.5B
│   ├── Semiconductors: $0.4B
│   └── Polymers/Composites: $0.2B
├── Genomics & Proteomics: $1.1B (13%)
├── Climate & Weather: $0.6B (7%)
├── Chemistry & Synthesis: $0.4B (5%)
└── Physics & Mathematics: $0.2B (3%)
```

---

## The Nobel Catalyst: AI for Science Goes Mainstream

### The 2024 Nobel Prize in Chemistry

The award to Hassabis, Jumper, and Baker was not merely symbolic — it fundamentally shifted how institutions, investors, and scientists perceive AI's role in discovery.

**Three laureates, three dimensions of AI-science impact:**

1. **Demis Hassabis & John Jumper (AlphaFold)**: Proving AI can solve a 50-year grand challenge in biology (protein structure prediction)
2. **David Baker (RoseTTAFold, ProteinMPNN, de novo design)**: Demonstrating AI can *create* novel proteins that don't exist in nature

### Institutional Response

Post-Nobel, we've seen:

- **NIH**: Created the "AI for Science" initiative with $500M in dedicated funding
- **EU**: Horizon Europe added "AI-Driven Discovery" as a priority area
- **China**: Launched the "AI for Science National Plan" with $2B in funding
- **UK**: AI Safety Institute expanded to include AI-for-science safety
- **Universities**: 140+ new "AI for Science" degree programs launched worldwide

### Cultural Shift in Science

The Nobel Prize accelerated a cultural transformation:

- **Before 2024**: AI was viewed as a "tool" — like a better microscope
- **After 2024**: AI is increasingly viewed as a "collaborator" — capable of generating hypotheses humans wouldn't conceive

This shift is reflected in how scientists talk about AI:

| Old Framing | New Framing |
|-------------|-------------|
| "We used AI to analyze our data" | "AI suggested this hypothesis" |
| "AI-accelerated screening" | "AI-designed molecule" |
| "Computational辅助" | "AI-driven discovery pipeline" |
| "ML model for prediction" | "Foundation model for science" |

---

## Core Application Domains

### 1. Drug Discovery & Development

**The Drug Discovery Timeline (Traditional vs. AI-Augmented)**

| Phase | Traditional | AI-Augmented | Time Saved |
|-------|-------------|--------------|------------|
| Target Discovery | 2–4 years | 6–18 months | 50–75% |
| Lead Discovery | 1–3 years | 3–12 months | 60–75% |
| Lead Optimization | 1–2 years | 3–9 months | 50–70% |
| Preclinical | 1–2 years | 6–12 months | 40–50% |
| Clinical Trials | 6–10 years | 4–7 years | 20–40% |
| **Total** | **11–21 years** | **5–10 years** | **40–60%** |

**Key AI Methods in Drug Discovery:**

- **Molecular Property Prediction**: Graph neural networks, transformers on SMILES strings
- **Virtual Screening**: Diffusion models for molecular generation
- **Protein-Ligand Docking**: AlphaFold3-based structure prediction + docking
- **ADMET Prediction**: Multi-task learning for absorption, distribution, metabolism, excretion, toxicity
- **De Novo Drug Design**: Reinforcement learning + generative models
- **Clinical Trial Optimization**: NLP for protocol design, patient matching

**Real-World Impact (2026):**

- 67 AI-discovered molecules in clinical trials (up from 15 in 2023)
- 4 AI-designed drugs approved by FDA (including Insilico's ISM001-055 for IPF)
- Average cost reduction: $500M–$1B per approved drug
- Success rate improvement: 2–3x for AI-designed vs. traditional candidates

### 2. Protein Engineering & Design

**Beyond Prediction: Creation**

The field has evolved from predicting existing protein structures to designing entirely new proteins:

- **De Novo Design**: Creating proteins with no natural homolog
- **Enzyme Engineering**: Optimizing catalytic activity, stability, selectivity
- **Antibody Design**: Computational antibody optimization
- **Protein-Protein Interactions**: Designing binders for specific targets

**Key Tools (2026):**

| Tool | Purpose | Accuracy | Speed |
|------|---------|----------|-------|
| AlphaFold3 | Structure prediction | Near-experimental | Minutes |
| RFdiffusion | De novo protein generation | High novelty + function | Seconds per design |
| ProteinMPNN | Sequence design | 50%+ recovery rate | Milliseconds per residue |
| ESMFold | Fast structure prediction | ~95% of AlphaFold quality | Milliseconds |
| Chai-1 | Multi-modal biomolecular | AlphaFold3-competitive | Minutes |
| Boltz-1 | Open-source biomolecular | Apache 2.0 license | Minutes |

### 3. Materials Science & Discovery

**AI-Accelerated Materials Discovery**

Google DeepMind's GNoME (Graph Networks for Materials Exploration) demonstrated AI can predict stability of millions of inorganic materials:

- **2.2M new stable crystals** predicted (9x increase in known stable materials)
- **381k materials** synthesized or available for synthesis
- Applications: batteries, superconductors, solar cells, catalysts

**Current Frontiers:**

- **Battery Materials**: Lithium-ion alternatives, solid-state electrolytes
- **Catalysts**: Green hydrogen production, CO2 conversion
- **Superconductors**: Room-temperature superconductor prediction
- **Semiconductors**: Novel chip materials beyond silicon
- **Polymers**: Biodegradable materials, high-performance composites

### 4. Genomics & Proteomics

**Foundation Models for Biology**

- **Evo**: 7B-parameter DNA language model from Arc Institute/Stanford
- **Nucleotide Transformer**: DNA foundation model for genomics
- **scGPT**: Single-cell RNA-seq foundation model
- **scFoundation**: Large-scale single-cell foundation model
- **ProGen**: Protein language model for enzyme generation

**Applications:**

- Gene expression prediction from sequence
- Variant effect prediction (pathogenicity scoring)
- Single-cell atlas construction
- Metagenomics analysis
- Epigenetic modification prediction

### 5. Climate & Weather Prediction

**AI Surpasses Physics-Based Models**

- **GraphCast** (Google DeepMind): Outperforms ECMWF's HRES on 90%+ of targets
- **GenCast** (Google DeepMind): Probabilistic ensemble forecasting, beats ECMWF ENS
- **Pangu-Weather** (Huawei): First ML model to outperform operational weather prediction
- **FourCastNet** (NVIDIA): Adaptive Fourier Neural Operator for weather

**Beyond Weather:**

- **Climate Modeling**: AI emulators for general circulation models
- **Extreme Event Prediction**: Heat waves, hurricanes, floods
- **Carbon Cycle**: AI for carbon flux estimation
- **Biodiversity**: Species distribution modeling

### 6. Mathematics & Formal Reasoning

**AI for Mathematical Discovery**

- **AlphaTensor**: Matrix multiplication algorithms (discovered faster than Strassen)
- **FunSearch** (DeepMind): LLM-guided search for mathematical conjectures
- **LeanDojo**: Theorem proving with LLMs
- **AlphaProof**: Automated theorem solving at IMO level

**Impact:**

- 2 new matrix multiplication algorithms discovered
- Several novel combinatorial bounds found
- Automated proof verification becoming routine
- AI assisting in conjecture generation for pure mathematics

### 7. Chemistry & Synthesis Planning

**Retrosynthesis and Reaction Prediction**

- **ASKCOS** (MIT): Automated retrosynthesis planning
- **RXN for Chemistry** (IBM): Transformer-based reaction prediction
- **Molecular SetProperty**: End-to-end molecular design
- **SynSpace**: Synthetic accessibility scoring

**Key Capabilities:**

- Retrosynthetic route planning for novel molecules
- Reaction condition optimization
- Yield prediction
- Side product prediction
- Green chemistry pathway selection

---

## Key Players & Ecosystem

### Tier 1: Deep Science Companies

| Company | Focus | Key Tech | Status (2026) |
|---------|-------|----------|---------------|
| Isomorphic Labs | Drug discovery | AlphaFold3, diffusion models | 2 Phase I trials |
| Insilico Medicine | End-to-end drug discovery | Pharma.AI suite | 1 FDA approval |
| Recursion Pharmaceuticals | Phenomics + drug discovery | OS (operating system) | 3 Phase II trials |
| Schrödinger Inc. | Molecular simulation | Physics + ML hybrid | $1.2B market cap |
| BenevolentAI | Knowledge graph drug discovery | BEN Platform | 2 Phase I trials |
| Exscientia | AI drug design | Centaur Chemist | Acquired by Recursion |

### Tier 2: Tech Giants with Science AI Divisions

| Division | Parent | Focus |
|----------|--------|-------|
| Google DeepMind | Alphabet | AlphaFold, materials, weather, math |
| Microsoft Research AI for Science | Microsoft | Molecular modeling, protein design |
| Meta FAIR (Science AI) | Meta | Materials, weather, protein structure |
| NVIDIA BioNeMo | NVIDIA | Simulation, drug discovery platforms |
| IBM Research | IBM | Chemistry, materials, genomics |
| Amazon Science | AWS | Climate, genomics, drug discovery |

### Tier 3: Academic Centers

| Center | Institution | Focus |
|--------|-------------|-------|
| Arc Institute | Stanford/UCSF | Foundation models for biology |
| Mila | Montreal | Generative chemistry, materials |
| Stanford AI4Science | Stanford | Multi-domain AI for science |
| MIT Jameel Clinic | MIT | AI for health |
| EMBL-EBI | Cambridge | Protein structure, genomics |
| Allen Institute for AI | Seattle | Semantic Scholar, science NLP |

### Tier 4: Open Source & Community

| Project | Purpose | Stars |
|---------|---------|-------|
| OpenFold | Open-source AlphaFold reimplementation | 1.8k |
| Boltz-1 | Open-source biomolecular prediction | 2.1k |
| RFdiffusion | De novo protein design | 3.2k |
| ProteinMPNN | Sequence design | 2.4k |
| DeepChem | Deep learning for chemistry | 5.3k |
| TorchDrug | Drug discovery library | 1.1k |
| GNoME data | Materials discovery dataset | 800+ |

---

## Technical Foundations

### Core Architecture Patterns

#### 1. Graph Neural Networks (GNNs) for Molecules

Molecules are naturally graphs — atoms are nodes, bonds are edges. GNNs excel here:

```python
# Example: Molecular property prediction with PyTorch Geometric
import torch
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data

class MoleculeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        # Node-level convolutions
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        # Global pooling to graph-level representation
        x = global_mean_pool(x, batch)
        return self.classifier(x)
```

#### 2. Transformer-Based Molecular Models

Treating molecules as sequences (SMILES strings) or sets (atom features):

```python
# SMILES-based molecular transformer
class MolecularTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.property_head = torch.nn.Linear(d_model, 1)  # regression head
    
    def forward(self, smiles_tokens):
        x = self.embedding(smiles_tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        # Pool over sequence
        x = x.mean(dim=1)
        return self.property_head(x)
```

#### 3. Diffusion Models for Molecular Generation

State-of-the-art approach for de novo molecule design:

```python
# Simplified diffusion model for molecular generation
class MolecularDiffusion(torch.nn.Module):
    """Denoising diffusion for 3D molecular conformation generation"""
    
    def __init__(self, atom_types=100, hidden_dim=256, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        # Noise schedule
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Denoising network (3D equivariant transformer)
        self.denoiser = EquivariantTransformer(
            atom_types=atom_types,
            hidden_dim=hidden_dim,
            num_layers=8
        )
    
    def forward(self, noisy_coords, atom_types, t):
        """Predict noise added to coordinates"""
        return self.denoiser(noisy_coords, atom_types, t)
    
    @torch.no_grad()
    def sample(self, num_atoms, atom_types):
        """Generate a new molecule from noise"""
        coords = torch.randn(num_atoms, 3)
        for t in reversed(range(self.timesteps)):
            noise_pred = self.forward(coords, atom_types, t)
            coords = self.p_sample(coords, noise_pred, t)
        return coords
```

### Foundation Models for Science

The emerging paradigm is domain-specific foundation models:

| Model | Domain | Parameters | Training Data |
|-------|--------|-----------|---------------|
| AlphaFold3 | Biomolecular structure | ~100M | PDB + AFDB |
| ESM-3 | Protein language | 1.4B | UniRef |
| Evo | DNA language | 7B | 2.7T nt |
| GNoME | Materials stability | ~50M | Materials Project |
| GenCast | Weather | ~50M | ERA5 reanalysis |
| Galactica | Science papers | 120B | 48M papers |

### Data Challenges in AI for Science

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Small datasets | Experimental data is expensive | Transfer learning, self-supervised |
| Noisy labels | Measurement uncertainty | Bayesian approaches, uncertainty quantification |
| Distribution shift | Lab conditions vary | Domain adaptation, meta-learning |
| Class imbalance | Rare events (e.g., adverse reactions) | Oversampling, focal loss |
| Temporal drift | Methods improve over time | Continuous learning, versioning |
| Multi-modal data | Images + spectra + text | Multi-modal architectures |

---

## Impact Metrics & ROI

### Drug Discovery ROI

| Metric | Traditional | AI-Augmented | Improvement |
|--------|-------------|--------------|-------------|
| Cost per approved drug | $2.6B | $1.2–1.8B | 30–55% |
| Time to IND | 5–7 years | 2–4 years | 40–60% |
| Clinical trial success rate | 8–12% | 15–25% | 2x |
| Target identification | 18 months | 3–6 months | 65–75% |
| Lead optimization | 24 months | 6–12 months | 50–75% |

### Materials Discovery ROI

- **GNoME**: Identified 2.2M stable materials in months vs. decades of manual search
- **Battery materials**: AI accelerated identification of solid-state electrolyte candidates by 10x
- **Catalysts**: 40% improvement in catalyst activity prediction accuracy

### Scientific Productivity

- **Paper output**: AI-augmented research groups publish 2–3x more papers
- **Citation impact**: AI-assisted papers cited 1.5x more on average
- **Grant success**: Groups using AI methods have 30% higher NIH/NSF funding rates
- **Time savings**: Routine analysis tasks reduced by 60–80%

---

## Challenges & Limitations

### Technical Challenges

1. **Out-of-Distribution Generalization**: Models trained on known chemistry may fail for truly novel compounds
2. **Physical Law Compliance**: Ensuring generated molecules obey quantum mechanics, thermodynamics
3. **Synthesizability**: Designing molecules that can actually be made in a lab
4. **Multi-Objective Optimization**: Balancing potency, selectivity, toxicity, solubility simultaneously
5. **Uncertainty Quantification**: Knowing when the model doesn't know

### Scientific Challenges

1. **Validation Gap**: AI predictions still need experimental verification (weeks to months)
2. **Causation vs. Correlation**: Models may learn spurious correlations in small datasets
3. **Reproducibility**: AI-designed experiments must be reproducible across labs
4. **Interpretability**: Black-box predictions are hard for scientists to trust and build upon

### Organizational Challenges

1. **Talent Scarcity**: Few people have both deep ML and domain science expertise
2. **Data Silos**: Pharmaceutical companies hoard data; sharing is limited
3. **Cultural Resistance**: Many scientists distrust "AI tells us what to study"
4. **IP Concerns**: Who owns an AI-designed molecule?
5. **Regulatory Uncertainty**: How should FDA evaluate AI-designed drugs?

---

## Regulatory Landscape

### FDA & AI

The FDA has been actively developing frameworks for AI in drug development:

- **2023**: Draft guidance on AI/ML in drug development
- **2024**: Final guidance on computer software assurance (CSA)
- **2025**: Pilot program for AI-designed drug review pathways
- **2026**: Expected guidance on AI-generated evidence in submissions

**Key FDA Positions:**
- AI-designed drugs must meet the same safety/efficacy standards
- AI models used in submissions must be validated and documented
- "Human in the loop" still required for critical decisions
- Real-world evidence from AI systems accepted with caveats

### EMA & Global Regulatory

- **EMA**: Created AI task force; expecting guidance by end of 2026
- **PMDA (Japan)**: Early adopter of AI-designed drug review
- **NMPA (China)**: Developing parallel framework
- **WHO**: Published principles for AI in health (2024)

### Intellectual Property

Open questions being debated:
- Can AI-invented molecules be patented? (Thaler v. Vidal implications)
- Who is the "inventor" when AI designs a drug?
- How to handle AI-generated data in patent claims?
- Trade secret vs. patent protection for AI models

---

## Future Outlook 2026–2030

### Near-Term (2026–2027)

- 100+ AI-designed drugs in clinical trials
- First AI-designed drug approved for a novel mechanism of action
- Foundation models become standard tools in biology labs
- Automated "self-driving labs" begin routine operation
- AI materials discovery moves from academic to industrial scale

### Medium-Term (2028–2030)

- AI designs entire drug candidates from target to IND in < 12 months
- Protein design becomes a routine engineering discipline
- AI-driven climate models achieve 10x resolution improvement
- Autonomous scientific agents begin conducting experiments
- First mathematical conjecture proven primarily by AI

### Long-Term (2030+)

- "AI scientist" agents autonomously discover new physics/chemistry/biology
- Personalized medicine: AI designs drugs for individual patients
- AI-designed materials enable room-temperature superconductors
- Full simulation of cellular processes at atomic resolution
- AI-accelerated discovery becomes the default mode of science

### Key Trends to Watch

1. **Foundation Model Proliferation**: Domain-specific models for every scientific field
2. **Lab Automation Integration**: AI design → robotic synthesis → automated testing
3. **Federated Learning for Pharma**: Collaborative models without data sharing
4. **Causal AI**: Moving from correlation to causation in scientific inference
5. **AI Ethics in Science**: Responsible use of AI in sensitive research (dual-use)

---

## Quick Reference: Essential Resources

| Resource | Type | Link |
|----------|------|------|
| AlphaFold DB | Database | https://alphafold.ebi.ac.uk/ |
| GNoME Data | Dataset | https://github.com/google-deepmind/materials_project |
| DeepChem | Library | https://deepchem.io/ |
| TorchDrug | Library | https://github.com/medigroup/torchdrug |
| BioNeMo | Platform | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/collections/bionemo |
| OpenFold | Framework | https://github.com/OpenFoldCollaboration/OpenFold |
| Boltz-1 | Model | https://github.com/boltz-community/boltz |
| RFdiffusion | Tool | https://github.com/RosettaCommons/RFdiffusion |
| ProteinMPNN | Tool | https://github.com/dauparas/ProteinMPNN |

---

*This document is part of the AI Base Knowledge Library. For the complete library index, see the root README.md.*
