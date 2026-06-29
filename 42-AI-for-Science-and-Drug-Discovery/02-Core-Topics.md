# 02 — Core Topics in AI for Science & Drug Discovery

> **Category**: 42-AI-for-Science-and-Drug-Discovery  
> **Focus**: Detailed technical domains, methods, and applications  
> **Cross-References**: [01-Overview.md](./01-Overview.md), [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md), [17-Research-Frontiers-2026/08-AI-for-Science.md](../17-Research-Frontiers-2026/08-AI-for-Science.md)

---

## Table of Contents

1. [Protein Structure Prediction & Design](#1-protein-structure-prediction--design)
2. [Small Molecule Drug Discovery](#2-small-molecule-drug-discovery)
3. [Biologics & Antibody Engineering](#3-biologics--antibody-engineering)
4. [Genomics & Transcriptomics](#4-genomics--transcriptomics)
5. [Materials Discovery](#5-materials-discovery)
6. [Chemical Synthesis Planning](#6-chemical-synthesis-planning)
7. [Climate & Earth Sciences](#7-climate--earth-sciences)
8. [Mathematics & Formal Reasoning](#8-mathematics--formal-reasoning)
9. [Medical Imaging & Diagnostics](#9-medical-imaging--diagnostics)
10. [Agricultural Science](#10-agricultural-science)

---

## 1. Protein Structure Prediction & Design

### 1.1 The AlphaFold Revolution

AlphaFold2 (2020) solved a 50-year grand challenge in biology. AlphaFold3 (2024) extended this to the entire biomolecular interactome.

**AlphaFold3 Architecture:**

```
Input: Protein sequence + Ligand + DNA/RNA
    ↓
[Single Embedding Module] → Per-token representations
    ↓
[Pair Embedding Module] → Inter-residue relationships
    ↓
[Triangular Attention] → 3D geometric consistency
    ↓
[Diffusion Module] → Raw atom coordinate prediction
    ↓
Output: 3D structure with confidence scores (pLDDT, PAE)
```

**Key Innovations:**
- **Diffusion-based coordinate prediction**: Directly predicts atomic coordinates rather than distance matrices
- **Multi-chain complexes**: Handles protein-protein, protein-ligand, protein-nucleic acid complexes
- **Post-translational modifications**: Glycosylation, phosphorylation, etc.
- **Confidence calibration**: Per-atom confidence scores (pLDDT > 90 = high confidence)

**Performance Benchmarks:**

| Target Type | AlphaFold2 Accuracy | AlphaFold3 Accuracy | Improvement |
|-------------|--------------------|--------------------|-------------|
| Protein-only | 87% (GDT) | 92% (GDT) | +6% |
| Protein-ligand | 55% (DockQ) | 77% (DockQ) | +40% |
| Protein-DNA | N/A | 73% (DockQ) | New capability |
| Protein-RNA | N/A | 68% (DockQ) | New capability |

### 1.2 De Novo Protein Design

Creating proteins that don't exist in nature:

**RFdiffusion Pipeline:**

```
Step 1: Design backbone using RFdiffusion
    → Generates novel protein backbones via denoising diffusion
    → Conditioned on desired function/shape/binding site
    
Step 2: Design sequence using ProteinMPNN
    → Assigns amino acid sequence that folds to the backbone
    → Recovery rate: 50%+ for designed backbones
    
Step 3: Validate with AlphaFold3
    → Predicts structure of designed sequence
    → Compares to intended design
    
Step 4: Experimental validation
    → Express, purify, characterize in vitro
    → X-ray/cryo-EM for ground truth
```

**Design Paradigms:**

| Paradigm | Description | Example | Success Rate |
|----------|-------------|---------|--------------|
| Scaffold design | Design around a functional motif | Binding site scaffolding | 60–80% |
| Motif scaffolding | Find backbone that supports a motif | Enzyme active sites | 40–60% |
| Binders | Design proteins that bind a target | Cytokine mimics | 30–50% |
| Enzymes | Design catalytic activity | Kemp eliminases | 10–30% |
| Self-assembling | Design higher-order structures | Nanocages | 20–40% |

### 1.3 Protein Language Models

Large-scale pre-trained models for protein understanding:

| Model | Parameters | Training Data | Key Capability |
|-------|-----------|---------------|----------------|
| ESM-3 | 1.4B | UniRef | Multi-modal protein understanding |
| ESM-2 | 15B | UniRef | Protein language modeling |
| ProtTrans | 3B | UniRef + BFD | Transfer learning for proteins |
| ProGen | 6.4B | UniProt | Protein generation |
| EVO | 7B | 2.7T nucleotides | DNA language model |
| DNABERT-2 | 400M | RefSeq | DNA understanding |

**Applications of Protein Language Models:**
- Variant effect prediction (pathogenicity)
- Protein function annotation
- Fitness landscape prediction
- Zero-shot mutation effect prediction
- Protein-protein interaction prediction

---

## 2. Small Molecule Drug Discovery

### 2.1 Molecular Representation

How molecules are encoded for ML:

**SMILES (Simplified Molecular Input Line Entry System):**
```
Aspirin: CC(=O)Oc1ccccc1C(=O)O
Caffeine: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Ibuprofen: CC(C)Cc1ccc(cc1)C(C)C(=O)O
```

**Molecular Graphs:**
```
Atoms → Nodes (features: element, charge, hybridization, etc.)
Bonds → Edges (features: type, conjugation, ring membership)
3D coords → Optional spatial features
```

**Fingerprints:**
- Morgan/ECFP: Circular fingerprints (most common)
- MACCS: Structural keys
- RDKit: Combination of features
- Learned: GNN-based embeddings

### 2.2 Virtual Screening

Finding hits from large compound libraries:

**Traditional Virtual Screening:**
```
1. Compile compound library (10^6 – 10^9 molecules)
2. Apply pharmacophore filter (reduce to ~10^5)
3. docking-based screening (reduce to ~10^3)
4. Expert review (reduce to ~10^2)
5. Experimental validation (yield ~10 hits)
```

**AI-Augmented Virtual Screening:**
```
1. Foundation model pre-filtering (10^6 → 10^4)
2. GNN-based property prediction (10^4 → 10^3)
3. Diversity selection (10^3 → 500)
4. Generative expansion around hits (500 → 2,000)
5. Experimental validation (yield ~50 hits)
```

**Key Methods:**

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Docking (AutoDock Vina) | Minutes/molecule | Moderate | Large libraries |
| GNN screening | Milliseconds/molecule | High | Focused libraries |
| Transformer screening | Milliseconds/molecule | High | Multi-property |
| Diffusion models | Seconds/molecule | Very High | Novel scaffolds |
| Pharmacophore + ML | Milliseconds/molecule | Moderate | Scaffold hopping |

### 2.3 ADMET Prediction

Predicting Absorption, Distribution, Metabolism, Excretion, and Toxicity:

**Multi-Task ADMET Model:**

```python
class ADMETPredictor(torch.nn.Module):
    """Multi-task molecular property prediction"""
    
    def __init__(self, mol_dim=512, hidden=256):
        super().__init__()
        self.encoder = MolecularEncoder(mol_dim, hidden)
        
        # Task-specific heads
        self.absorption = nn.Linear(hidden, 3)  # Caco-2, Pgp, HIA
        self.distribution = nn.Linear(hidden, 3)  # BBB, Vd, PPB
        self.metabolism = nn.Linear(hidden, 2)  # CYP inhibition, half-life
        self.excretion = nn.Linear(hidden, 2)  # CL, renal clearance
        self.toxicity = nn.Linear(hidden, 5)  # hERG, Ames, DILI, LD50, skin sens
    
    def forward(self, mol_features):
        h = self.encoder(mol_features)
        return {
            'absorption': self.absorption(h),
            'distribution': self.distribution(h),
            'metabolism': self.metabolism(h),
            'excretion': self.excretion(h),
            'toxicity': self.toxicity(h)
        }
```

**ADMET Databases:**
- ADMETlab 2.0: 81 datasets, 700k+ compounds
- Therapeutics Data Commons: 230+ ML datasets
- MoleculeNet: Benchmark datasets for molecular ML
- ChEMBL: Bioactivity data

### 2.4 Generative Drug Design

Creating novel molecules with desired properties:

**Generative Approaches:**

| Approach | Method | Strength | Weakness |
|----------|--------|----------|----------|
| VAE | Variational autoencoder | Smooth latent space | Mode collapse |
| GAN | Generative adversarial | High quality samples | Training instability |
| Flow | Normalizing flows | Exact likelihood | Limited expressiveness |
| Diffusion | Denoising diffusion | State-of-the-art quality | Slow sampling |
| RL | Reinforcement learning | Goal-directed | Reward hacking |
| LLM | Language model on SMILES | Leverages text pretraining |Validity issues |

**Reinforcement Learning for Drug Design:**

```python
# Simplified RL agent for molecule generation
class DrugDesignAgent:
    def __init__(self, generator, oracle):
        self.generator = generator  # SMILES generative model
        self.oracle = oracle        # Property prediction oracle
    
    def generate_and_evaluate(self, target_properties):
        molecule = self.generator.sample()
        properties = self.oracle.predict(molecule)
        
        # Multi-objective reward
        reward = (
            0.3 * properties['potency_score'] +
            0.2 * properties['selectivity_score'] +
            0.2 * properties['admet_score'] +
            0.2 * properties['novelty_score'] +
            0.1 * properties['synthesizability_score']
        )
        return molecule, reward
    
    def train_step(self, batch_size=64):
        molecules, rewards = zip(*[self.generate_and_evaluate() for _ in range(batch_size)])
        # REINFORCE or PPO update
        self.generator.update(molecules, rewards)
```

---

## 3. Biologics & Antibody Engineering

### 3.1 Antibody Design

Monoclonal antibodies are the fastest-growing drug class. AI is transforming their design:

**Antibody Structure:**
- Heavy chain + Light chain
- Complementarity-determining regions (CDRs): 6 loops that determine binding
- Framework regions: Structural scaffold

**AI Approaches:**

| Task | Method | Tools |
|------|--------|-------|
| CDR design | Sequence design | ProteinMPNN, Absolut |
| Binding prediction | Docking + ML | AlphaFold3, IgGSM |
| Humanization | Framework optimization | RAbD, abLSTM |
| Stability optimization | Mutagenesis prediction | ThermoMPNN, ProteinGym |
| Aggregation prediction | Sequence features | OBGPred, MAbsolve |

### 3.2 mRNA Vaccine Design

AI for mRNA therapeutics (post-COVID acceleration):

- **Codon optimization**: AI-optimized codons for expression
- **UTR design**: AI-designed 5' and 3' UTRs for stability
- **Antigen design**: Optimizing immunogenic protein sequences
- **Lipid nanoparticle design**: AI-optimized delivery vehicles
- **Structure prediction**: mRNA secondary structure prediction

### 3.3 Cell & Gene Therapy

- **CAR-T cell design**: Optimizing chimeric antigen receptor sequences
- **Gene editing guide RNA**: CRISPR gRNA design and off-target prediction
- **Cell type conversion**: AI-guided cellular reprogramming
- **Immunogenicity prediction**: Predicting immune responses to engineered cells

---

## 4. Genomics & Transcriptomics

### 4.1 DNA Language Models

Foundation models for understanding genomic sequences:

| Model | Scale | Architecture | Key Finding |
|-------|-------|--------------|-------------|
| EVO | 7B | Mamba (SSM) | Predicts gene essentiality from sequence |
| DNABERT-2 | 400M | Transformer | Multi-species DNA understanding |
| Nucleotide Transformer | 2.5B | Transformer | Zero-shot gene expression prediction |
| HyenaDNA | 1B | Hyena | Long-range genomic understanding |
| NaviGenie | 500M | GPT-like | Genomic instruction following |

**Key Tasks:**
- Promoter prediction
- Enhancer identification
- Splice site prediction
- Variant effect prediction
- Gene expression prediction from sequence

### 4.2 Single-Cell AI

The single-cell revolution meets AI:

**scGPT / scFoundation:**
- Pre-trained on millions of single-cell RNA-seq profiles
- Zero-shot cell type annotation
- Gene perturbation prediction
- Multi-omic integration (RNA + ATAC + protein)

**Applications:**
- Cancer heterogeneity mapping
- Developmental biology trajectory inference
- Drug response prediction at single-cell resolution
- Spatial transcriptomics integration

### 4.3 Cancer Genomics

AI for precision oncology:

- **Tumor mutational burden prediction**
- **Neoantigen prediction** (for immunotherapy)
- **Drug sensitivity prediction** from genomic profiles
- **Resistance mechanism identification**
- **Metastasis prediction**

---

## 5. Materials Discovery

### 5.1 The GNoME Revolution

Google DeepMind's Graph Networks for Materials Exploration:

**GNoME Results:**
- 2.2 million new stable crystals predicted
- 381,000 materials with synthesis data
- 736,000 materials with DFT-computed properties
- 9x expansion of known stable materials

**How GNoME Works:**

```
Input: Crystal structure (atomic positions + lattice)
    ↓
[Graph Neural Network] → Stability prediction
    ↓
[Structure Search] → Candidate enumeration
    ↓
[DFT Validation] → Ground truth confirmation
    ↓
Output: Stable material with predicted properties
```

### 5.2 Battery Materials

AI for next-generation energy storage:

| Challenge | AI Approach | Progress |
|-----------|-------------|----------|
| Solid electrolytes | GNN stability prediction | 10x candidates identified |
| Cathode design | Generative models | Novel Ni-rich compositions |
| Anode materials | Graphite alternatives | Silicon-carbon composites |
| Interface stability | MD + ML | Degradation prediction |

### 5.3 Catalyst Design

AI for chemical catalysis:

- **Active site prediction**: Identifying catalytic centers
- **Selectivity optimization**: AI-guided catalyst tuning
- **Mechanism elucidation**: ML-assisted reaction pathway analysis
- **High-throughput screening**: Virtual catalyst libraries

### 5.4 Superconductor Discovery

The holy grail of materials science:

- **Room-temperature superconductor prediction**: Graph networks on crystal databases
- **Pressure-induced superconductivity**: AI-guided high-pressure experiments
- **Mechanism understanding**: AI for electron-phonon coupling analysis

---

## 6. Chemical Synthesis Planning

### 6.1 Retrosynthetic Analysis

Planning backwards from target to starting materials:

**ASKCOS (MIT):**
- Transformer-based retrosynthesis
- Conditions recommendation
- Reaction prediction
- Score-based route ranking

**RXN for Chemistry (IBM):**
- Transformer on reaction SMILES
- Top-1 accuracy: 90%+ for single-step
- Multi-step planning with tree search

### 6.2 Reaction Condition Optimization

AI for optimizing experimental conditions:

```python
# Bayesian optimization for reaction conditions
class ReactionOptimizer:
    def __init__(self, reaction_type):
        self.surrogate = GaussianProcessRegressor()
        self.acquisition = ExpectedImprovement()
    
    def optimize(self, n_iterations=50):
        for i in range(n_iterations):
            # Suggest next experiment
            conditions = self.acquisition.suggest(self.surrogate)
            
            # Run experiment (or simulation)
            yield = self.run_experiment(conditions)
            
            # Update model
            self.surrogate.fit(conditions, yield)
        
        return self.get_best_conditions()
```

### 6.3 Flow Chemistry + AI

Automated synthesis with AI control:

- **Real-time monitoring**: UV-Vis, IR, mass spec
- **Adaptive control**: AI adjusts conditions on-the-fly
- **Safety optimization**: Predictive safety monitoring
- **Scale-up guidance**: Lab → pilot → production

---

## 7. Climate & Earth Sciences

### 7.1 Weather Prediction

AI has surpassed the best physics-based weather models:

| Model | Resolution | Speed | Accuracy vs. ECMWF |
|-------|-----------|-------|---------------------|
| GraphCast | 0.25° | Seconds | Better on 90% targets |
| GenCast | 0.25° | Minutes | Better on 97% targets |
| Pangu-Weather | 0.25° | Seconds | Better on 85% targets |
| FourCastNet | 0.25° | Seconds | Comparable |
| Aurora | 0.1° | Minutes | High-res, multi-variable |

### 7.2 Climate Modeling

AI emulators for climate simulation:

- **General circulation model acceleration**: 1000x faster than traditional GCMs
- **Downscaling**: Global → regional → local predictions
- **Tipping point prediction**: Early warning signals for climate transitions
- **Carbon cycle modeling**: AI for global carbon budget estimation

### 7.3 Earth Observation

AI for satellite data analysis:

- **Deforestation monitoring**: Real-time forest cover change detection
- **Crop yield prediction**: AI for agricultural forecasting
- **Disaster response**: Flood, fire, earthquake damage assessment
- **Urban planning**: Land use change prediction

---

## 8. Mathematics & Formal Reasoning

### 8.1 AI for Mathematical Discovery

**AlphaTensor:**
- Found faster matrix multiplication algorithms
- Improved Strassen's algorithm for specific sizes
- Discovered algorithms humans hadn't found in 50 years

**FunSearch (DeepMind):**
- LLM + evolutionary search for mathematical functions
- Found new upper bounds for cap set problems
- Novel combinatorial constructions

**AlphaProof (DeepMind):**
- Automated theorem proving in Lean 4
- Solved IMO-level problems
- First AI to achieve IMO silver medal level

### 8.2 AI-Assisted Proof Development

- **Lean 4 + AI**: Automated tactic generation
- **Coq + ML**: Proof term synthesis
- **Formal verification**: AI for software correctness proofs
- **Type theory**: AI for dependent type inference

### 8.3 Conjecture Generation

AI as a hypothesis generator:

- Pattern recognition in mathematical objects
- Analogy-based conjecture formation
- Empirical testing of mathematical statements
- Guidance for human mathematicians

---

## 9. Medical Imaging & Diagnostics

### 9.1 Radiology AI

- **Chest X-ray**: Pneumonia, TB, lung nodule detection
- **Mammography**: Breast cancer screening
- **CT**: Organ segmentation, tumor detection
- **MRI**: Brain tumor classification, cardiac analysis

### 9.2 Pathology AI

- **Whole slide image analysis**: Cancer grading
- **Digital pathology**: Automated cell counting, feature extraction
- **Molecular pathology**: Predicting molecular markers from H&E

### 9.3 Ophthalmology AI

- **Diabetic retinopathy screening**: FDA-approved AI systems
- **Glaucoma detection**: Optic nerve analysis
- **Age-related macular degeneration**: Early detection

### 9.4 Dermatology AI

- **Skin lesion classification**: Melanoma detection
- **Psoriasis severity**: AI-assisted scoring
- **Wound healing assessment**: Automated monitoring

---

## 10. Agricultural Science

### 10.1 Crop Science

- **Genomic selection**: AI for crop breeding optimization
- **Disease detection**: Plant disease identification from images
- **Yield prediction**: Field-level crop yield forecasting
- **Pesticide optimization**: Targeted application recommendations

### 10.2 Soil Science

- **Soil composition analysis**: Spectroscopy + ML
- **Carbon sequestration prediction**: Soil carbon modeling
- **Nutrient management**: AI-guided fertilization

### 10.3 Livestock

- **Animal health monitoring**: Wearable sensors + AI
- **Feed optimization**: Precision nutrition
- **Breeding optimization**: Genomic selection for livestock

---

## Summary: Core Topics Map

```
AI for Science & Drug Discovery
├── Protein Science
│   ├── Structure Prediction (AlphaFold3, ESMFold, Chai-1)
│   ├── De Novo Design (RFdiffusion, ProteinMPNN)
│   └── Protein Language Models (ESM-3, ProGen)
├── Small Molecule Discovery
│   ├── Virtual Screening (GNN, Transformer)
│   ├── ADMET Prediction (Multi-task learning)
│   └── Generative Design (Diffusion, RL)
├── Biologics
│   ├── Antibody Engineering
│   ├── mRNA Design
│   └── Cell/Gene Therapy
├── Genomics
│   ├── DNA Language Models (EVO, DNABERT-2)
│   ├── Single-Cell AI (scGPT)
│   └── Cancer Genomics
├── Materials Science
│   ├── GNoME (Stability prediction)
│   ├── Battery Materials
│   ├── Catalyst Design
│   └── Superconductor Discovery
├── Chemistry
│   ├── Retrosynthesis (ASKCOS, RXN)
│   ├── Reaction Optimization
│   └── Flow Chemistry + AI
├── Climate & Earth Science
│   ├── Weather Prediction (GraphCast, GenCast)
│   ├── Climate Modeling
│   └── Earth Observation
├── Mathematics
│   ├── Algorithm Discovery (AlphaTensor)
│   ├── Theorem Proving (AlphaProof)
│   └── Conjecture Generation
├── Medical Diagnostics
│   ├── Radiology, Pathology, Ophthalmology
│   └── Dermatology
└── Agricultural Science
    ├── Crop Science
    ├── Soil Science
    └── Livestock
```

---

*This document is part of the AI Base Knowledge Library. For the complete library index, see the root README.md.*
