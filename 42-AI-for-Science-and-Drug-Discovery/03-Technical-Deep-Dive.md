# 03 — Technical Deep Dive: AI for Science & Drug Discovery

> **Category**: 42-AI-for-Science-and-Drug-Discovery  
> **Focus**: Advanced architectures, training methods, evaluation, and production systems  
> **Cross-References**: [02-Core-Topics.md](./02-Core-Topics.md), [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md), [06-Advanced/](../06-Advanced/)

---

## Table of Contents

1. [Model Architectures for Scientific Data](#1-model-architectures-for-scientific-data)
2. [Training Paradigms](#2-training-paradigms)
3. [Evaluation & Benchmarking](#3-evaluation--benchmarking)
4. [Data Pipelines & Feature Engineering](#4-data-pipelines--feature-engineering)
5. [Uncertainty Quantification](#5-uncertainty-quantification)
6. [Multi-Modal Scientific Learning](#6-multi-modal-scientific-learning)
7. [Active Learning & Closed-Loop Discovery](#7-active-learning--closed-loop-discovery)
8. [Production Deployment](#8-production-deployment)
9. [Case Study: End-to-End Drug Discovery Pipeline](#9-case-study-end-to-end-drug-discovery-pipeline)

---

## 1. Model Architectures for Scientific Data

### 1.1 Equivariant Neural Networks

Physical systems have symmetries — rotation, translation, permutation. Equivariant networks respect these:

**SE(3)-Equivariant Networks:**

```python
import torch
from e3nn import o3

class SE3EquivariantLayer(torch.nn.Module):
    """Equivariant layer for 3D molecular data"""
    
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out)
        self.norm = o3.LayerNorm(irreps_out)
    
    def forward(self, x, edge_src, edge_dst, edge_attr):
        # x: node features (irreps_in)
        # edge_attr: edge features (irreps_in)
        # equivariant: rotations of input → rotations of output
        
        # Message passing with equivariant features
        messages = self.tp(x[edge_src], x[edge_dst], edge_attr)
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(messages), messages)
        
        return self.norm(out)
```

**Key Equivariant Architectures:**

| Architecture | Symmetry | Application | Parameters |
|-------------|----------|-------------|-----------|
| EGNN | E(n) | Molecular dynamics | 1M–10M |
| PaiNN | SE(3) | Molecular property | 3M–30M |
| SchNet | SE(3) | Potential energy | 1M–10M |
| DimeNet++ | SE(3) | Molecular property | 5M–50M |
| GemNet | SE(3) | High-accuracy property | 10M–100M |
| MACE | SE(3) | Materials, molecules | 5M–50M |

### 1.2 Graph Neural Networks for Molecules

**Message Passing Neural Networks (MPNNs):**

```python
class MPNNLayer(torch.nn.Module):
    """General message passing layer for molecular graphs"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_fn = nn.GRUCell(hidden_dim, node_dim)
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        # Compute messages
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.message_fn(msg_input)
        
        # Attention-weighted aggregation
        attn_weights = self.attention_fn(messages)
        weighted_msgs = messages * attn_weights
        
        # Aggregate
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_msgs), weighted_msgs)
        
        # Update
        x = self.update_fn(aggregated, x)
        return x
```

**Advanced GNN Variants:**

| Variant | Innovation | Best For |
|---------|-----------|----------|
| GAT | Attention over edges | Selective feature learning |
| MPNN | General message passing | Molecular property |
| SchNet | Continuous filter convolution | Potential energy surfaces |
| DimeNet | Directional message passing | Accurate molecular property |
| GemNet | Multi-order interactions | High-fidelity prediction |
| GIN | Expressive graph isomorphism | Graph classification |
| GINE | Edge-enhanced GIN | Molecular fingerprints |

### 1.3 Transformer Architectures for Science

**Molecule Transformer (for SMILES):**

```python
class MoleculeTransformer(nn.Module):
    """Transformer for SMILES-based molecular modeling"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, 
                 num_layers=12, max_seq_len=256):
        super().__init__()
        
        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Task-specific heads
        self.property_head = nn.Linear(d_model, 1)  # regression
        self.classification_head = nn.Linear(d_model, 2)  # binary
    
    def forward(self, smiles_tokens, task='property'):
        seq_len = smiles_tokens.size(1)
        positions = torch.arange(seq_len, device=smiles_tokens.device).unsqueeze(0)
        
        x = self.token_emb(smiles_tokens) + self.pos_emb(positions)
        
        # Causal mask (autoregressive generation)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        mask = mask.to(smiles_tokens.device)
        
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1)  # global average pooling
        
        if task == 'property':
            return self.property_head(x)
        return self.classification_head(x)
```

### 1.4 Diffusion Models for Molecular Generation

**Score-Based Generative Models for Molecules:**

```python
class MolecularScoreMatching(nn.Module):
    """Score matching for molecular conformation generation"""
    
    def __init__(self, atom_types, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Score network (predicts noise at each step)
        self.score_network = EquivariantTransformer(
            atom_types=atom_types,
            hidden_dim=hidden_dim,
            num_layers=8
        )
    
    def add_noise(self, x_0, t):
        """Forward process: add Gaussian noise"""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
    
    def training_loss(self, x_0, atom_types):
        """Simple denoising score matching loss"""
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
        x_t, noise = self.add_noise(x_0, t)
        
        predicted_noise = self.score_network(x_t, atom_types, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, num_atoms, atom_types, num_steps=100):
        """Reverse process: generate molecule from noise"""
        x = torch.randn(num_atoms, 3)
        
        for t in reversed(range(0, self.num_timesteps, self.num_timesteps // num_steps)):
            t_tensor = torch.full((num_atoms,), t, device=x.device)
            predicted_noise = self.score_network(x, atom_types, t_tensor)
            
            # DDIM update step
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
            )
        
        return x
```

### 1.5 Scientific Foundation Models

The paradigm shift: pre-train large models on scientific data, then fine-tune for specific tasks.

**Architecture Pattern:**

```
[Scientific Foundation Model]
├── Pre-training
│   ├── Self-supervised: Predict masked atoms/residues/genes
│   ├── Contrastive: Align molecular structure ↔ properties
│   ├── Generative: Denoise corrupted scientific data
│   └── Multi-modal: Align structure ↔ function ↔ text
│
└── Fine-tuning (Downstream Tasks)
    ├── Property prediction (regression)
    ├── Activity classification (binary/multi-class)
    ├── Structure prediction (seq2seq)
    ├── Generation (diffusion/autoregressive)
    └── Reasoning (chain-of-thought for science)
```

**Training Data Sources:**

| Data Type | Source | Scale | Quality |
|-----------|--------|-------|---------|
| Protein structures | PDB, AlphaFold DB | 214M+ structures | High |
| Chemical compounds | ChEMBL, PubChem | 200M+ compounds | Medium |
| Materials | Materials Project, ICSD | 150k+ entries | High |
| Genomic sequences | RefSeq, GenBank | Billions of nt | Medium |
| Reaction data | USPTO, Reaxys | 10M+ reactions | High |
| Scientific papers | PubMed, arXiv | 40M+ papers | Medium |

---

## 2. Training Paradigms

### 2.1 Self-Supervised Pre-training

**Masked Token Prediction (BERT-style):**

```python
def pretrain_molecular_bert(model, dataloader, mask_prob=0.15):
    """Pre-train molecular transformer with masked token prediction"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch in dataloader:
        smiles = batch['smiles_tokens']
        
        # Create masked version
        mask = torch.bernoulli(torch.full_like(smiles, mask_prob, dtype=torch.float))
        masked_smiles = smiles.clone()
        masked_smiles[mask.bool()] = MASK_TOKEN
        
        # Predict masked tokens
        logits = model(masked_smiles)
        loss = F.cross_entropy(
            logits[mask.bool()], 
            smiles[mask.bool()]
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Contrastive Learning (aligning modalities):**

```python
def contrastive_pretraining(structure_encoder, property_encoder, batch):
    """Align molecular structures with their properties"""
    structures = structure_encoder(batch['molecular_graph'])
    properties = property_encoder(batch['property_descriptions'])
    
    # Normalize
    structures = F.normalize(structures, dim=1)
    properties = F.normalize(properties, dim=1)
    
    # InfoNCE loss
    logits = structures @ properties.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss
```

### 2.2 Multi-Task Learning

Training on multiple related tasks simultaneously:

**Benefits:**
- Shared representations improve generalization
- Data efficiency (less data per task)
- Implicit regularization

```python
class MultiTaskScientificModel(nn.Module):
    """Multi-task model for molecular property prediction"""
    
    def __init__(self, mol_encoder, tasks):
        super().__init__()
        self.encoder = mol_encoder
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(512, n_outputs)
            for name, n_outputs in tasks.items()
        })
    
    def forward(self, mol_features):
        shared_repr = self.encoder(mol_features)
        return {name: head(shared_repr) for name, head in self.task_heads.items()}
    
    def compute_loss(self, predictions, targets, task_weights):
        total_loss = 0
        for task_name in predictions:
            if task_name in targets:
                loss = F.mse_loss(predictions[task_name], targets[task_name])
                total_loss += task_weights[task_name] * loss
        return total_loss
```

**Task Grouping for Drug Discovery:**

| Task Group | Tasks | Benefit |
|-----------|-------|---------|
| ADMET | Absorption, Distribution, Metabolism, Excretion, Toxicity | Shared pharmacokinetic features |
| Binding | Target binding, Selectivity, Off-target | Shared molecular recognition |
| Synthesis | Retrosynthesis, Reaction prediction, Yield | Shared chemical knowledge |
| Safety | Mutagenicity, Carcinogenicity, Reproductive toxicity | Shared toxicity mechanisms |

### 2.3 Transfer Learning Strategies

**From General to Specific:**

```
Step 1: Pre-train on large, general dataset
    → E.g., ChEMBL (2M compounds, 15K tasks)
    → Learns general chemical knowledge
    
Step 2: Fine-tune on medium, domain-specific dataset
    → E.g., Kinase inhibitor data (50K compounds)
    → Adapts to specific target class
    
Step 3: Adapt to small, specific dataset
    → E.g., Novel target (500 compounds)
    → Few-shot adaptation
```

**Effective Fine-Tuning Methods:**

| Method | Parameters Updated | Data Required | Performance |
|--------|-------------------|---------------|-------------|
| Full fine-tuning | All | Large | Best |
| LoRA | Low-rank adapters | Small | Near-best |
| Prompt tuning | Soft prompts | Very small | Good |
| In-context learning | None | None | Moderate |
| Adapter layers | Small adapters | Small | Good |

### 2.4 Active Learning for Science

Optimizing the experiment cycle:

```python
class ActiveLearningLoop:
    """Active learning for molecular discovery"""
    
    def __init__(self, model, oracle, pool, budget=100):
        self.model = model
        self.oracle = oracle  # Experimental validation
        self.pool = pool      # Unlabeled compound library
        self.budget = budget
    
    def run(self):
        for iteration in range(self.budget):
            # 1. Model predicts on pool
            predictions = self.model.predict(self.pool)
            
            # 2. Select most informative molecules
            selected = self.select_by_uncertainty(predictions)
            
            # 3. Get labels (run experiments)
            labels = self.oracle.validate(selected)
            
            # 4. Update model
            self.model.update(selected, labels)
            
            # 5. Remove labeled from pool
            self.pool = self.pool - selected
    
    def select_by_uncertainty(self, predictions):
        """Uncertainty sampling strategy"""
        # Compute uncertainty (e.g., ensemble disagreement)
        uncertainties = self.compute_uncertainty(predictions)
        
        # Select top-k most uncertain
        top_k_indices = uncertainties.argsort()[-10:]
        return self.pool[top_k_indices]
```

---

## 3. Evaluation & Benchmarking

### 3.1 Standard Benchmarks

**Molecular Property Prediction:**

| Benchmark | Tasks | Metrics | SOTA (2026) |
|-----------|-------|---------|-------------|
| MoleculeNet | 8 datasets | ROC-AUC, RMSE | 0.92 avg AUC |
| OGB-MolPCBA | 128 tasks | AP | 0.326 AP |
| Therapeutics Data Commons | 70+ tasks | Various | Task-dependent |
| ADMETlab | 81 datasets | Various | Task-dependent |
| TDC ADMET | 22 datasets | Various | Task-dependent |

**Protein Understanding:**

| Benchmark | Tasks | Metrics | SOTA |
|-----------|-------|---------|------|
| FLIP | 22 mutational landscapes | Spearman ρ | 0.85+ |
| PEER | 53 protein tasks | Task-dependent | Task-dependent |
| ProteinGym | 217 mutational datasets | Spearman ρ | 0.60+ |
| CATH | Fold classification | Accuracy | 85%+ |
| UniProt | Function prediction | Fmax | 0.68+ |

### 3.2 Evaluation Metrics

**For Molecular Generation:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Validity | % chemically valid molecules | >95% |
| Uniqueness | % unique molecules | >95% |
| Novelty | % not in training set | >50% |
| QED | Drug-likeness score | >0.5 |
| SA Score | Synthetic accessibility | <4.0 |
| Diversity | Internal diversity | High |
| FCD | Fréchet ChemNet Distance | Low |
| KL divergence | Distribution match to training | Low |

**For Structure Prediction:**

| Metric | What It Measures | Good Threshold |
|--------|-----------------|----------------|
| GDT-TS | Global distance test | >80 |
| lDDT | Local distance difference test | >0.8 |
| TM-score | Topology match | >0.7 |
| RMSD | Root mean square deviation | <2.0 Å |
| DockQ | Docking quality | >0.5 |
| pLDDT | Confidence per residue | >70 |

### 3.3 Ablation Studies Framework

```python
def ablation_study(model, dataset, components):
    """Systematic ablation study for scientific ML models"""
    results = {}
    
    for component in components:
        # Remove component
        model_ablated = remove_component(model, component)
        
        # Train from scratch
        model_ablated = train(model_ablated, dataset)
        
        # Evaluate
        metrics = evaluate(model_ablated, dataset.test)
        results[component] = metrics
    
    # Baseline (full model)
    results['full_model'] = evaluate(model, dataset.test)
    
    # Compute importance
    for component in components:
        importance = results['full_model'] - results[component]
        print(f"{component}: importance = {importance:.4f}")
    
    return results
```

---

## 4. Data Pipelines & Feature Engineering

### 4.1 Molecular Feature Engineering

**From Raw Data to ML-Ready Features:**

```
Raw Data Sources
├── Compound Libraries (SDF, MOL2, SMILES)
├── Protein Structures (PDB, mmCIF)
├── Bioassays (Activity, Selectivity)
├── Genomic Data (VCF, BED, BigWig)
└── Literature (PDFs, Text)

Feature Engineering Pipeline
├── Molecular Features
│   ├── Fingerprints (Morgan, MACCS)
│   ├── Descriptors (RDKit, Mordred)
│   ├── Graph representations
│   └── 3D coordinates + conformers
├── Protein Features
│   ├── Sequence embeddings (ESM, ProtTrans)
│   ├── Structural features (distance matrices, angles)
│   ├── Evolutionary features (MSA, pSSM)
│   └── Functional annotations (GO terms)
├── Interaction Features
│   ├── Binding site features
│   ├── Docking scores
│   └── Pharmacophore features
└── Contextual Features
    ├── Experimental conditions
    ├── Cell type / tissue
    └── Time point
```

### 4.2 Data Augmentation for Scientific Data

```python
class MolecularAugmentation:
    """Data augmentation strategies for molecules"""
    
    @staticmethod
    def atom_dropout(mol, p=0.1):
        """Randomly remove atoms"""
        atoms = mol.GetAtoms()
        mask = torch.bernoulli(torch.full((len(atoms),), 1-p))
        return mol_from_subset(mol, mask)
    
    @staticmethod
    def bond_perturbation(mol, p=0.1):
        """Randomly change bond types"""
        for bond in mol.GetBonds():
            if random.random() < p:
                bond_type = random.choice([SINGLE, DOUBLE, TRIPLE])
                bond.SetBondType(bond_type)
        return mol
    
    @staticmethod
    def conformer_ensemble(mol, n_conformers=10):
        """Generate multiple 3D conformers"""
        conformers = []
        for _ in range(n_conformers):
            conf = EmbedMolecule(mol, randomSeed=random.randint(0, 1000))
            conformers.append(conf)
        return conformers
    
    @staticmethod
    def SMILES_canonicalization(smiles):
        """Canonical SMILES with random augmentations"""
        mol = Chem.MolFromSmiles(smiles)
        # Different canonical forms
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
```

### 4.3 Data Quality Control

```python
class ScientificDataQC:
    """Quality control for scientific ML datasets"""
    
    def __init__(self):
        self.checks = [
            self.check_chemical_validity,
            self.check_property_ranges,
            self.check_duplicates,
            self.check_label_quality,
            self.check_distribution_shift,
        ]
    
    def run_qc(self, dataset):
        report = {}
        for check in self.checks:
            report[check.__name__] = check(dataset)
        return report
    
    def check_chemical_validity(self, dataset):
        valid = sum(1 for smi in dataset.smiles if Chem.MolFromSmiles(smi))
        return {'valid': valid, 'total': len(dataset), 'ratio': valid/len(dataset)}
    
    def check_label_quality(self, dataset):
        """Check for suspicious label patterns"""
        # High variance in repeated measurements
        # Systematic biases across plates/batches
        # Known problematic compounds
        pass
    
    def check_distribution_shift(self, train, test):
        """Check for train-test distribution shift"""
        from scipy.stats import ks_2samp
        train_props = compute_molecular_descriptors(train)
        test_props = compute_molecular_descriptors(test)
        
        shifts = {}
        for prop in train_props:
            stat, pval = ks_2samp(train_props[prop], test_props[prop])
            shifts[prop] = {'statistic': stat, 'pvalue': pval}
        return shifts
```

---

## 5. Uncertainty Quantification

### 5.1 Why Uncertainty Matters in Science

In scientific applications, knowing when the model is uncertain is critical:

- **Drug discovery**: Avoid predicting high confidence for unexplored chemical space
- **Clinical decisions**: Flag uncertain predictions for human review
- **Materials design**: Guide exploration toward regions of high uncertainty

### 5.2 Methods

**Ensemble Uncertainty:**

```python
class EnsembleModel:
    """Deep ensemble for uncertainty estimation"""
    
    def __init__(self, model_class, n_ensemble=5):
        self.models = [model_class() for _ in range(n_ensemble)]
    
    def predict_with_uncertainty(self, x):
        predictions = torch.stack([model(x) for model in self.models])
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = variance.mean(dim=-1)
        
        return mean, epistemic
    
    def select_for_labeling(self, pool, budget):
        """Select most uncertain molecules for experimental validation"""
        mean, uncertainty = self.predict_with_uncertainty(pool)
        top_k = uncertainty.argsort()[-budget:]
        return pool[top_k]
```

**Monte Carlo Dropout:**

```python
class MCDropoutModel(nn.Module):
    """Model with MC Dropout for uncertainty estimation"""
    
    def __init__(self, base_model, n_samples=20):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
    
    def forward(self, x):
        # Enable dropout during inference
        self.base_model.train()
        
        predictions = torch.stack([
            self.base_model(x) for _ in range(self.n_samples)
        ])
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
```

**Bayesian Neural Networks:**

```python
import bayesian_optimization as bo

class BayesianMolecularModel:
    """Bayesian approach to molecular property prediction"""
    
    def __init__(self):
        self.kernel = bo.MaternKernel()
        self.gp = bo.GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10
        )
    
    def fit(self, X_train, y_train):
        self.gp.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.gp.predict(X_test, return_std=True)
    
    def acquisition(self, X_pool, strategy='ucb'):
        """Select next molecule to evaluate"""
        mean, std = self.predict(X_pool)
        
        if strategy == 'ucb':
            # Upper confidence bound
            acquisition = mean + 2 * std
        elif strategy == 'ei':
            # Expected improvement
            best = self.gp.y_train_.max()
            z = (mean - best) / std
            acquisition = std * (z * norm.cdf(z) + norm.pdf(z))
        
        return acquisition.argmax()
```

### 5.3 Calibration

Ensuring model confidence matches actual accuracy:

```python
def calibration_analysis(model, test_set, n_bins=10):
    """Analyze model calibration"""
    predictions = model.predict(test_set)
    
    # Bin by confidence
    confidences = predictions.confidence
    accuracies = predictions.accuracy
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(accuracies[mask].mean())
            bin_confidences.append(confidences[mask].mean())
    
    # Expected Calibration Error
    ece = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    
    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences
    }
```

---

## 6. Multi-Modal Scientific Learning

### 6.1 Aligning Modalities

Scientific data comes in multiple modalities:

- **Structure** (3D coordinates)
- **Sequence** (SMILES, amino acids, DNA)
- **Spectra** (NMR, IR, mass spec)
- **Images** (microscopy, X-ray)
- **Text** (papers, protocols)
- **Properties** (experimental measurements)

**Multi-Modal Architecture:**

```python
class MultiModalScientificModel(nn.Module):
    """Align multiple scientific modalities"""
    
    def __init__(self, d_model=512):
        super().__init__()
        
        # Modality-specific encoders
        self.structure_encoder = StructureEncoder(d_model)
        self.sequence_encoder = SequenceEncoder(d_model)
        self.spectra_encoder = SpectraEncoder(d_model)
        self.text_encoder = TextEncoder(d_model)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8)
        
        # Shared projection space
        self.shared_projection = nn.Linear(d_model, d_model)
    
    def forward(self, structure=None, sequence=None, spectra=None, text=None):
        representations = []
        
        if structure is not None:
            representations.append(self.structure_encoder(structure))
        if sequence is not None:
            representations.append(self.sequence_encoder(sequence))
        if spectra is not None:
            representations.append(self.spectra_encoder(spectra))
        if text is not None:
            representations.append(self.text_encoder(text))
        
        # Stack and apply cross-modal attention
        stacked = torch.stack(representations, dim=0)
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Average over modalities
        return self.shared_projection(attended.mean(dim=0))
```

### 6.2 Vision-Language Models for Science

```python
class ScientificVLM(nn.Module):
    """Vision-Language model for scientific images"""
    
    def __init__(self, vision_model, language_model, projection_dim=512):
        super().__init__()
        self.vision = vision_model  # e.g., ViT
        self.language = language_model  # e.g., LLaMA
        self.vision_projection = nn.Linear(vision_model.d_model, projection_dim)
        self.language_projection = nn.Linear(language_model.d_model, projection_dim)
    
    def forward(self, images, text):
        # Encode image
        vision_features = self.vision(images)
        vision_projected = self.vision_projection(vision_features.mean(dim=1))
        
        # Encode text
        language_features = self.language(text)
        language_projected = self.language_projection(language_features.mean(dim=1))
        
        # Contrastive alignment
        vision_projected = F.normalize(vision_projected, dim=1)
        language_projected = F.normalize(language_projected, dim=1)
        
        return vision_projected, language_projected
```

---

## 7. Active Learning & Closed-Loop Discovery

### 7.1 Self-Driving Labs

The ultimate vision: AI designs experiments, robots execute them, AI analyzes results, repeat.

```
┌─────────────────────────────────────────────────────┐
│                    Self-Driving Lab                   │
│                                                      │
│  [AI Model] ──proposes──→ [Lab Automation]           │
│       ↑                        │                     │
│       │                        ↓                     │
│  [Data Analysis] ←──measures── [Experiment]          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Real-World Examples (2026):**
- **Emerald Cloud Lab**: Full cloud lab with AI-guided experiments
- **Strateos**: Automated drug discovery platform
- **Kebotix**: AI-guided materials synthesis
- **A-Lab (LBNL)**: Autonomous materials synthesis robot

### 7.2 Bayesian Optimization for Experiment Design

```python
class BayesianExperimentDesigner:
    """Bayesian optimization for experimental design"""
    
    def __init__(self, objective_fn, search_space):
        self.objective = objective_fn
        self.search_space = search_space
        self.gp = GaussianProcessRegressor()
        self.history = []
    
    def suggest_next_experiment(self, n_suggestions=1):
        """Suggest the most informative next experiment"""
        if len(self.history) < 5:
            # Random exploration
            return self.search_space.sample(n_suggestions)
        
        # Fit GP to history
        X = np.array([h['params'] for h in self.history])
        y = np.array([h['result'] for h in self.history])
        self.gp.fit(X, y)
        
        # Optimize acquisition function
        def acquisition(x):
            mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return mean + 2 * std  # UCB
        
        candidates = []
        for _ in range(n_suggestions):
            result = differential_evolution(
                lambda x: -acquisition(x),
                bounds=self.search_space.bounds
            )
            candidates.append(result.x)
        
        return candidates
    
    def update(self, params, result):
        """Record experiment result"""
        self.history.append({'params': params, 'result': result})
```

### 7.3 Reinforcement Learning for Experimental Design

```python
class ExperimentalDesignAgent:
    """RL agent for experimental design"""
    
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
    
    def select_experiment(self, state):
        """Select next experiment based on current knowledge"""
        action_probs = self.policy(state)
        action = torch.multinomial(action_probs, 1)
        return action
    
    def update(self, trajectory):
        """Update policy based on experiment outcomes"""
        states, actions, rewards = trajectory
        
        # Compute advantages
        values = self.value(states)
        advantages = rewards - values
        
        # Policy gradient
        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus for exploration
        entropy = self.policy.entropy(states).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        return loss
```

---

## 8. Production Deployment

### 8.1 Model Serving for Scientific Applications

```python
# FastAPI-based model server for molecular predictions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI(title="Molecular Property Prediction API")

class MoleculeRequest(BaseModel):
    smiles: str
    task: str = "admet"
    confidence_threshold: float = 0.8

class PredictionResponse(BaseModel):
    smiles: str
    prediction: float
    confidence: float
    uncertainty: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MoleculeRequest):
    try:
        # Validate SMILES
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        
        # Get prediction with uncertainty
        model = load_model(request.task)
        prediction, uncertainty = model.predict_with_uncertainty(
            smiles_to_tensor(request.smiles)
        )
        
        # Check confidence threshold
        if uncertainty > (1 - request.confidence_threshold):
            return PredictionResponse(
                smiles=request.smiles,
                prediction=prediction.item(),
                confidence=1 - uncertainty.item(),
                uncertainty=uncertainty.item(),
                model_version=model.version
            )
        
        return PredictionResponse(
            smiles=request.smiles,
            prediction=prediction.item(),
            confidence=1 - uncertainty.item(),
            uncertainty=uncertainty.item(),
            model_version=model.version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 8.2 Model Monitoring

```python
class ScientificModelMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, model, reference_data):
        self.model = model
        self.reference_data = reference_data
        self.predictions = []
        self.alerts = []
    
    def check_prediction_distribution(self, new_predictions):
        """Check if new predictions drift from training distribution"""
        from scipy.stats import ks_2samp
        
        ref_preds = self.model.predict(self.reference_data)
        stat, pval = ks_2samp(ref_preds, new_predictions)
        
        if pval < 0.05:
            self.alerts.append({
                'type': 'distribution_drift',
                'pvalue': pval,
                'timestamp': datetime.now()
            })
    
    def check_confidence_calibration(self, predictions, true_labels):
        """Monitor if model confidence is well-calibrated"""
        ece = compute_ece(predictions, true_labels)
        if ece > 0.1:
            self.alerts.append({
                'type': 'miscalibration',
                'ece': ece,
                'timestamp': datetime.now()
            })
    
    def generate_report(self):
        """Generate monitoring report"""
        return {
            'n_predictions': len(self.predictions),
            'n_alerts': len(self.alerts),
            'alert_types': [a['type'] for a in self.alerts],
            'recent_alerts': self.alerts[-10:]
        }
```

### 8.3 CI/CD for Scientific Models

```yaml
# .github/workflows/science-model-ci.yml
name: Scientific Model CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install rdkit-pypi
      
      - name: Run unit tests
        run: pytest tests/unit/
      
      - name: Run integration tests
        run: pytest tests/integration/
      
      - name: Run benchmark tests
        run: pytest tests/benchmarks/ --benchmark
      
      - name: Check model performance
        run: |
          python scripts/evaluate_model.py \
            --model models/latest.pt \
            --test-data data/test/ \
            --thresholds config/thresholds.yaml
      
      - name: Check for data leakage
        run: python scripts/check_data_leakage.py
      
      - name: Deploy to staging
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/deploy.py --env staging
```

---

## 9. Case Study: End-to-End Drug Discovery Pipeline

### 9.1 Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-Driven Drug Discovery Pipeline             │
│                                                                 │
│  Phase 1: Target Discovery                                      │
│  ├── Genomic analysis (EVO, scGPT)                             │
│  ├── Literature mining (PubMedBERT)                             │
│  └── Causal inference (CausalNex)                               │
│                                                                 │
│  Phase 2: Hit Discovery                                         │
│  ├── Virtual screening (GNN + Transformer)                     │
│  ├── Fragment-based design (RFdiffusion)                       │
│  └── DNA-encoded library (DEL) analysis                        │
│                                                                 │
│  Phase 3: Lead Optimization                                     │
│  ├── Multi-parameter optimization (Bayesian)                   │
│  ├── ADMET prediction (multi-task)                              │
│  ├── De novo design (diffusion models)                          │
│  └── Retrosynthetic analysis (ASKCOS)                          │
│                                                                 │
│  Phase 4: Preclinical                                           │
│  ├── Toxicity prediction (multi-task)                           │
│  ├── PK/PD modeling (mechanistic + ML)                         │
│  ├── Biomarker identification                                   │
│  └── Formulation optimization                                   │
│                                                                 │
│  Phase 5: Clinical Trials                                       │
│  ├── Patient stratification (foundation models)                 │
│  ├── Endpoint prediction                                        │
│  ├── Adaptive trial design                                      │
│  └── Safety monitoring                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Example: Insilico Medicine's Pipeline

**ISM001-055 (Idiopathic Pulmonary Fibrosis):**
- **Target**: TNIK (Traf2- and NCK-interacting kinase)
- **AI discovery time**: 18 months (target to IND)
- **Traditional timeline**: 4–5 years
- **Phase**: Approved by FDA (2025)

**Key Technologies Used:**
- Pharma.AI (target discovery + molecule generation)
- Chemistry42 (generative chemistry)
- InClinico (clinical trial prediction)

### 9.3 Cost-Benefit Analysis

| Metric | Traditional | AI-Augmented | Savings |
|--------|-------------|--------------|---------|
| Target-to-IND | 4–5 years | 1–2 years | 60–75% |
| Cost to IND | $200–500M | $50–150M | 60–70% |
| Failure rate (Phase I) | 60–70% | 30–40% | 50% |
| Failure rate (Phase II) | 70–80% | 40–50% | 40% |
| Overall success rate | 5–10% | 15–25% | 2–3x |

---

*This document is part of the AI Base Knowledge Library. For the complete library index, see the root README.md.*
