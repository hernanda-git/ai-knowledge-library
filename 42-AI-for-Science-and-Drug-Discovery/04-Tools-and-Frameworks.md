# 04 — Tools and Frameworks for AI in Science & Drug Discovery

> **Category**: 42-AI-for-Science-and-Drug-Discovery  
> **Focus**: Libraries, platforms, cloud services, and open-source tools  
> **Cross-References**: [02-Core-Topics.md](./02-Core-Topics.md), [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md), [08-Reference/](../08-Reference/)

---

## Table of Contents

1. [Molecular Modeling Libraries](#1-molecular-modeling-libraries)
2. [Deep Learning for Science](#2-deep-learning-for-science)
3. [Protein Science Tools](#3-protein-science-tools)
4. [Drug Discovery Platforms](#4-drug-discovery-platforms)
5. [Materials Science Tools](#5-materials-science-tools)
6. [Genomics & Bioinformatics](#6-genomics--bioinformatics)
7. [Chemical Informatics](#7-chemical-informatics)
8. [Cloud Platforms & Services](#8-cloud-platforms--services)
9. [Visualization & Analysis](#9-visualization--analysis)
10. [Comparison Tables](#10-comparison-tables)

---

## 1. Molecular Modeling Libraries

### 1.1 RDKit

The de facto standard for cheminformatics:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw

# Basic molecular operations
mol = Chem.MolFromSmiles('c1ccc(CC(=O)O)cc1')  # Ibuprofen
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)  # 3D conformer

# Compute descriptors
descriptors = {
    'MW': Descriptors.MolWt(mol),
    'LogP': Descriptors.MolLogP(mol),
    'HBD': Descriptors.NumHDonors(mol),
    'HBA': Descriptors.NumHAcceptors(mol),
    'TPSA': Descriptors.TPSA(mol),
    'RotBonds': Descriptors.NumRotatableBonds(mol),
    'QED': Descriptors.qed(mol),
}

# Molecular fingerprints
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Substructure search
query = Chem.MolFromSmarts('c1ccc(-c2ccccc2)cc1')  # biphenyl
matches = mol.GetSubstructMatches(query)
```

**Key Features:**
- 2D/3D molecular manipulation
- 200+ molecular descriptors
- Fingerprint generation (Morgan, MACCS, RDKit)
- Substructure search and enumeration
- Reaction handling
- Conformer generation
- PDB file I/O

### 1.2 DeepChem

Deep learning for chemistry:

```python
import deepchem as dc

# Load MoleculeNet dataset
tasks, datasets, transformers = dc.molnet.load_delaney()
train, valid, test = datasets

# Build a graph convolution model
model = dc.models.GraphConvModel(
    n_tasks=1,
    mode='regression',
    dropout=0.2,
    batch_normalize=True
)

# Train
model.fit(train, nb_epoch=50)

# Evaluate
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print(model.evaluate(test, [metric]))

# Use for prediction
preds = model.predict(test)
```

**Key Features:**
- 40+ datasets (MoleculeNet, Tox21, etc.)
- GNN models (GCN, GAT, MPNN, AttentiveFP)
- Transformer-based models
- Uncertainty quantification
- Featurization (ECFP, GraphConv, Weave)
- Splitters (scaffold, random, butina)
- Hyperparameter optimization

### 1.3 TorchDrug / TorchCells

PyTorch-based drug discovery:

```python
import torchdrug
from torchdrug import models, tasks, datasets

# Load dataset
dataset = datasets.BACE("data/")

# Build model
model = models.GIN(
    num_atom_type=dataset.num_atom_type,
    num_bond_type=dataset.num_bond_type,
    hidden_dim=256,
    num_layer=6,
    dropout=0.2
)

# Create task
task = tasks.PropertyPrediction(
    model,
    task=dataset.tasks,
    criterion="binary_cross_entropy",
    num_epoch=100
)

# Train and evaluate
task.fit(dataset.train, dataset.valid)
results = task.evaluate(dataset.test)
```

### 1.4 PyG (PyTorch Geometric)

Graph neural networks for molecular data:

```python
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, global_add_pool

# Load molecular dataset
dataset = MoleculeNet(root='data/', name='ESOL')

class MolecularGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x
```

### 1.5 OGB (Open Graph Benchmark)

Standard benchmarks for graph ML:

```python
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# Load dataset
dataset = PygGraphPropPredDataset(root='data/', name='ogbg-molhiv')

# Standard splits
split_idx = dataset.get_idx_split()
train_loader = DataLoader(
    dataset[split_idx['train']], batch_size=32, shuffle=True
)
valid_loader = DataLoader(
    dataset[split_idx['valid']], batch_size=32, shuffle=False
)
test_loader = DataLoader(
    dataset[split_idx['test']], batch_size=32, shuffle=False
)

# Evaluate with OGB evaluator
evaluator = Evaluator(name='ogbg-molhiv')
pred = model(data)
train_result = evaluator.eval({
    'y_true': train_targets,
    'y_pred': pred
})
```

---

## 2. Deep Learning for Science

### 2.1 PyTorch

Primary framework for scientific ML:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Scientific model with uncertainty
class ScientificModelWithUncertainty(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_outputs=1, n_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_outputs)
            ) for _ in range(n_models)
        ])
    
    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models])
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std

# Training loop
model = ScientificModelWithUncertainty(input_dim=2048)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        x, y = batch
        mean, std = model(x)
        
        # Heteroscedastic loss
        loss = (0.5 * torch.log(std**2) + 0.5 * (y - mean)**2 / std**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
```

### 2.2 TensorFlow / Keras

Alternative framework with strong deployment:

```python
import tensorflow as tf
from tensorflow import keras

# Molecular property prediction model
def build_molecular_model(vocab_size, max_len, embed_dim=128, hidden_dim=256):
    inputs = keras.Input(shape=(max_len,))
    
    # Embedding + positional encoding
    x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    positions = tf.range(start=0, limit=max_len, delta=1)
    x = x + keras.layers.Embedding(max_len, embed_dim)(positions)
    
    # Transformer blocks
    for _ in range(6):
        x = transformer_block(x, hidden_dim, num_heads=8)
    
    # Global pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Output head
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)
```

### 2.3 JAX / Haiku / Flax

High-performance scientific computing:

```python
import jax
import jax.numpy as jnp
import haiku as hk

def molecular_transformer(x, is_training=True):
    """JAX-based molecular transformer"""
    # Embedding
    x = hk.Embed(vocab_size=100, embed_dim=256)(x)
    
    # Transformer layers
    for _ in range(6):
        # Multi-head attention
        attn = hk.MultiHeadAttention(
            num_heads=8, key_size=32, value_size=32
        )(x, x, x)
        x = x + attn
        x = hk.LayerNorm(axis=-1)(x)
        
        # Feed-forward
        ff = hk.Sequential([
            hk.Linear(1024),
            jax.nn.gelu,
            hk.Linear(256),
        ])
        x = x + ff(x)
        x = hk.LayerNorm(axis=-1)(x)
    
    return x.mean(axis=1)  # Global average pooling

# JIT compile for speed
model = hk.without_apply_rng(hk.transform(molecular_transformer))
jax_fn = jax.jit(model.apply)
```

### 2.4 e3nn (Equivariant Neural Networks)

Equivariant networks for 3D molecular data:

```python
import e3nn
from e3nn import o3

class EquivariantMolecularNetwork(hk.Module):
    """Equivariant network for 3D molecular data"""
    
    def __init__(self, irreps_in, irreps_hidden, irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out
        
        # Equivariant layers
        self.layers = []
        for i in range(4):
            irreps_mid = irreps_hidden if i < 3 else irreps_out
            self.layers.append(
                o3.Convolution(self.irreps_in, irreps_mid, irreps_mid)
            )
    
    def __call__(self, node_features, edge_src, edge_dst, edge_attr):
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_src, edge_dst, edge_attr)
        return x
```

### 2.5 DGL (Deep Graph Library)

Alternative graph library:

```python
import dgl
import dgl.nn as dglnn
import torch

class MolecularGNN_DGL(torch.nn.Module):
    """GNN for molecular property prediction using DGL"""
    
    def __init__(self, in_feats, hidden_feats=128, n_classes=1):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv3 = dglnn.GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        self.readout = dglnn.AvgPooling()
        self.classify = torch.nn.Linear(hidden_feats, n_classes)
    
    def forward(self, g):
        h = g.ndata['feat']
        h = torch.relu(self.conv1(g, h))
        h = torch.relu(self.conv2(g, h))
        h = torch.relu(self.conv3(g, h))
        
        with g.local_scope():
            g.ndata['h'] = h
            hg = self.readout(g)
        
        return self.classify(hg)
```

---

## 3. Protein Science Tools

### 3.1 AlphaFold / OpenFold

```bash
# AlphaFold3 inference
python run_alphafold.py \
    --fasta_path=proteins.fasta \
    --output_dir=output/ \
    --model_preset=multimer \
    --db_preset=full_dbs

# OpenFold (open-source alternative)
python scripts/inference.py \
    --fasta_path=proteins.fasta \
    --output_dir=output/ \
    --config_preset=alphafold2
```

**Key Differences:**

| Feature | AlphaFold3 | OpenFold |
|---------|-----------|---------|
| License | Non-commercial | Apache 2.0 |
| Biomolecular complexes | Full support | Limited |
| GPU memory | ~40GB | ~24GB |
| Speed | Minutes | Minutes |
| Accuracy | State-of-the-art | Near state-of-the-art |

### 3.2 RFdiffusion / ProteinMPNN

```python
# RFdiffusion for de novo protein design
from rfdiffusion import run_inference

# Design a protein binder for a target
config = {
    'target_pdb': 'target.pdb',
    'target_chain': 'A',
    'contig_residues': '50-100',
    'output_prefix': 'design',
    'num_designs': 100,
    'denoising_steps': 100,
}

run_inference(config)

# ProteinMPNN for sequence design
from protein_mpnn_run import run_protein_mpnn

# Design sequence for a backbone
mpnn_config = {
    'pdb_path': 'design_backbone.pdb',
    'chain_id': 'A',
    'fixed_residues': [],
    'temperature': 0.1,
    'num_seq_per_target': 8,
}

sequences = run_protein_mpnn(mpnn_config)
```

### 3.3 ESM (Evolutionary Scale Modeling)

```python
import torch
from esm.pretrained import esm2_t33_650M_UR50D

# Load ESM-2 model
model, alphabet = esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Predict structure from sequence
data = [
    ("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"),
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-token embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

token_representations = results["representations"][33]

# Predict contacts
contacts = results["contacts"]
```

### 3.4 PyRosetta

Python interface to Rosetta:

```python
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.protocols.relax import FastRelax

# Initialize
pyrosetta.init()

# Load pose
pose = Pose()
pose.assign_from_sequence("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV")

# Energy evaluation
scorefxn = pyrosetta.create_score_function('ref2015')
score = scorefxn(pose)

# Structure optimization
fast_relax = FastRelax()
fast_relax.set_scorefxn(scorefxn)
fast_relax.apply(pose)

# Design
from pyrosetta.rosetta.protocols.packprot import PackRotamersMover
packer = PackRotamersMover()
packer.apply(pose)
```

### 3.5 PyMOL

Visualization of molecular structures:

```python
# PyMOL scripting
from pymol import cmd

# Load structure
cmd.load("protein.pdb")

# Color by chain
cmd.color("chain A", "chain A")
cmd.color("chain B", "chain B")

# Surface
cmd.set("surface_mode", 1)
cmd.show("surface", "chain A")

# Export high-res image
cmd.png("output.png", width=3000, height=2000, dpi=300)
```

---

## 4. Drug Discovery Platforms

### 4.1 Commercial Platforms

| Platform | Company | Features | Pricing |
|----------|---------|----------|---------|
| Schrödinger Suite | Schrödinger | Molecular modeling, FEP+ | $50k–500k/yr |
| MOE | CCG | Drug design suite | $30k–200k/yr |
| KNIME Analytics | KNIME | Workflow automation | Free/Enterprise |
| Pipeline Pilot | BIOVIA | Enterprise informatics | Enterprise |
| Spotfire | TIBCO | Data visualization | $10k–100k/yr |

### 4.2 Open-Source Drug Discovery

**Open Source Malaria:**
- Fully open drug discovery project
- GitHub-based data sharing
- Community-driven compound optimization

**Open Source COVID Drugs:**
- COVID Moonshot: Crowdsourced antiviral design
- 200k+ compounds evaluated
- Multiple leads in development

**Therapeutics Data Commons (TDC):**
```python
from tdc.single_pred import ADME
from tdc.single_pred import Tox

# Load ADMET datasets
data = ADME(name='Caco2_Wang')
split = data.get_split()

# Load toxicity datasets
data = Tox(name='LD50_Zhu')
split = data.get_split()
```

### 4.3 AI Drug Discovery Pipelines

| Company | Pipeline Stage | AI Methods | Notable Results |
|---------|---------------|------------|-----------------|
| Insilico Medicine | End-to-end | Generative AI, target ID | ISM001-055 approved |
| Recursion | Phenomics | Computer vision, GNN | 3 Phase II trials |
| Isomorphic Labs | Structure-based | AlphaFold3, diffusion | 2 Phase I trials |
| Exscientia | Lead optimization | Centaur Chemist | First AI drug in Phase II |
| BenevolentAI | Knowledge graphs | NLP, graph learning | AIDDISCOVER platform |

---

## 5. Materials Science Tools

### 5.1 Materials Project API

```python
from mp_api.client import MPRester

# Query Materials Project database
with MPRester("YOUR_API_KEY") as mpr:
    # Search for materials
    docs = mpr.materials.summary.search(
        band_gap=[0.5, 1.5],
        is_stable=True,
        elements=["Si", "O"],
        num_elements=2,
        fields=["material_id", "formula_pretty", "band_gap"]
    )
    
    # Get elastic properties
    elastic = mpr.materials.elasticity.get_data_by_id("mp-149")
    
    # Get phonon data
    phonon = mpr.materials.phonon.get_data_by_id("mp-149")
```

### 5.2 ASE (Atomic Simulation Environment)

```python
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.build import bulk

# Create a crystal
si = bulk('Si', 'diamond', a=5.43)

# Set calculator
si.calc = EMT()

# Optimize geometry
dyn = BFGS(si)
dyn.run(fmax=0.01)

# Get energy and forces
energy = si.get_potential_energy()
forces = si.get_forces()
```

### 5.3 Pymatgen

Python Materials Genomics:

```python
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester

# Create structure
lattice = Lattice.cubic(5.43)
structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# Query Materials Project
with MPRester("API_KEY") as mpr:
    entries = mpr.get_entries_in_chemical_system(["Si", "O"])
    pd = PhaseDiagram(entries)
    
    # Get decomposition
    decomp = pd.get_decomposition(structure.composition)
    
    # Get phase diagram data
    form_energy = pd.get_form_energy_per_atom(entries[0])
```

### 5.4 ASE + ML Potentials

```python
from ase.calculators.mlp import MLP

# Use a pre-trained ML potential
calc = MLP(model="mace-mpa-0")  # MACE foundation model

# Or use NequIP/Allegro
from ase.calculators.nequip import NequIP
calc = NequIP(model="path/to/model.pth")

# Run MD simulation
from ase.md.langevin import Langevin
from ase import units

dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.002
)

dyn.run(1000)  # 1 ps
```

---

## 6. Genomics & Bioinformatics

### 6.1 Single-Cell Analysis

```python
import scanpy as sc
import scvi

# Load single-cell data
adata = sc.read_10x_mtx('data/')

# Preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# scVI for batch correction
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model = scvi.model.SCVI(adata)
model.train()

# Get latent representation
adata.obsm['X_scVI'] = model.get_latent_representation()

# Clustering and visualization
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden', 'batch'])
```

### 6.2 Variant Calling with Deep Learning

```python
import torch
from dvCaller import DeepVariantModel

# Load model
model = DeepVariantModel.from_pretrained("dvCaller-large")

# Call variants from pileup images
pileups = load_pileup_images("sample.bam", "reference.fa")
predictions = model.predict(pileups)

# Convert to VCF
write_vcf(predictions, "output.vcf")
```

### 6.3 Gene Expression Prediction

```python
from enformer import Enformer

# Load Enformer model
model = Enformer.from_pretrained("enformer-base")

# Predict gene expression from sequence
sequence = "ACGTACGT..."  # DNA sequence
predictions = model.predict(sequence)

# Predict variant effects
reference_expr = model.predict(reference_sequence)
alt_expr = model.predict(alt_sequence)
delta = alt_expr - reference_expr
```

---

## 7. Chemical Informatics

### 7.1 Mordred (Descriptor Calculation)

```python
from mordred import Calculator, descriptors
from rdkit import Chem

calc = Calculator(descriptors, ignore_3D=False)

# Calculate all descriptors
mol = Chem.MolFromSmiles("c1ccc(CC(=O)O)cc1")
descriptors = calc(mol)

# Convert to pandas DataFrame
df = descriptors.pandas()
```

### 7.2 Chemical Space Analysis

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def chemical_space_analysis(smiles_list, method='umap'):
    """Visualize chemical space"""
    # Compute fingerprints
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(np.array(fp))
    
    fps = np.array(fps)
    
    # Dimensionality reduction
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embedding = reducer.fit_transform(fps)
    return embedding
```

### 7.3 Retrosynthesis Planning

```python
from rxn4chemistry import RXN4Chemistry

# Initialize RXN
rxn = RXN4Chemistry(api_key="YOUR_API_KEY")

# Predict retrosynthesis
target = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
results = rxn.predict_retrosynthesis(target)

# Get synthetic routes
for route in results['retrosynthesis']:
    print(f"Reaction: {route['reaction']}")
    print(f"Confidence: {route['confidence']}")
    print(f"SMILES: {route['smiles']}")
    print()
```

---

## 8. Cloud Platforms & Services

### 8.1 AWS for Science

| Service | Use Case | Features |
|---------|----------|----------|
| SageMaker | Model training | Managed ML, distributed training |
| HealthOmics | Genomics | Variant calling, RNA-seq |
| ParallelCluster | HPC | Compute clusters |
| S3 + Lake Formation | Data lake | Petabyte-scale data |
| Batch | Compute | Spot instances, containerized jobs |
| Deep Learning AMIs | Pre-configured | PyTorch, TensorFlow, JAX |

### 8.2 Google Cloud for Science

| Service | Use Case | Features |
|---------|----------|----------|
| Vertex AI | ML platform | AutoML, custom training |
| Life Sciences API | Genomics | WGS, RNA-seq pipelines |
| BigQuery ML | Data analysis | SQL-based ML |
| TPU | Accelerators | Cost-effective training |
| Cloud Life Sciences | Pipelines | Workflow orchestration |
| Deep Learning VMs | Compute | Pre-configured instances |

### 8.3 NVIDIA NGC

```bash
# Pull pre-built containers
docker pull nvcr.io/nvidia/clara/bionemo:latest

# Run inference
docker run --gpus all -v /data:/data \
    nvcr.io/nvidia/clara/bionemo:latest \
    python inference.py --model esm2 --input /data/proteins.fasta

# Access pre-trained models
ngc registry model list --org nvidia --team clara
```

**BioNeMo Models:**
- ESM-2 (protein language)
- AlphaFold2 (structure prediction)
- OpenFold (open-source AF)
- ProtGPT2 (protein generation)
- MoLeR (molecular generation)

### 8.4 Microsoft Azure for Science

| Service | Use Case | Features |
|---------|----------|----------|
| Azure ML | ML platform | Managed training |
| Azure Genomics | Genomics | WGS, variant calling |
| Azure HPC | Compute | HPC clusters |
| Azure OpenAI | LLM access | GPT-4, Codex |
| Azure Quantum | Quantum | Hybrid quantum-classical |

---

## 9. Visualization & Analysis

### 9.1 RDKit Visualization

```python
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Draw molecules
mol1 = Chem.MolFromSmiles('c1ccc(CC(=O)O)cc1')
mol2 = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')

# 2D grid
img = Draw.MolsToGridImage(
    [mol1, mol2],
    molsPerRow=2,
    subImgSize=(300, 300),
    legends=['Benzoic acid', 'Aspirin']
)
img.save('molecules.png')
```

### 9.2 Plotly for Interactive Visualizations

```python
import plotly.graph_objects as go
import plotly.express as px

def plot_chemical_space(df, color_by='activity'):
    """Interactive chemical space plot"""
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color=color_by,
        hover_data=['SMILES', 'name'],
        title='Chemical Space',
        opacity=0.7
    )
    fig.update_layout(
        template='plotly_white',
        width=800,
        height=600
    )
    return fig
```

### 9.3 NGLView (3D Molecular Visualization)

```python
import nglview as nv

# View protein structure
view = nv.show_pdbid("1crn")
view.add_ball_and_stick()
view.add_surface(opacity=0.3)

# View from specific angle
view.view.camera = 'perspective'
view.view.orientation = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

### 9.4 Matplotlib/Seaborn for Scientific Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction_vs_actual(y_true, y_pred, title=""):
    """Publication-quality scatter plot"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect prediction')
    
    # R² score
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'{title}\nR² = {r2:.3f}, RMSE = {rmse:.3f}', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    return fig
```

---

## 10. Comparison Tables

### 10.1 GNN Frameworks Comparison

| Framework | Backend | Speed | Ease of Use | Community | Best For |
|-----------|---------|-------|-------------|-----------|----------|
| PyG | PyTorch | Fast | Good | Large | Research |
| DGL | PyTorch/MXNet | Very Fast | Moderate | Large | Scale |
| DeepChem | PyTorch | Moderate | Excellent | Medium | Drug discovery |
| Spektral | TensorFlow | Moderate | Good | Medium | Quick prototyping |
| Jraph | JAX | Very Fast | Moderate | Small | Performance |

### 10.2 Molecular Generation Methods

| Method | Sample Quality | Speed | Diversity | Novelty | Best For |
|--------|---------------|-------|-----------|---------|----------|
| VAE | Good | Fast | High | Moderate | General use |
| GAN | High | Fast | Low | Moderate | Image-like |
| Flow | High | Moderate | High | Moderate | Likelihood |
| Diffusion | Very High | Slow | Very High | High | State-of-the-art |
| RL | Variable | Slow | Variable | High | Goal-directed |
| LLM | High | Fast | High | High | SMILES-based |

### 10.3 Protein Structure Prediction

| Tool | Speed | Accuracy | License | GPU Memory | Input |
|------|-------|----------|---------|-----------|-------|
| AlphaFold3 | Minutes | SOTA | Non-commercial | ~40GB | Sequence |
| OpenFold | Minutes | Near-SOTA | Apache 2.0 | ~24GB | Sequence |
| ESMFold | Milliseconds | ~95% of AF | Non-commercial | ~16GB | Sequence |
| Chai-1 | Minutes | AF3-competitive | Research | ~32GB | Sequence |
| Boltz-1 | Minutes | Near-SOTA | Apache 2.0 | ~24GB | Sequence |
| RoseTTAFold | Minutes | Good | Non-commercial | ~16GB | Sequence |

### 10.4 Benchmark Performance (Molecular Property)

| Model | MoleculeNet (AUC) | OGB-MolPCBA (AP) | Speed | Parameters |
|-------|-------------------|-------------------|-------|-----------|
| GIN | 0.89 | 0.299 | Fast | 1M |
| GAT | 0.88 | 0.293 | Fast | 2M |
| SchNet | 0.91 | 0.297 | Fast | 3M |
| DimeNet++ | 0.92 | 0.305 | Moderate | 10M |
| AttentiveFP | 0.92 | 0.301 | Fast | 5M |
| Uni-Mol | 0.94 | 0.326 | Slow | 30M |

### 10.5 Cloud Computing Costs for Science

| Instance | GPU | RAM | $/hr | Best For |
|----------|-----|-----|------|----------|
| AWS g4dn.xlarge | T4 | 16GB | $0.73 | Inference |
| AWS p3.2xlarge | V100 | 61GB | $3.82 | Training |
| AWS p4d.24xlarge | 8×A100 | 1.1TB | $37.69 | Large training |
| GCP a2-highgpu-1g | A100 | 85GB | $3.67 | Training |
| Azure NC24ads_A100_v4 | A100 | 88GB | $3.67 | Training |
| Lambda Cloud | A100 | 250GB | $1.10 | Cost-effective |

---

*This document is part of the AI Base Knowledge Library. For the complete library index, see the root README.md.*
