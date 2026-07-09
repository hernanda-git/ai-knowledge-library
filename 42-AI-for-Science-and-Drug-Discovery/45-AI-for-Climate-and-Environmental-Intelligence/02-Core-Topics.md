# 02 - AI for Climate & Environmental Intelligence: Core Topics

> **Category:** 45-AI-for-Climate-and-Environmental-Intelligence
> **Last updated:** July 1, 2026
> **Cross-references:** `03-Agents/01-Agent-Architectures.md`, `04-RAG/01-RAG-Architectures.md`, `06-Advanced/01-Multimodal-AI.md`, `10-Industry/03-AI-for-Robotics.md`

---

## Table of Contents

1. [AI Weather Forecasting: The Core Discipline](#1-ai-weather-forecasting-the-core-discipline)
2. [Climate Modeling with AI](#2-climate-modeling-with-ai)
3. [Environmental Monitoring and Remote Sensing](#3-environmental-monitoring-and-remote-sensing)
4. [Agriculture and Food Security](#4-agriculture-and-food-security)
5. [Biodiversity and Ecosystem Intelligence](#5-biodiversity-and-ecosystem-intelligence)
6. [Ocean and Marine Systems](#6-ocean-and-marine-systems)
7. [Urban Climate Resilience](#7-urban-climate-resilience)
8. [Carbon Accounting and MRV](#8-carbon-accounting-and-mrv)
9. [Climate Finance and Risk Analytics](#9-climate-finance-and-risk-analytics)
10. [Key Technical Patterns](#10-key-technical-patterns)

---

## 1. AI Weather Forecasting: The Core Discipline

### 1.1 The Architecture Revolution

Weather forecasting has undergone a fundamental architectural shift. Traditional Numerical Weather Prediction (NWP) solves discretized forms of the Navier-Stokes equations on a grid. AI models learn statistical relationships from historical data and reanalysis products.

**Traditional NWP Pipeline:**
```
Observation -> Data Assimilation -> Model Integration -> Post-Processing -> Forecast Product
  (hours)        (hours)              (hours)             (minutes)       (minutes)
```

**AI Weather Pipeline:**
```
Observation/Data -> Preprocessing -> Neural Network Inference -> Post-Processing -> Forecast Product
  (minutes)          (minutes)         (seconds-minutes)        (minutes)        (minutes)
```

The speed advantage is dramatic: AI models can produce ensemble forecasts (hundreds of members) in the time it takes NWP to produce one deterministic run.

### 1.2 Model Architectures

#### GraphCast-Style (Graph Neural Networks)

GraphCast, developed by DeepMind, treats the atmosphere as a graph:
- **Nodes**: Grid points on a multi-resolution icosahedral mesh
- **Edges**: Connections between nearby nodes
- **Message passing**: Information flows along edges at each step
- **Autoregressive**: Predicts state at T+6h, then uses that as input for T+12h

```python
# Simplified GraphCast-style architecture
import torch
import torch_geometric

class GraphCastModel(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, out_channels=3):
        super().__init__()
        self.encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.processor = torch_geometric.nn.GATConv(
            hidden_channels, hidden_channels, heads=4, concat=False
        )
        self.decoder = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = torch.relu(self.processor(x, edge_index))
        return self.decoder(x)
```

#### Pangu-Weather Style (3D Earth-Specific Transformer)

Huawei's Pangu-Weather uses a 3D transformer that respects Earth's geometry:
- **Vertical attention**: Processes the full atmospheric column at each grid point
- **Horizontal attention**: Captures spatial dependencies across grid points
- **Pressure-level encoding**: Explicitly represents the vertical structure of the atmosphere

#### FourCastNet Style (Adaptive Fourier Neural Operators)

NVIDIA's approach uses Fourier Neural Operators (FNOs):
- Operates in frequency domain for efficient global pattern capture
- Captures multi-scale interactions without explicit hierarchical structure
- Particularly efficient for high-resolution outputs

#### GenCast Style (Diffusion Models)

DeepMind's latest approach applies diffusion models to weather:
- **Stochastic generation**: Produces multiple plausible future states
- **Calibrated uncertainty**: Naturally generates ensemble spreads
- **Generative modeling**: Learns the full distribution of possible weather states

### 1.3 Training Data

AI weather models are primarily trained on reanalysis data:

| Dataset | Resolution | Time Period | Provider |
|---------|-----------|-------------|----------|
| **ERA5** | 0.25 deg (~31km) | 1979-present | ECMWF |
| **ERA5-Land** | 0.1 deg (~9km) | 1979-present | ECMWF |
| **MERRA-2** | 0.5 deg (~50km) | 1980-present | NASA |
| **JRA-55** | 0.5625 deg (~56km) | 1958-present | JMA |

### 1.4 Evaluation Metrics

- **ACC (Anomaly Correlation Coefficient)**: Measures pattern correlation of anomalies. >0.6 is generally considered useful
- **RMSE (Root Mean Square Error)**: Point-wise error. Lower is better.
- **CRPS (Continuous Ranked Probability Score)**: For probabilistic forecasts
- **Spread-skill ratio**: For ensembles, measures whether ensemble spread matches forecast error

---

## 2. Climate Modeling with AI

### 2.1 Emulation vs. Replacement

1. **Climate model emulation**: Train a fast ML model to approximate an expensive climate model
2. **Physics-AI hybrid models**: Replace specific parameterized processes with neural networks

### 2.2 NeuralGCM: The Hybrid Approach

Google's NeuralGCM represents a landmark hybrid model:
- Retains the dynamical core (solving the primitive equations)
- Replaces parameterized processes (convection, radiation, cloud microphysics) with neural networks
- Trained on reanalysis data to produce physically consistent atmospheric states
- Can be run forward for decades, unlike pure ML models that diverge

### 2.3 Downscaling

| Method | Description | Resolution Gain |
|--------|-------------|----------------|
| **Statistical downscaling** | Regression from large-scale to local variables | Limited by training data |
| **Dynamic downscaling** | Running a regional model nested in a GCM | Computationally expensive |
| **ML downscaling** | Neural network super-resolution | 10-50x resolution increase |
| **ClimaX** | Foundation model fine-tuned for downscaling | Flexible, multi-task |

---

## 3. Environmental Monitoring and Remote Sensing

### 3.1 Satellite Data Sources

| Satellite/Constellation | Type | Resolution | Revisit Time | Key Application |
|------------------------|------|-----------|-------------|-----------------|
| **Sentinel-2** | Multispectral optical | 10m | 5 days | Land cover, agriculture |
| **Sentinel-1** | SAR radar | 5m | 6-12 days | Flood mapping, soil moisture |
| **Landsat 8/9** | Multispectral optical | 30m | 16 days | Long-term land change |
| **GOES-R** | Geostationary | 2km | 1-15 min | Real-time weather, fire |
| **MODIS** | Multispectral | 250m-1km | Daily | Global vegetation, fire |

### 3.2 Deep Learning for Satellite Imagery

```python
import torch
import torch.nn as nn

class DeforestationDetector(nn.Module):
    """U-Net style model for deforestation detection"""
    def __init__(self, in_channels=10, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_classes, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

---

## 4. Agriculture and Food Security

### 4.1 Crop Type Mapping

AI models classify satellite imagery into crop types using phenological signatures.

### 4.2 Yield Prediction

Combining satellite vegetation indices (NDVI, EVI), weather data, soil properties, and historical yields.

### 4.3 Precision Agriculture

- Variable rate application: AI-guided fertilization, irrigation
- Weed detection: Computer vision for selective herbicide application
- Harvest optimization: Predicting optimal harvest timing

---

## 5. Biodiversity and Ecosystem Intelligence

| System | Focus | Method |
|--------|-------|--------|
| **BirdNET** | Bird species ID | CNN on spectrograms |
| **Arbimon** | Tropical monitoring | Pattern matching + ML |
| **Rainforest Connection** | Poaching + biodiversity | Real-time audio analysis |

---

## 6. Ocean and Marine Systems

- **SST prediction**: Critical for hurricane forecasting and coral bleaching alerts
- **ENSO prediction**: Long-lead El Nino/La Nina forecasting
- **Harmful algal blooms**: Satellite + ML for early warning

---

## 7. Urban Climate Resilience

- **Urban heat island mapping**: Landsat thermal data + ML
- **Flood risk assessment**: AI-driven pluvial, fluvial, and coastal flood modeling
- **Green infrastructure planning**: AI optimization of urban green spaces

---

## 8. Carbon Accounting and MRV

| Method | Application | Companies |
|--------|-------------|-----------|
| **Methane detection** | Point source from satellites | GHGSat, Carbon Mapper |
| **CO2 monitoring** | Column measurements | OCO-3 |
| **Deforestation tracking** | Forest carbon loss | Global Forest Watch |

---

## 9. Climate Finance and Risk Analytics

- Physical risk assessment: AI-enhanced hazard, exposure, and vulnerability modeling
- Transition risk: Stranded asset analysis, carbon policy modeling
- Climate insurance: Parametric triggers, claims processing, risk pricing

---

## 10. Key Technical Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Encoder-decoder** | U-Net style for spatial mapping | Land cover, flood mapping |
| **Spatiotemporal Transformer** | Attention over space and time | Weather, crop monitoring |
| **Graph Neural Network** | Irregular spatial relationships | Ocean, urban networks |
| **Diffusion model** | Generative uncertainty | Ensemble forecasting |
| **Fourier Neural Operator** | Efficient global patterns | PDE solving |
| **Physics-informed NN** | Embedding physical constraints | Climate projection |

---

## Related Library Documents

- `03-Agents/01-Agent-Architectures.md` -- Agent patterns for climate monitoring
- `06-Advanced/01-Multimodal-AI.md` -- Multimodal for satellite + weather fusion
- `10-Industry/03-AI-for-Robotics.md` -- Robotics for environmental monitoring
- `17-Research-Frontiers-2026/08-AI-for-Science.md` -- AI for science broadly
- `42-AI-for-Science-and-Drug-Discovery/01-Overview.md` -- Adjacent AI-for-science domain

---

*This document is part of the AI Base Knowledge Library -- an open, structured collection of AI knowledge for practitioners, researchers, and learners.*
