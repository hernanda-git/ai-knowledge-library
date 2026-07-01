# 03 - AI for Climate & Environmental Intelligence: Technical Deep Dive

> **Category:** 45-AI-for-Climate-and-Environmental-Intelligence
> **Last updated:** July 1, 2026
> **Cross-references:** `06-Advanced/02-Diffusion-Models.md`, `06-Advanced/08-Adversarial-ML.md`, `29-Reasoning-and-Inference-Scaling/01-Overview-and-Architecture.md`

---

## Table of Contents

1. [Data Assimilation for AI Weather Models](#1-data-assimilation-for-ai-weather-models)
2. [Training Strategies and Loss Functions](#2-training-strategies-and-loss-functions)
3. [Physics-Informed Constraints](#3-physics-informed-constraints)
4. [Uncertainty Quantification Deep Dive](#4-uncertainty-quantification-deep-dive)
5. [Resolution and Super-Resolution](#5-resolution-and-super-resolution)
6. [Multi-Task and Transfer Learning](#6-multi-task-and-transfer-learning)
7. [Foundation Models for Earth Science](#7-foundation-models-for-earth-science)
8. [Distributed Training for Climate Data](#8-distributed-training-for-climate-data)
9. [Evaluation Frameworks](#9-evaluation-frameworks)
10. [Common Failure Modes](#10-common-failure-modes)

---

## 1. Data Assimilation for AI Weather Models

### 1.1 The Assimilation Challenge

Traditional data assimilation (DA) combines observations with a prior forecast to produce an optimal analysis. For AI weather models, the challenge is different: how do you feed observations into a model trained on reanalysis?

**Approach 1: Train on Reanalysis, Infer from Reanalysis**
- Model receives ERA5 gridded fields as input
- Problem: ERA5 is already a blended product, not raw observations
- Advantage: Consistent, gap-free input fields

**Approach 2: Direct Observation Ingestion**
- Model receives raw observations (satellite radiances, radiosondes, etc.)
- Requires learnable observation operators
- Closer to traditional DA but more flexible

**Approach 3: Hybrid Approaches**
- Use reanalysis for training but incorporate observation uncertainty
- Learn to correct for reanalysis biases
- Emerging approach in 2026 research

### 1.2 Observation Types and Characteristics

| Observation Type | Spatial Coverage | Temporal Frequency | Accuracy | Key Variables |
|-----------------|-----------------|-------------------|----------|---------------|
| **Radiosondes** | Sparse (land only) | Twice daily | High | T, q, u, v profiles |
| **Satellite radiances** | Global | Every 1-6 hours | Moderate | T, q, precipitation |
| **Aircraft (AMDAR)** | Along flight paths | Continuous | High | T, wind, turbulence |
| **Surface stations** | Dense (land) | Hourly | High | T, p, wind, precip |
| **GNSS-RO** | Global | Dense | Very high | Temperature, humidity |
| **Scatterometry** | Ocean surface | Every 1-2 days | Moderate | Ocean surface winds |

### 1.3 Bias Correction

```python
import numpy as np

def correct_observation_bias(obs, model_background, bias_coefficients):
    """Apply variational bias correction to satellite observations"""
    predictors = np.column_stack([
        np.ones(len(obs)),           # constant bias
        obs - model_background,      # innovation
        np.cos(np.radians(obs_lat)), # latitudinal dependence
    ])
    bias = predictors @ bias_coefficients
    return obs - bias
```

---

## 2. Training Strategies and Loss Functions

### 2.1 Loss Function Design

AI weather models use composite loss functions:

```python
import torch
import torch.nn.functional as F

class WeatherForecastLoss(torch.nn.Module):
    def __init__(self, variable_weights, lead_time_weights):
        super().__init__()
        self.var_weights = variable_weights
        self.lt_weights = lead_time_weights

    def forward(self, pred, target, lead_times):
        mse = F.mse_loss(pred, target, reduction='none')
        weighted_mse = mse * self.var_weights
        lt_weight = self.lt_weights[lead_times]
        weighted_mse = weighted_mse * lt_weight
        spectral_loss = self.spectral_penalty(pred - target)
        return weighted_mse.mean() + 0.1 * spectral_loss
```

### 2.2 Spectral Loss

A key innovation penalizing errors at different spatial scales:

```python
def spectral_penalty(error_field):
    """Penalize errors at different spatial frequencies"""
    error_fft = torch.fft.rfft2(error_field)
    magnitude = torch.abs(error_fft)
    kx = torch.fft.rfftfreq(error_field.shape[-1])
    ky = torch.fft.fftfreq(error_field.shape[-2])
    K = torch.sqrt(kx**2 + ky**2)
    weighted = magnitude * K
    return weighted.mean()
```

### 2.3 Training Curriculum

Most successful models use a multi-stage training curriculum:

1. **Stage 1: Short lead times (0-24h)** - Learn atmospheric dynamics, lower learning rate
2. **Stage 2: Medium lead times (1-5 days)** - Autoregressive rollouts, higher learning rate
3. **Stage 3: Long lead times (5-15 days)** - Full predictions, ensemble generation

### 2.4 Data Augmentation

Geometric augmentations for spherical data:
- Random longitude rotation
- East-west mirroring
- Initial condition perturbation for ensemble generation

---

## 3. Physics-Informed Constraints

### 3.1 Conservation Laws

| Conservation Law | Mathematical Form | Enforcement Method |
|-----------------|-------------------|-------------------|
| **Mass conservation** | div(rho * v) = 0 | Penalty term in loss |
| **Energy conservation** | dE/dt = source - sink | Hard constraint via architecture |
| **Angular momentum** | Conservation on sphere | Spherical harmonic projection |
| **Water mass** | Total water vapor conserved | Normalization layer |

### 3.2 Hard vs. Soft Constraints

**Hard constraints** (architecture-level):
- Output layers that project onto physically valid states
- Normalization that preserves total mass/energy
- Pressure levels as explicit architectural features

**Soft constraints** (loss-level):
- Penalty terms that discourage violations
- More flexible but do not guarantee physical consistency

### 3.3 Hybrid Physics-AI Architecture

```python
class PhysicsAwareWeatherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_component = AtmosphericTransformer()
        self.physics_component = PrimitiveEquationsSolver()

    def forward(self, state, dt=21600):
        nn_tendency = self.neural_component(state)
        physical_state = self.physics_component(state, nn_tendency, dt)
        physical_state = self.enforce_physical_bounds(physical_state)
        return physical_state
```

---

## 4. Uncertainty Quantification Deep Dive

### 4.1 Sources of Uncertainty

1. **Initial condition uncertainty**: Imperfect knowledge of current state
2. **Model uncertainty**: Limitations of the learned model
3. **Observation uncertainty**: Noise and bias in input data
4. **Chaos sensitivity**: Sensitive dependence on initial conditions

### 4.2 Ensemble Generation Methods

| Method | Approach | Pros | Cons |
|--------|---------|------|------|
| **MC Dropout** | Multiple passes with dropout | Simple | May underestimate uncertainty |
| **Deep Ensembles** | Train multiple models | Well-calibrated | Expensive to train |
| **Stochastic perturbation** | Noise on initial conditions | Physically motivated | Limited model uncertainty |
| **Diffusion models** | Learn full distribution | Most expressive | Complex training |
| **Evidential deep learning** | Learn uncertainty distribution params | Single forward pass | Less flexible |

### 4.3 Calibrated Ensembles

```python
def evaluate_ensemble_calibration(ensemble_predictions, observations):
    """Check if ensemble spread matches forecast error"""
    mean = ensemble_predictions.mean(dim=0)
    spread = ensemble_predictions.std(dim=0)
    error = (mean - observations).abs()
    spread_skill = spread.mean() / error.mean()

    n_members = ensemble_predictions.shape[0]
    ranks = []
    for i in range(len(observations)):
        rank = (ensemble_predictions[:, i] < observations[i]).sum()
        ranks.append(rank)
    rank_histogram = torch.histc(torch.tensor(ranks), bins=n_members+1, min=0, max=n_members)

    return {
        'spread_skill_ratio': spread_skill,
        'rank_histogram': rank_histogram,
    }
```

---

## 5. Resolution and Super-Resolution

### 5.1 The Resolution Challenge

AI weather models typically train at 0.25 degree (~31km) but many applications need higher resolution:
- Urban heat island mapping: ~1km
- Flash flood prediction: ~100m
- Agricultural monitoring: ~10m

### 5.2 Neural Super-Resolution

NVIDIA's CorrDiff approach:
- **Input**: Coarse (0.25 deg) AI weather forecast
- **Output**: Fine (2-3km) regional forecast
- **Method**: Conditional diffusion model trained on high-res ERA5 + observations

### 5.3 Downscaling Applications

| Application | Coarse Input | Fine Output | Method |
|-------------|-------------|-------------|--------|
| **Precipitation** | 50km GCM | 1km rainfall maps | Conditional GAN |
| **Temperature** | 25km reanalysis | 1km surface temp | CNN + topographic features |
| **Wind** | 25km ERA5 | 100m wind maps | GNN + terrain data |
| **Humidity** | 25km ERA5 | 1km near-surface RH | Multi-scale U-Net |

---

## 6. Multi-Task and Transfer Learning

### 6.1 Multi-Task Weather Prediction

```python
class MultiTaskWeatherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = AtmosphericTransformer()
        self.variable_heads = torch.nn.ModuleDict({
            'temperature': PredictionHead(256, 1),
            'humidity': PredictionHead(256, 1),
            'wind_u': PredictionHead(256, 1),
            'wind_v': PredictionHead(256, 1),
            'geopotential': PredictionHead(256, 1),
        })

    def forward(self, x):
        features = self.shared_backbone(x)
        return {name: head(features) for name, head in self.variable_heads.items()}
```

### 6.2 Transfer Learning Strategies

1. **Pretrain on global, fine-tune on region**: Train on ERA5 globally, fine-tune for specific region at higher resolution
2. **Pretrain on weather, fine-tune on climate**: Use weather weights for climate projection
3. **Transfer across variables**: Use wind/temperature to improve precipitation prediction

---

## 7. Foundation Models for Earth Science

### 7.1 The Foundation Model Paradigm

| Model | Organization | Pre-training Data | Fine-tuning Tasks |
|-------|-------------|-------------------|-------------------|
| **ClimaX** | Microsoft | CMIP6 ensemble | Downscaling, interpolation, projection |
| **Aurora** | Microsoft | Multiple atmospheric datasets | Weather, air quality, ocean waves |
| **Prithvi** | NASA/IBM | Harmonized Landsat-Sentinel | Land cover, flood mapping |
| **Gencast** | DeepMind | ERA5 | Weather forecasting with uncertainty |

### 7.2 Fine-Tuning Strategy

```python
from climaix import ClimaXFoundation

model = ClimaXFoundation.from_pretrained('microsoft/climax-base')
for param in model.backbone.parameters():
    param.requires_grad = False

flood_head = torch.nn.Sequential(
    torch.nn.Linear(768, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 1), torch.nn.Sigmoid()
)
```

---

## 8. Distributed Training for Climate Data

### 8.1 Scale of Climate Data

- ERA5: ~5TB total
- CMIP6: ~30PB across all models
- Satellite data: Petabytes per year

### 8.2 Training Infrastructure

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 8.3 Memory Optimization

- **Gradient checkpointing**: Recompute activations during backward pass
- **Mixed precision**: FP16/BF16 for forward/backward
- **Model parallelism**: Split model across GPUs
- **Data parallelism**: Split batch across GPUs

---

## 9. Evaluation Frameworks

### 9.1 Comprehensive Evaluation

```python
class WeatherForecastEvaluator:
    def __init__(self, baselines, metrics):
        self.baselines = baselines
        self.metrics = metrics

    def evaluate(self, predictions, observations):
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn(predictions, observations)
        for baseline_name, baseline_pred in self.baselines.items():
            baseline_score = self.metrics['acc'](baseline_pred, observations)
            ai_score = self.metrics['acc'](predictions, observations)
            results[f'versus_{baseline_name}'] = ai_score - baseline_score
        return results
```

### 9.2 Extreme Event Evaluation

- **POD (Probability of Detection)**: Fraction of events correctly detected
- **FAR (False Alarm Ratio)**: Fraction of alarms that were false
- **CSI (Critical Success Index)**: Balanced detection measure
- **Reliability diagrams**: Predicted probabilities vs observed frequencies

---

## 10. Common Failure Modes

### 10.1 Distribution Shift

AI models trained on 20th-century climate may not generalize to:
- Novel weather patterns (heat domes, unprecedented atmospheric rivers)
- Tipping points (abrupt climate state changes)
- Compound events (simultaneous extremes not in training data)

### 10.2 Physics Violations

Pure ML models can produce:
- Negative humidity or pressure values
- Mass/energy non-conservation
- Unstable integrations during autoregressive rollout
- Spatially incoherent features

### 10.3 Autoregressive Error Accumulation

Small errors at short lead times compound at long lead times, and model may drift toward climatology.

### 10.4 Mitigation Strategies

| Failure Mode | Mitigation |
|-------------|-----------|
| Distribution shift | Train on diverse scenarios, test on novel events |
| Physics violations | Add physics-informed loss, use hard constraints |
| Error accumulation | Curriculum training, regularization at long lead times |
| Catastrophic predictions | Ensemble averaging, post-processing calibration |

---

## Related Library Documents

- `06-Advanced/02-Diffusion-Models.md` -- Diffusion model techniques used in GenCast
- `06-Advanced/08-Adversarial-ML.md` -- Adversarial robustness for climate models
- `29-Reasoning-and-Inference-Scaling/01-Overview-and-Architecture.md` -- Scaling laws for climate AI
- `36-Long-Context-AI/01-Overview.md` -- Long-context techniques for temporal data
- `42-AI-for-Science-and-Drug-Discovery/03-Technical-Deep-Dive.md` -- Adjacent technical patterns

---

*This document is part of the AI Base Knowledge Library -- an open, structured collection of AI knowledge for practitioners, researchers, and learners.*
