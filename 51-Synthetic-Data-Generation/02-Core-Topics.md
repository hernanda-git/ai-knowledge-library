# Synthetic Data Generation: Core Topics

> This document covers the foundational techniques, algorithms, and architectural patterns that power modern synthetic data generation. It provides practitioners with the knowledge needed to select, implement, and evaluate synthetic data solutions for their specific use cases.

---

## Table of Contents

1. [Generative Model Architectures](#generative-model-architectures)
2. [Training Strategies](#training-strategies)
3. [Quality Metrics and Evaluation](#quality-metrics-and-evaluation)
4. [Privacy-Preserving Techniques](#privacy-preserving-techniques)
5. [Domain-Specific Approaches](#domain-specific-approaches)
6. [Data Types and Modalities](#data-types-and-modalities)
7. [Evaluation Frameworks](#evaluation-frameworks)

---

## Generative Model Architectures

### Variational Autoencoders (VAEs)

VAEs learn a compressed latent representation of data and generate new samples by sampling from this latent space and decoding.

**Architecture**:
```
Input → Encoder → μ, σ (latent parameters) → Sample z → Decoder → Output
         ↓                                       ↓
    Reparametrization trick: z = μ + σ * ε, where ε ~ N(0,1)
```

**Strengths**:
- Stable training dynamics
- Smooth latent space enables interpolation
- Well-understood theoretical foundations
- Good for tabular and sequential data

**Weaknesses**:
- Blurry outputs for high-dimensional data (images)
- Posterior collapse in complex scenarios
- Lower sample quality than GANs for images

**Implementation**:
```python
import torch
import torch.nn as nn

class SyntheticDataVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def generate(self, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)
```

### Generative Adversarial Networks (GANs)

GANs consist of two networks competing in a minimax game: a generator that creates synthetic samples and a discriminator that tries to distinguish real from fake.

**Architecture**:
```
Random noise z → Generator G → Fake data
                                   ↓
                    Discriminator D → Real or Fake?
                                   ↑
                          Real data → Real
```

**Training Objective**:
```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

**Key Variants for Synthetic Data**:

| Variant | Best For | Key Innovation |
|---------|----------|----------------|
| **WGAN-GP** | Tabular, general | Wasserstein distance + gradient penalty |
| **CTGAN** | Tabular (conditional) | Conditional generation + mode-specific normalization |
| **TGAN** | Tabular (time series) | Temporal dependencies |
| **StyleGAN2/3** | Images | Style-based generation, high fidelity |
| **Pix2Pix** | Image-to-image | Paired image synthesis |
| **CycleGAN** | Unpaired translation | Cycle consistency loss |

**CTGAN Implementation for Tabular Data**:
```python
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# CTGAN is the standard for tabular synthetic data
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# Configure CTGAN with privacy-preserving options
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    discriminator_dim=(256, 256),
    generator_dim=(256, 256),
    discriminator_decay=1e-6,
    generator_decay=1e-6,
    pac=10,  # Pac parameter for training stability
    verbose=True
)

synthesizer.fit(real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
```

### Diffusion Models

Diffusion models generate data by learning to reverse a gradual noising process. They have become the dominant architecture for high-quality image, video, and audio synthesis.

**Two-Phase Process**:
```
Forward process (fixed):  x₀ → x₁ → x₂ → ... → xₜ (pure noise)
Backlearned):  xₜ → xₜ₋₁ → ... → x₁ → x₀ (generated sample)
```

**Core Equation** (Denoising Diffusion Probabilistic Models):
```
p(xₜ₋₁ | xₜ) = N(xₜ₋₁; μθ(xₜ, t), σt²I)
```

**Applications in Synthetic Data**:
- **Images**: Stable Diffusion, DALL-E, Midjourney
- **Video**: Sora, Runway, Pika
- **Audio**: AudioLDM, MusicGen
- **3D**: Point-E, Shap-E
- **Tabular**: TabDDPM, SynthCity

```python
# Example: Using TabDDPM for tabular synthetic data
from synthcity.plugins.core.models.tabular_ddpm import TabularDDPM

# Configure DDPM for tabular data
model = TabularDDPM(
    n_iter=1000,
    batch_size=256,
    diffusion_iterations=1000,
    dim=128,
    dim_mults=(1, 2, 4),
    random_features=False,
    num_time_dim=64
)

model.fit(X_real)
synthetic_tabular = model.generate(n_samples=10000)
```

### Transformer-Based Generators

Transformers have emerged as powerful generative models beyond language:

**Applications**:
- **Table Transformer**: Generates synthetic tabular data
- **Vision Transformer (ViT)** based generators for images
- **Audio Spectrogram Transformer (AST)** for audio synthesis
- **Time Series Transformers** for temporal data generation

```python
# Example: Time series synthetic data with TimeGrad
# (Autoregressive diffusion model)
from timegrad.reversibles import ReversibleGraphNet
from timegrad.distributions import GaussianDistribution

# TimeGrad uses reversible networks with diffusion
# for efficient autoregressive time series generation
model = ReversibleGraphNet(
    input_size=num_features,
    n_blocks=5,
    n_layers=4,
    hidden_size=128,
    forecast_history=24,
    forecast_horizon=12
)
```

---

## Training Strategies

### Data Preprocessing for Synthesis

Before training any generative model, data must be preprocessed:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SyntheticDataPreprocessor:
    """Preprocess real data for synthetic generation."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
    
    def fit_transform(self, df):
        """Fit preprocessors and transform data."""
        processed = df.copy()
        
        # Identify column types
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                self.categorical_cols.append(col)
                le = LabelEncoder()
                processed[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                self.numerical_cols.append(col)
                scaler = StandardScaler()
                processed[col] = scaler.fit_transform(
                    df[[col]].fillna(0)
                )
                self.scalers[col] = scaler
        
        return processed
    
    def inverse_transform(self, synthetic_df):
        """Convert synthetic data back to original format."""
        result = synthetic_df.copy()
        
        for col in self.numerical_cols:
            if col in self.scalers:
                result[col] = self.scalers[col].inverse_transform(
                    result[[col]]
                )
        
        for col in self.categorical_cols:
            if col in self.encoders:
                result[col] = self.encoders[col].inverse_transform(
                    result[col].clip(0, len(self.encoders[col].classes_) - 1).astype(int)
                )
        
        return result
```

### Handling Mixed Data Types

Real-world datasets almost always contain a mix of numerical, categorical, and ordinal features:

```python
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import (
    Positive,
    Unique,
    FixedCombinations,
    Inequality
)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# Add constraints for data integrity
constraints = [
    # Ensure positive values
    Positive(
        column_name='age',
        strict_boundaries=False
    ),
    
    # Ensure unique identifiers
    Unique(column_name='patient_id'),
    
    # Ensure valid category combinations
    FixedCombinations(
        column_name_1='department',
        column_name_2='role',
        combinations=[
            ('Engineering', 'Developer'),
            ('Engineering', 'Manager'),
            ('Marketing', 'Analyst'),
            # ... more valid combos
        ]
    ),
    
    # Ensure income > expenses
    Inequality(
        low_column_name='expenses',
        high_column_name='income',
        strict_boundaries=False
    )
]

# Train with constraints
synthesizer = GaussianCopulaSynthesizer(
    metadata,
    constraints=constraints,
    enforce_min_max_values=True,
    enforce_rounding=True,  # Round integer columns
    default_distribution='truncnorm'  # Good for bounded data
)

synthesizer.fit(real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
```

### Conditional Generation

Generating synthetic data conditioned on specific attributes is critical for creating balanced datasets:

```python
# Generate synthetic data with specific class distribution
synthesizer = CTGANSynthesizer(metadata, epochs=200)

# Fit on real data
synthesizer.fit(real_data)

# Generate balanced classes (50/50 fraud vs legitimate)
synthetic_fraud = synthesizer.sample_remaining_columns(
    synthetic_data=pd.DataFrame({
        'is_fraud': [1] * 5000
    })
)

synthetic_legit = synthesizer.sample_remaining_columns(
    synthetic_data=pd.DataFrame({
        'is_fraud': [0] * 5000
    })
)

balanced_synthetic = pd.concat([synthetic_fraud, synthetic_legit])
```

---

## Quality Metrics and Evaluation

### Fidelity Metrics

How well does synthetic data match real data?

```python
from sdv.evaluation.single_table import evaluate_quality

# SDV Quality Report
quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# Key metrics:
# - Column Shapes: How well individual column distributions match
# - Column Pair Trends: How well correlations between columns are preserved
# - Overall Quality Score: Weighted combination (0-1, higher is better)
```

**Custom Quality Metrics**:

```python
from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np

def evaluate_synthetic_quality(real_df, synthetic_df, numerical_cols):
    """Compute custom quality metrics for synthetic data."""
    metrics = {}
    
    for col in numerical_cols:
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(
            real_df[col].dropna(),
            synthetic_df[col].dropna()
        )
        
        # Wasserstein distance
        w_dist = wasserstein_distance(
            real_df[col].dropna(),
            synthetic_df[col].dropna()
        )
        
        # Jensen-Shannon divergence (on histograms)
        real_hist, _ = np.histogram(real_df[col].dropna(), bins=50, density=True)
        synth_hist, _ = np.histogram(synthetic_df[col].dropna(), bins=50, density=True)
        
        # Avoid log(0)
        real_hist = real_hist + 1e-10
        synth_hist = synth_hist + 1e-10
        
        # Normalize
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()
        
        m = 0.5 * (real_hist + synth_hist)
        js_div = 0.5 * (
            np.sum(real_hist * np.log(real_hist / m)) +
            np.sum(synth_hist * np.log(synth_hist / m))
        )
        
        metrics[col] = {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_pval,
            'wasserstein_distance': w_dist,
            'js_divergence': js_div
        }
    
    return metrics
```

### Utility Metrics

How useful is synthetic data for downstream tasks?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_synthetic_utility(real_data, synthetic_data, 
                                target_col, task='classification'):
    """Evaluate if models trained on synthetic data perform well."""
    
    # Train on real data
    X_real = real_data.drop(columns=[target_col])
    y_real = real_data[target_col]
    
    model_real = RandomForestClassifier(n_estimators=100, random_state=42)
    real_scores = cross_val_score(model_real, X_real, y_real, cv=5, scoring='f1')
    
    # Train on synthetic data
    X_synth = synthetic_data.drop(columns=[target_col])
    y_synth = synthetic_data[target_col]
    
    model_synth = RandomForestClassifier(n_estimators=100, random_state=42)
    synth_scores = cross_val_score(model_synth, X_synth, y_synth, cv=5, scoring='f1')
    
    # Test synthetic-trained model on real data
    model_synth.fit(X_synth, y_synth)
    y_pred = model_synth.predict(X_real)
    transfer_score = f1_score(y_real, y_pred)
    
    return {
        'real_data_f1': np.mean(real_scores),
        'synthetic_data_f1': np.mean(synth_scores),
        'transfer_f1': transfer_score,
        'utility_ratio': np.mean(synth_scores) / np.mean(real_scores)
    }
```

### Privacy Metrics

```python
def evaluate_synthetic_privacy(real_data, synthetic_data, 
                                sensitive_cols, model_class=None):
    """Evaluate privacy protection of synthetic data."""
    from sklearn.neighbors import NearestNeighbors
    
    metrics = {}
    
    # Nearest neighbor distance ratio
    nn_real = NearestNeighbors(n_neighbors=2)
    nn_real.fit(real_data[sensitive_cols].fillna(0))
    
    # For each synthetic record, find nearest real neighbor
    distances, _ = nn_real.kneighbors(
        synthetic_data[sensitive_cols].fillna(0)
    )
    
    metrics['min_nn_distance'] = distances[:, 0].min()
    metrics['mean_nn_distance'] = distances[:, 0].mean()
    metrics['records_within_threshold'] = (
        distances[:, 0] < 0.1  # Very close to real data
    ).mean()
    
    # Membership inference attack simulation
    if model_class:
        from sklearn.model_selection import train_test_split
        
        # Split real data into members and non-members
        members, non_members = train_test_split(
            real_data, test_size=0.5, random_state=42
        )
        
        # Train membership classifier
        # (simplified — in practice, use shadow models)
        X_train = pd.concat([members, non_members])
        y_train = np.concatenate([
            np.ones(len(members)), 
            np.zeros(len(non_members))
        ])
        
        # Attack accuracy indicates privacy leakage
        attack_model = RandomForestClassifier(n_estimators=100)
        attack_scores = cross_val_score(
            attack_model, X_train[sensitive_cols], y_train, cv=3
        )
        metrics['membership_inference_accuracy'] = np.mean(attack_scores)
    
    return metrics
```

---

## Privacy-Preserving Techniques

### Differential Privacy for Synthetic Data

Differential privacy provides mathematical guarantees that individual records cannot be identified from the synthetic data:

```python
# DP-SGD training for synthetic data generation
import diffprivlib.tools as dp
from diffprivlib.mechanisms import Gaussian

def dp_synthetic_generation(real_data, epsilon=1.0, delta=1e-5):
    """Generate synthetic data with differential privacy guarantees."""
    
    # Add calibrated noise to training statistics
    n = len(real_data)
    sensitivity = 1.0  # Depends on data preprocessing
    
    # Gaussian mechanism for continuous data
    mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    
    # Generate noisy statistics
    noisy_mean = mechanism.randomise(real_data.mean().values)
    noisy_cov = mechanism.randomise(real_data.cov().values)
    
    # Generate synthetic data from noisy distribution
    synthetic_data = np.random.multivariate_normal(
        mean=noisy_mean,
        cov=noisy_cov + np.eye(noisy_cov.shape[0]) * 0.01,  # Regularization
        size=n
    )
    
    return pd.DataFrame(synthetic_data, columns=real_data.columns)
```

### Membership Inference Defense

```python
def defend_against_membership_inference(
    real_data, synthetic_data, 
    augmentation_factor=3
):
    """Augment synthetic data to reduce membership inference risk."""
    
    augmented_synthetic = []
    
    for _ in range(augmentation_factor):
        # Add noise to synthetic data
        noise = np.random.normal(0, 0.01, synthetic_data.shape)
        augmented = synthetic_data + noise
        
        # Clip to valid ranges
        augmented = np.clip(
            augmented, 
            real_data.min().values, 
            real_data.max().values
        )
        
        augmented_synthetic.append(pd.DataFrame(
            augmented, 
            columns=synthetic_data.columns
        ))
    
    return pd.concat(augmented_synthetic, ignore_index=True)
```

---

## Domain-Specific Approaches

### Medical Data Synthesis

Medical data requires special handling due to:
- **Missing values**: Clinical data often has 30-50% missing values
- **Temporal patterns**: Patient trajectories evolve over time
- **Imbalanced classes**: Rare diseases may have <100 cases
- **Multi-site heterogeneity**: Different hospitals use different protocols

```python
from sdv.single_table import GaussianCopulaSynthesizer

# Medical data synthesizer with clinical constraints
medical_metadata = SingleTableMetadata()
medical_metadata.detect_from_dataframe(clinical_df)

# Add clinical constraints
constraints = [
    # Systolic > Diastolic
    Inequality(
        low_column_name='diastolic_bp',
        high_column_name='systolic_bp'
    ),
    # Age > 0
    Positive(column_name='age'),
    # BMI in valid range
    # (handled by enforce_min_max_values)
]

medical_synth = GaussianCopulaSynthesizer(
    medical_metadata,
    constraints=constraints,
    default_distribution='truncnorm',
    enforce_min_max_values=True
)

medical_synth.fit(clinical_df)
synthetic_patients = medical_synth.sample(num_rows=50000)
```

### Financial Transaction Synthesis

Financial data has unique characteristics:
- **Temporal autocorrelation**: Transactions follow patterns
- **Multi-scale patterns**: Daily, weekly, monthly cycles
- **Rare events**: Fraud is <0.1% of transactions
- **Network structure**: Transactions form graphs

```python
# Time-series aware synthetic transaction generation
def generate_synthetic_transactions(
    real_transactions,
    num_days=365,
    fraud_rate=0.001
):
    """Generate synthetic transactions preserving temporal patterns."""
    
    from sdv.timeseries import PARSynthesizer
    
    # Configure time series synthesizer
    synthesizer = PARSynthesizer(
        datetime_column='timestamp',
        context_columns=['customer_id'],
        sequence_length=24,  # 24-hour windows
        epochs=100,
        batch_size=64,
        learning_rate=1e-3
    )
    
    synthesizer.fit(real_transactions)
    
    # Generate full year of synthetic transactions
    synthetic_transactions = synthesizer.sample(
        num_sequences=num_days * 24,
        context_column_values=None  # Generate new customers
    )
    
    return synthetic_transactions
```

### Time Series Synthesis

```python
# Using TimeGrad for complex temporal dependencies
from timegrad.models import TimeGrad

config = {
    'input_size': 10,  # Number of features
    'hidden_size': 128,
    'num_layers': 4,
    'num_blocks': 2,
    'dropout': 0.1,
    'forecast_history': 48,
    'forecast_horizon': 24,
    'context_size': 16,
    'num_timesteps': 1000,  # Diffusion steps
    'beta_schedule': 'cosine',
    'loss_type': 'l2'
}

model = TimeGrad(**config)
model.fit(train_loader, val_loader, epochs=200)
synthetic_series = model.generate(n_samples=1000, horizon=168)
```

---

## Data Types and Modalities

### Structured/Tabular Data

| Technique | Pros | Cons | Best For |
|-----------|------|------|----------|
| Gaussian Copula | Fast, interpretable | Limited to Gaussian relationships | Simple tabular data |
| CTGAN | Handles mixed types, mode coverage | Slow training | Complex tabular data |
| TVAE | Stable training, good quality | Slower than CTGAN | Medium-sized datasets |
| TabDDPM | State-of-the-art quality | Computationally expensive | Large tabular datasets |
| GReaT (LLM-based) | Natural language conditioning | Requires LLM | Mixed data, natural language queries |

### Unstructured Data (Text)

```python
# LLM-based synthetic text generation
from anthropic import Anthropic

def generate_synthetic_training_data(
    task_description, 
    num_examples=100,
    model="claude-sonnet-4-20250514"
):
    """Generate synthetic training examples using an LLM."""
    
    client = Anthropic()
    
    prompt = f"""Generate {num_examples} synthetic training examples for:
    
Task: {task_description}

Requirements:
- Each example must be realistic and diverse
- Include edge cases and boundary conditions
- Vary the complexity and style
- Include both positive and negative examples where appropriate

Output as JSON array with fields: "input", "output", "metadata"

Generate exactly {num_examples} examples."""
    
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    return json.loads(response.content[0].text)
```

### Image Data

```python
# Stable Diffusion for synthetic image generation
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

def generate_synthetic_images(
    prompts,
    output_dir="synthetic_images",
    num_images_per_prompt=10,
    guidance_scale=7.5,
    num_inference_steps=30
):
    """Generate synthetic images using Stable Diffusion."""
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generated = []
    for prompt_idx, prompt in enumerate(prompts):
        for img_idx in range(num_images_per_prompt):
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(
                    prompt_idx * 1000 + img_idx
                )
            ).images[0]
            
            filename = f"{output_dir}/synth_{prompt_idx}_{img_idx}.png"
            image.save(filename)
            generated.append(filename)
    
    return generated
```

### Audio Data

```python
# TTS-based synthetic audio generation
def generate_synthetic_audio_dataset(
    text_samples,
    voice_ids=None,
    output_dir="synthetic_audio"
):
    """Generate synthetic audio from text using TTS."""
    from elevenlabs import ElevenLabs
    
    client = ElevenLabs(api_key="your-api-key")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, text in enumerate(text_samples):
        # Generate audio for each text sample
        audio = client.generate(
            text=text,
            voice=voice_ids[idx % len(voice_ids)] if voice_ids else "Rachel"
        )
        
        with open(f"{output_dir}/audio_{idx}.mp3", "wb") as f:
            for chunk in audio:
                f.write(chunk)
```

---

## Evaluation Frameworks

### Comprehensive Evaluation Pipeline

```python
class SyntheticDataEvaluator:
    """Complete evaluation framework for synthetic data."""
    
    def __init__(self, real_data, synthetic_data, metadata):
        self.real = real_data
        self.synthetic = synthetic_data
        self.metadata = metadata
        self.results = {}
    
    def run_full_evaluation(self):
        """Run all evaluation metrics."""
        
        self.results['fidelity'] = self.evaluate_fidelity()
        self.results['utility'] = self.evaluate_utility()
        self.results['privacy'] = self.evaluate_privacy()
        self.results['diversity'] = self.evaluate_diversity()
        self.results['overall'] = self.compute_overall_score()
        
        return self.results
    
    def evaluate_fidelity(self):
        """How well does synthetic match real distributions?"""
        from sdv.evaluation.single_table import evaluate_quality
        
        quality_report = evaluate_quality(
            self.real, self.synthetic, self.metadata
        )
        return quality_report.get_details()
    
    def evaluate_utility(self):
        """Can models trained on synthetic perform well on real?"""
        target_col = self._identify_target()
        if target_col is None:
            return {"status": "no target column identified"}
        
        X_real = self.real.drop(columns=[target_col])
        y_real = self.real[target_col]
        X_synth = self.synthetic.drop(columns=[target_col])
        y_synth = self.synthetic[target_col]
        
        # Train on real, test on real
        from sklearn.ensemble import RandomForestClassifier
        model_real = RandomForestClassifier(n_estimators=100, random_state=42)
        real_scores = cross_val_score(model_real, X_real, y_real, cv=5)
        
        # Train on synthetic, test on real
        model_synth = RandomForestClassifier(n_estimators=100, random_state=42)
        model_synth.fit(X_synth, y_synth)
        transfer_scores = cross_val_score(model_synth, X_real, y_real, cv=5)
        
        return {
            "real_data_accuracy": np.mean(real_scores),
            "transfer_accuracy": np.mean(transfer_scores),
            "utility_ratio": np.mean(transfer_scores) / np.mean(real_scores)
        }
    
    def evaluate_privacy(self):
        """How well does synthetic data protect privacy?"""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(self.real.select_dtypes(include=[np.number]))
        
        distances, _ = nn.kneighbors(
            self.synthetic.select_dtypes(include=[np.number])
        )
        
        return {
            "mean_nn_distance": float(distances[:, 0].mean()),
            "min_nn_distance": float(distances[:, 0].min()),
            "pct_close_to_real": float((distances[:, 0] < 0.05).mean())
        }
    
    def evaluate_diversity(self):
        """Does synthetic data cover the full range of variation?"""
        from scipy.spatial.distance import pdist, squareform
        
        # Compute pairwise distances in synthetic data
        synth_numeric = self.synthetic.select_dtypes(include=[np.number])
        distances = pdist(synth_numeric.sample(min(1000, len(synth_numeric))))
        
        return {
            "mean_pairwise_distance": float(np.mean(distances)),
            "min_pairwise_distance": float(np.min(distances)),
            "coverage_score": float(np.std(distances) / np.mean(distances))
        }
    
    def compute_overall_score(self):
        """Weighted overall quality score."""
        scores = []
        
        if 'fidelity' in self.results and isinstance(self.results['fidelity'], dict):
            fidelity_score = self.results['fidelity'].get('overall_quality', 0.5)
            scores.append(('fidelity', fidelity_score, 0.3))
        
        if 'utility' in self.results and 'utility_ratio' in self.results['utility']:
            utility_score = min(self.results['utility']['utility_ratio'], 1.0)
            scores.append(('utility', utility_score, 0.4))
        
        if 'privacy' in self.results:
            privacy_score = max(0, 1 - self.results['privacy']['pct_close_to_real'])
            scores.append(('privacy', privacy_score, 0.3))
        
        if scores:
            total_weight = sum(w for _, _, w in scores)
            overall = sum(s * w for _, s, w in scores) / total_weight
        else:
            overall = 0.5
        
        return {
            "overall_score": overall,
            "component_scores": {name: score for name, score, _ in scores},
            "recommendation": self._get_recommendation(overall)
        }
    
    def _get_recommendation(self, score):
        if score >= 0.8:
            return "Excellent — synthetic data is production-ready"
        elif score >= 0.6:
            return "Good — suitable for most use cases with monitoring"
        elif score >= 0.4:
            return "Fair — consider additional tuning or real data supplementation"
        else:
            return "Poor — significant quality issues, review generation approach"
    
    def _identify_target(self):
        """Heuristic to identify the most likely target column."""
        for col in self.real.columns:
            if self.real[col].nunique() <= 20 and self.real[col].dtype != 'object':
                return col
        return None
```

---

## See Also

- [01-Overview.md](01-Overview.md) — Introduction and market overview
- [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) — Advanced techniques and architecture details
- [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) — Complete tooling guide
- [05-Future-Outlook.md](05-Future-Outlook.md) — Future trends and research directions

---

*Last updated: July 4, 2026*
*Part of the AI Knowledge Library — Category 51: Synthetic Data Generation*
