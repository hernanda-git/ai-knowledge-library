# Synthetic Data Generation: Tools and Frameworks

> This document provides a comprehensive guide to the tools, libraries, platforms, and frameworks available for synthetic data generation in 2026. It covers open-source libraries, commercial platforms, cloud services, and specialized toolkits for different data modalities and use cases.

---

## Table of Contents

1. [Open-Source Libraries](#open-source-libraries)
2. [Commercial Platforms](#commercial-platforms)
3. [Cloud Services](#cloud-services)
4. [Simulation Engines](#simulation-engines)
5. [Domain-Specific Toolkits](#domain-specific-toolkits)
6. [Evaluation and Benchmarking Tools](#evaluation-and-benchmarking-tools)
7. [Integration Patterns](#integration-patterns)

---

## Open-Source Libraries

### SDV (Synthetic Data Vault)

SDV is the most comprehensive open-source framework for tabular synthetic data generation.

**Overview**:
- **GitHub**: github.com/sdv-dev/SDV
- **License**: BSL 1.1 (Business Source License)
- **Language**: Python
- **Stars**: 8,000+

**Key Features**:
- Multiple generative models (GaussianCopula, CTGAN, TVAE, CopulaGAN)
- Automatic metadata detection
- Quality evaluation built-in
- Time series support (PARSynthesizer)
- Hierarchical/relational data support

```python
# SDV Quick Start
pip install sdv

from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer
)
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# 1. Load and detect metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# 2. Choose and train synthesizer
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    verbose=True
)
synthesizer.fit(real_data)

# 3. Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=10000)

# 4. Evaluate quality
quality = evaluate_quality(real_data, synthetic_data, metadata)
print(quality.get_details())

# 5. Save/load
synthesizer.save('my_synthesizer.pkl')
loaded = GaussianCopulaSynthesizer.load('my_synthesizer.pkl')
```

**Supported Models**:

| Model | Type | Best For | Training Speed | Quality |
|-------|------|----------|---------------|---------|
| GaussianCopula | Statistical | Simple tabular | Very Fast | Good |
| CTGAN | GAN | Complex mixed types | Slow | Very Good |
| TVAE | VAE | Medium datasets | Medium | Good |
| CopulaGAN | Hybrid | Balanced quality/speed | Medium | Very Good |

### Gretel SDK

Gretel provides both open-source and cloud-based synthetic data generation.

**Overview**:
- **GitHub**: github.com/gretelai/gretel-sdk
- **License**: Apache 2.0
- **Language**: Python
- **Key Innovation**: LLM-powered synthesis (Gretel Navigator)

```python
# Gretel Synthetic Data Generation
pip install gretel-sdk

from gretel_client import Gretel

# Configure synthetic data generation
gretel = Gretel(project_name="my-project")

# Using Navigator (LLM-based)
config = {
    "schema_version": "1.0",
    "models": [
        {
            "name": "navigator",
            "params": {
                "model_type": "gretel-navigator",
                "data_source": "path/to/data.csv",
                "num_records": 10000,
                "fields": ["name", "email", "age", "salary"]
            }
        }
    ]
}

# Run generation
job = gretel.submit_job(config)
job.wait()

# Download results
results = job.get_artifacts()
```

### Mostly AI

Mostly AI focuses on privacy-preserving synthetic data with enterprise features.

**Overview**:
- **License**: Free tier available, enterprise pricing
- **Key Features**: Differential privacy, fairness constraints, smart masking
- **Strengths**: Financial services, healthcare

```python
# Mostly AI SDK
pip install mostlyai

from mostlyai import MostlyAI

mostly = MostlyAI(api_key="your-api-key")

# Create a synthesizer
synthesizer = mostly.synthesizers.create(
    name="customer-synthesis",
    source_data=real_data,
    tabular_config={
        "max_sample_size": 1000000,
        "privacy_regexes": [
            {"column": "ssn", "regex": r"\d{3}-\d{2}-\d{4}"},
            {"column": "email", "regex": r".*@.*"}
        ],
        "constraints": [
            {"type": "unique", "columns": ["customer_id"]},
            {"type": "range", "column": "age", "min": 0, "max": 120}
        ]
    }
)

# Generate synthetic data
synthetic = mostly.synthesizers.generate(
    synthesizer_id=synthesizer.id,
    sample_size=50000
)
```

### CTGAN (Standalone)

The original CTGAN implementation for tabular data.

```python
# CTGAN standalone
pip install ctgan

from ctgan import CTGANSynthesizer

# Define metadata
continuous_columns = ['age', 'income', 'balance']
categorical_columns = ['gender', 'education', 'occupation']

# Create and train
synthesizer = CTGANSynthesizer(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    batch_size=500,
    pac=10,
    epochs=300,
    verbose=True
)

synthesizer.fit(
    data=real_data,
    discrete_columns=categorical_columns
)

# Generate
synthetic = synthesizer.sample(num_rows=10000)

# Save/load
synthesizer.save('ctgan_model.pkl')
loaded = CTGANSynthesizer.load('ctgan_model.pkl')
```

### SynthCity

A library specifically designed for differentially private synthetic data.

```python
# SynthCity
pip install synthcity

from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints

# List available plugins
plugins = Plugins()
print(plugins.list())

# Using DP-GAN (differentially private)
dp_plugin = plugins.get("dpgan")

# Configure privacy
constraints = Constraints(
    rules=[
        {"column": "age", "constraint_type": "range", "lower_bound": 0, "upper_bound": 120}
    ]
)

dp_plugin.fit(
    real_data,
    protected_fields=["ssn", "email"],  # Fields to protect
    constraints=constraints,
    epsilon=1.0,  # Differential privacy budget
    delta=1e-5
)

synthetic = dp_plugin.generate(n_samples=10000)
```

### Synthpop (R)

For R users, Synthpop is a well-established package for synthetic tabular data.

```r
# Synthpop in R
install.packages("synthpop")

library(synthpop)

# Simple synthetic data generation
syn <- syn(data = real_data, m = 1)  # m = number of synthetic datasets

# Access synthetic data
synthetic_data <- syn$syn

# Evaluate quality
compare(syn, real_data, stat = "mean")

# Generate with specific methods
syn <- syn(
  data = real_data,
  method = c(
    "norm",           # age: normal
    "lognorm",        # income: lognormal
    "sample",         # gender: categorical
    "cart",           # education: decision tree
    "pmm"             # occupation: predictive mean matching
  ),
  visit.sequence = c(1, 2, 3, 4, 5),
  m = 5  # Generate 5 synthetic datasets
)
```

### DataSynthesizer

A tool from MIT for private synthetic data generation.

```python
# DataSynthesizer
pip install DataSynthesizer

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector

# Step 1: Describe real data
describer = DataDescriber(category_threshold=10)
describer.describe_dataset_independent_attributes(real_data)
describer.describe_dataset_dependent_attributes(real_data)
describer.save_dataset_description_to_file("data_description.json")

# Step 2: Generate synthetic data
generator = DataGenerator()
generator.generate_dataset_in_dependent_attributes(num_tuples=10000)
generator.generate_dataset_dependent_attributes()
synthetic_data = generator.generate_tuples()
```

---

## Commercial Platforms

### Comparison Matrix

| Platform | Focus | Pricing | Key Strength | Privacy Model |
|----------|-------|---------|--------------|---------------|
| **Mostly AI** | Enterprise tabular | Enterprise | Financial/healthcare | Differential privacy |
| **Gretel.ai** | Multi-modal | Free tier + enterprise | LLM-powered, open source core | Synthetic + DP |
| **Tonic.ai** | Developer/DB | Enterprise | Database subsetting | De-identification |
| **Hazy** | Financial | Enterprise | Real-time generation | k-anonymity |
| **Tonic.ai** | Test data | Enterprise | DB integration | Masking |
| **Hazy** | Financial | Enterprise | Real-time streaming | Statistical privacy |
| **Datagen** | Computer vision | Enterprise | 3D rendering | Physical simulation |
| **AI.Reverie** | Simulation | Enterprise | Game engine integration | Simulation-based |
| **Mostly AI** | Tabular | Enterprise | Fairness controls | Multiple DP methods |
| **SDV Cloud** | Tabular | Free/Pro | SDV ecosystem | Statistical |

### Tonic Structural

```python
# Tonic Structural for database synthetic data
# Unique approach: subset and anonymize real databases

# CLI usage
tonic-structural config.yaml

# Python SDK
from tonic import TonicStructural

structural = TonicStructural(
    api_key="your-api-key",
    workspace="my-workspace"
)

# Configure synthetic data generation from database
config = {
    "source_db": {
        "type": "postgresql",
        "host": "mydb.example.com",
        "database": "production",
        "port": 5432
    },
    "tables": ["users", "orders", "products"],
    "generators": {
        "users.email": "EmailGenerator",
        "users.ssn": "US_SSN_Generator",
        "users.name": "NameGenerator",
        "orders.amount": {
            "type": "NumericGenerator",
            "min": 0,
            "max": 10000,
            "distribution": "normal"
        }
    },
    "output": {
        "type": "postgresql",
        "host": "synthetic-db.example.com",
        "database": "synthetic"
    }
}

job = structural.create_job(config)
job.run()
```

### Gretel Navigator (LLM-Powered)

Gretel Navigator represents the next generation of synthetic data generation using large language models:

```python
# Gretel Navigator for complex data generation
from gretel_client import Gretel

gretel = Gretel(project_name="navigator-demo")

# Define generation with natural language
config = {
    "models": [{
        "name": "navigator",
        "params": {
            "data_source": "my_data.csv",
            "num_records": 10000,
            "fields": [
                {
                    "name": "customer_email",
                    "type": "string",
                    "description": "Realistic customer email addresses"
                },
                {
                    "name": "purchase_amount",
                    "type": "float",
                    "description": "Dollar amounts between 10 and 5000",
                    "constraints": {"min": 10, "max": 5000}
                },
                {
                    "name": "product_category",
                    "type": "categorical",
                    "categories": ["Electronics", "Clothing", "Food", "Home"]
                }
            ],
            # Natural language instruction
            "instruction": "Generate realistic e-commerce transaction data "
                          "with realistic email formats and purchase patterns. "
                          "Electronics purchases tend to be higher value."
        }
    }]
}

job = gretel.submit_job(config)
```

---

## Cloud Services

### AWS

```python
# AWS Synthetics for cloud-based synthetic data
import boto3

# Amazon SageMaker Data Wrangler
# (UI-based, but also scriptable)

# AWS Glue DataBrew for data preparation + synthesis
client = boto3.client('databrew')

# Create a recipe for synthetic data generation
response = client.create_project(
    name='synthetic-data-project',
    dataset_name='real-dataset',
    recipe_name='synthesis-recipe'
)

# AWS Bedrock for LLM-based text synthesis
bedrock = boto3.client('bedrock-runtime')

def generate_synthetic_text(prompt, n_samples=100):
    """Generate synthetic text using AWS Bedrock."""
    responses = []
    
    for _ in range(n_samples):
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.8
            })
        )
        
        result = json.loads(response['body'].read())
        responses.append(result['completion'])
    
    return responses
```

### Google Cloud

```python
# Google Cloud Vertex AI for synthetic data
from google.cloud import aiplatform

# BigQuery ML for synthetic data
from google.cloud import bigquery

client = bigquery.Client()

# Generate synthetic data using BigQuery ML
query = """
CREATE OR REPLACE MODEL `my_dataset.synthetic_model`
OPTIONS(
    model_type='BOOSTED_TREE_CLASSIFIER',
    input_label_cols=['target']
) AS
SELECT * FROM `my_dataset.real_data`;

-- Generate synthetic predictions
SELECT *
FROM ML.PREDICT(
    MODEL `my_dataset.synthetic_model`,
    (SELECT * FROM `my_dataset.feature_distribution`)
);
"""
client.query(query)

# Vertex AI for generative models
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Use PaLM for text synthesis
from google.cloud.aiplatform import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@002")

def generate_synthetic_records(prompt, n=100):
    records = []
    for _ in range(n):
        response = model.predict(
            prompt,
            temperature=0.8,
            max_output_tokens=1024,
            top_p=0.95
        )
        records.append(response.text)
    return records
```

### Azure

```python
# Azure Synapse + Azure OpenAI for synthetic data
from azure.identity import DefaultAzureCredential
from azure.ai.textanalytics import TextAnalyticsClient
from openai import OpenAI

# Azure OpenAI for text synthesis
client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)

def generate_synthetic_medical_records(n=1000):
    """Generate synthetic medical records using Azure OpenAI."""
    records = []
    
    for i in range(n):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "You are a medical data generator. Generate realistic "
                          "but synthetic patient records. Use fake names and IDs."
            }, {
                "role": "user",
                "content": f"Generate a synthetic patient record with demographics, "
                          f"vital signs, diagnosis codes (ICD-10), and lab results."
            }],
            temperature=0.9,
            max_tokens=500
        )
        
        records.append(json.loads(
            response.choices[0].message.content
        ))
    
    return records
```

---

## Simulation Engines

### NVIDIA Omniverse

```python
# NVIDIA Omniverse for physics-based synthetic data
# Primarily used for robotics and autonomous driving

# Isaac Sim for robotics synthetic data
# See: https://developer.nvidia.com/isaac-sim

# Python API (simplified)
import omni.isaac

# Configure simulation
config = {
    "scene": "warehouse.usd",
    "robots": ["franka_panda", "ur5"],
    "objects": ["boxes", "cylinders"],
    "lighting": ["natural", "artificial"],
    "camera_configs": [
        {"type": "rgb", "resolution": (1920, 1080)},
        {"type": "depth", "resolution": (640, 480)},
        {"type": "segmentation", "resolution": (640, 480)}
    ],
    "num_variations": 10000
}

# Generate synthetic data
pipeline = omni.isaac.DataGenerator(config)
pipeline.run()
```

### Unity Perception

```csharp
// Unity Perception for synthetic data generation
// Primarily for computer vision training data

using Unity.Perception.Sampling;
using UnityEngine;

public class SyntheticDataGenerator : MonoBehaviour
{
    public Camera sensorCamera;
    public int framesToGenerate = 10000;
    
    void Start()
    {
        // Configure perception
        var perceptionCamera = sensorCamera.gameObject
            .AddComponent<PerceptionCamera>();
        
        // Add annotation labels
        perceptionCamera.AddAnnotation<KeypointAnnotation>();
        perceptionCamera.AddAnnotation<InstanceSegmentationAnnotation>();
        
        // Configure randomization
        var randomizer = gameObject.AddComponent<KeypointRandomizer>();
        randomizer.AddRandomizer<GameObjectSpawner>();
        randomizer.AddRandomizer<MaterialRandomizer>();
        randomizer.AddRandomizer<LightRandomizer>();
        randomizer.AddRandomizer<CameraPositionRandomizer>();
        
        // Generate frames
        StartCoroutine(GenerateFrames());
    }
    
    IEnumerator GenerateFrames()
    {
        for (int i = 0; i < framesToGenerate; i++)
        {
            // Randomize scene
            Randomizerizer.Sample();
            
            // Capture frame
            yield return new WaitForEndOfFrame();
            
            // Annotations are automatically saved
        }
    }
}
```

### CARLA Simulator

```python
# CARLA for autonomous driving synthetic data
import carla
import random

def generate_driving_scenarios(n_scenarios=100):
    """Generate synthetic driving scenarios with CARLA."""
    
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    
    # Get blueprint library
    blueprints = world.get_blueprint_library()
    
    scenarios = []
    
    for i in range(n_scenarios):
        # Spawn ego vehicle
        ego_bp = blueprints.find('vehicle.tesla.model3')
        ego_spawn = random.choice(world.get_map().get_spawn_points())
        ego = world.spawn_actor(ego_bp, ego_spawn)
        
        # Spawn random traffic
        n_vehicles = random.randint(5, 20)
        for _ in range(n_vehicles):
            vehicle_bp = random.choice(blueprints.filter('vehicle'))
            spawn = random.choice(world.get_map().get_spawn_points())
            vehicle = world.spawn_actor(vehicle_bp, spawn)
            vehicle.set_autopilot(True)
        
        # Add weather variation
        weather = carla.WeatherParameters(
            sun_altitude_angle=random.uniform(-90, 90),
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 100),
            precipitation_deposits=random.uniform(0, 100)
        )
        world.set_weather(weather)
        
        # Capture sensor data
        sensor_data = capture_sensors(ego, [
            'camera.rgb.front',
            'camera.rgb.back',
            'lidar.ray_cast',
            'radar.ray_cast'
        ])
        
        scenarios.append({
            'weather': weather,
            'vehicles': n_vehicles,
            'sensor_data': sensor_data
        })
        
        # Cleanup
        ego.destroy()
        for vehicle in world.get_actors().filter('vehicle.*'):
            vehicle.destroy()
    
    return scenarios
```

### NVIDIA DRIVE Sim

For high-fidelity autonomous driving synthetic data:

```python
# NVIDIA DRIVE Sim (simplified API)
import drive_sim as ds

# Configure simulation
sim_config = ds.SimulationConfig(
    map="san_francisco",
    weather="rainy_night",
    traffic_density="heavy",
    ego_vehicle="model_3",
    sensors=[
        ds.Sensor(name="front_camera", type="camera", fov=120),
        ds.Sensor(name="lidar", type="velodyne", channels=128),
        ds.Sensor(name="radar", type="radar", range=200)
    ],
    duration_seconds=600,  # 10 minutes
    fps=30
)

# Run simulation
simulator = ds.Simulator(sim_config)
results = simulator.run()

# Export synthetic data
results.export_to_coco_format("synthetic_driving_data/")
results.export_to_kitti_format("synthetic_driving_data_kitti/")
```

---

## Domain-Specific Toolkits

### Healthcare

```python
# Synthea - Synthetic Patient Generator (Java, but Python-wrappable)
# Generates realistic (but not real) patient medical records

import subprocess
import json

def generate_synthetic_patients(n_patients=1000):
    """Generate synthetic patient records using Synthea."""
    
    # Synthea command line
    cmd = [
        "java", "-jar", "synthea.jar",
        "-p", str(n_patients),
        "-o", "output/synthetic_patients",
        "--exporter.fhir.dstu3", "true",
        "--exporter.csv", "true"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Parse generated FHIR resources
    patients = []
    for f in os.listdir("output/synthetic_patients/fhir"):
        if f.endswith(".json"):
            with open(f"output/synthetic_patients/fhir/{f}") as fh:
                resource = json.load(fh)
                if resource['resourceType'] == 'Patient':
                    patients.append(resource)
    
    return patients
```

### Financial Data

```python
# FinSim for financial transaction synthesis
# (Conceptual example)

class FinancialSynthesizer:
    """Generate synthetic financial transactions."""
    
    def __init__(self, real_transactions):
        self.real = real_transactions
        self.merchant_distribution = self._compute_merchant_dist()
        self.amount_distribution = self._compute_amount_dist()
        self.temporal_pattern = self._compute_temporal_pattern()
    
    def generate(self, n_customers=10000, n_days=365):
        """Generate synthetic financial transactions."""
        transactions = []
        
        for customer_id in range(n_customers):
            # Each customer has spending patterns
            customer_profile = self._generate_customer_profile()
            
            for day in range(n_days):
                # Generate daily transactions based on profile
                n_txns = self._sample_daily_txns(customer_profile)
                
                for _ in range(n_txns):
                    txn = {
                        'customer_id': customer_id,
                        'timestamp': self._generate_timestamp(day),
                        'amount': self._sample_amount(customer_profile),
                        'merchant': self._sample_merchant(),
                        'category': self._sample_category(),
                        'is_fraud': self._sample_fraud(customer_profile)
                    }
                    transactions.append(txn)
        
        return pd.DataFrame(transactions)
```

---

## Evaluation and Benchmarking Tools

### SDV Quality Evaluation

```python
from sdv.evaluation.single_table import (
    evaluate_quality,
    run_diagnostic,
    get_column_pair_plot
)

# Comprehensive quality evaluation
quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# Get overall score
print(f"Overall Quality: {quality_report.get_details()['Quality Score']}")

# Get per-column quality
details = quality_report.get_details()
for _, row in details.iterrows():
    if row['Quality Score'] < 0.8:
        print(f"  Low quality column: {row['Column']} ({row['Quality Score']:.2f})")
```

### SynthCity Evaluation

```python
from synthcity.plugins.core.evaluation import (
    evaluate,
    quality,
    privacy,
    fidelity
)

# SynthCity built-in evaluation
eval_results = evaluate(
    real_data,
    synthetic_data,
    metrics=[
        'quality.detection',
        'quality.stats',
        'privacy.nndistance',
        'privacy.mst',
        'fidelity.kurtosis',
        'fidelity.kstest'
    ]
)

print(eval_results)
```

### Custom Benchmarking Suite

```python
class SyntheticDataBenchmark:
    """Comprehensive benchmarking for synthetic data."""
    
    def __init__(self, real_data, synthetic_data, metadata):
        self.real = real_data
        self.synthetic = synthetic_data
        self.metadata = metadata
        self.results = {}
    
    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite."""
        self.results['statistical'] = self.benchmark_statistical()
        self.results['machine_learning'] = self.benchmark_ml_utility()
        self.results['privacy'] = self.benchmark_privacy()
        self.results['diversity'] = self.benchmark_diversity()
        self.results['downstream'] = self.benchmark_downstream_tasks()
        
        return self._generate_report()
    
    def benchmark_statistical(self):
        """Statistical similarity benchmarks."""
        from scipy.stats import ks_2samp, chi2_contingency
        
        results = {}
        numeric_cols = self.real.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            ks_stat, ks_pval = ks_2samp(
                self.real[col].dropna(),
                self.synthetic[col].dropna()
            )
            results[f'{col}_ks'] = ks_stat
        
        return results
    
    def benchmark_ml_utility(self):
        """Machine learning utility benchmarks."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        target_col = self._find_target()
        if not target_col:
            return {"error": "no target column"}
        
        X_real = self.real.drop(columns=[target_col])
        y_real = self.real[target_col]
        X_synth = self.synthetic.drop(columns=[target_col])
        y_synth = self.synthetic[target_col]
        
        # Train on real, test on real
        model_real = RandomForestClassifier(n_estimators=50, random_state=42)
        real_scores = cross_val_score(model_real, X_real, y_real, cv=3)
        
        # Train on synthetic, test on real
        model_synth = RandomForestClassifier(n_estimators=50, random_state=42)
        model_synth.fit(X_synth, y_synth)
        synth_scores = cross_val_score(model_synth, X_real, y_real, cv=3)
        
        return {
            'real_accuracy': float(np.mean(real_scores)),
            'synthetic_accuracy': float(np.mean(synth_scores)),
            'utility_ratio': float(np.mean(synth_scores) / np.mean(real_scores))
        }
    
    def benchmark_privacy(self):
        """Privacy protection benchmarks."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(self.real.select_dtypes(include=[np.number]).fillna(0))
        
        distances, _ = nn.kneighbors(
            self.synthetic.select_dtypes(include=[np.number]).fillna(0)
        )
        
        return {
            'mean_nn_distance': float(distances[:, 0].mean()),
            'min_nn_distance': float(distances[:, 0].min()),
            'pct_close_to_real': float((distances[:, 0] < 0.05).mean()),
            'privacy_score': float(1 - (distances[:, 0] < 0.05).mean())
        }
    
    def _generate_report(self):
        """Generate human-readable benchmark report."""
        report = []
        report.append("=" * 60)
        report.append("SYNTHETIC DATA BENCHMARK REPORT")
        report.append("=" * 60)
        
        # Overall score
        stat_score = 1 - np.mean(list(self.results['statistical'].values()))
        ml_score = self.results['machine_learning'].get('utility_ratio', 0)
        priv_score = self.results['privacy'].get('privacy_score', 0)
        
        overall = (stat_score * 0.3 + ml_score * 0.4 + priv_score * 0.3)
        
        report.append(f"\nOverall Score: {overall:.2%}")
        report.append(f"  Statistical Fidelity: {stat_score:.2%}")
        report.append(f"  ML Utility: {ml_score:.2%}")
        report.append(f"  Privacy Protection: {priv_score:.2%}")
        
        # Recommendations
        report.append("\nRecommendations:")
        if overall >= 0.8:
            report.append("  ✓ Synthetic data is production-ready")
        elif overall >= 0.6:
            report.append("  ⚠ Consider additional tuning for production use")
        else:
            report.append("  ✗ Review generation approach")
        
        return "\n".join(report)
    
    def _find_target(self):
        """Find target column for ML benchmark."""
        for col in self.real.columns:
            if self.real[col].nunique() <= 20:
                return col
        return None
```

---

## Integration Patterns

### Data Pipeline Integration

```python
class SyntheticDataPipeline:
    """Integrate synthetic data into data pipelines."""
    
    def __init__(self, config):
        self.config = config
        self.synthesizer = self._load_synthesizer()
    
    def run(self, real_data, output_path):
        """Run the synthetic data pipeline."""
        # 1. Preprocess
        processed = self._preprocess(real_data)
        
        # 2. Generate synthetic data
        synthetic = self.synthesizer.sample(
            num_rows=self.config['num_rows']
        )
        
        # 3. Post-process
        post_processed = self._postprocess(synthetic)
        
        # 4. Validate
        validation = self._validate(real_data, post_processed)
        
        if not validation['passed']:
            raise ValueError(
                f"Validation failed: {validation['errors']}"
            )
        
        # 5. Export
        self._export(post_processed, output_path)
        
        return {
            'synthetic_data': post_processed,
            'validation': validation,
            'output_path': output_path
        }
```

### API Service

```python
# FastAPI service for synthetic data generation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI(title="Synthetic Data API")

# Load trained synthesizer
with open("synthesizer.pkl", "rb") as f:
    synthesizer = pickle.load(f)

class GenerateRequest(BaseModel):
    num_rows: int = 1000
    conditions: dict = {}
    seed: int = None

class GenerateResponse(BaseModel):
    data: list[dict]
    metadata: dict

@app.post("/generate", response_model=GenerateResponse)
def generate_synthetic_data(request: GenerateRequest):
    """Generate synthetic data on demand."""
    try:
        if request.seed:
            import random
            random.seed(request.seed)
        
        synthetic = synthesizer.sample(num_rows=request.num_rows)
        
        return GenerateResponse(
            data=synthetic.to_dict(orient='records'),
            metadata={
                'num_rows': len(synthetic),
                'columns': list(synthetic.columns),
                'types': {col: str(synthetic[col].dtype) 
                         for col in synthetic.columns}
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "synthesizer_loaded": True}
```

---

## See Also

- [01-Overview.md](01-Overview.md) — Introduction and market overview
- [02-Core-Topics.md](02-Core-Topics.md) — Core techniques and algorithms
- [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) — Advanced techniques
- [05-Future-Outlook.md](05-Future-Outlook.md) — Future trends and research directions

---

*Last updated: July 4, 2026*
*Part of the AI Knowledge Library — Category 51: Synthetic Data Generation*
