# 04 — Model Portability and Interoperability

## Moving AI Models Between Clouds

This document provides a comprehensive guide to making AI models portable across cloud providers. It covers open standard formats (ONNX, OpenVINO), framework-specific formats (TensorFlow SavedModel, PyTorch), containerization strategies, model registries, migration patterns, and best practices for ensuring model behavior consistency across environments.

---

## Table of Contents

1. Why Model Portability Matters
2. Overview of Model Formats
3. ONNX — Open Neural Network Exchange
4. OpenVINO — Intel's Cross-Platform Format
5. TensorFlow SavedModel
6. PyTorch — TorchScript and Dynamo
7. Hugging Face Model Hub
8. Containerization for Model Portability
9. Model Registry for Multi-Cloud
10. Model Conversion Toolkit
11. Migration Case Studies
12. Portability Testing
13. Fine-Tuning and Portability
14. Embedding Model Portability
15. Large Language Model Portability
16. Governance and Versioning
17. Automation and CI/CD
18. Performance Optimization
19. Common Pitfalls and Solutions
20. Future Trends

---

## 1. Why Model Portability Matters

In a multi-cloud AI strategy, model portability is the ability to take a trained model and deploy it on any cloud provider with minimal modification. Without portability, the multi-cloud strategy collapses into a set of independent single-cloud deployments.

### Business Benefits

- **Provider Independence.** Switch from one cloud AI service to another without retraining models. This protects against price increases, service degradation, or strategic shifts.
- **Cost Optimization.** Deploy models on the cheapest available infrastructure. For example, use AWS Trainium for training and OCI for inference at lower GPU cost.
- **Resilience.** If one cloud's AI service goes down, deploy the same model on another cloud within minutes.
- **Workload Mobility.** Move workloads between development (GCP), production (AWS), and disaster recovery (Azure).
- **Talent Efficiency.** Data scientists train once in their preferred framework; engineers deploy anywhere.

### Portability Spectrum

```
Low Portability ──────────────────────────────────> High Portability

Custom CUDA         Containerized       ONNX Format      Pure Python
kernels             PyTorch model       + ONNX Runtime    + Transformers
Tied to specific    Runs on any GPU     Runs on any       Runs anywhere
GPU architecture    environment         ONNX-compatible   with Python
(no portability)                         runtime          (slowest)
```

---

## 2. Overview of Model Formats

### Format Comparison

| Format | Type | Framework Support | Cloud Support | Performance | Maturity |
|---|---|---|---|---|---|
| **ONNX** | Open standard | PyTorch, TF, JAX, scikit-learn | All major clouds | Near-native | Production |
| **OpenVINO** | Intel open | TF, PyTorch, MXNet | Intel-based clouds | Optimized for Intel | Mature |
| **TensorFlow SavedModel** | Google format | TF, Keras | GCP native, others via conversion | Native on GCP | Production |
| **PyTorch TorchScript** | Meta format | PyTorch | AWS native, others via conversion | Native on AWS | Production |
| **PyTorch Dynamo** | Meta (newer) | PyTorch 2.x | Growing support | Better than TorchScript | Early production |
| **JAX SavedModel** | Google format | JAX, Flax, Haiku | GCP native | Excellent on TPU | Growing |
| **Core ML** | Apple format | PyTorch, TF | Apple devices | Native on Apple | Mature |
| **TFLite** | Google mobile | TF, Keras | Mobile/edge | Optimized for mobile | Mature |
| **PMML** | Open standard | scikit-learn, R | Limited | N/A (classical ML) | Legacy |
| **MLflow Model** | Meta format | Any (wrapper) | Any (brings own runtime) | Varies | Production |

### Format Selection Decision Tree

```
Is the model a deep neural network?
├── Yes → Will it run on multiple hardware types?
│   ├── Yes → ONNX (best portability)
│   └── No → Native format (TorchScript/SavedModel)
└── No (classical ML) → PMML or MLflow Model

Is the model a large language model?
├── Yes → ONNX (for portability) or containerized + API
└── No → Framework-native with ONNX export

Is inference speed critical?
├── Yes → ONNX Runtime or TensorRT
└── No → Containerized + Transformers library
```

---

## 3. ONNX — Open Neural Network Exchange

### What is ONNX?

ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models. Developed jointly by Microsoft, Facebook (Meta), and AWS, it has become the de facto standard for model interoperability.

**Supported by:**
- AWS: SageMaker, Bedrock (via custom model import)
- Azure: Azure OpenAI Service, Azure ML
- GCP: Vertex AI, AI Platform Prediction
- OCI: OCI Data Science, OCI Model Deployment
- IBM: watsonx.ai

### ONNX Export Examples

```python
# Export PyTorch model to ONNX
import torch
import torch.onnx

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = MyModel()
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 10)

# Export to ONNX
torch.onnx.export(
    model,                    # model to export
    dummy_input,              # example input
    "model.onnx",             # output file
    export_params=True,       # store trained weights
    opset_version=17,         # ONNX opset version
    do_constant_folding=True, # optimize constants
    input_names=["input"],    # input names
    output_names=["output"],  # output names
    dynamic_axes={            # variable batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
print("Model exported to ONNX format")
```

```python
# Export TensorFlow model to ONNX with tf2onnx
import tensorflow as tf
import tf2onnx

# Load or create TF model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Convert to ONNX
spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec,
    opset=17,
    output_path=output_path
)
print(f"Model converted to ONNX: {output_path}")
```

### ONNX Runtime Inference

```python
# Cross-cloud inference with ONNX Runtime
import onnxruntime as ort
import numpy as np

# Create inference session (same code on any cloud)
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        "CUDAExecutionProvider",  # GPU
        "CPUExecutionProvider"    # CPU fallback
    ]
)

# Get input/output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = session.run([output_name], {input_name: input_data})
print(f"Output: {outputs[0]}")

# ONNX Runtime is available on all clouds:
# pip install onnxruntime (CPU)
# pip install onnxruntime-gpu (GPU, CUDA 12.x)
```

### ONNX Optimization Tools

```bash
# ONNX Simplifier - reduces model complexity
python -m onnxsim model.onnx model_simplified.onnx

# ONNX Optimizer - graph optimization
python -m onnxoptimizer model.onnx model_optimized.onnx

# Quantization - reduce model size and improve speed
python -m onnxruntime.quantization.preprocess --input model.onnx --output model_quant.onnx

# Shape inference
python -m onnxruntime.shape_inference --input model.onnx --output model_with_shape.onnx
```

### Supported Operations by Opset Version

| Opset | PyTorch | TensorFlow | Status |
|---|---|---|---|
| 9 | 1.3+ | 2.0+ | Legacy |
| 13 | 1.8+ | 2.4+ | Common baseline |
| 15 | 1.10+ | 2.6+ | Recommended |
| 17 | 1.13+ | 2.9+ | Latest stable |
| 18 | 2.0+ | 2.12+ | Cutting edge |
| 19 | 2.1+ | 2.14+ | Preview |
| 20 | 2.2+ | 2.16+ | Future |

**Recommendation:** Target opset 17 for maximum compatibility across all cloud providers.

---

## 4. OpenVINO — Intel's Cross-Platform Format

OpenVINO (Open Visual Inference and Neural Network Optimization) is Intel's toolkit for optimizing and deploying AI models across Intel hardware (CPU, GPU, NPU, FPGA).

### When to Use OpenVINO

- Deploying on Intel-based infrastructure (most cloud instances use Intel Xeon)
- Optimizing for CPU inference (many AI inference workloads run on CPU)
- Edge deployments requiring Intel hardware optimization
- Existing investment in Intel ecosystem

### Model Conversion

```python
# Convert PyTorch model to OpenVINO
from openvino.tools import mo
import torch

model = torch.load("model.pth")
model.eval()

# Convert to OpenVINO IR format
ov_model = mo.convert_model(
    model,
    input_shape=[1, 3, 224, 224],
    mean_values=[123.675, 116.28, 103.53],
    scale_values=[58.395, 57.12, 57.375]
)

# Save OpenVINO model
from openvino.runtime import serialize
serialize(ov_model, "model.xml", "model.bin")
```

### OpenVINO Inference

```python
# OpenVINO inference (optimized for Intel)
from openvino.runtime import Core
import numpy as np

core = Core()
model = core.read_model("model.xml")
compiled = core.compile_model(model, "CPU")

infer_request = compiled.create_infer_request()
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

output = infer_request.infer({"input": input_data})
```

---

## 5. TensorFlow SavedModel

TensorFlow SavedModel is Google's format for saving and serving TensorFlow models. It is the native format for Vertex AI and Google Cloud ML.

### Exporting SavedModel

```python
import tensorflow as tf

# Create and train model
model = tf.keras.Sequential([...])
model.compile(...)
model.fit(x_train, y_train, epochs=10)

# Save as SavedModel
model.save("saved_model/")

# The saved model directory contains:
# - saved_model.pb (graph definition)
# - variables/ (trained weights)
# - assets/ (external files)

# To see the model signature (for inference)
imported = tf.saved_model.load("saved_model/")
print(list(imported.signatures.keys()))
# Output: ['serving_default']
```

### Loading SavedModel on Different Clouds

```python
# Same code runs on GCP Vertex AI, AWS SageMaker, Azure ML
import tensorflow as tf

# Load the model (works anywhere)
model = tf.keras.models.load_model("saved_model/")

# Run inference
predictions = model.predict(input_data)

# For TensorFlow Serving (production)
# docker run -p 8501:8501 --mount type=bind,source=/path/to/saved_model/,target=/models/model -e MODEL_NAME=model tensorflow/serving
```

### Converting SavedModel to Other Formats

```python
# SavedModel → ONNX
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("saved_model/")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17)
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

# SavedModel → TFLite (for edge)
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 6. PyTorch — TorchScript and Dynamo

PyTorch models are natively supported by AWS SageMaker (optimized) and are increasingly portable through TorchScript (legacy) and TorchDynamo (new).

### TorchScript (Stable, Legacy)

```python
import torch

class MyModel(torch.nn.Module):
    def forward(self, x, hidden_state=None):
        # Model logic
        return output

model = MyModel()
model.eval()

# Method 1: Tracing (for models with fixed control flow)
dummy_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Method 2: Scripting (for models with data-dependent control flow)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### TorchDynamo (Modern, Recommended)

```python
# PyTorch 2.x Dynamo - graph capture and optimization
import torch

model = MyModel()
model.eval()

# Compile with Dynamo (automatically captures graph)
compiled_model = torch.compile(model, backend="inductor")

# Use normally - Dynamo captures the graph
output = compiled_model(input_data)

# Export to ONNX via Dynamo (better quality)
torch.onnx.dynamo_export(model, dummy_input).save("model_dynamo.onnx")

# Export to TorchScript via Dynamo
exported = torch.export.export(model, (dummy_input,))
torch.export.save(exported, "model_exported.pt")
```

### Running PyTorch Models on Non-AWS Clouds

```python
# Works on any cloud - just need PyTorch installed
import torch

# Load the saved model
model = torch.jit.load("model_scripted.pt")

# Or load state dict (more portable between PyTorch versions)
model = MyModel()
model.load_state_dict(torch.load("model_state.pth", map_location="cpu"))
model.eval()

# Inference
with torch.no_grad():
    output = model(input_data)
```

---

## 7. Hugging Face Model Hub

Hugging Face has become the central repository for open-source models. Its Transformers library provides framework-agnostic inference that works across clouds.

### Using Hugging Face Models Portably

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer (works on any cloud)
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Exporting Hugging Face Models to Portable Formats

```python
# Convert to ONNX for maximum portability
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load and export to ONNX with optimum
ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save ONNX model and tokenizer
ort_model.save_pretrained("./mistral-onnx/")
tokenizer.save_pretrained("./mistral-onnx/")

# Load for inference (any cloud)
ort_model = ORTModelForCausalLM.from_pretrained("./mistral-onnx/")
tokenizer = AutoTokenizer.from_pretrained("./mistral-onnx/")

# Using ONNX Runtime for inference
from optimum.onnxruntime import ORTModelForCausalLM
inputs = tokenizer("Hello!", return_tensors="pt")
outputs = ort_model.generate(**inputs, max_new_tokens=50)
```

### Hugging Face + ONNX Runtime Optimization

```python
# Quantize Hugging Face model for faster inference
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Create quantizer
quantizer = ORTQuantizer.from_pretrained(ort_model)

# Apply dynamic quantization
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(
    save_dir="./model-quantized/",
    quantization_config=dqconfig
)

# Load quantized model
quantized_model = ORTModelForCausalLM.from_pretrained(
    "./model-quantized/",
    use_merged=True
)
```

---

## 8. Containerization for Model Portability

Containerization is the most practical approach for deploying complex models across clouds. A Docker container encapsulates the model, its dependencies, and the serving infrastructure.

### Dockerfile for Portable Model Serving

```dockerfile
# Multi-stage build for minimal container size
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build stage
FROM base AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM base
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Model files (mounted from volume in production)
ENV MODEL_PATH=/models/model.onnx
ENV HF_HOME=/cache/huggingface

# Copy serving code
COPY serve.py /app/
RUN mkdir -p /cache/huggingface

EXPOSE 8080
CMD ["uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Requirements File for Maximum Portability

```txt
# requirements.txt - Works across all clouds
torch>=2.2.0,<3.0.0
transformers>=4.38.0,<5.0.0
onnxruntime-gpu>=1.17.0,<2.0.0  # or onnxruntime for CPU
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
pydantic>=2.5.0,<3.0.0
sentence-transformers>=2.2.0,<3.0.0  # for embeddings
numpy>=1.24.0,<2.0.0
```

### Serving Code (Cloud-Agnostic)

```python
# serve.py - Cloud-agnostic model serving
import os
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Portable Model Server")

# Load model once at startup
class ModelServer:
    def __init__(self):
        model_path = os.environ.get("MODEL_PATH", "/models/model.onnx")
        
        # Try GPU first, fall back to CPU
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: input_data})

model_server = ModelServer()

class PredictRequest(BaseModel):
    data: list
    shape: list[int] | None = None

class PredictResponse(BaseModel):
    predictions: list
    model: str
    provider: str

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        input_array = np.array(request.data, dtype=np.float32)
        if request.shape:
            input_array = input_array.reshape(request.shape)
        
        output = model_server.predict(input_array)
        
        return PredictResponse(
            predictions=output[0].tolist(),
            model=os.environ.get("MODEL_NAME", "unknown"),
            provider=model_server.session.get_providers()[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": os.environ.get("MODEL_NAME", "unknown")}
```

### Multi-Cloud Container Registry

```bash
# Build once, deploy to any cloud
docker build -t my-ai-model:1.0.0 .

# Push to all cloud registries
docker tag my-ai-model:1.0.0 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ai-model:1.0.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ai-model:1.0.0

docker tag my-ai-model:1.0.0 myregistry.azurecr.io/my-ai-model:1.0.0
docker push myregistry.azurecr.io/my-ai-model:1.0.0

docker tag my-ai-model:1.0.0 gcr.io/my-project/my-ai-model:1.0.0
docker push gcr.io/my-project/my-ai-model:1.0.0

# Or use a multi-cloud compatible registry
# Docker Hub, GitHub Container Registry, GitLab Container Registry
docker tag my-ai-model:1.0.0 ghcr.io/my-org/my-ai-model:1.0.0
docker push ghcr.io/my-org/my-ai-model:1.0.0
```

### Kubernetes Deployment (Cloud-Agnostic)

```yaml
# k8s-deployment.yaml - Works on EKS, AKS, GKE
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-server
  labels:
    app: ai-model
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-model
  template:
    metadata:
      labels:
        app: ai-model
        version: "1.0.0"
    spec:
      containers:
      - name: model-server
        image: ghcr.io/my-org/my-ai-model:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/models/model.onnx"
        - name: MODEL_NAME
          value: "my-model-v1"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            memory: "8Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

---

## 9. Model Registry for Multi-Cloud

A model registry is the central source of truth for model metadata, versions, and artifacts. It enables consistent model deployment across clouds.

### MLflow Model Registry

```python
import mlflow
import mlflow.pyfunc

# Set tracking URI (shared across clouds)
mlflow.set_tracking_uri("https://mlflow.organization.com")

# Log a model
with mlflow.start_run():
    mlflow.log_param("model_type", "bert-classifier")
    mlflow.log_metric("accuracy", 0.95)
    
    # Log the model (stores artifact in MLflow artifact store)
    mlflow.pyfunc.log_model(
        "model",
        python_model=wrapped_model,
        artifacts={"model_path": "model.onnx"},
        registered_model_name="bert-classifier"
    )

# Register a new version
client = mlflow.tracking.MlflowClient()
client.create_registered_model("bert-classifier")
result = client.create_model_version(
    name="bert-classifier",
    source="s3://mlflow-artifacts/1/abc123/artifacts/model",
    run_id="abc123",
    description="v1 - ONNX format, accuracy 0.95"
)

# Transition version to production
client.transition_model_version_stage(
    name="bert-classifier",
    version=1,
    stage="Production"
)

# Deploy from registry to any cloud
model = mlflow.pyfunc.load_model(
    model_uri="models:/bert-classifier/Production"
)
predictions = model.predict(input_data)
```

### MLflow with Multi-Cloud Artifact Store

```python
# Configure MLflow to use multi-cloud artifact stores
# AWS S3
mlflow.set_tracking_uri("https://mlflow.organization.com")
mlflow.set_experiment("multi-cloud-models")

# Store artifacts in S3 (accessible from any cloud)
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.us-east-1.amazonaws.com"

# Or use a cloud-agnostic artifact store
# MinIO (self-hosted, S3-compatible)
# Artifactory
# Google Cloud Storage
# Azure Blob Storage
```

### DVC (Data Version Control) for Model Portability

```bash
# Initialize DVC
dvc init

# Add model files to DVC
dvc add models/model.onnx

# Configure remote storage (multiple clouds)
dvc remote add -d aws_remote s3://my-bucket/dvc-store
dvc remote add azure_remote azure://mycontainer/dvc-store
dvc remote add gcs_remote gs://my-bucket/dvc-store

# Push to all remotes
dvc push -r aws_remote
dvc push -r azure_remote
dvc push -r gcs_remote

# Pull from any remote
dvc pull -r aws_remote  # On AWS
dvc pull -r azure_remote  # On Azure
```

### Custom Model Registry (Simplified)

```python
# model_registry.py - Simplified multi-cloud model registry
import json
import boto3
import os
from datetime import datetime
from typing import Optional

class ModelRegistry:
    """Lightweight model registry for multi-cloud deployment"""
    
    def __init__(self, backend: str = "s3", bucket: str = "model-registry"):
        self.backend = backend
        self.bucket = bucket
        
        if backend == "s3":
            self.client = boto3.client("s3")
        elif backend == "gcs":
            from google.cloud import storage
            self.client = storage.Client()
        # Add Azure Blob support
    
    def register_model(
        self,
        name: str,
        version: str,
        artifact_path: str,
        model_format: str,
        metadata: dict
    ):
        """Register a model version in the registry"""
        registry_entry = {
            "name": name,
            "version": version,
            "artifact_path": artifact_path,
            "model_format": model_format,
            "metadata": metadata,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "staging"
        }
        
        # Store registry entry
        key = f"registry/{name}/{version}/metadata.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(registry_entry)
        )
        
        # Update latest version pointer
        self.client.put_object(
            Bucket=self.bucket,
            Key=f"registry/{name}/latest",
            Body=version.encode()
        )
    
    def get_model(self, name: str, version: Optional[str] = None) -> dict:
        """Get model metadata"""
        if version is None:
            # Get latest version
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=f"registry/{name}/latest"
            )
            version = response["Body"].read().decode()
        
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=f"registry/{name}/{version}/metadata.json"
        )
        return json.loads(response["Body"].read())
    
    def promote_to_production(self, name: str, version: str):
        """Promote a model version to production"""
        key = f"registry/{name}/production"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=version.encode()
        )

# Usage
registry = ModelRegistry(backend="s3", bucket="my-org-model-registry")
registry.register_model(
    name="sentiment-classifier",
    version="1.2.0",
    artifact_path="s3://model-artifacts/sentiment/v1.2.0/model.onnx",
    model_format="onnx",
    metadata={"accuracy": 0.96, "framework": "pytorch", "dataset": "imdb"}
)
```

---

## 10. Model Conversion Toolkit

### Automated Conversion Pipeline

```python
# convert.py — Automated model format conversion
import argparse
import os
from pathlib import Path

class ModelConverter:
    SUPPORTED_CONVERSIONS = {
        "pytorch:onnx": "convert_pytorch_to_onnx",
        "tensorflow:onnx": "convert_tf_to_onnx",
        "pytorch:openvino": "convert_pytorch_to_openvino",
        "tensorflow:openvino": "convert_tf_to_openvino",
        "onnx:tensorflow": "convert_onnx_to_tf",
        "onnx:pytorch": "convert_onnx_to_pytorch",
    }
    
    def convert(self, input_path: str, input_format: str, output_format: str, output_path: str):
        conversion_key = f"{input_format}:{output_format}"
        if conversion_key not in self.SUPPORTED_CONVERSIONS:
            raise ValueError(f"Unsupported conversion: {conversion_key}")
        
        converter = getattr(self, self.SUPPORTED_CONVERSIONS[conversion_key])
        converter(input_path, output_path)
    
    def convert_pytorch_to_onnx(self, input_path: str, output_path: str):
        import torch
        model = torch.jit.load(input_path)
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy, output_path, opset_version=17)
    
    def convert_tf_to_onnx(self, input_path: str, output_path: str):
        import tensorflow as tf
        import tf2onnx
        model = tf.keras.models.load_model(input_path)
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17)
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
    
    def convert_onnx_to_tf(self, input_path: str, output_path: str):
        import onnx
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(input_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output_path)

# CLI usage
# python convert.py --input model.pth --input-format pytorch --output-format onnx --output model.onnx
```

### Conversion Validation

```python
# validate_conversion.py — Verify model output consistency after conversion
import numpy as np
import onnxruntime as ort

def validate_conversion(
    original_model,  # Original framework model
    converted_path: str,  # Path to converted ONNX model
    test_inputs: list[np.ndarray],
    tolerance: float = 1e-3
) -> bool:
    """Validate that conversion preserves model behavior"""
    
    # Get original outputs
    original_outputs = [original_model(x) for x in test_inputs]
    
    # Get converted outputs
    session = ort.InferenceSession(converted_path)
    input_name = session.get_inputs()[0].name
    
    all_match = True
    for i, test_input in enumerate(test_inputs):
        converted_output = session.run(None, {input_name: test_input})[0]
        original_output = original_outputs[i]
        
        if isinstance(original_output, list):
            original_output = original_output[0]
        
        if isinstance(original_output, np.ndarray):
            diff = np.max(np.abs(converted_output - original_output))
            if diff > tolerance:
                print(f"Test {i}: FAILED (max diff: {diff:.6f})")
                all_match = False
            else:
                print(f"Test {i}: PASSED (max diff: {diff:.6f})")
    
    return all_match
```

---

## 11. Migration Case Studies

### Case Study 1: NLP Model from GCP to AWS

**Scenario:** A sentiment analysis model trained on GCP using TensorFlow and deployed on Vertex AI needs to be migrated to AWS Bedrock.

**Steps:**
1. Export TensorFlow SavedModel from Vertex AI Model Registry
2. Validate model outputs with test dataset
3. Convert SavedModel to ONNX format using tf2onnx
4. Test ONNX model equivalence (output diff < 1e-5)
5. Upload ONNX model to AWS Bedrock Custom Model Import
6. Create Bedrock endpoint with provisioned throughput
7. Validate latency and cost on AWS
8. Update DNS to route 10% of traffic to AWS, monitoring errors
9. Gradually increase traffic to 100%
10. Decommission GCP deployment after stabilization

**Duration:** 2 weeks total (3 days conversion, 4 days testing, 7 days gradual migration)

**Results:**
- Latency: Increased 15% (ONNX overhead) → mitigated by upgrading to larger instance
- Cost: Reduced 35% (AWS lower GPU pricing)
- Accuracy: 99.97% identical (within tolerance)

### Case Study 2: LLM from Azure to GCP

**Scenario:** A custom fine-tuned Llama 3.1 model deployed on Azure ML needs to be migrated to GCP Vertex AI.

**Challenges:**
- Fine-tuning was done with Azure-specific tooling
- Model format was Hugging Face (PyTorch) — already portable
- Need to maintain the same serving infrastructure

**Steps:**
1. Download fine-tuned model from Azure ML Model Registry
2. Convert to ONNX with optimum (for both Azure and GCP compatibility)
3. Containerize with Docker (transformers + ONNX Runtime)
4. Deploy container to GCP Vertex AI Prediction
5. Set up GCP load balancer with same endpoint structure
6. A/B test between Azure and GCP (10% traffic on GCP)
7. Full cutover after 1 week of successful testing

**Duration:** 1 week

**Results:**
- Zero accuracy degradation (same model weights)
- 10% lower latency on GCP (better GPU availability)
- 25% cost reduction (GCP committed use discounts)

### Case Study 3: Computer Vision Model to OCI

**Scenario:** A production ResNet-50 classification model needs to run on OCI for a new customer region.

**Steps:**
1. Export PyTorch model to ONNX
2. Optimize ONNX model for OCI's GPU architecture (Ampere)
3. Deploy using OCI Data Science Model Deployment
4. Configure autoscaling on OCI
5. Set up multi-cloud load balancer routing based on customer region

**Duration:** 3 days

**Results:**
- Model ran on OCI with no code changes
- 40% cost savings vs. hyperscaler
- Same latency performance

---

## 12. Portability Testing

### Automated Portability Test Suite

```python
# test_portability.py
import unittest
import numpy as np
import onnxruntime as ort

class TestModelPortability(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model_path = "models/model.onnx"
        cls.session = ort.InferenceSession(cls.model_path)
        cls.input_name = cls.session.get_inputs()[0].name
        cls.output_name = cls.session.get_outputs()[0].name
        cls.input_shape = cls.session.get_inputs()[0].shape
    
    def test_basic_inference(self):
        """Test basic forward pass works"""
        dummy_input = np.random.randn(1, *self.input_shape[1:]).astype(np.float32)
        output = self.session.run([self.output_name], {self.input_name: dummy_input})
        self.assertIsNotNone(output)
        self.assertGreater(len(output[0][0]), 0)
    
    def test_batch_inference(self):
        """Test batching works"""
        batch_size = 8
        batch_input = np.random.randn(batch_size, *self.input_shape[1:]).astype(np.float32)
        output = self.session.run([self.output_name], {self.input_name: batch_input})
        self.assertEqual(output[0].shape[0], batch_size)
    
    def test_different_providers(self):
        """Test model runs on both CPU and GPU"""
        cpu_session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )
        gpu_session = ort.InferenceSession(
            self.model_path,
            providers=["CUDAExecutionProvider"]
        )
        
        test_input = np.random.randn(1, *self.input_shape[1:]).astype(np.float32)
        cpu_output = cpu_session.run([self.output_name], {self.input_name: test_input})
        gpu_output = gpu_session.run([self.output_name], {self.input_name: test_input})
        
        # Outputs should be very close (small diff due to numerical precision)
        max_diff = np.max(np.abs(cpu_output[0] - gpu_output[0]))
        self.assertLess(max_diff, 1e-4)
    
    def test_output_consistency(self):
        """Test deterministic output"""
        session1 = ort.InferenceSession(self.model_path)
        session2 = ort.InferenceSession(self.model_path)
        
        test_input = np.random.randn(1, *self.input_shape[1:]).astype(np.float32)
        output1 = session1.run([self.output_name], {self.input_name: test_input})
        output2 = session2.run([self.output_name], {self.input_name: test_input})
        
        np.testing.assert_array_almost_equal(output1[0], output2[0], decimal=6)
    
    def test_model_size(self):
        """Test model file size is reasonable"""
        import os
        size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        self.assertGreater(size_mb, 0.1)  # Model should be non-trivial
```

### Cloud-Specific Validation

```python
# validate_on_clouds.py — Test model on each cloud provider
import asyncio
import aiohttp

CLOUD_ENDPOINTS = {
    "aws": "https://ai-gateway.aws.internal/predict",
    "azure": "https://ai-gateway.azure.internal/predict",
    "gcp": "https://ai-gateway.gcp.internal/predict",
    "oci": "https://ai-gateway.oci.internal/predict",
}

async def validate_on_cloud(provider: str, endpoint: str, test_data: dict) -> dict:
    """Validate model inference on a specific cloud provider"""
    async with aiohttp.ClientSession() as session:
        start = asyncio.get_event_loop().time()
        async with session.post(endpoint, json=test_data) as response:
            latency = (asyncio.get_event_loop().time() - start) * 1000
            result = await response.json()
            
    return {
        "provider": provider,
        "status": response.status,
        "latency_ms": latency,
        "predictions": result.get("predictions"),
        "model": result.get("model")
    }

async def cross_cloud_validation(test_data: dict):
    """Run validation across all clouds"""
    tasks = [
        validate_on_cloud(provider, endpoint, test_data)
        for provider, endpoint in CLOUD_ENDPOINTS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    predictions = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"Cloud failed: {result}")
        else:
            predictions[result["provider"]] = result["predictions"]
            print(f"{result['provider']}: {result['status']}, {result['latency_ms']:.1f}ms")
    
    # Check consistency across clouds
    if len(predictions) >= 2:
        first_provider = list(predictions.keys())[0]
        for provider, pred in predictions.items():
            if provider != first_provider:
                diff = np.max(np.abs(np.array(predictions[first_provider]) - np.array(pred)))
                print(f"Diff {first_provider} vs {provider}: {diff:.6f}")
    
    return results
```

---

## 13. Fine-Tuning and Portability

### Portable Fine-Tuning Workflow

```python
# 1. Fine-tune with Hugging Face (cloud-agnostic)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    push_to_hub=False,  # Don't push to HF hub; we'll export manually
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune
trainer.train()

# 2. Save in portable format
model.save_pretrained("./fine-tuned-model")

# 3. Convert to ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification
ort_model = ORTModelForSequenceClassification.from_pretrained(
    "./fine-tuned-model",
    export=True
)
ort_model.save_pretrained("./fine-tuned-onnx/")

# 4. Upload to model registry (accessible from any cloud)
# s3://model-registry/fine-tuned-bert/v1/
```

### Fine-Tuning Considerations for Portability

| Technique | Portable? | Notes |
|---|---|---|
| **Full fine-tuning** | ✅ Yes | Save model weights → convert to ONNX |
| **LoRA/QLoRA** | ✅ Yes | Merge LoRA weights before conversion |
| **Prefix tuning** | ✅ Yes | Can be exported as part of model |
| **Adapter layers** | ✅ Yes | Separate adapter weights, merge after |
| **RLHF/DPO** | ⚠️ Partial | Reward model needs separate export |
| **DeepSpeed** | ⚠️ Partial | ZeRO stage 3 sharding must be collapsed |
| **FSDP** | ⚠️ Partial | Must consolidate shards before export |
| **PEFT + quantization** | ⚠️ Partial | Quantization may not be ONNX-compatible |

---

## 14. Embedding Model Portability

Embedding models present unique portability challenges because they must produce consistent vector representations across clouds.

### Consistent Embedding Generation

```python
# Ensure embeddings are consistent across clouds
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the same model on any cloud
model = SentenceTransformer("all-MiniLM-L6-v2")

# Normalize embeddings (critical for consistency)
def get_normalized_embedding(text: str) -> np.ndarray:
    embedding = model.encode(text)
    return embedding / np.linalg.norm(embedding)

# Cosine similarity will be the same regardless of cloud
emb1 = get_normalized_embedding("Hello world")
emb2 = get_normalized_embedding("Hi there")
similarity = np.dot(emb1, emb2)  # Cosine for normalized vectors
```

### Embedding Dimension Alignment

| Provider | Default Model | Dimensions | Consistent? |
|---|---|---|---|
| **Azure OpenAI** | text-embedding-3-large | 3072 (default), 256+ configurable | ✅ Configurable |
| **Azure OpenAI** | text-embedding-3-small | 1536 (default), 512 configurable | ✅ Configurable |
| **AWS Bedrock** | Titan Embeddings G1 | 1536 | ✅ Fixed |
| **GCP Vertex AI** | textembedding-gecko | 768 | ✅ Fixed |
| **GCP Vertex AI** | multimodalembedding | 1408 | ✅ Fixed |
| **OCI AI** | Cohere Embed | 1024 or 4096 | ✅ Configurable |
| **IBM watsonx** | Granite Embedding | 768 | ✅ Fixed |

**Strategy for multi-cloud embedding portability:** 
1. Choose a fixed dimension (e.g., 768 or 1024)
2. If needed, project embeddings to lower dimension using PCA or a learned projection
3. Normalize all embeddings before storing
4. Use the same vector database regardless of embedding provider

---

## 15. Large Language Model Portability

LLMs present the greatest portability challenge due to their size, specialized hardware requirements, and framework-specific optimizations.

### LLM Portability Options

| Approach | Portability | Performance | Complexity |
|---|---|---|---|
| **ONNX Runtime** | High | Good (some overhead) | Medium |
| **Containerized + vLLM** | Medium | Excellent | Medium |
| **Containerized + TGI** | Medium | Excellent | Medium |
| **AWS SageMaker + DJL** | Low (AWS-focused) | Excellent | Low |
| **Hugging Face TGI** | Medium | Good | Low |
| **TensorRT-LLM** | Medium | Excellent (NVIDIA) | High |

### Portable LLM Deployment with vLLM

```python
# vllm_serve.py — Cloud-agnostic LLM serving with vLLM
import os
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize LLM (works on any cloud with GPU)
model_path = os.environ.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
llm = LLM(
    model=model_path,
    tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
    dtype="bfloat16",
    max_model_len=8192,
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    text: str
    usage: dict

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return GenerateResponse(
            text=generated_text,
            usage={
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Running vLLM on Different Clouds

```bash
# Same container runs on any cloud
# AWS EKS
kubectl apply -f k8s-vllm-deployment.yaml

# Azure AKS
kubectl apply -f k8s-vllm-deployment.yaml

# GCP GKE
kubectl apply -f k8s-vllm-deployment.yaml

# OCI OKE
kubectl apply -f k8s-vllm-deployment.yaml

# Only difference: GPU node pool configuration:
# AWS: nvidia.com/gpu resource
# Azure: nvidia.com/gpu or azure.com/nd-h100-v5
# GCP: nvidia.com/gpu resource
# OCI: oci.oraclecloud.com/gpu resource
```

---

## 16. Governance and Versioning

### Model Lineage Tracking

```python
# model_lineage.yaml
models:
  - name: sentiment-classifier
    versions:
      - version: 1.0.0
        source_framework: pytorch
        export_format: onnx
        opset: 17
        trained_on: gcp_vertex_ai
        deployed_on: [aws_bedrock, azure_ml]
        accuracy: 0.95
        converted_by: auto-convert-pipeline-v2
        validation_passed: true
        commit: abc123def456
        trained_at: 2026-01-15T10:30:00Z
        
      - version: 1.1.0
        source_framework: pytorch
        export_format: onnx
        opset: 18
        trained_on: aws_sagemaker
        deployed_on: [aws_bedrock, azure_ml, gcp_vertex_ai]
        accuracy: 0.97
        converted_by: auto-convert-pipeline-v3
        validation_passed: true
        commit: ghi789jkl012
        trained_at: 2026-03-20T14:00:00Z
```

### Git LFS for Model Files

```bash
# Track model files with Git LFS (portable across CI/CD systems)
git lfs track "*.onnx"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.safetensors"

# Add and commit
git add model.onnx .gitattributes
git commit -m "Add ONNX model v1.1.0"

# Push (works with GitHub, GitLab, Bitbucket)
git push origin main
```

---

## 17. Automation and CI/CD

### GitHub Actions for Model Portability

```yaml
# .github/workflows/model-portability.yml
name: Model Portability Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'conversion/**'

jobs:
  convert-and-validate:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target-format: [onnx, openvino]
    
    steps:
    - uses: actions/checkout@v4
      with:
        lfs: true
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install torch torchvision onnx onnxruntime tf2onnx openvino-dev optimum
    
    - name: Convert model
      run: |
        python conversion/convert.py \
          --input models/source/model.pth \
          --input-format pytorch \
          --output-format ${{ matrix.target-format }} \
          --output models/converted/model.${{ matrix.target-format }}
    
    - name: Validate conversion
      run: |
        python conversion/validate.py \
          --original models/source/model.pth \
          --converted models/converted/model.${{ matrix.target-format }}
    
    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: model-${{ matrix.target-format }}
        path: models/converted/model.${{ matrix.target-format }}
    
    - name: Push to model registry
      run: |
        python registry/push.py \
          --model-path models/converted/model.${{ matrix.target-format }} \
          --format ${{ matrix.target-format }} \
          --name my-model \
          --version ${{ github.sha }}
```

### Multi-Cloud Deployment Pipeline

```yaml
# Deploy model to all clouds
deploy:
  steps:
    - name: Deploy to AWS SageMaker
      run: |
        aws sagemaker create-model \
          --model-name my-model-v1 \
          --primary-container Image=123456789.dkr.ecr.us-east-1.amazonaws.com/my-model:1.0.0
        aws sagemaker create-endpoint-config \
          --endpoint-config-name my-model-v1-config \
          --production-variants VariantName=default,ModelName=my-model-v1,InitialInstanceCount=2
        aws sagemaker create-endpoint \
          --endpoint-name my-model-v1 --endpoint-config-name my-model-v1-config
    
    - name: Deploy to Azure ML
      run: |
        az ml model create --name my-model --version 1 --path ./model
        az ml endpoint create --name my-model-endpoint --type managed
    
    - name: Deploy to GCP Vertex AI
      run: |
        gcloud ai models upload \
          --region=us-central1 \
          --display-name=my-model-v1 \
          --container-image-uri=gcr.io/my-project/my-model:1.0.0
        
        gcloud ai endpoints deploy-model \
          --region=us-central1 \
          --endpoint=my-model-endpoint \
          --model=my-model-v1 \
          --traffic-split=0=100
```

---

## 18. Performance Optimization

### ONNX Runtime Optimization Levels

| Optimization | Speedup | Description |
|---|---|---|
| **Graph optimization (basic)** | 10–30% | Operator fusion, constant folding |
| **Graph optimization (extended)** | 20–50% | Layout optimization, transpose optimization |
| **INT8 quantization** | 2–4x | Reduce precision to 8-bit integer |
| **FP16 quantization** | 1.5–2x | Half precision floating point |
| **NUMA optimization** | 10–20% | CPU core affinity tuning |
| **TensorRT integration** | 2–3x (NVIDIA) | NVIDIA-specific optimizations |
| **OpenVINO integration** | 1.5–2x (Intel) | Intel-specific optimizations |

### Optimization Configuration

```python
# ONNX Runtime optimization options
import onnxruntime as ort

# Session options
sess_options = ort.SessionOptions()

# Enable all graph optimizations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable parallel execution
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.inter_op_num_threads = 4
sess_options.intra_op_num_threads = 4

# Enable memory pattern optimization
sess_options.enable_cpu_mem_arena = True
sess_options.enable_mem_reuse = True

# Create session with optimizations
session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options,
    providers=[
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }),
        "CPUExecutionProvider"
    ]
)
```

---

## 19. Common Pitfalls and Solutions

### Pitfall 1: Opset Version Mismatch

**Problem:** Model exported with opset version not supported by target cloud's ONNX Runtime.

**Solution:** Pin to opset 17 (widely supported). Check cloud provider's supported ONNX Runtime version.

```python
# Check opset compatibility
import onnx
model = onnx.load("model.onnx")
print(f"Opset version: {model.opset_import[0].version}")
```

### Pitfall 2: Dynamic Shapes Not Handled

**Problem:** Model fails with variable-length inputs.

**Solution:** Use dynamic axes in ONNX export.

```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
)
```

### Pitfall 3: Custom Operators

**Problem:** Model uses custom CUDA kernels or operations not in ONNX standard.

**Solution:** 
- Replace custom ops with standard equivalents before export
- Use ONNX Custom Operator support if necessary
- Fall back to containerized deployment with native framework

### Pitfall 4: Numerical Differences Between Frameworks

**Problem:** Outputs differ slightly between original framework and ONNX Runtime.

**Solution:**
- Normalize outputs after conversion
- Verify tolerance is acceptable for your use case
- Use FP32 instead of FP16 for maximum precision
- Test with representative real-world data, not just random inputs

### Pitfall 5: Tokenizer Incompatibility

**Problem:** LLM tokenizer behaves differently across environments.

**Solution:**
- Bundle tokenizer with model in container
- Pin tokenizer version and configuration
- Test tokenizer outputs across environments

```python
# Always save and load tokenizer with model
tokenizer.save_pretrained("./model-with-tokenizer/")
model.save_pretrained("./model-with-tokenizer/")

# Load together
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("./model-with-tokenizer/")
model = AutoModelForCausalLM.from_pretrained("./model-with-tokenizer/")
```

### Pitfall 6: Large Model Serialization

**Problem:** Models > 5GB are slow to load and may hit provider limits.

**Solution:**
- Use safetensors format (faster loading, safer)
- Shard large models across multiple files
- Use progressive loading (load first layer, then rest)

```python
# Save with safetensors for faster loading
model.save_pretrained(
    "./large-model/",
    safe_serialization=True,
    max_shard_size="2GB"
)
```

---

## 20. Future Trends

### ONNX 2.0 (Projected)

ONNX 2.0 is expected to bring:
- Native support for dynamic control flow (loops, conditionals)
- Improved LLM operation coverage
- Better interop with JAX and Flax
- Built-in quantization and compression
- Standardized model serving interface

### Open Standard for Agents

The industry is converging on open standards for agent interoperability:
- OpenAPI/Swagger for tool definitions
- Standard agent communication protocol
- Portable agent state serialization

### Model Compression Standardization

BitNet and 1-bit LLM architectures make portability trivial (model fits in any environment).

### WebAssembly for AI

WASM-based model execution promises true portability across any platform (cloud, edge, browser).

---

## References

- ONNX Specification: https://github.com/onnx/onnx
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- OpenVINO: https://github.com/openvinotoolkit/openvino
- MLflow: https://mlflow.org
- Hugging Face Optimum: https://huggingface.co/docs/optimum
- PyTorch Export: https://pytorch.org/docs/stable/export.html
- TensorFlow SavedModel: https://www.tensorflow.org/guide/saved_model
- vLLM: https://github.com/vllm-project/vllm
- ad-hoc model portability tools: https://github.com/onnx/tutorials
