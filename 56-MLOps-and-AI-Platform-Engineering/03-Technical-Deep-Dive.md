# MLOps & AI Platform Engineering — Technical Deep-Dive

> **Category:** 56 — MLOps & AI Platform Engineering  
> **Last Updated:** July 2026  
> **Cross-references:** [02-Core-Topics.md](02-Core-Topics.md), [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md)

---

## Table of Contents

1. [Kubernetes for AI Workloads](#1-kubernetes-for-ai-workloads)
2. [GPU Operator & NVIDIA Stack](#2-gpu-operator--nvidia-stack)
3. [Model Serving Deep-Dive](#3-model-serving-deep-dive)
4. [Distributed Training Implementation](#4-distributed-training-implementation)
5. [Feature Store Implementation](#5-feature-store-implementation)
6. [Monitoring Stack Setup](#6-monitoring-stack-setup)
7. [Cost Engineering](#7-cost-engineering)
8. [Security Implementation](#8-security-implementation)
9. [Platform as Code](#9-platform-as-code)
10. [Disaster Recovery & High Availability](#10-disaster-recovery--high-availability)

---

## 1. Kubernetes for AI Workloads

### Cluster Architecture

```yaml
# AI-optimized Kubernetes cluster
apiVersion: eks.amazonaws.com/v1
kind: ClusterConfig
metadata:
  name: ai-production
  region: us-west-2

nodeGroups:
  # GPU node group for training
  - name: gpu-training
    instanceTypes:
      - p5.48xlarge    # 8x H100 80GB
      - p4d.24xlarge   # 8x A100 40GB
    minSize: 0
    maxSize: 50
    desiredSize: 4
    labels:
      node-type: gpu
      workload: training
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    
  # GPU node group for inference
  - name: gpu-inference
    instanceTypes:
      - g5.12xlarge    # 4x A10G
      - g6.xlarge      # 1x L4
    minSize: 2
    maxSize: 100
    desiredSize: 8
    labels:
      node-type: gpu
      workload: inference
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    
  # CPU node group for data processing
  - name: cpu-processing
    instanceTypes:
      - m5.4xlarge
      - m5.8xlarge
    minSize: 3
    maxSize: 20
    desiredSize: 5
    labels:
      node-type: cpu
      workload: processing
```

### GPU Scheduling with NVIDIA GPU Operator

```yaml
# GPU Operator Helm values
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: gpu-operator
  namespace: gpu-operator
spec:
  interval: 5m
  chart:
    spec:
      chart: gpu-operator
      version: "24.6.0"
      sourceRef:
        kind: HelmRepository
        name: nvidia
  values:
    driver:
      enabled: true
      version: "535.129.03"
    
    toolkit:
      enabled: true
      version: "1.14.0"
    
    devicePlugin:
      enabled: true
    
    gfd:
      enabled: true  # GPU Feature Discovery
    
    mig:
      enabled: true  # Multi-Instance GPU
      strategy: single  # or mixed
    
    dcgm:
      enabled: true  # Data Center GPU Manager
      exporter:
        enabled: true
    
    # Time-sharing for inference
    timeSlicing:
      enabled: true
      resources:
        - name: nvidia.com/gpu
          replicas: 4  # 4 virtual GPUs per physical
```

### Resource Quotas for AI

```yaml
# Resource quotas to prevent GPU waste
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ai-gpu-quota
  namespace: ai-team
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "32"
    requests.cpu: "128"
    requests.memory: "512Gi"
    persistentvolumeclaims: "20"
```

### Node Affinity & Pod Placement

```yaml
# Schedule training job on appropriate GPU
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-fine-tune
spec:
  template:
    spec:
      nodeSelector:
        node-type: gpu
        workload: training
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-H100-80GB-HBM3
                      - NVIDIA-A100-SXM4-80GB
      containers:
        - name: trainer
          image: company/llm-trainer:latest
          resources:
            limits:
              nvidia.com/gpu: "4"
              memory: "256Gi"
              cpu: "16"
            requests:
              nvidia.com/gpu: "4"
              memory: "256Gi"
              cpu: "16"
```

---

## 2. GPU Operator & NVIDIA Stack

### NVIDIA GPU Feature Discovery

```python
# Query GPU features for scheduling
import subprocess
import json

def get_gpu_features():
    """Get GPU features from NVIDIA DCGM"""
    result = subprocess.run(
        ["dcgm-exporter", "-format", "json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# GPU labels available for scheduling:
# nvidia.com/gpu.product: "NVIDIA-H100-80GB-HBM3"
# nvidia.com/gpu.memory: "81920"
# nvidia.com/gpu.count: "8"
# nvidia.com/gpu.family: "Hopper"
# nvidia.com/gpu.machine: "p5.48xlarge"
```

### Dynamic GPU Partitioning (MIG)

```bash
# Enable MIG on A100/H100
# Create MIG instances
nvidia-smi -i 0 -mig 1

# Create GPU instances
nvidia-smi mig -i 0 -cgi 19,19,19 -C  # 3x 3g.20gb

# List MIG instances
nvidia-smi mig -lgi
```

### GPU Monitoring with DCGM

```yaml
# DCGM monitoring stack
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    spec:
      containers:
        - name: dcgm-exporter
          image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
          ports:
            - containerPort: 9400
          env:
            - name: DCGM_EXPORTER_COLLECTORS
              value: "/etc/dcgm-exporter/custom-counters.csv"
          volumeMounts:
            - name: custom-counters
              mountPath: /etc/dcgm-exporter/custom-counters.csv
      volumes:
        - name: custom-counters
          configMap:
            name: dcgm-custom-counters
---
# Custom GPU metrics
apiVersion: v1
kind: ConfigMap
metadata:
  name: dcgm-custom-counters
data:
  custom-counters.csv: |
    # Format: DCGM_FI_field_id, type, promote_to_regex, description
    DCGM_FI_DEV_GPU_UTIL, gauge, gpu_utilization, GPU utilization percentage
    DCGM_FI_DEV_MEM_COPY_UTIL, gauge, mem_utilization, Memory utilization percentage
    DCGM_FI_DEV_ENC_UTIL, gauge, encoder_utilization, Encoder utilization
    DCGM_FI_DEV_DEC_UTIL, gauge, decoder_utilization, Decoder utilization
    DCGM_FI_DEV_FB_USED, gauge, fb_used, Framebuffer memory used
    DCGM_FI_DEV_FB_FREE, gauge, fb_free, Framebuffer memory free
    DCGM_FI_DEV_temperature, gauge, gpu_temperature, GPU temperature
    DCGM_FI_DEV_POWER_USAGE, gauge, power_usage, Power usage
    DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, energy_consumption, Total energy consumption
```

---

## 3. Model Serving Deep-Dive

### vLLM Production Configuration

```python
# vLLM server configuration for production
from dataclasses import dataclass
from typing import Optional

@dataclass
class VLLMConfig:
    # Model configuration
    model: str = "meta-llama/Llama-3-70B-Instruct"
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    max_model_len: int = 8192
    
    # Memory optimization
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    block_size: int = 16
    
    # Batching
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_num_prefills: int = 8
    
    # Quantization
    quantization: Optional[str] = "awq"
    
    # Serving
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_log_level: str = "info"
    
    # Safety
    trust_remote_code: bool = True
    disable_log_requests: bool = True

# Start vLLM server
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-3-70B-Instruct \
#   --tensor-parallel-size 4 \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.9 \
#   --enable-prefix-caching \
#   --quantization awq
```

### Load Balancing for Multi-Replica Serving

```yaml
# NGINX load balancing for vLLM
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    upstream vllm_backend {
        least_conn;
        server vllm-0:8000 weight=3;
        server vllm-1:8000 weight=3;
        server vllm-2:8000 weight=2;
        server vllm-3:8000 weight=2;
        
        keepalive 32;
    }
    
    server {
        listen 80;
        
        location /v1/ {
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # Timeouts for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 8k;
            proxy_buffers 8 8k;
        }
        
        location /health {
            return 200 'healthy';
            add_header Content-Type text/plain;
        }
    }
```

### Request Queue Management

```python
# Priority queue for model serving
import asyncio
from dataclasses import dataclass
from enum import Enum
import heapq

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class InferenceRequest:
    prompt: str
    priority: Priority
    max_tokens: int
    temperature: float
    request_id: str
    submitted_at: float
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

class RequestQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = []
        self.max_size = max_size
        self.semaphore = asyncio.Semaphore(max_size)
    
    async def enqueue(self, request: InferenceRequest) -> bool:
        """Add request to queue"""
        if len(self.queue) >= self.max_size:
            return False
        
        heapq.heappush(self.queue, request)
        return True
    
    async def dequeue(self) -> Optional[InferenceRequest]:
        """Get next request by priority"""
        if self.queue:
            return heapq.heappop(self.queue)
        return None
    
    async def process_queue(self, model):
        """Process queue with priority scheduling"""
        while True:
            request = await self.dequeue()
            if request:
                # Process with timeout
                try:
                    result = await asyncio.wait_for(
                        model.generate(request.prompt, request.max_tokens),
                        timeout=30.0
                    )
                    await self.send_response(request.request_id, result)
                except asyncio.TimeoutError:
                    await self.send_error(request.request_id, "Timeout")
                finally:
                    self.semaphore.release()
            else:
                await asyncio.sleep(0.1)
```

---

## 4. Distributed Training Implementation

### FSDP (Fully Sharded Data Parallel)

```python
# FSDP training setup
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaConfig

# Model sharding configuration
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Initialize FSDP model
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    torch_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
    use_orig_params=True,
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(epochs):
    for batch in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### DeepSpeed Configuration

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### Multi-Node Training Setup

```bash
# Distributed training launch script
#!/bin/bash

# Configuration
MASTER_ADDR="10.0.1.100"
MASTER_PORT=29500
NNODES=4
NPROC_PER_NODE=8

# Launch training
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
  --model meta-llama/Llama-3-70B \
  --dataset s3://datasets/training_data \
  --output s3://models/checkpoints \
  --epochs 3 \
  --learning_rate 1e-5 \
  --batch_size 32 \
  --gradient_accumulation_steps 8 \
  --bf16 true \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}'
```

---

## 5. Feature Store Implementation

### Feast Setup

```python
# feast_feature_store.py
from feast import FeatureStore
from feast import Entity, Feature, FeatureView, FileSource
from datetime import timedelta

# Initialize feature store
store = FeatureStore(repo_path=".")

# Define entity
user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

# Define feature view
user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="avg_query_length", dtype=ValueType.FLOAT),
        Feature(name="total_queries_24h", dtype=ValueType.INT64),
        Feature(name="preferred_model", dtype=ValueType.STRING),
        Feature(name="embedding_cluster", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=FileSource(
        path="s3://features/user_features.parquet",
        event_timestamp_column="event_timestamp"
    )
)

# Get features for inference
feature_vector = store.get_online_features(
    features=[
        "user_features:avg_query_length",
        "user_features:total_queries_24h",
        "user_features:preferred_model"
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()
```

### Real-Time Feature Pipeline with Kafka

```python
# kafka_feature_pipeline.py
from kafka import KafkaConsumer, KafkaProducer
from json import loads, dumps
import redis
import time

class RealTimeFeatureProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'user_events',
            bootstrap_servers=['kafka:9092'],
            value_deserializer=lambda x: loads(x.decode('utf-8')),
            group_id='feature-processor'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['kafka:9092'],
            value_serializer=lambda x: dumps(x).encode('utf-8')
        )
        self.redis = redis.Redis(host='redis', port=6379)
    
    def process_event(self, event):
        """Process incoming event and update features"""
        user_id = event['user_id']
        event_type = event['event_type']
        timestamp = event['timestamp']
        
        # Update rolling features in Redis
        key = f"user:{user_id}:events"
        
        # Add to sorted set (score = timestamp)
        self.redis.zadd(key, {dumps(event): timestamp})
        
        # Remove events older than 24 hours
        cutoff = time.time() - 86400
        self.redis.zremrangebyscore(key, 0, cutoff)
        
        # Compute features
        all_events = self.redis.zrange(key, 0, -1)
        features = self.compute_features(all_events)
        
        # Store features
        feature_key = f"user:{user_id}:features"
        self.redis.hset(feature_key, mapping=features)
        self.redis.expire(feature_key, 3600)  # TTL 1 hour
        
        # Publish feature update
        self.producer.send('feature_updates', {
            'user_id': user_id,
            'features': features,
            'timestamp': timestamp
        })
    
    def compute_features(self, events):
        """Compute features from event stream"""
        events = [loads(e) for e in events]
        
        return {
            'total_events_24h': len(events),
            'avg_response_time': sum(e.get('response_time', 0) for e in events) / max(len(events), 1),
            'unique_event_types': len(set(e['event_type'] for e in events)),
            'last_active': max(e['timestamp'] for e in events) if events else 0
        }
```

---

## 6. Monitoring Stack Setup

### Prometheus Configuration

```yaml
# prometheus.yml for AI monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ai_alerts.yml"

scrape_configs:
  - job_name: 'vllm-serving'
    static_configs:
      - targets: ['vllm-0:8000', 'vllm-1:8000', 'vllm-2:8000']
    metrics_path: /metrics
    
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['dcgm-exporter:9400']
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Custom AI Metrics

```python
# ai_metrics.py - Custom metrics for AI systems
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
REQUEST_COUNT = Counter(
    'ai_requests_total',
    'Total AI requests',
    ['model', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ai_request_latency_seconds',
    'AI request latency',
    ['model', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Model metrics
MODEL_QUALITY = Gauge(
    'ai_model_quality_score',
    'Model quality score',
    ['model', 'metric']
)

MODEL_DRIFT = Gauge(
    'ai_model_drift_score',
    'Model drift detection score',
    ['model']
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    'ai_gpu_utilization',
    'GPU utilization percentage',
    ['gpu_id', 'instance']
)

GPU_MEMORY = Gauge(
    'ai_gpu_memory_bytes',
    'GPU memory usage',
    ['gpu_id', 'instance']
)

# Cost metrics
INFERENCE_COST = Summary(
    'ai_inference_cost_dollars',
    'Cost per inference',
    ['model', 'team']
)

# Decorator for tracking
def track_inference(model: str, team: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_COUNT.labels(model=model, endpoint='inference', status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(model=model, endpoint='inference', status='error').inc()
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(model=model, endpoint='inference').observe(latency)
                cost = calculate_cost(model, latency)
                INFERENCE_COST.labels(model=model, team=team).observe(cost)
        return wrapper
    return decorator
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI Platform Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_requests_total[5m])",
            "legendFormat": "{{model}} - {{status}}"
          }
        ]
      },
      {
        "title": "Latency Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(ai_request_latency_seconds_bucket[5m]))",
            "legendFormat": "p99 {{model}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "ai_gpu_utilization",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ],
        "thresholds": [
          {"value": 30, "color": "red"},
          {"value": 70, "color": "yellow"},
          {"value": 90, "color": "green"}
        ]
      },
      {
        "title": "Cost per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(ai_inference_cost_dollars[1h])) * 3600",
            "legendFormat": "Total Cost"
          }
        ]
      }
    ]
  }
}
```

---

## 7. Cost Engineering

### GPU Cost Calculator

```python
# gpu_cost_calculator.py
class GPUCostCalculator:
    # Pricing per hour (on-demand)
    PRICING = {
        # AWS
        'p3.2xlarge': {'gpu': 'V100-16GB', 'cost': 3.06},
        'p3.8xlarge': {'gpu': 'V100-32GB', 'cost': 12.24},
        'p3.16xlarge': {'gpu': 'V100-32GB', 'cost': 24.48},
        'p4d.24xlarge': {'gpu': 'A100-40GB', 'cost': 32.77},
        'p5.48xlarge': {'gpu': 'H100-80GB', 'cost': 98.32},
        'g5.xlarge': {'gpu': 'A10G-24GB', 'cost': 1.01},
        'g5.2xlarge': {'gpu': 'A10G-24GB', 'cost': 1.21},
        'g5.4xlarge': {'gpu': 'A10G-24GB', 'cost': 1.62},
        'g5.12xlarge': {'gpu': 'A10G-24GB', 'cost': 5.67},
        'g6.xlarge': {'gpu': 'L4-24GB', 'cost': 0.80},
        'g6.2xlarge': {'gpu': 'L4-24GB', 'cost': 0.97},
        
        # GCP
        'a2-highgpu-1g': {'gpu': 'A100-40GB', 'cost': 3.67},
        'a2-highgpu-2g': {'gpu': 'A100-40GB', 'cost': 7.35},
        'a2-highgpu-4g': {'gpu': 'A100-40GB', 'cost': 14.69},
        'a2-highgpu-8g': {'gpu': 'A100-40GB', 'cost': 29.39},
        'a3-highgpu-8g': {'gpu': 'H100-80GB', 'cost': 101.22},
        'g2-standard-4': {'gpu': 'L4-24GB', 'cost': 0.70},
        'g2-standard-8': {'gpu': 'L4-24GB', 'cost': 0.85},
    }
    
    SPOT_DISCOUNT = 0.70  # 70% discount for spot instances
    
    def calculate_monthly_cost(
        self,
        instance_type: str,
        hours_per_day: float = 24,
        days_per_month: int = 30,
        use_spot: bool = False
    ) -> dict:
        """Calculate monthly GPU cost"""
        pricing = self.PRICING.get(instance_type)
        if not pricing:
            raise ValueError(f"Unknown instance type: {instance_type}")
        
        hourly_cost = pricing['cost']
        if use_spot:
            hourly_cost *= (1 - self.SPOT_DISCOUNT)
        
        monthly_hours = hours_per_day * days_per_month
        monthly_cost = hourly_cost * monthly_hours
        
        return {
            'instance_type': instance_type,
            'gpu': pricing['gpu'],
            'hourly_cost': round(hourly_cost, 2),
            'monthly_hours': monthly_hours,
            'monthly_cost': round(monthly_cost, 2),
            'annual_cost': round(monthly_cost * 12, 2),
            'use_spot': use_spot
        }
    
    def compare_options(self, workload: dict) -> list:
        """Compare different instance options for a workload"""
        results = []
        for instance_type in self.PRICING:
            try:
                cost = self.calculate_monthly_cost(
                    instance_type,
                    hours_per_day=workload.get('hours_per_day', 24),
                    use_spot=workload.get('use_spot', False)
                )
                cost['fits_requirement'] = self.check_requirement(
                    instance_type, workload
                )
                results.append(cost)
            except:
                continue
        
        return sorted(results, key=lambda x: x['monthly_cost'])
```

### Cost Optimization Engine

```python
# cost_optimizer.py
class CostOptimizer:
    def __init__(self, cost_calculator: GPUCostCalculator):
        self.calculator = cost_calculator
        self.historical_usage = []
    
    def analyze_usage(self, gpu_id: str, usage_data: list) -> dict:
        """Analyze GPU usage patterns"""
        avg_utilization = sum(usage_data) / len(usage_data)
        peak_utilization = max(usage_data)
        
        recommendations = []
        
        if avg_utilization < 30:
            recommendations.append({
                'type': 'downsize',
                'reason': 'Low average utilization',
                'savings': '40-60%'
            })
        
        if peak_utilization < 50:
            recommendations.append({
                'type': 'time_sharing',
                'reason': 'Peak utilization is low',
                'savings': '50-70%'
            })
        
        if avg_utilization > 80:
            recommendations.append({
                'type': 'upscale',
                'reason': 'High utilization may cause queuing',
                'impact': 'Improved latency'
            })
        
        return {
            'gpu_id': gpu_id,
            'avg_utilization': avg_utilization,
            'peak_utilization': peak_utilization,
            'recommendations': recommendations
        }
    
    def optimize_inference(self, inference_config: dict) -> dict:
        """Optimize inference configuration for cost"""
        current_cost = self.calculator.calculate_monthly_cost(
            inference_config['instance_type'],
            use_spot=inference_config.get('use_spot', False)
        )
        
        optimizations = []
        
        # Check if quantization is possible
        if not inference_config.get('quantized'):
            optimizations.append({
                'type': 'quantization',
                'method': 'AWQ',
                'estimated_savings': '40-60%'
            })
        
        # Check if model distillation is possible
        if inference_config.get('model_size', 0) > 70:
            optimizations.append({
                'type': 'distillation',
                'target_model': '7B',
                'estimated_savings': '70-80%'
            })
        
        # Check if caching can help
        if inference_config.get('repeat_queries', 0) > 0.3:
            optimizations.append({
                'type': 'caching',
                'method': 'Redis',
                'estimated_savings': '20-40%'
            })
        
        return {
            'current_cost': current_cost,
            'optimizations': optimizations,
            'estimated_total_savings': sum(
                o.get('estimated_savings_percent', 0) 
                for o in optimizations
            )
        }
```

---

## 8. Security Implementation

### Model Security Scanner

```python
# model_security_scanner.py
import hashlib
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class SecurityScanResult:
    model_name: str
    model_hash: str
    vulnerabilities: List[Dict]
    risk_score: float
    recommendations: List[str]

class ModelSecurityScanner:
    def __init__(self):
        self.vulnerability_db = self.load_vulnerability_db()
    
    def scan_model(self, model_path: str) -> SecurityScanResult:
        """Comprehensive security scan of a model"""
        model_hash = self.calculate_hash(model_path)
        
        vulnerabilities = []
        recommendations = []
        
        # Check 1: Model provenance
        provenance = self.check_provenance(model_path)
        if not provenance.get('verified'):
            vulnerabilities.append({
                'type': 'unverified_provenance',
                'severity': 'HIGH',
                'description': 'Model origin cannot be verified'
            })
            recommendations.append('Use only verified model sources')
        
        # Check 2: Weight integrity
        integrity = self.check_weight_integrity(model_path)
        if not integrity.get('passed'):
            vulnerabilities.append({
                'type': 'weight_tampering',
                'severity': 'CRITICAL',
                'description': 'Model weights may have been modified'
            })
            recommendations.append('Download model from official source')
        
        # Check 3: Known vulnerabilities
        known_vulns = self.check_known_vulnerabilities(model_hash)
        vulnerabilities.extend(known_vulns)
        
        # Check 4: Backdoor detection
        backdoor_score = self.detect_backdoors(model_path)
        if backdoor_score > 0.7:
            vulnerabilities.append({
                'type': 'potential_backdoor',
                'severity': 'HIGH',
                'description': f'Backdoor detection score: {backdoor_score}'
            })
            recommendations.append('Run additional backdoor detection tests')
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(vulnerabilities)
        
        return SecurityScanResult(
            model_name=model_path,
            model_hash=model_hash,
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def calculate_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
```

### Prompt Injection Defense

```python
# prompt_injection_defense.py
from typing import List, Tuple
import re

class PromptInjectionDefense:
    def __init__(self):
        self.patterns = self.load_patterns()
        self.allowed_topics = set()
    
    def validate_input(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate prompt for injection attempts"""
        issues = []
        
        # Check 1: Known injection patterns
        for pattern in self.patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"Matches injection pattern: {pattern}")
        
        # Check 2: System prompt extraction attempts
        extraction_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"you\s+are\s+now",
            r"new\s+instructions:",
            r"disregard\s+(previous|above)",
        ]
        for pattern in extraction_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"Possible extraction attempt: {pattern}")
        
        # Check 3: Role manipulation
        role_patterns = [
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+if",
            r"you\s+are\s+now\s+a",
            r"switch\s+to\s+mode",
        ]
        for pattern in role_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"Role manipulation detected: {pattern}")
        
        # Check 4: Output manipulation
        output_patterns = [
            r"output\s+the\s+following",
            r"respond\s+with",
            r"format\s+your\s+response",
        ]
        for pattern in output_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"Output manipulation: {pattern}")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    def sanitize_input(self, prompt: str) -> str:
        """Sanitize prompt before processing"""
        # Remove potential injection content
        sanitized = prompt
        
        # Remove system-like prefixes
        sanitized = re.sub(
            r"^(system|assistant|human):\s*",
            "",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Remove instruction-like content
        sanitized = re.sub(
            r"\[INST\].*?\[/INST\]",
            "",
            sanitized,
            flags=re.DOTALL
        )
        
        return sanitized.strip()
```

---

## 9. Platform as Code

### Terraform AI Platform

```hcl
# main.tf - AI Platform Infrastructure
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.0"

  cluster_name    = "ai-production"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    gpu_training = {
      min_size     = 0
      max_size     = 50
      desired_size = 4

      instance_types = ["p5.48xlarge", "p4d.24xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        node-type = "gpu"
        workload  = "training"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }

    gpu_inference = {
      min_size     = 2
      max_size     = 100
      desired_size = 8

      instance_types = ["g5.12xlarge", "g6.xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        node-type = "gpu"
        workload  = "inference"
      }
    }
  }
}

# S3 Bucket for Models
resource "aws_s3_bucket" "models" {
  bucket = "ai-platform-models-${data.aws_caller_identity.current.account_id}"
}

# ElastiCache for Feature Store
resource "aws_elasticache_cluster" "feature_store" {
  cluster_id           = "ai-feature-store"
  engine               = "redis"
  node_type            = "cache.r6g.xlarge"
  num_cache_nodes      = 3
  parameter_group_name = "default.redis7"
  port                 = 6379
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "ai_platform" {
  dashboard_name = "AI-Platform-Dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AI/Inference", "RequestCount"],
            ["AI/Inference", "Latency"],
            ["AI/GPU", "Utilization"]
          ]
          period = 300
          stat   = "Average"
          region = "us-west-2"
          title  = "AI Platform Metrics"
        }
      }
    ]
  })
}
```

### Pulumi AI Platform

```python
# pulumi_ai_platform.py
import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s

# EKS Cluster
cluster = aws.eks.Cluster(
    "ai-production",
    role_arn=cluster_role.arn,
    vpc_config=aws.eks.ClusterVpcConfigArgs(
        subnet_ids=subnet_ids,
        security_group_ids=[sg.id],
    ),
)

# GPU Node Group
gpu_node_group = aws.eks.NodeGroup(
    "gpu-training",
    cluster_name=cluster.name,
    node_group_name="gpu-training",
    node_role_arn=node_role.arn,
    subnet_ids=subnet_ids,
    instance_types=["p5.48xlarge"],
    scaling_config=aws.eks.NodeGroupScalingConfigArgs(
        desired_size=4,
        min_size=0,
        max_size=50,
    ),
    labels={
        "node-type": "gpu",
        "workload": "training",
    },
    taints=[{
        "key": "nvidia.com/gpu",
        "value": "true",
        "effect": "NO_SCHEDULE",
    }],
)

# Kubernetes deployment for model serving
vllm_deployment = k8s.apps.v1.Deployment(
    "vllm-serving",
    metadata={"name": "vllm-serving"},
    spec={
        "replicas": 3,
        "selector": {"matchLabels": {"app": "vllm"}},
        "template": {
            "metadata": {"labels": {"app": "vllm"}},
            "spec": {
                "nodeSelector": {"node-type": "gpu", "workload": "inference"},
                "tolerations": [{
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }],
                "containers": [{
                    "name": "vllm",
                    "image": "vllm/vllm-openai:latest",
                    "ports": [{"containerPort": 8000}],
                    "resources": {
                        "limits": {"nvidia.com/gpu": "4"},
                        "requests": {"nvidia.com/gpu": "4"},
                    },
                    "env": [
                        {"name": "MODEL", "value": "meta-llama/Llama-3-70B-Instruct"},
                        {"name": "TENSOR_PARALLEL_SIZE", "value": "4"},
                    ],
                }],
            },
        },
    },
)
```

---

## 10. Disaster Recovery & High Availability

### High Availability Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│              (Global Accelerator / Route53)              │
├─────────────────────────────────────────────────────────┤
│                    API Gateway                           │
│           (Kong / AWS API Gateway / Envoy)               │
├─────────────────────────────────────────────────────────┤
│                  Model Serving Tier                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ vLLM-0  │  │ vLLM-1  │  │ vLLM-2  │  │ vLLM-3  │   │
│  │ (AZ-1)  │  │ (AZ-2)  │  │ (AZ-3)  │  │ (AZ-1)  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
├─────────────────────────────────────────────────────────┤
│                  Model Registry                          │
│        (S3 + DynamoDB / MLflow / Custom)                 │
├─────────────────────────────────────────────────────────┤
│                  Monitoring Tier                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │Prometheus│  │ Grafana │  │ Langfuse│                │
│  │(Primary) │  │(Primary)│  │(Primary)│                │
│  └─────────┘  └─────────┘  └─────────┘                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │Prometheus│  │ Grafana │  │ Langfuse│                │
│  │(Replica) │  │(Replica)│  │(Replica)│                │
│  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Backup Strategy

```python
# backup_manager.py
import boto3
import json
from datetime import datetime, timedelta

class BackupManager:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.backup_bucket = 'ai-platform-backups'
    
    def backup_model_registry(self):
        """Backup model registry state"""
        timestamp = datetime.now().isoformat()
        
        # Export model registry
        models = self.export_model_registry()
        
        # Upload to S3
        self.s3.put_object(
            Bucket=self.backup_bucket,
            Key=f"model-registry/{timestamp}/registry.json",
            Body=json.dumps(models)
        )
        
        # Retention: keep daily for 30 days, weekly for 90 days
        self.cleanup_old_backups('model-registry', days=30)
    
    def backup_experiment_tracking(self):
        """Backup experiment tracking data"""
        timestamp = datetime.now().isoformat()
        
        # Export experiments
        experiments = self.export_experiments()
        
        # Upload to S3
        self.s3.put_object(
            Bucket=self.backup_bucket,
            Key=f"experiments/{timestamp}/experiments.json",
            Body=json.dumps(experiments)
        )
    
    def restore_from_backup(self, backup_date: str, component: str):
        """Restore from a specific backup"""
        # Download backup
        response = self.s3.get_object(
            Bucket=self.backup_bucket,
            Key=f"{component}/{backup_date}/data.json"
        )
        
        data = json.loads(response['Body'].read())
        
        # Restore based on component
        if component == 'model-registry':
            self.restore_model_registry(data)
        elif component == 'experiments':
            self.restore_experiments(data)
```

### Failover Testing

```python
# failover_tester.py
class FailoverTester:
    def __init__(self, serving_endpoints: list):
        self.endpoints = serving_endpoints
    
    def test_endpoint_failure(self, endpoint: str):
        """Test failover when an endpoint fails"""
        # Stop the endpoint
        self.stop_endpoint(endpoint)
        
        # Send requests
        results = self.send_test_requests(count=100)
        
        # Check that all requests succeeded
        success_rate = sum(1 for r in results if r.status == 200) / len(results)
        
        # Check latency
        avg_latency = sum(r.latency for r in results) / len(results)
        
        # Start the endpoint
        self.start_endpoint(endpoint)
        
        return {
            'endpoint': endpoint,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'passed': success_rate > 0.99 and avg_latency < 0.5
        }
    
    def test_zone_failure(self, zone: str):
        """Test failover when an entire AZ fails"""
        # Stop all endpoints in the zone
        for endpoint in self.endpoints:
            if endpoint.zone == zone:
                self.stop_endpoint(endpoint.name)
        
        # Send requests
        results = self.send_test_requests(count=1000)
        
        # Check results
        success_rate = sum(1 for r in results if r.status == 200) / len(results)
        
        # Start all endpoints
        for endpoint in self.endpoints:
            if endpoint.zone == zone:
                self.start_endpoint(endpoint.name)
        
        return {
            'zone': zone,
            'success_rate': success_rate,
            'passed': success_rate > 0.999
        }
```

---

## Cross-References

- **[01-Overview.md](01-Overview.md)** — High-level overview
- **[02-Core-Topics.md](02-Core-Topics.md)** — Core topics and patterns
- **[04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md)** — Tool landscape
- **[05-Future-Outlook.md](05-Future-Outlook.md)** — Future trends

---

*Next: [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) — Tool landscape and selection guide*
