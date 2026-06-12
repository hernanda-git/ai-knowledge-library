# Real-Time AI Systems

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 02-AI-Agent-Development.md, 06-RAG-Retrieval-Systems.md, 08-Edge-AI-Inference.md, 10-AI-Governance-Compliance.md

---

## Table of Contents

1. [Market Context & Demand](#1-market-context--demand)
2. [Streaming Inference Architectures](#2-streaming-inference-architectures)
   - 2.1 Apache Kafka + ML Integration
   - 2.2 Apache Flink + TensorFlow
   - 2.3 Kafka + Python ML (KStreams, Faust)
   - 2.4 Architecture Diagrams
3. [Online Learning Algorithms](#3-online-learning-algorithms)
   - 3.1 River (formerly Creme)
   - 3.2 Vowpal Wabbit
   - 3.3 scikit-multiflow
   - 3.4 Incremental Learning Techniques
   - 3.5 Adaptive Models & Concept Drift Detection
4. [Feature Stores for Real-Time](#4-feature-stores-for-real-time)
   - 4.1 Feast (Feature Store)
   - 4.2 Tecton
   - 4.3 Hopsworks Feature Store
   - 4.4 Real-Time Feature Engineering
   - 4.5 Serving Architecture Comparison
5. [Low-Latency Model Serving](#5-low-latency-model-serving)
   - 5.1 NVIDIA Triton Inference Server
   - 5.2 TorchServe
   - 5.3 TensorFlow Serving
   - 5.4 BentoML
   - 5.5 Seldon Core / MLServer
   - 5.6 Deployment Configurations
6. [Edge Streaming Architectures](#6-edge-streaming-architectures)
   - 6.1 Edge vs Cloud Trade-offs
   - 6.2 Edge Inference Patterns
   - 6.3 Azure IoT Edge + ONNX Runtime
   - 6.4 AWS IoT Greengrass + SageMaker
   - 6.5 Edge ML Optimization Techniques
   - 6.6 Architecture Diagrams
7. [Real-Time Monitoring & Observability](#7-real-time-monitoring--observability)
   - 7.1 Prometheus + Custom AI Metrics
   - 7.2 Grafana Dashboards for ML
   - 7.3 WhyLabs
   - 7.4 Arize AI
   - 7.5 ML-Specific Monitoring Strategies
   - 7.6 Alerting & Incident Response
8. [Anomaly Detection Pipelines](#8-anomaly-detection-pipelines)
   - 8.1 Twitter's AnomalyDetection (Seasonal Decomposition)
   - 8.2 Numenta NAB (Numenta Anomaly Benchmark)
   - 8.3 Real-Time Anomaly Detection Architectures
   - 8.4 Use Cases (Fraud, Infra, Security, IoT)
9. [Real-Time ML System Design Patterns](#9-real-time-ml-system-design-patterns)
   - 9.1 Lambda Architecture
   - 9.2 Kappa Architecture
   - 9.3 Comparison & Decision Framework
   - 9.4 Hybrid Patterns
10. [Performance Benchmarks for Real-Time Inference](#10-performance-benchmarks-for-real-time-inference)
    - 10.1 Latency Benchmarks
    - 10.2 Throughput Benchmarks
    - 10.3 Serving Infrastructure Comparison
    - 10.4 Optimization Techniques
11. [Future Outlook](#11-future-outlook)

---

## 1. Market Context & Demand

Real-time AI — AI models that process data and produce predictions in sub-second or streaming fashion — has become a critical infrastructure category in June 2026. The shift from batch processing to real-time has been driven by demand for immediate, context-aware AI decisions in applications ranging from fraud detection to autonomous systems.

**Key market signals:**
- Real-time inference infrastructure market: $7.2B in 2026, growing at 34% CAGR
- 68% of ML models in production now serve predictions in real-time (vs. 32% in 2023)
- Streaming ML pipelines have grown 4x year-over-year in enterprise deployments
- Sub-10ms inference latency is now a baseline requirement for 55% of production systems
- Real-time feature stores are deployed in 72% of enterprises with >10 ML models in production

**Why now?**
- **Streaming data ubiquity** — Kafka, Pulsar, and Kinesis are enterprise-standard; data is born in streams
- **Chip advances** — NVIDIA H200/B200, AMD MI300X, and custom AI accelerators enable real-time inference at scale
- **Model optimization** — Quantization, pruning, distillation reduce latency 10-100x
- **Edge maturity** — ARM-based inference, mobile NPUs, and IoT hardware are production-ready
- **Business imperative** — Milliseconds matter: 100ms additional latency costs Amazon 1% in sales; real-time fraud prevention saves billions

**The real-time AI stack:**

```
┌─────────────────────────────────────────────────────┐
│              APPLICATION LAYER                       │
│  (Dashboards, APIs, Mobile, Web, IoT)               │
├─────────────────────────────────────────────────────┤
│            REAL-TIME PREDICTION SERVING              │
│  Triton │ TorchServe │ TFServing │ BentoML │ Seldon │
├─────────────────────────────────────────────────────┤
│             REAL-TIME FEATURE STORES                 │
│        Feast │ Tecton │ Hopsworks │ Custom           │
├─────────────────────────────────────────────────────┤
│          STREAM PROCESSING ENGINES                   │
│    Kafka │ Flink │ Spark Streaming │ Pulsar │       │
├─────────────────────────────────────────────────────┤
│           ONLINE LEARNING / ADAPTIVE ML              │
│        River │ Vowpal Wabbit │ Incremental          │
├─────────────────────────────────────────────────────┤
│            MONITORING & OBSERVABILITY                │
│   Prometheus │ Arize │ WhyLabs │ Grafana │          │
└─────────────────────────────────────────────────────┘
```

---

## 2. Streaming Inference Architectures

Streaming inference processes data as it arrives, producing predictions without waiting for batch windows.

### 2.1 Apache Kafka + ML Integration

Kafka is the backbone of most real-time ML architectures, serving as the event bus for both inference requests and prediction outputs.

**Pattern 1: Kafka → ML Service → Kafka**
```
                          ┌─────────────┐
Producer → [topic: input] │  ML Model   │ → [topic: predictions]
                          │  Service    │
                          └─────────────┘
```

**Pattern 2: Kafka Streams (KStreams) for ML**
```java
// Kafka Streams with embedded ML model
Serde<Prediction> predictionSerde = new JsonSerde<>(Prediction.class);

KStream<String, Transaction> transactions = builder
    .stream("transactions", Consumed.with(Serdes.String(), txnSerde));

transactions
    .mapValues(transaction -> {
        // Feature extraction
        Features features = featureExtractor.extract(transaction);
        // Model inference
        double score = fraudModel.predict(features.toArray());
        // Anomaly threshold check
        return new Prediction(transaction.getId(), score, score > 0.95);
    })
    .to("fraud-predictions", Produced.with(Serdes.String(), predictionSerde));
```

**Pattern 3: Kafka Connect + Sink for Model Outputs**
```
Kafka Connect:
  Source connectors (DB, API, IoT) → Kafka topics
    ↓
  Kafka Streams / ksqlDB (feature computation)
    ↓
  Kafka → ML Service (REST/gRPC) → Kafka
    ↓
  Sink connectors: predictions to DB, dashboard, alert system
```

**Pattern 4: µServic with Kafka and gRPC inference:**

```yaml
# docker-compose excerpt for streaming inference pipeline
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6
    ports:
      - "2181:2181"
  
  kafka:
    image: confluentinc/cp-kafka:7.6
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
  
  feature-computation:
    image: stream-feature-job:latest
    depends_on: [kafka]
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      INPUT_TOPIC: raw-events
      OUTPUT_TOPIC: features
  
  ml-inference:
    image: triton-inference-server:23.12
    depends_on: [kafka, feature-computation]
    ports:
      - "8001:8001"  # gRPC
      - "8000:8000"  # HTTP
    volumes:
      - ./model-repository:/models
    command: tritonserver --model-repository=/models --grpc-port=8001
  
  inference-consumer:
    image: inference-consumer:latest
    depends_on: [kafka, ml-inference]
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      INPUT_TOPIC: features
      OUTPUT_TOPIC: predictions
      INFERENCE_SERVER: ml-inference:8001
```

### 2.2 Apache Flink + TensorFlow

Apache Flink provides stateful stream processing with exactly-once semantics, ideal for ML pipelines that require feature computation and model inference in a unified stream processing framework.

**Flink ML inference pipeline:**

```java
// Flink job for real-time ML inference
public class FlinkMLInferenceJob {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        DataStream<Transaction> transactions = env
            .addSource(new FlinkKafkaConsumer<>("transactions", 
                new JSONDeserializationSchema(), kafkaProps));
        
        DataStream<Prediction> predictions = transactions
            .keyBy(Transaction::getUserId)
            .flatMap(new RichFlatMapFunction<Transaction, Prediction>() {
                private transient TFModel model;
                private transient FeatureStoreClient featureStore;
                
                @Override
                public void open(Configuration parameters) {
                    // Load TensorFlow SavedModel
                    model = TFModel.load("/models/fraud-detection/1");
                    // Initialize feature store connection
                    featureStore = new FeastClient("feast-serving:6566");
                }
                
                @Override
                public void flatMap(Transaction txn, Collector<Prediction> out) {
                    // Get real-time features from feature store
                    Map<String, Float> features = featureStore
                        .getOnlineFeatures("fraud_features", 
                            txn.getUserId(), txn.getTimestamp());
                    
                    // Run inference
                    float score = model.predict(tensorFrom(features));
                    out.collect(new Prediction(txn.getId(), score));
                }
            });
        
        predictions.addSink(new FlinkKafkaProducer<>("predictions",
            new JSONSerializationSchema(), kafkaProps));
        
        env.execute("Real-Time Fraud Detection Pipeline");
    }
}
```

**Flink ML operations (batch prediction in streaming context):**

```python
# PyFlink with model inference
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
import numpy as np
import joblib

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define source table (Kafka)
t_env.execute_sql("""
    CREATE TABLE transactions (
        transaction_id STRING,
        user_id STRING,
        amount DOUBLE,
        timestamp BIGINT,
        merchant STRING
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'transactions',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'json'
    )
""")

# Load pre-trained model
model = joblib.load('/models/xgboost_fraud.pkl')

# SQL streaming query with UDF inference
t_env.create_temporary_system_function("predict_fraud", 
    lambda amount, merchant: 
        float(model.predict_proba(np.array([[amount, hash(merchant)]]))[0][1])
)

t_env.execute_sql("""
    INSERT INTO predictions
    SELECT 
        transaction_id,
        user_id,
        predict_fraud(amount, merchant) AS fraud_score,
        CASE 
            WHEN predict_fraud(amount, merchant) > 0.95 THEN 'ALERT'
            WHEN predict_fraud(amount, merchant) > 0.80 THEN 'REVIEW'
            ELSE 'OK'
        END AS action
    FROM transactions
""")
```

### 2.3 Kafka + Python ML (KStreams, Faust, Bytewax)

**Faust streaming application example:**

```python
import faust
import joblib
import numpy as np

app = faust.App(
    'fraud-detection',
    broker='kafka://kafka:9092',
    store='rocksdb://',
    value_serializer='json'
)

# Define data models
class Transaction(faust.Record, serializer='json'):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    timestamp: int

class Prediction(faust.Record, serializer='json'):
    transaction_id: str
    fraud_score: float
    action: str
    latency_ms: float

# Kafka topics
transactions_topic = app.topic('transactions', value_type=Transaction)
predictions_topic = app.topic('predictions', value_type=Prediction)

# Load model
model = joblib.load('/models/fraud_model.pkl')

@app.agent(transactions_topic)
async def predict(stream):
    async for transaction in stream:
        import time
        start = time.time()
        
        # Feature engineering
        features = np.array([[
            transaction.amount,
            hash(transaction.merchant) % 1000,
            transaction.timestamp % 86400  # time of day
        ]])
        
        # Inference
        score = float(model.predict_proba(features)[0][1])
        
        # Action
        if score > 0.95:
            action = 'BLOCK'
        elif score > 0.80:
            action = 'REVIEW'
        else:
            action = 'OK'
        
        latency = (time.time() - start) * 1000  # ms
        
        # Emit prediction
        await predictions_topic.send(
            value=Prediction(
                transaction_id=transaction.transaction_id,
                fraud_score=score,
                action=action,
                latency_ms=latency
            )
        )
```

**Bytewax streaming ML example:**

```python
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.connectors.kafka import KafkaSource, KafkaSink
import joblib

flow = Dataflow("fraud-detection")

# Kafka input
kin = KafkaSource(brokers=["kafka:9092"], topics=["transactions"])
stream = op.input("input", flow, kin)

# Parse JSON
parsed = op.map("parse", stream, json.loads)

# Feature extraction + model inference
model = joblib.load("/models/fraud_model.pkl")

def infer(event):
    features = [event["amount"], event["merchant_category"]]
    score = model.predict_proba([features])[0][1]
    return {
        "transaction_id": event["transaction_id"],
        "fraud_score": score,
        "action": "ALERT" if score > 0.9 else "OK",
        "timestamp": event["timestamp"]
    }

results = op.map("inference", parsed, infer)

# Kafka output
kout = KafkaSink(brokers=["kafka:9092"], topic="predictions")
op.output("output", results, kout)
```

### 2.4 Architecture Diagrams

**End-to-end real-time ML architecture:**

```
                        ┌──────────────────────┐
                        │   Event Producers     │
                        │ (App, Web, IoT, DB)   │
                        └───────┬──────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────┐
│               STREAMING DATA LAYER                 │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  Kafka   │  │  Pulsar  │  │  Kinesis       │  │
│  │(Primary) │  │(Geo-Dist)│  │(AWS Managed)   │  │
│  └────┬─────┘  └────┬─────┘  └───────┬────────┘  │
│       │              │                │            │
│       └──────────────┴────────────────┘            │
│                          │                         │
└──────────────────────────┼─────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────┐
│              STREAM PROCESSING                     │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  Flink   │  │Spark Str │  │  ksqlDB       │    │
│  │(Stateful)│  │(Micro-   │  │(SQL on Streams)│    │
│  │          │  │ batches) │  │               │    │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘    │
│       │              │                │            │
│       └──────────────┴────────────────┘            │
│                          │                         │
│                   Feature Computation              │
└──────────────────────────┼─────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────┐
│             FEATURE STORE LAYER                    │
│                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │  Online    │  │  Offline   │  │  Registry  │  │
│  │  Serving   │  │  Store     │  │  (Metadata)│  │
│  │(Redis/DDB) │  │(S3/GCS)   │  │            │  │
│  └─────┬──────┘  └─────┬──────┘  └──────┬─────┘  │
│        │                │                │        │
│        └────────────────┴────────────────┘        │
└──────────────────────────┼─────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────┐
│             MODEL INFERENCE SERVING                │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  Triton  │  │TorchServe│  │  TF Serving   │    │
│  │(GPU Opt) │  │(PyTorch) │  │(TensorFlow)  │    │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘    │
│       │              │                │            │
│  ┌────▼──────────────▼────────────────▼───────┐    │
│  │       Model Repository (NFS / S3 / PV)      │   │
│  └─────────────────────────────────────────────┘    │
│                          │                         │
└──────────────────────────┼─────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────┐
│            PREDICTION OUTPUT LAYER                 │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  Kafka   │  │  Alert   │  │   Database    │    │
│  │(Events)  │  │  System  │  │  (Predictions)│    │
│  └──────────┘  └──────────┘  └──────────────┘    │
└───────────────────────────────────────────────────┘
```

**Multi-model inference serving with Triton:**

```
                         ┌────────────┐
                         │ HTTP/gRPC  │
                         │ Clients    │
                         └──────┬─────┘
                                │
                                ▼
                  ┌─────────────────────┐
                  │  Triton Inference   │
                  │  Server Instance    │
                  │                     │
                  │  ┌───────────────┐  │
                  │  │ Scheduler     │  │
                  │  │ (Dynamic      │  │
                  │  │  Batch +      │  │
                  │  │  Concurrent)  │  │
                  │  └───────┬───────┘  │
                  │          │          │
                  │  ┌───────┴───────┐  │
                  │  │ Model Manager │  │
                  │  └───────┬───────┘  │
                  └──────────┼──────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Model A     │    │  Model B     │    │  Model C     │
│  (Fraud-XGB) │    │(Credit-NN)  │    │(NLP-BERT)    │
│  CPU Opt     │    │ GPU Opt     │    │ GPU + Batch  │
│  p99 < 5ms   │    │ p99 < 20ms  │    │ p99 < 100ms  │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 3. Online Learning Algorithms

Online learning algorithms update incrementally as new data arrives, without full retraining. This is critical for real-time AI systems that must adapt to concept drift.

### 3.1 River (formerly Creme)

River is the leading Python library for online/streaming ML, offering incremental versions of common algorithms.

**Key capabilities:**
- All models support partial_fit() — train instance by instance
- Built-in drift detection modules
- Feature engineering for streaming data
- Evaluation with progressive validation (no train/test split needed)
- Native integration with Kafka and dataframe streaming

**Installation and basic usage:**

```bash
pip install river
```

```python
from river import linear_model, metrics, preprocessing, optim
from river import drift, ensemble, tree, neighbors

# Online logistic regression
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(
        optimizer=optim.SGD(0.01),
        loss=linear_model.logistic.LogLoss()
    )
)

# Progressive validation
metric = metrics.Accuracy() + metrics.F1() + metrics.LogLoss()

# Stream data one sample at a time
for x, y in data_stream:
    y_pred = model.predict_one(x)       # Predict
    metric.update(y, y_pred)            # Update metric
    model.learn_one(x, y)               # Update model

print(f"Accuracy: {metric['Accuracy']:.4f}")
print(f"F1: {metric['F1']:.4f}")
```

**Online decision trees with River:**

```python
from river import tree

# Hoeffding Tree (Very Fast Decision Tree)
ht = tree.HoeffdingTreeClassifier(
    grace_period=200,
    delta=1e-7,
    leaf_prediction='nb',  # Naive Bayes at leaves
    nb_threshold=10
)

# Adaptive Random Forest
arf = ensemble.AdaptiveRandomForestClassifier(
    n_models=10,
    seed=42,
    leaf_model='nb',
    drift_detection_method=drift.ADWIN()
)

# Streaming evaluation
for x, y in stream:
    y_pred = arf.predict_one(x)
    arf.learn_one(x, y)
```

**Concept drift detection built-in:**

```python
from river import drift

# ADWIN (Adaptive Windowing)
adwin = drift.ADWIN(delta=0.002)

for val in data_stream:
    adwin.update(val)
    if adwin.drift_detected:
        print(f"Drift detected at index {adwin.n_detections}")
        # Reset or adapt model here

# Page-Hinkley test
ph = drift.PageHinkley(threshold=50, alpha=0.9999)

for val in data_stream:
    ph.update(val)
    if ph.drift_detected:
        print(f"Change detected")

# DDM (Drift Detection Method)
ddm = drift.DDM()

for val in data_stream:
    ddm.update(val)
    if ddm.drift_detected:
        print(f"Drift detected! Warning level: {ddm.warning_zone}")
    elif ddm.warning_zone:
        print(f"Warning zone entered")
```

### 3.2 Vowpal Wabbit

Vowpal Wabbit (VW) is a high-performance online learning system designed for large-scale, real-time applications. It excels at regression, classification, and learning-to-rank with sub-linear scaling.

**Installation:**

```bash
pip install vowpalwabbit
```

**Basic usage:**

```python
import pylibvw as vw

# Create VW model with online learning
model = vw.VW("--oaa 10 --learning_rate 0.1 -l 0.1 --power_t 0.5 --passes 1 -c -k")

# Train one example at a time
for example in training_stream:
    label = example['label']
    features = " ".join([f"{k}:{v}" for k, v in example['features'].items()])
    line = f"{label} | {features}"
    model.learn(line)

# Predict
prediction = model.predict(f"| {test_features}")
print(f"Prediction: {prediction}")

# Online regression with SGD
vw_reg = vw.VW("--loss_function squared -l 0.01")
for x, y in stream:
    vw_reg.learn(f"{y} | f1:{x['f1']} f2:{x['f2']} f3:{x['f3']}")

# Save and load model
model.save("vw_model.vw")
loaded = vw.VW("--oaa 10")
loaded.load("vw_model.vw")
```

**VW for real-time CTR prediction:**

```python
# Command-line usage example
vw_data = """
0 | user_history:0.8 time_of_day:0.5 device:mobile
1 | user_history:0.2 time_of_day:0.1 device:desktop
0 | user_history:0.6 time_of_day:0.9 device:tablet
"""

# Train on the fly via subprocess (production pattern)
import subprocess
vw_process = subprocess.Popen(
    ['vw', '--oaa', '10', '--learning_rate', '0.1', '-f', 'model.vw'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Feed training examples
for example in real_time_stream:
    # Format: label | namespace feature:value feature:value
    line = f"{example['click']} | user:{user_features} context:{context_features}\n"
    vw_process.stdin.write(line.encode())
    vw_process.stdin.flush()

# Prediction via separate process with --testonly
vw_predict = subprocess.Popen(
    ['vw', '-i', 'model.vw', '-t', '--oaa', '10'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
stdout, _ = vw_predict.communicate(
    f"| user:{features} context:{ctx}\n".encode()
)
```

**Advanced VW features for real-time:**

- **Hash trick** — Feature hashing for high-cardinality categoricals (no memory blowup)
- **Online SGD with adaptive learning rates** — AdaGrad, Adam, etc.
- **Multi-line data format** — For multi-label and structured prediction
- **Reduction-based learning** — Cost-sensitive classification, contextual bandits
- **Saving/loading with progressive updates** — Hot-swap models without downtime

### 3.3 scikit-multiflow

scikit-multiflow provides streaming ML algorithms with a scikit-learn-like API:

```python
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import KNNAdwinClassifier
from skmultiflow.data import DataStream

# Create stream
stream = DataStream(X, y)
stream.prepare_for_use()

# Adaptive model
model = AdaptiveRandomForestClassifier(
    n_models=25,
    max_features='sqrt',
    lambda_value=6,
    metric='accuracy',
    disable_weighted_vote=True
)

# Stream-train-evaluate
n_samples = 0
correct = 0

while stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = model.predict(X)
    
    if y_pred[0] == y[0]:
        correct += 1
    
    model.partial_fit(X, y)
    n_samples += 1

print(f"Accuracy: {correct / n_samples:.4f}")
```

### 3.4 Incremental Learning Techniques

**Core online learning algorithms:**

| Algorithm | Library | Update Type | Speed | Use Case |
|---|---|---|---|---|
| Hoeffding Tree (VFDT) | River, MOA | Incremental | Fast | Classification with evolving streams |
| Online Gradient Descent | VW, River | SGD step | Very fast | Regression, classification |
| Adaptive Random Forest | River, MOA | Ensemble + drift | Moderate | General purpose streaming |
| SGD Classifier/Regressor | sklearn (partial_fit) | Mini-batch | Moderate | When batch is impractical |
| Incremental PCA | River | Covariance update | Fast | Dimensionality reduction |
| Online K-means | River | Streaming update | Very fast | Clustering streams |
| KNN with ADWIN | River, MOA | Sliding window | Moderate | Lazy learning with drift |

**Comparison with batch learning:**

```
Batch Learning                Online Learning
┌──────────────────┐          ┌──────────────────┐
│ Load all data    │          │ Stream one       │
│ Train once       │          │ instance at a    │
│ Deploy static    │          │ time, update     │
│ model            │          │ model each step  │
│                  │          │                  │
│ High latency     │          │ Low latency      │
│ No drift handling│          │ Drift-adaptive   │
│ Periodic retrain │          │ Continuous learn │
│ High compute     │          │ Low compute/inst │
│ burst on retrain │          │                  │
└──────────────────┘          └──────────────────┘
```

**When to use online learning:**
- Data arrives as a stream (IoT sensors, clickstreams, trades)
- Concept drift is expected (fraud patterns, user behavior)
- Labeled data must be used immediately (active learning)
- Model must not go stale between retraining cycles
- Training data is too large to fit in memory

### 3.5 Adaptive Models & Concept Drift Detection

**Drift detection methods:**

| Method | Type | Detects | Pros | Cons |
|---|---|---|---|---|
| ADWIN (Adaptive Windowing) | Window-based | Mean shifts | Adaptive window, no threshold tuning | Sensitive to variance |
| DDM (Drift Detection Method) | Sequential | Classification error increase | Simple, interpretable | Only classification |
| EDDM (Early DDM) | Sequential | Earlier drift detection | Faster than DDM | More false positives |
| Page-Hinkley | Sequential | Mean changes | Simple, parametric | Assumes distribution |
| K-S Test | Window-based | Distribution change | Non-parametric | Requires storage |
| HDDM (Hellinger Distance) | Window-based | Distribution drift | Robust to outliers | Computationally heavy |

**Building a drift-adaptive model pipeline:**

```python
from river import drift, linear_model, preprocessing, compose
import numpy as np

class DriftAdaptiveModel:
    """Model that detects drift and triggers adaptation."""
    
    def __init__(self, base_model=None, drift_detector=None):
        self.model = base_model or (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        )
        self.drift_detector = drift_detector or drift.ADWIN(delta=0.001)
        self.drift_history = []
        self.current_performance = metrics.Accuracy()
        self.adaptation_count = 0
    
    def predict_one(self, x):
        return self.model.predict_one(x)
    
    def learn_one(self, x, y):
        y_pred = self.model.predict_one(x)
        
        # Track performance for drift detection
        correct = 1 if y_pred == y else 0
        self.drift_detector.update(correct)
        self.current_performance.update(y, y_pred)
        
        # Learn from the instance
        self.model.learn_one(x, y)
        
        # Check for drift
        if self.drift_detector.drift_detected:
            self._handle_drift()
    
    def _handle_drift(self):
        self.drift_history.append({
            'timestamp': datetime.now(),
            'performance': self.current_performance.get(),
            'n_detections': self.drift_detector.n_detections
        })
        self.adaptation_count += 1
        
        # Reset model with same structure but fresh weights
        self.model = (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        )
        self.current_performance = metrics.Accuracy()
        
        print(f"Drift adapted (count={self.adaptation_count})")
```

**Ensemble adaptation strategy:**

```python
from river import ensemble, drift, tree

class AdaptiveEnsemble:
    """Dynamic ensemble that adds/removes models on drift."""
    
    def __init__(self, n_models=10):
        self.models = [
            ensemble.AdaptiveRandomForestClassifier(
                n_models=5, seed=i
            ) for i in range(n_models)
        ]
        self.weights = np.ones(n_models) / n_models
        self.drift_detectors = [drift.ADWIN() for _ in range(n_models)]
    
    def predict_proba_one(self, x):
        # Weighted voting
        predictions = {}
        for w, model in zip(self.weights, self.models):
            pred = model.predict_proba_one(x)
            for cls, prob in pred.items():
                predictions[cls] = predictions.get(cls, 0) + w * prob
        return predictions
    
    def learn_one(self, x, y):
        for i, (model, detector) in enumerate(zip(self.models, self.drift_detectors)):
            y_pred = model.predict_one(x)
            correct = 1 if y_pred == y else 0
            detector.update(correct)
            
            model.learn_one(x, y)
            
            # Adjust weight based on drift state
            if detector.drift_detected:
                self.weights[i] *= 0.5
            elif detector.warning_zone:
                self.weights[i] *= 0.9
            else:
                self.weights[i] = min(1.0, self.weights[i] * 1.01)
        
        # Normalize weights
        self.weights /= self.weights.sum()
```

---

## 4. Feature Stores for Real-Time

Feature stores provide a centralized platform for feature engineering, storage, and serving across both training and inference.

### 4.1 Feast (Feature Store)

Feast is the leading open-source feature store, supporting both offline (batch) and online (real-time) feature serving.

**Core architecture:**

```
┌────────────────────────────────────────────────────────┐
│                   FEAST ARCHITECTURE                    │
├──────────────┬────────────────────┬────────────────────┤
│   OFFLINE    │     ONLINE         │     REGISTRY       │
│   STORE      │     STORE          │     (Metadata)     │
├──────────────┼────────────────────┼────────────────────┤
│  - BigQuery  │  - Redis           │  - SQL database    │
│  - Snowflake │  - Firestore       │  - Feature views   │
│  - S3/Parquet│  - DynamoDB        │  - Data sources    │
│  - Databricks│  - Cassandra        │  - Entity defs     │
└──────────────┴────────────────────┴────────────────────┘
```

**Defining features:**

```python
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define entity
user = Entity(
    name="user",
    value_type=ValueType.STRING,
    description="User identifier"
)

# Define batch feature source
user_stats_source = FileSource(
    path="/data/features/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature view
user_stats = FeatureView(
    name="user_stats",
    entities=["user"],
    ttl="24h",
    schema=[
        Field(name="avg_transaction_amount", dtype=Float32),
        Field(name="transaction_count_7d", dtype=Int64),
        Field(name="max_transaction_amount", dtype=Float32),
        Field(name="days_since_last_transaction", dtype=Int64),
    ],
    source=user_stats_source,
)

# Streaming features
from feast.stream_feature_view import stream_feature_view
from feast.data_source import KafkaSource

kafka_source = KafkaSource(
    name="kafka_transactions",
    kafka_bootstrap_servers="kafka:9092",
    topic="transactions",
    timestamp_field="timestamp",
    message_format="json",
)

@stream_feature_view(
    entities=[user],
    ttl="1h",
    mode="spark",
    source=kafka_source,
    schema=[Field(name="rolling_avg_5min", dtype=Float32)],
)
def user_rolling_avg(df: DataFrame):
    return df.groupBy("user_id").agg(
        avg("amount").alias("rolling_avg_5min")
    )
```

**Online feature retrieval:**

```python
import feast

# Initialize feature store
fs = feast.FeatureStore(repo_path="./feature_repo")

# Get online features for real-time inference
features = fs.get_online_features(
    features=[
        "user_stats:avg_transaction_amount",
        "user_stats:transaction_count_7d",
        "user_stats:max_transaction_amount",
        "user_rolling_avg:rolling_avg_5min",
    ],
    entity_rows=[{"user": user_id}],
).to_dict()

# Use in inference
fraud_score = model.predict(features)
```

**Deployment configuration:**

```yaml
# feature_store.yaml
project: fraud_detection
provider: gcp
registry:
  registry_type: sql
  path: postgresql://feast:password@host:5432/feast
online_store:
  type: redis
  connection_string: redis://redis:6379
offline_store:
  type: file
feature_server:
  enabled: True
  port: 6566
```

### 4.2 Tecton

Tecton is a managed feature platform (built on Feast + additional capabilities) with enterprise-grade streaming feature support.

**Key differentiators:**
- Declarative feature definition with automatic pipeline generation
- Native streaming aggregation via Spark Structured Streaming
- Point-in-time correct feature joins (no data leakage)
- Automatic feature materialization and backfill
- Built-in feature monitoring and data quality checks
- Feature serving SLAs with automatic failover

**Tecton feature definition:**

```python
import tecton
from tecton import (
    StreamFeatureView, BatchFeatureView, 
    Aggregation, FilteredSource
)

# Define a stream aggregation
@tecton.stream_feature_view(
    source=FilteredSource(
        source=transactions_stream,
        filter_expr="amount > 0"
    ),
    entities=[user],
    mode="spark",
    aggregation_interval="5 minutes",
    batch_schedule="1h",
    online=True,
    owner="ml-team",
)
def user_transaction_stats(transactions):
    from pyspark.sql import functions as F
    
    return transactions.groupBy("user_id").agg(
        F.count("transaction_id").alias("txn_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
        F.max("amount").alias("max_amount"),
        F.expr("percentile_approx(amount, 0.95)").alias("p95_amount"),
    )

# Feature serving
@tecton.online_feature_service(
    features=[
        user_transaction_stats,
        user_credit_history,
        merchant_risk_features,
    ],
    enable_cache=True,
    cache_ttl="30s",
)
def fraud_features():
    pass
```

### 4.3 Hopsworks Feature Store

Hopsworks provides an open-source feature store with strong MLOps integration:

**Key capabilities:**
- Feature groups (logical grouping of related features)
- Training dataset generation with point-in-time joins
- Online feature store with RonDB (MySQL Cluster) for low latency
- Feature monitoring and validation
- Integration with Kafka for streaming features
- Feature lineage and metadata tracking

**Hopsworks feature creation:**

```python
import hsfs

conn = hsfs.connection()
fs = conn.get_feature_store()

# Create feature group
transaction_features = fs.create_feature_group(
    name="transaction_features",
    version=1,
    description="Real-time transaction features",
    primary_key=["user_id"],
    event_time="timestamp",
    online_enabled=True,
    stream=True,  # Enable streaming ingestion
)

# Insert batch data
transaction_features.insert(batch_df)

# Streaming ingestion from Kafka
transaction_features.insert_stream(
    kafka_topic="transactions",
    kafka_bootstrap_servers="kafka:9092",
    starting_offsets="latest",
)

# Online retrieval for inference
feature_vector = fs.get_feature_vector(
    feature_view_name="fraud_features",
    entry={"user_id": user_id},
    return_type="dict",
)
```

### 4.4 Real-Time Feature Engineering

**Key techniques for streaming feature computation:**

1. **Windowed aggregations:**
```python
# Tumbling window (non-overlapping)
tumbling_avg = transactions \
    .window_by("10 minutes") \
    .aggregate(avg("amount"))

# Sliding window (overlapping, updated continuously)
sliding_avg = transactions \
    .window_by("10 minutes", slide="1 minute") \
    .aggregate(avg("amount"))

# Session window (event-triggered)
session_features = transactions \
    .window_by("session", gap="30 minutes") \
    .aggregate(count("transaction_id"), sum("amount"))
```

2. **Feature backfill:**
- Historical features must be computed with point-in-time correctness
- Use feature store's point-in-time join for training data
- Ensure features for time T are computed using only data before T

3. **Feature freshness SLA:**
```
Feature freshness tiers:
┌──────────────────────┬────────────────┬──────────────┐
│ Tier                 │ Freshness SLA  │ Update Freq  │
├──────────────────────┼────────────────┼──────────────┤
│ Real-time (critical) │ < 10 seconds   │ Streaming    │
│ Near-real-time       │ < 1 minute     │ Micro-batch  │
│ Batch                │ < 24 hours     │ Scheduled    │
│ Historical           │ > 24 hours     │ On-demand    │
└──────────────────────┴────────────────┴──────────────┘
```

### 4.5 Serving Architecture Comparison

| Feature Store | Online Store | Latency | Scalability | Deployment |
|---|---|---|---|---|
| **Feast** | Redis, Firestore, DynamoDB | <5ms (Redis) | Horizontal | Self-managed or cloud |
| **Tecton** | DynamoDB, Redis | <3ms | Managed auto-scale | Tecton cloud |
| **Hopsworks** | RonDB, Redis | <5ms | Horizontal + vertical | Self-managed or cloud |
| **Feast + Redis** | Redis Cluster | <2ms (p99) | 10K+ QPS per node | Kubernetes |
| **SageMaker Feature Store** | S3 + DynamoDB | <10ms | AWS auto-scale | AWS only |

**Choosing a feature store:**

```
Decision matrix:
┌──────────────────────┬────────┬────────┬────────────┐
│ Requirement          │ Feast  │ Tecton │ Hopsworks  │
├──────────────────────┼────────┼────────┼────────────┤
│ Open source          │ ✓ Full │ ✗      │ ✓ Core     │
│ Managed service      │ ✗      │ ✓      │ ✓          │
│ Streaming features   │ ✓      │ ✓      │ ✓          │
│ Point-in-time joins  │ ✓      │ ✓      │ ✓          │
│ Feature monitoring   │ Manual │ Auto   │ ✓          │
│ Multi-cloud          │ ✓      │ ✓      │ Partial    │
│ Enterprise support   │ Community│ ✓     │ ✓          │
│ GPU inference        │ ✓      │ ✓      │ ✓          │
│ Cost                 │ Free   │ $$$    │ Free/$$    │
└──────────────────────┴────────┴────────┴────────────┘
```

---

## 5. Low-Latency Model Serving

Low-latency model serving is the core infrastructure for real-time AI. The choice of serving infrastructure directly impacts inference latency, throughput, and cost.

### 5.1 NVIDIA Triton Inference Server

Triton is the leading high-performance inference server, supporting all major frameworks with GPU optimization.

**Key features:**
- Multi-framework: TensorFlow, PyTorch, ONNX, TensorRT, vLLM, Python
- Dynamic batching — automatically groups requests for GPU efficiency
- Concurrent model execution — multiple models on same GPU
- Model ensembles — pipeline multiple models as inference graph
- BLS (Business Logic Scripting) — custom pre/post-processing in Python
- GPU/CPU partitioning — models can target specific devices
- Model versioning — multiple versions with canary/rollback
- Metrics — Prometheus-native latency, throughput, queue metrics

**Deployment configuration:**

```yaml
# config.pbtxt for Triton model repository
name: "fraud_model"
platform: "onnxruntime_onnx"
max_batch_size: 1024
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [300]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [2]
  }
]
dynamic_batching {
  preferred_batch_size: [32, 64, 128]
  max_queue_delay_microseconds: 500
}
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
version_policy: {
  specific: { versions: [1, 2] }
  # latest: { num_versions: 2 }
}
```

**Ensemble pipeline:**

```yaml
name: "fraud_pipeline"
platform: "ensemble"
max_batch_size: 64

ensemble_scheduler {
  step [
    {
      model_name: "feature_engineering"
      model_version: 1
      input_map: { raw_input: "input" }
      output_map: { features: "features" }
    },
    {
      model_name: "fraud_model"
      model_version: 2
      input_map: { input: "features" }
      output_map: { fraud_score: "output" }
    },
    {
      model_name: "post_processing"
      model_version: 1
      input_map: { score: "fraud_score" }
      output_map: { decision: "output" }
    }
  ]
}
```

**Client-side inference:**

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient("triton:8001")

# Prepare input
input_data = np.random.randn(1, 300).astype(np.float32)
inputs = [grpcclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Inference
output = grpcclient.InferRequestedOutput("output")
response = client.infer(
    model_name="fraud_model",
    inputs=inputs,
    outputs=[output],
    request_id="req-001",
    sequence_id=100,
    sequence_start=False,
    priority=1,
    timeout_ms=100,
)

result = response.as_numpy("output")
print(f"Fraud score: {result[0][1]:.4f}")
print(f"Inference time: {response.get_response().inference_header.timestamp}")
```

### 5.2 TorchServe

TorchServe is the official PyTorch model serving framework with enterprise features:

**Key features:**
- Multi-model serving with GPU auto-scaling
- Adaptive batching (configurable batch size and timeout)
- Model versioning with A/B testing support
- Custom request/response handlers for preprocessing
- Metrics endpoint (Prometheus format)
- gRPC and HTTP/REST endpoints
- Distributed inference for large models
- TorchScript and eager mode support

**Model handler (custom inference logic):**

```python
# fraud_handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np

class FraudHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.feature_names = ['amount', 'merchant_cat', 'time_of_day', 
                               'user_tenure', 'txn_frequency']
    
    def preprocess(self, data):
        """Convert raw request to model input tensor."""
        inputs = []
        for row in data:
            features = [
                row['amount'],
                row['merchant_category'],
                row['time_of_day'],
                row['user_tenure_days'],
                row['txn_last_hour']
            ]
            inputs.append(features)
        return torch.tensor(inputs, dtype=torch.float32)
    
    def inference(self, input_tensor):
        """Run model and apply business logic."""
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probabilities = torch.softmax(output, dim=1)
        return probabilities
    
    def postprocess(self, inference_output):
        """Format response with decision and score."""
        scores = inference_output.numpy()
        results = []
        for score in scores:
            fraud_prob = float(score[1])
            results.append({
                'fraud_score': fraud_prob,
                'decision': 'BLOCK' if fraud_prob > 0.95 
                            else 'REVIEW' if fraud_prob > 0.80 
                            else 'ALLOW'
            })
        return results
```

**TorchServe deployment:**

```bash
# Create model archive
torch-model-archiver \
    --model-name fraud_detection \
    --version 1.0 \
    --model-file model.py \
    --serialized-file fraud_model.pth \
    --handler fraud_handler.py \
    --extra-files index_to_class.json \
    --export-path model_store

# Start TorchServe
torchserve --start \
    --model-store model_store \
    --models fraud_detection=fraud_detection.mar \
    --enable-model-api \
    --metrics-mode prometheus

# Configuration file (config.properties)
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
number_of_netty_threads=16
job_queue_size=1000
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{
    "fraud_detection":{"1.0":{"defaultVersion":true,"marName":"fraud_detection.mar",
    "minWorkers":2,"maxWorkers":8,"batchSize":32,"maxBatchDelay":100,"responseTimeout":30}}
}}
```

### 5.3 TensorFlow Serving

TensorFlow Serving provides high-performance serving for TensorFlow/SavedModel models:

**Key features:**
- Native SavedModel support (versioned, with signature definitions)
- Dynamic batching with configurable parameters
- Model warmup to avoid cold-start latency
- gRPC and HTTP/REST endpoints
- Model version policy (latest, specific, or label-based)
- Resource-based autoscaling
- Prometheus monitoring integration

**Deployment configuration:**

```dockerfile
# Dockerfile for TF Serving
FROM tensorflow/serving:2.14.0

COPY models /models

ENV MODEL_NAME=fraud_detection
ENV MODEL_BASE_PATH=/models
ENV MODEL_CONFIG_FILE=/models/models.config
ENV MONITORING_CONFIG_FILE=/models/monitoring.config
ENV TENSORFLOW_INTER_OP_PARALLELISM=2
ENV TENSORFLOW_INTRA_OP_PARALLELISM=8
ENV OMP_NUM_THREADS=8

# models.config
echo 'model_config_list: {
  config: {
    name: "fraud_detection",
    base_path: "/models/fraud_detection",
    model_platform: "tensorflow",
    model_version_policy: {
      specific: { versions: [1, 2] }
    },
    version_labels: {
      key: "stable", value: 1,
      key: "canary", value: 2
    }
  }
}' > /models/models.config
```

**Client (gRPC):**

```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel('tf-serving:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'fraud_detection'
request.model_spec.version_label = 'stable'

# Prepare input tensor
input_tensor = tf.make_tensor_proto(
    values=features_array,
    shape=features_array.shape,
    dtype=tf.float32
)
request.inputs['input'].CopyFrom(input_tensor)

# Inference with timeout
response = stub.Predict(request, timeout=0.1)  # 100ms timeout
result = tf.make_ndarray(response.outputs['output'])
```

### 5.4 BentoML

BentoML provides a simple python-native approach to model serving with built-in optimization:

**Key features:**
- Python-native model serving with @svc.api decorators
- Automatic OpenAPI/Swagger docs
- Built-in model management and versioning
- Adaptive batching (micro-batching)
- GPU acceleration out of the box
- Docker container export
- Distributed serving with BentoCloud

**Service definition:**

```python
import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np

# Load model from BentoML model store
fraud_runner = bentoml.sklearn.get("fraud_model:latest").to_runner()

svc = bentoml.Service("fraud_detection", runners=[fraud_runner])

@svc.api(input=JSON(), output=JSON())
async def predict(input_data: dict) -> dict:
    """Real-time fraud prediction endpoint."""
    import time
    start = time.time()
    
    # Extract features from input
    features = np.array([[
        input_data['amount'],
        input_data['merchant_category'],
        input_data['time_of_day'],
        input_data['user_tenure_days'],
    ]], dtype=np.float32)
    
    # Run inference
    score = await fraud_runner.predict_proba.async_run(features)
    fraud_prob = float(score[0][1])
    
    latency_ms = (time.time() - start) * 1000
    
    return {
        "fraud_score": fraud_prob,
        "decision": "BLOCK" if fraud_prob > 0.95 else "REVIEW" if fraud_prob > 0.80 else "ALLOW",
        "latency_ms": round(latency_ms, 2)
    }

@svc.api(input=JSON(), output=JSON())
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "model": "fraud_model:latest"}
```

**Batch inference configuration:**

```python
import bentoml
from bentoml.io import JSON

svc = bentoml.Service("fraud_batch")

@svc.api(
    input=JSON(),
    output=JSON(),
    batchable=True,
    max_batch_size=256,
    max_latency_ms=50,
)
async def batch_predict(inputs: list) -> list:
    """Batch inference with micro-batching."""
    features = np.array([
        [i['amount'], i['merchant_category'], i['time_of_day']]
        for i in inputs
    ])
    scores = model.predict_proba(features)
    return [{"fraud_score": float(s[1])} for s in scores]
```

### 5.5 Seldon Core / MLServer

Seldon Core with MLServer provides standardized ML model serving on Kubernetes:

**SeldonDeployment YAML:**

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: fraud-detection
spec:
  name: fraud
  predictors:
  - name: default
    graph:
      name: fraud-classifier
      implementation: MLFLOW_SERVER
      modelUri: s3://models/fraud/1
      serviceAccountName: s3-reader
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: fraud-classifier
          env:
          - name: MLSERVER_MODEL_PARALLEL_WORKERS
            value: "4"
          - name: MLSERVER_INFER_OPTIMISED
            value: "true"
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 4Gi
              nvidia.com/gpu: 1
    replicas: 3
    traffic: 100
```

### 5.6 Deployment Configurations

**Kuberenetes deployment for Triton:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
  labels:
    app: triton-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton-server
  template:
    metadata:
      labels:
        app: triton-server
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.06-py3
        args: ["tritonserver", "--model-repository=/models", 
               "--strict-model-config=false",
               "--model-control-mode=explicit",
               "--load-model=fraud_model",
               "--grpc-port=8001", "--http-port=8000",
               "--metrics-port=8002"]
        ports:
        - containerPort: 8000  # HTTP
        - containerPort: 8001  # gRPC
        - containerPort: 8002  # Metrics
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
          requests:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /models
        readinessProbe:
          tcpSocket:
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-repo-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
spec:
  selector:
    app: triton-server
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: grpc
    port: 8001
    targetPort: 8001
  - name: metrics
    port: 8002
    targetPort: 8002
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: triton_inference_queue_size
      target:
        type: AverageValue
        averageValue: 10
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 6. Edge Streaming Architectures

Edge AI inference moves computation closer to the data source, reducing latency and bandwidth requirements.

### 6.1 Edge vs Cloud Trade-offs

| Aspect | Cloud Inference | Edge Inference | Hybrid |
|---|---|---|---|
| **Latency** | 20-100ms (network round-trip) | <5ms (local) | <10ms (edge with cloud fallback) |
| **Bandwidth** | High (send raw data) | Low (send predictions/tokens) | Medium |
| **Compute** | Unlimited (GPU clusters) | Limited (CPU, NPU, tiny GPU) | Tiered |
| **Model size** | Any | <500MB typical, <100MB for real-time | Split (edge for preproc, cloud for heavy compute) |
| **Privacy** | Data leaves device | Data stays local | Edge for sensitive, cloud for aggregate |
| **Connectivity** | Requires internet | Works offline | Graceful degradation |
| **Update frequency** | Immediate | OTA updates, periodic | Staged rollout |
| **Cost** | Pay-per-inference | Fixed hardware cost | Mixed |

### 6.2 Edge Inference Patterns

**Pattern 1: Fully on-device inference**

```
┌─────────────────────┐
│   Edge Device        │
│                     │
│  ┌───────────────┐ │
│  │ Sensor/Camera │ │
│  └───────┬───────┘ │
│          │         │
│          ▼         │
│  ┌───────────────┐ │
│  │ Preprocessing │ │
│  └───────┬───────┘ │
│          │         │
│          ▼         │
│  ┌───────────────┐ │
│  │ Edge ML Model │ │
│  │ (TFLite/ONNX) │ │
│  └───────┬───────┘ │
│          │         │
│          ▼         │
│  ┌───────────────┐ │
│  │ Predict/Act   │ │
│  └───────────────┘ │
└─────────────────────┘
```

**Pattern 2: Edge + Cloud hybrid**

```
┌──────────────┐    ┌──────────────────────┐
│ Edge Device   │    │   Cloud              │
│               │    │                      │
│  ┌─────────┐  │    │  ┌────────────────┐  │
│  │ Sensor  │  │    │  │  Feature Store  │  │
│  └────┬────┘  │    │  └────────────────┘  │
│       │       │    │        ↑              │
│       ▼       │    │        │              │
│  ┌─────────┐  │    │  ┌────────────────┐  │
│  │ Feature │──┼────┼─→│  Cloud Model   │  │
│  │ Extract │  │    │  │  (High Acc)    │  │
│  └────┬────┘  │    │  └───────┬────────┘  │
│       │       │    │          │            │
│       ▼       │    │          ▼            │
│  ┌─────────┐  │    │  ┌────────────────┐  │
│  │ Edge    │←─┼────┼──│  Fallback/     │  │
│  │ Model   │  │    │  │  Ensemble      │  │
│  └────┬────┘  │    │  └────────────────┘  │
│       │       │    └──────────────────────┘
│       ▼       │
│  ┌─────────┐  │
│  │ Decision│  │
│  └─────────┘  │
└──────────────┘
```

### 6.3 Azure IoT Edge + ONNX Runtime

```json
{
  "modules": {
    "modelModule": {
      "version": "1.0",
      "type": "docker",
      "status": "running",
      "restartPolicy": "always",
      "settings": {
        "image": "mcr.microsoft.com/onnxruntime:latest",
        "createOptions": {
          "Env": ["MODEL_PATH=/models/fraud.onnx"],
          "HostConfig": {
            "Memory": 512000000,
            "CpuShares": 512,
            "PortBindings": {"5001/tcp": [{"HostPort": "5001"}]}
          },
          "Binds": ["/data/models:/models"]
        }
      }
    },
    "preprocessingModule": {
      "version": "1.0",
      "type": "docker",
      "status": "running",
      "restartPolicy": "always",
      "settings": {
        "image": "myregistry.azurecr.io/edge-preprocessor:1.0",
        "createOptions": {
          "Env": ["INPUT_TOPIC=raw_sensor", "OUTPUT_TOPIC=features"]
        }
      }
    },
    "streamAnalyticsModule": {
      "version": "1.0",
      "type": "docker",
      "status": "running",
      "restartPolicy": "always",
      "settings": {
        "image": "mcr.microsoft.com/azure-stream-analytics/iotedge:1.0"
      }
    }
  },
  "routes": {
    "rawToPreprocess": "FROM /messages/modules/sensor INTO BrokeredEndpoint(\"/modules/preprocessingModule/inputs/input1\")",
    "preprocessToModel": "FROM /messages/modules/preprocessingModule/outputs/output1 INTO BrokeredEndpoint(\"/modules/modelModule/inputs/input1\")",
    "modelToCloud": "FROM /messages/modules/modelModule/outputs/output1 INTO $upstream"
  }
}
```

### 6.4 AWS IoT Greengrass + SageMaker

```python
# AWS IoT Greengrass ML component
import greengrasssdk
import time
import numpy as np
import onnxruntime as ort

client = greengrasssdk.client('iot-data')

# Load ONNX model
session = ort.InferenceSession('/models/fraud_model.onnx')
input_name = session.get_inputs()[0].name

def inference_handler(event, context):
    """Greengrass Lambda handler for edge inference."""
    
    # Extract features
    features = np.array([[
        event['amount'],
        event['merchant_category'],
        event['time_of_day'],
    ]], dtype=np.float32)
    
    # Run inference
    start = time.time()
    outputs = session.run(None, {input_name: features})
    latency = (time.time() - start) * 1000
    
    score = float(outputs[0][0][1])
    
    # Send result to cloud
    client.publish(
        topic='fraud/predictions',
        payload={
            'device_id': context.client_id,
            'transaction_id': event['transaction_id'],
            'fraud_score': score,
            'latency_ms': latency,
            'action': 'BLOCK' if score > 0.95 else 'ALLOW'
        }
    )
    
    return {'status': 'success', 'score': score}
```

### 6.5 Edge ML Optimization Techniques

**Model optimization for edge deployment:**

| Technique | Latency Reduction | Accuracy Impact | Tools |
|---|---|---|---|
| **Quantization (INT8)** | 2-4x | <1% loss | TensorRT, ONNX Runtime, TFLite |
| **Quantization (FP16)** | 1.5-2x | Negligible | TensorRT, ONNX Runtime |
| **Pruning** | 1.5-3x | 0-3% loss (tunable) | TensorFlow Model Optimization, PyTorch |
| **Knowledge Distillation** | 2-10x (smaller student) | 1-5% loss | Custom training pipeline |
| **Operator Fusion** | 1.2-2x | None | TensorRT, ONNX Runtime, XLA |
| **Model Architecture Search** | 2-5x | Task-dependent | NAS, Once-for-All, EdgeTPU |
| **ONNX Runtime** | 1.5-3x | None | Cross-platform optimization |
| **TensorRT** | 2-10x (GPU) | Minimal (FP16) | NVIDIA GPU optimization |

**Quantization example:**

```python
# Post-training quantization with ONNX Runtime
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "fraud_model.onnx"
model_int8 = "fraud_model_int8.onnx"

quantized_model = quantize_dynamic(
    model_fp32,
    model_int8,
    weight_type=QuantType.QUInt8
)

# Compare sizes
import os
print(f"FP32 size: {os.path.getsize(model_fp32) / 1e6:.2f} MB")
print(f"INT8 size: {os.path.getsize(model_int8) / 1e6:.2f} MB")
```

**Benchmarking optimized models on edge hardware:**

```bash
# Benchmark ONNX Runtime on edge
python -m onnxruntime.benchmark \
    --model fraud_model_int8.onnx \
    --input_shape 1,300 \
    --iterations 1000 \
    --warmup 100 \
    --device cpu

# TensorRT optimization
trtexec --onnx=fraud_model.onnx \
        --fp16 \
        --workspace=1024 \
        --minShapes=input:1x300 \
        --optShapes=input:64x300 \
        --maxShapes=input:1024x300 \
        --saveEngine=fraud_model_fp16.plan

# Benchmark TensorRT engine
trtexec --loadEngine=fraud_model_fp16.plan \
        --shapes=input:1x300 \
        --iterations=1000 \
        --duration=30
```

### 6.6 Architecture Diagrams

**Complete edge streaming architecture:**

```
                           ┌──────────────────────────────────────┐
                           │             CLOUD                    │
                           │                                      │
                           │  ┌──────────────────────────────┐   │
                           │  │  Model Registry + Training    │   │
                           │  │  (SageMaker / Vertex AI)     │   │
                           │  └────────┬─────────────────────┘   │
                           │           │ OTA Model Updates        │
                           │           ▼                         │
                           │  ┌──────────────────────────────┐   │
                           │  │  Cloud Inference (Fallback)   │   │
                           │  │  Triton / BentoML             │   │
                           │  └────────┬─────────────────────┘   │
                           │           │                          │
                           │  ┌────────▼─────────────────────┐   │
                           │  │  Monitoring & Analytics      │   │
                           │  │  (Arize / WhyLabs / Grafana) │   │
                           │  └──────────────────────────────┘   │
                           └──────────────────┬───────────────────┘
                                              │ MQTT / HTTP
                                              │
        ┌─────────────────────────────────────┼──────────────────────┐
        │                 EDGE                │                      │
        │                                     ▼                      │
        │  ┌──────────────────────────────────────────────────┐     │
        │  │          Edge Gateway (Greengrass / IoT Edge)    │     │
        │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │     │
        │  │  │ Data     │  │ Feature  │  │ Model        │   │     │
        │  │  │Ingestion │→│ Compute  │→│ Inference   │   │     │
        │  │  └──────────┘  └──────────┘  └──────┬───────┘   │     │
        │  │                                      │           │     │
        │  │  ┌───────────────────────────────────▼──────┐   │     │
        │  │  │      Decision & Action Engine            │   │     │
        │  │  └──────────────────────────────────────────┘   │     │
        │  └──────────────────────────────────────────────────┘     │
        │             │              │              │                │
        │     ┌───────▼──┐   ┌───────▼──┐   ┌───────▼──┐            │
        │     │ Camera   │   │  IoT     │   │  Sensor  │            │
        │     │(Vision)  │   │  Device  │   │  Array   │            │
        │     └──────────┘   └──────────┘   └──────────┘            │
        └───────────────────────────────────────────────────────────┘
```

---

## 7. Real-Time Monitoring & Observability

Monitoring real-time AI systems requires tracking model performance, data drift, infrastructure health, and business impact concurrently.

### 7.1 Prometheus + Custom AI Metrics

**Custom ML metrics exposition:**

```python
# prometheus_metrics.py
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import time

# Prediction counters
predictions_total = Counter(
    'model_predictions_total', 
    'Total number of predictions',
    ['model_name', 'model_version', 'decision']
)

# Prediction latency
prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name', 'model_version'],
    buckets=[.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0]
)

# Feature drift score (tracked as Gauge)
feature_drift = Gauge(
    'feature_drift_score',
    'Population stability index for features',
    ['model_name', 'feature_name']
)

# Model accuracy (rolling window)
rolling_accuracy = Gauge(
    'model_rolling_accuracy',
    'Rolling accuracy over last 1000 predictions',
    ['model_name']
)

# Queue depth
inference_queue_depth = Gauge(
    'inference_queue_depth',
    'Current inference request queue depth',
    ['server_name']
)

# Model version tracking
current_model_version = Gauge(
    'model_version_active',
    'Currently active model version number',
    ['model_name']
)

def record_prediction(model_name, version, decision, latency_ms):
    predictions_total.labels(
        model_name=model_name,
        model_version=version,
        decision=decision
    ).inc()
    
    prediction_latency.labels(
        model_name=model_name,
        model_version=version
    ).observe(latency_ms / 1000.0)
```

**Integrating with inference server (Triton example):**

```yaml
# Triton metrics configuration
metrics_config {
  counters: [
    { name: "inference_count", labels: ["model", "version"] },
    { name: "inference_failure_count", labels: ["model", "version"] },
    { name: "response_sent", labels: [] }
  ]
  histograms: [
    { name: "inference_duration_us", labels: ["model", "version"], 
      buckets: [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000] },
    { name: "queue_duration_us", labels: ["model", "version"],
      buckets: [100, 250, 500, 1000, 5000, 10000] }
  ]
  gauges: [
    { name: "inference_queue_size", labels: [] },
    { name: "model_memory_usage_bytes", labels: ["model", "version"] },
    { name: "gpu_utilization", labels: ["gpu_uuid"] },
    { name: "gpu_memory_used_bytes", labels: ["gpu_uuid"] }
  ]
}
```

### 7.2 Grafana Dashboards for ML

**Dashboard panels for real-time ML monitoring:**

```json
{
  "dashboard": {
    "title": "Real-Time ML System Overview",
    "panels": [
      {
        "title": "Prediction Latency (p50/p95/p99)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.50, sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le, model_name))",
          "legendFormat": "p50 - {{model_name}}"
        }, {
          "expr": "histogram_quantile(0.95, sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le, model_name))",
          "legendFormat": "p95 - {{model_name}}"
        }, {
          "expr": "histogram_quantile(0.99, sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le, model_name))",
          "legendFormat": "p99 - {{model_name}}"
        }],
        "yaxis": {"format": "s", "decimals": 0}
      },
      {
        "title": "Prediction Volume & Decision Distribution",
        "type": "bar-gauge",
        "targets": [{
          "expr": "sum(rate(model_predictions_total[5m])) by (decision)",
          "legendFormat": "{{decision}}"
        }]
      },
      {
        "title": "Feature Drift Scores (PSI)",
        "type": "table",
        "targets": [{
          "expr": "feature_drift_score",
          "legendFormat": "{{feature_name}}"
        }]
      },
      {
        "title": "Rolling Accuracy",
        "type": "graph",
        "targets": [{
          "expr": "model_rolling_accuracy",
          "legendFormat": "{{model_name}}"
        }],
        "alert": {
          "alertRuleTags": {"severity": "critical"},
          "conditions": [{
            "evaluator": {"type": "lt", "params": [0.80]},
            "operator": {"type": "and"}
          }]
        }
      },
      {
        "title": "GPU Utilization & Memory",
        "type": "graph",
        "targets": [{
          "expr": "triton_gpu_utilization",
          "legendFormat": "GPU {{gpu_uuid}}"
        }, {
          "expr": "triton_gpu_memory_used_bytes / 1024 / 1024 / 1024",
          "legendFormat": "GPU {{gpu_uuid}} Memory (GB)"
        }]
      },
      {
        "title": "Inference Queue Depth",
        "type": "graph",
        "targets": [{
          "expr": "inference_queue_depth",
          "legendFormat": "{{server_name}}"
        }],
        "alert": {
          "conditions": [{
            "evaluator": {"type": "gt", "params": [100]}
          }]
        }
      }
    ]
  }
}
```

### 7.3 WhyLabs

WhyLabs provides AI observability with automatic monitoring, alerting, and root cause analysis:

**Key features:**
- Pre-configured monitors for data drift, model performance, and data quality
- Automatic baseline estimation
- Segment analysis by feature values
- Root cause investigation with drill-down
- Integration with ML pipeline (batch + streaming)
- Incident management workflow

**Integration example:**

```python
import whylogs as why
from whylogs.api.writer import WhyLabsWriter
from whylogs.core.schema import DeclarativeSchema

# Configure WhyLabs logger
writer = WhyLabsWriter(
    org_id="org-xxxx",
    api_key="API_KEY",
    dataset_id="model-1"
)

# Log predictions in real-time
with why.logger(mode="rolling", interval_minutes=5) as logger:
    
    for transaction in real_time_stream:
        features = extract_features(transaction)
        prediction = model.predict(features)
        
        # Log feature and prediction data
        logger.log({
            "amount": features['amount'],
            "merchant_category": features['merchant_category'],
            "fraud_score": prediction['fraud_score'],
            "decision": prediction['decision'],
            "latency_ms": prediction['latency_ms']
        })
        
        # Flush to WhyLabs periodically
        if logger.eager:
            logger.flush()

# Alternative: direct WhyLabs upload
profile = why.log({"x": features}).profile()
profile.set_dataset_timestamp(timestamp)
writer.write(profile)
```

**Monitor configuration (WhyLabs dashboard):**

```yaml
monitors:
  - name: "Fraud Score Distribution Drift"
    type: "data_drift"
    metric: "distribution_distance"
    threshold: 0.2  # Hellinger distance
    severity: "warning"
    frequency: "PT5M"
    
  - name: "Prediction Latency Spike"
    type: "data_drift"
    metric: "mean"
    column: "latency_ms"
    threshold: 3  # Std deviations from baseline
    severity: "critical"
    frequency: "PT1M"
    
  - name: "Missing Features"
    type: "data_quality"
    metric: "missing_ratio"
    threshold: 0.01
    severity: "critical"
    frequency: "PT10M"
    
  - name: "Fraud Rate Drift"
    type: "model_performance"
    metric: "accuracy"
    threshold: 0.10  # Percentage point drop
    severity: "critical"
    frequency: "PT1H"
```

### 7.4 Arize AI

Arize provides ML observability with rich performance visualization:

**Integration:**

```python
from arize.api import Client
from arize.pandas.logger import Client as PandasClient
import pandas as pd
import numpy as np

arize_client = Client(space_id="space-xxx", api_key="API_KEY")

# Log real-time predictions
for transaction, prediction in zip(stream, predictions):
    response = arize_client.log(
        model_id="fraud-detection-v2",
        model_version="2.1.0",
        prediction_id=transaction["transaction_id"],
        features={
            "amount": transaction["amount"],
            "merchant_category": transaction["merchant_category"],
            "time_of_day": transaction["time_of_day"],
            "user_tenure_days": transaction["user_tenure_days"],
        },
        prediction_score=prediction["fraud_score"],
        actual_label=transaction["is_fraud"] if "is_fraud" in transaction else None,
        # Embedding for deep learning models
        prediction_label=prediction["decision"],
        tags={"env": "production", "region": "us-east-1"},
        # For LLM monitoring
        prompt=transaction.get("prompt"),
        response=transaction.get("response"),
    )
    
    if response.status_code != 200:
        print(f"Logging error: {response.text}")
```

**Arize monitoring capabilities:**
- **Drift detection:** PSI, KS-test, Hellinger distance
- **Performance monitoring:** Accuracy, AUC, F1, precision, recall, custom metrics
- **Bias monitoring:** Demographic parity, equal opportunity
- **Ranking and LLM metrics:** NDCG, MRR, hallucination rate, relevance
- **Root cause analysis:** Segment comparison, feature importance drift
- **Automated alerts:** Multi-condition, suppressible, with escalation

### 7.5 ML-Specific Monitoring Strategies

**Key monitoring dimensions for real-time AI:**

```
┌─────────────────────────────────────────────────────────────┐
│               REAL-TIME ML MONITORING                       │
├────────────┬──────────────┬──────────────┬──────────────────┤
│ DATA DRIFT │  MODEL DRIFT │  INFRA       │  BUSINESS        │
├────────────┼──────────────┼──────────────┼──────────────────┤
│ Feature    │ Accuracy     │ GPU          │ Fraud rate       │
│ distribution│ drop        │ utilization  │                   │
│            │              │              │                   │
│ Prediction │ Confusion    │ Memory       │ Approval rate    │
│ distribution│ matrix shift│ usage        │ change           │
│            │              │              │                   │
│ Missing    │ Calibration  │ Request      │ Revenue impact   │
│ values     │ drift        │ rate         │                   │
│            │              │              │                   │
│ New        │ Error        │ Response     │ False positive   │
│ categories │ analysis     │ status       │ rate              │
│            │              │              │                   │
│ Data       │ Concept      │ Request      │ Customer         │
│ quality    │ drift        │ size         │ complaints        │
└────────────┴──────────────┴──────────────┴──────────────────┘
```

**Drift detection thresholds:**

| Metric | Method | Warning Threshold | Critical Threshold |
|---|---|---|---|
| Feature distribution | Population Stability Index (PSI) | > 0.1 | > 0.25 |
| Feature distribution | Hellinger Distance | > 0.15 | > 0.30 |
| Feature distribution | K-S Test p-value | < 0.05 | < 0.01 |
| Prediction distribution | PSI on predictions | > 0.1 | > 0.25 |
| Model accuracy | Rolling 1000 predictions | Drop > 3% | Drop > 10% |
| Model calibration | Expected Calibration Error | > 0.05 | > 0.10 |
| Missing features | Fraction of nulls | > 1% | > 5% |
| Latency p99 | Statistical threshold | > 100ms | > 500ms |
| Error rate | Fraction of 5xx responses | > 1% | > 5% |

### 7.6 Alerting & Incident Response

**Alertmanager configuration for ML alerts:**

```yaml
# alertmanager.yml
route:
  receiver: 'ml-team-slack'
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
  - match:
      severity: critical
    receiver: 'ml-team-pagerduty'
    repeat_interval: 30m
  - match:
      type: model_drift
    receiver: 'ml-team-email'
    repeat_interval: 2h

receivers:
- name: 'ml-team-slack'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/xxx'
    channel: '#ml-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ .CommonAnnotations.description }}'
    
- name: 'ml-team-pagerduty'
  pagerduty_configs:
  - service_key: 'xxx'
    description: '{{ .CommonAnnotations.description }}'
    severity: 'critical'
    
- name: 'ml-team-email'
  email_configs:
  - to: 'ml-team@company.com'
    from: 'ml-alerts@company.com'
    smarthost: 'smtp.company.com:587'
```

**Alert rules for ML (PrometheusRule):**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-monitoring-rules
spec:
  groups:
  - name: ml-alerts
    rules:
    - alert: HighPredictionLatency
      expr: histogram_quantile(0.99, rate(model_prediction_latency_seconds_bucket[5m])) > 0.1
      for: 5m
      labels:
        severity: critical
        type: infrastructure
      annotations:
        summary: "p99 latency > 100ms for {{ $labels.model_name }}"
        description: "Model {{ $labels.model_name }} v{{ $labels.model_version }} has p99 latency {{ $value }}s"
    
    - alert: FeatureDriftDetected
      expr: feature_drift_score > 0.25
      for: 30m
      labels:
        severity: warning
        type: model_drift
      annotations:
        summary: "Feature drift on {{ $labels.feature_name }}"
        description: "Feature {{ $labels.feature_name }} in model {{ $labels.model_name }} has PSI {{ $value }}"
    
    - alert: AccuracyDrop
      expr: model_rolling_accuracy < 0.80
      for: 1h
      labels:
        severity: critical
        type: model_drift
      annotations:
        summary: "Accuracy below threshold"
        description: "Model {{ $labels.model_name }} rolling accuracy is {{ $value }}"
    
    - alert: QueueDepthCritical
      expr: inference_queue_depth > 500
      for: 2m
      labels:
        severity: critical
        type: infrastructure
      annotations:
        summary: "Inference queue depth critical"
        description: "Server {{ $labels.server_name }} has {{ $value }} requests queued"
```

---

## 8. Anomaly Detection Pipelines

Real-time anomaly detection is a primary use case for streaming ML, spanning fraud detection, infrastructure monitoring, cybersecurity, and IoT.

### 8.1 Twitter's AnomalyDetection (Seasonal Decomposition)

Twitter's AnomalyDetection (R package) uses Seasonal Hybrid ESD (S-H-ESD) for detecting anomalies in time series data.

**Algorithm overview:**
1. Decompose time series into seasonal, trend, and residual components (STL)
2. Apply Generalized ESD (Extreme Studentized Deviate) test on the residual
3. Handle both global and local anomalies
4. Works with daily, weekly, yearly seasonal patterns

**Python reimplementation for real-time:**

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy import stats

class SeasonalESDDetector:
    """Seasonal ESD anomaly detection adapted for streaming."""
    
    def __init__(self, period=24, max_anomalies=0.1):
        self.period = period
        self.max_anomalies = max_anomalies
        self.history = []
        self.seasonal_pattern = None
        
    def fit(self, series):
        """Fit seasonal decomposition on baseline data."""
        self.history = list(series)
        stl = STL(series, period=self.period, robust=True)
        result = stl.fit()
        self.seasonal_pattern = result.seasonal
        return self
        
    def detect(self, value, timestamp=None):
        """Detect if new value is anomalous."""
        self.history.append(value)
        
        if len(self.history) < self.period * 2:
            return False, 0.0
        
        # Use sliding window
        window = self.history[-self.period * 4:]
        series = pd.Series(window)
        
        # STL decomposition
        stl = STL(series, period=self.period, robust=True)
        result = stl.fit()
        
        residual = result.resid
        n = len(residual)
        
        # GESD test
        max_outliers = int(n * self.max_anomalies)
        
        if max_outliers < 1:
            return False, 0.0
            
        r_values = []
        for i in range(max_outliers):
            abs_resid = np.abs(residual)
            max_idx = np.argmax(abs_resid)
            r = abs_resid[max_idx]
            r_values.append((r, max_idx))
            residual = np.delete(residual.values, max_idx) if hasattr(residual, 'values') \
                      else np.delete(residual, max_idx)
        
        # Critical values
        for i, (r, idx) in enumerate(r_values):
            p = 1 - (i + 1) / (n + 1)
            t_crit = stats.t.ppf(1 - p / (2 * n), n - i - 2)
            lambda_crit = (n - i - 1) * t_crit / np.sqrt((n - i - 2 + t_crit**2) * (n - i))
            
            if r > lambda_crit:
                # The last detected anomaly is the one that just arrived
                if idx == len(series) - 1:
                    return True, r / lambda_crit
        
        return False, 0.0
```

### 8.2 Numenta NAB (Numenta Anomaly Benchmark)

NAB provides benchmark anomaly detection algorithms and a standardized evaluation framework:

**Core algorithms in NAB:**

```python
# Numenta HTM (Hierarchical Temporal Memory) based detection
# Simplified real-time detector
from collections import deque
import numpy as np

class NumentaDetector:
    """Streaming anomaly detector inspired by Numenta's approach."""
    
    def __init__(self, window_size=100, threshold_percentile=95):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.threshold_percentile = threshold_percentile
        self.mean = 0
        self.std = 1
        self.initialized = False
    
    def update(self, value):
        """Update model and return anomaly score."""
        if not self.initialized:
            self.window.append(value)
            if len(self.window) >= self.window_size:
                self.mean = np.mean(self.window)
                self.std = max(np.std(self.window), 1e-6)
                self.initialized = True
            return 0.0
        
        # Prediction (naive: last value, can be replaced with ARIMA/Prophet)
        if len(self.window) > 1:
            prediction = self.window[-1]  # Naive forecast
        else:
            prediction = self.mean
        
        # Error
        error = abs(value - prediction)
        
        # Normalize error
        normalized_error = (error - np.mean(self.errors)) / max(np.std(self.errors), 1e-6) \
                          if len(self.errors) > 10 else error / self.std
        
        self.errors.append(error)
        self.window.append(value)
        
        # Adaptive threshold
        if len(self.errors) >= self.window_size:
            threshold = np.percentile(self.errors, self.threshold_percentile)
        else:
            threshold = 3 * self.std
        
        # Anomaly score
        score = error / max(threshold, 1e-6)
        
        return score
    
    def is_anomaly(self, score, threshold=3.0):
        """Determine if score indicates anomaly."""
        return score > threshold
```

### 8.3 Real-Time Anomaly Detection Architectures

**Fraud detection pipeline (real-time):**

```
Transaction Event
       │
       ▼
┌──────────────────┐
│  Rule Engine     │ ← Fast path: known fraud patterns
│  (CEL / Drools)  │    Block instantly, no ML needed
└──────┬───────────┘
       │ Pass-through
       ▼
┌──────────────────┐
│  Feature         │ ← Real-time feature store lookup
│  Computation     │    User history, session stats
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  ML Model        │ ← Ensemble: XGBoost + RNN
│  (Low Latency)   │    Target: <10ms inference
└──────┬───────────┘
       │ Score
       ▼
┌──────────────────┐
│  Decision Engine │ ← Thresholds + business rules
│  (Score + Rules) │    BLOCK / REVIEW / ALLOW
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Deep Analysis   │ ← Async path: Graph NN, NLP
│  (High Latency)  │    For REVIEW queue items only
│  (Enrichment)    │    Feedback to model updates
└──────────────────┘
```

**Infrastructure anomaly detection pipeline:**

```
Metrics Stream (Prometheus / Telegraf)
       │
       ▼
┌──────────────────────┐
│  Metric Aggregation  │ ← 1m, 5m, 15m windows
│  (Downsampling)      │    Handle seasonality
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Baseline Estimation │ ← Moving average + std
│  (Seasonal Decomp)   │    Holt-Winters, STL
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Anomaly Scoring     │ ← Multiple methods
│  (3-Sigma / MAD /     │    Ensemble of detectors
│   GESD / Isolation   │
│   Forest / LSTM-AE)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Correlation Engine  │ ← Link related anomalies
│  (Graph-based)       │    Reduce alert fatigue
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Alert / Auto-Remed  │ ← PagerDuty, Jenkins
│  (Threshold + Triage)│    Auto-scale, restart, etc.
└──────────────────────┘
```

### 8.4 Use Cases

**Fraud Detection:**
- Credit card fraud: sub-100ms decision required
- Account takeover: anomaly in login pattern, device fingerprint
- Synthetic identity: graph-based anomaly detection
- Money laundering: temporal pattern anomalies in transaction chains

**Infrastructure Monitoring:**
- Server metrics: CPU, memory, disk, network anomalies
- Application performance: latency spikes, error rate surges
- Kubernetes: pod restarts, resource contention
- Database: query latency anomalies, connection pool exhaustion

**Security:**
- Network intrusion detection: packet-level anomaly scoring
- User behavior analytics: UEBA for insider threats
- DDoS detection: traffic volume and pattern anomalies
- API abuse: rate anomalies, abnormal request patterns

**IoT:**
- Predictive maintenance: vibration sensor anomaly detection
- Energy grid: consumption pattern anomalies
- Manufacturing: quality control anomaly detection
- Smart city: traffic flow, pollution sensor anomaly detection

---

## 9. Real-Time ML System Design Patterns

### 9.1 Lambda Architecture

Lambda Architecture combines batch and stream processing for comprehensive ML:

```
┌──────────────────────────────────────────────────────────────┐
│                    LAMBDA ARCHITECTURE                       │
├────────────────────────────┬─────────────────────────────────┤
│   SPEED LAYER              │   BATCH LAYER                   │
│   (Real-Time)              │   (Historical)                  │
│                            │                                 │
│  ┌────────────┐            │  ┌────────────────────┐         │
│  │ Stream     │            │  │ Raw Data Store     │         │
│  │ Ingestion  │            │  │ (S3 / HDFS)        │         │
│  └─────┬──────┘            │  └─────────┬──────────┘         │
│        │                   │            │                    │
│        ▼                   │            ▼                    │
│  ┌────────────┐            │  ┌────────────────────┐         │
│  │ Real-time  │            │  │ Batch Computation  │         │
│  │ Feature    │            │  │ (Spark / Hive)     │         │
│  │ Engineering│            │  └─────────┬──────────┘         │
│  └─────┬──────┘            │            │                    │
│        │                   │            ▼                    │
│        ▼                   │  ┌────────────────────┐         │
│  ┌────────────┐            │  │ Batch Views        │         │
│  │ Online     │            │  │ (Parquet tables)   │         │
│  │ Model      │            │  └─────────┬──────────┘         │
│  └─────┬──────┘            │            │                    │
│        │                   │            ▼                    │
│        └──────────┬────────┼────┐                             │
│                   │        │    │                             │
│                   ▼        ▼    │                             │
│            ┌──────────────────┐ │                             │
│            │  Serving Layer   │←┘                             │
│            │  (Merge Results) │                               │
│            └──────────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

**Pros and cons:**
- **Pros:** Complete (handles both real-time and historical), well-understood, fault-tolerant
- **Cons:** Two codebases (batch + stream), complex reconciliation, maintenance overhead

### 9.2 Kappa Architecture

Kappa Architecture simplifies by using a single stream processing pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                    KAPPA ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐                                              │
│  │ Event      │──→ Kafka / Pulsar (immutable log)           │
│  │ Producers  │                                              │
│  └────────────┘                                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Stream Processor (Kafka Streams / Flink) │    │
│  │                                                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │    │
│  │  │ Feature  │  │ Online   │  │ Model Serving    │   │    │
│  │  │ Compute  │→│ Learning │→│ & Evaluation     │   │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │    │
│  └──────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                 Serving Layer                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │    │
│  │  │ Fast Path │  │ Slow Path│  │ Feature Store    │   │    │
│  │  │(<5ms)    │  │(reprocess)│  │ (for training)   │   │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           Historical Replay (Training Data Gen)       │    │
│  │  Replay Kafka from offset T → recompute features     │    │
│  │  → generate training records with point-in-time      │    │
│  │  correctness                                         │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key advantage:** Single processing pipeline — no batch/stream reconciliation

**Implementation principle:** "Everything is a stream." Batch is just replaying the stream at a different rate.

### 9.3 Comparison & Decision Framework

| Criteria | Lambda | Kappa | Hybrid |
|---|---|---|---|
| **Complexity** | High (two codebases) | Medium (one codebase) | Medium-High |
| **Latency** | Low (speed layer) | Low (native stream) | Low |
| **Accuracy** | High (batch correction) | Medium (approximate only) | High |
| **Reprocessing** | Trivial (batch recompute) | Requires stream replay | Flexible |
| **State management** | Separated | Unified | Unified |
| **Cost** | Higher (two pipelines) | Lower (one pipeline) | Medium |
| **Team expertise** | Harder to find | Growing talent pool | Specialized |
| **Best for** | Legacy systems, strict accuracy requirements | New systems, streaming-native teams | Complex real-time + historical needs |

**Decision tree for architecture selection:**

```
Q: Do you have strict accuracy requirements (e.g., financial reconciliation)?
├── YES → Lambda Architecture
│   └── Do you also need sub-second latency?
│       ├── YES → Lambda (speed for fast, batch for correct)
│       └── NO → Lambda (batch only is sufficient)
└── NO → 
    ├── Q: Can you accept approximate results that converge to correct?
    │   ├── YES → Kappa Architecture
    │   └── NO → Hybrid (Lambda-lite)
    │
    └── Q: Is your team experienced with stream processing?
        ├── YES → Kappa (Flink / Kafka Streams)
        └── NO → Lambda (familiar batch + streaming)
```

### 9.4 Hybrid Patterns

**Lambda-lite (practical hybrid):**

```
Stream Path (sub-second):
  Raw Input → Feature Store (online) → Model Serving → Prediction
           ↓                          ↑
Batch Path (hourly):                  │
  Raw Storage → Batch Feature Compute ┘
              → Model Retraining
              → Training Dataset Generation
              → Evaluation & Validation
```

**Feature-based hybrid:**
- **Hot features** (frequently updated, low latency): Stream processing
- **Warm features** (periodically updated): Micro-batch processing
- **Cold features** (rarely updated): Batch processing

**Model-based hybrid:**
- **Fast model** (lightweight, lower accuracy): In-stream, sub-ms inference
- **Slow model** (complex, higher accuracy): Parallel path, results compared
- **Ensemble:** Fast model provides immediate action, slow model verifies and corrects

---

## 10. Performance Benchmarks for Real-Time Inference

### 10.1 Latency Benchmarks

**Inference latency by server (p99, ms, single request, INT8 quantized):**

| Model | Triton (GPU) | TorchServe (GPU) | TF Serving (GPU) | BentoML (CPU) | Triton (CPU) |
|---|---|---|---|---|---|
| ResNet-50 | 0.8 ms | 1.2 ms | 1.0 ms | 3.5 ms | 2.5 ms |
| BERT-base | 3.2 ms | 4.5 ms | 3.8 ms | 12 ms | 8.1 ms |
| XGBoost (300 feat) | 0.3 ms | 0.5 ms | 0.6 ms | 1.2 ms | 0.4 ms |
| LSTM (128 hidden) | 1.5 ms | 2.0 ms | 1.8 ms | 5.0 ms | 3.2 ms |
| GPT-2 small | 8.5 ms | 12 ms | 10 ms | 35 ms | 22 ms |
| Transformer (4 layer) | 2.1 ms | 3.0 ms | 2.5 ms | 8.0 ms | 5.5 ms |

**Batch size impact on latency (BERT-base, Triton GPU):**

```
Batch Size │ p50 Latency │ p95 Latency │ p99 Latency │ Throughput
───────────┼─────────────┼─────────────┼─────────────┼───────────
    1      │   2.8 ms    │   3.2 ms    │   3.5 ms    │  357 QPS
    4      │   3.5 ms    │   4.1 ms    │   4.8 ms    │ 1142 QPS
    8      │   4.2 ms    │   5.0 ms    │   6.1 ms    │ 1904 QPS
    16     │   5.8 ms    │   7.2 ms    │   8.5 ms    │ 2758 QPS
    32     │   8.5 ms    │  10.5 ms    │  12.0 ms    │ 3764 QPS
    64     │  14.2 ms    │  18.0 ms    │  21.0 ms    │ 4507 QPS
    128    │  25.1 ms    │  32.0 ms    │  38.5 ms    │ 5099 QPS
    256    │  48.0 ms    │  60.0 ms    │  72.0 ms    │ 5333 QPS
```

### 10.2 Throughput Benchmarks

**Throughput (requests/second) by serving infrastructure:**

```
Hardware: NVIDIA A100-80GB, Intel Xeon 8480C (48 cores), 256GB RAM

Model: BERT-base (sequence length=128, batch=optimal)

Serving System      │ Throughput (RPS) │ Latency p99 │ Memory
────────────────────┼──────────────────┼─────────────┼────────
Triton (TensorRT)   │    32,450 RPS    │   8.2 ms    │ 4.2 GB
TorchServe          │    18,720 RPS    │  12.5 ms    │ 5.8 GB
TF Serving          │    21,100 RPS    │  10.8 ms    │ 4.8 GB
BentoML + ONNX      │    15,400 RPS    │  14.2 ms    │ 3.9 GB
Seldon + MLServer   │    19,800 RPS    │  11.5 ms    │ 4.5 GB
Ray Serve           │    14,200 RPS    │  16.0 ms    │ 5.2 GB
KServe (Triton)     │    30,100 RPS    │   8.8 ms    │ 4.3 GB

Model: XGBoost (300 features, 500 trees)

Triton (CPU)        │   125,000 RPS    │   0.4 ms    │ 1.2 GB
BentoML (CPU)       │    85,000 RPS    │   0.8 ms    │ 1.1 GB
MLServer (CPU)      │    95,000 RPS    │   0.6 ms    │ 1.3 GB
```

### 10.3 Serving Infrastructure Comparison

| Feature | Triton | TorchServe | TF Serving | BentoML | Seldon/MLServer |
|---|---|---|---|---|---|
| **GPU support** | ✓ Best | ✓ Good | ✓ Good | ✓ Good | ✓ Good |
| **CPU optimization** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Dynamic batching** | ✓ Optimal | ✓ | ✓ | ✓ Micro-batch | ✓ |
| **Model ensembles** | ✓ Native | ✓ Custom | ✗ | ✓ Pipeline | ✓ Graph |
| **gRPC** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **REST** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Multi-framework** | ✓ All | ✓ PyTorch | ✓ TF | ✓ Many | ✓ Many |
| **Prometheus metrics** | ✓ Built-in | ✓ Built-in | ✓ Built-in | ✓ Plugin | ✓ Built-in |
| **Model versioning** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **A/B testing** | ✓ | ✓ | ✓ Via labels | ✓ | ✓ Traffic split |
| **Kubernetes native** | ✓ | ✓ | ✓ | ✓ | ✓ (Seldon) |
| **Cost** | Free | Free | Free | Free (OSS) | Free (OSS) |
| **GPU memory mgmt** | ✓ Best | ✓ Good | ✓ Good | ✓ Managed | ✓ Configurable |
| **Performance** | Best | Good | Very Good | Good | Very Good |

### 10.4 Optimization Techniques

**End-to-end latency optimization checklist:**

```
□ 1. Model Optimization
   □ Quantize to INT8 (if accuracy permits)
   □ Prune redundant weights
   □ Distill large model into smaller student
   □ Export to TensorRT / ONNX Runtime
   □ Fuse operations (layernorm, activation, etc.)

□ 2. Serving Optimization
   □ Enable dynamic batching (find optimal batch size)
   □ Set appropriate max_queue_delay_microseconds
   □ Use gRPC instead of REST (faster serialization)
   □ Configure instance groups (KIND_GPU / KIND_CPU)
   □ Set number of model instances based on model size

□ 3. Infrastructure Optimization
   □ Use GPU with high memory bandwidth (H100/B200)
   □ Enable NUMA-aware inference (bind to CPU cores)
   □ Use RDMA networking (InfiniBand / RoCE)
   □ Pin model to specific GPU via CUDA_VISIBLE_DEVICES
   □ Configure Kubernetes HPA with ML-specific metrics

□ 4. Data Path Optimization
   □ Pre-encode categorical features as integers
   □ Pre-allocate numpy arrays (avoid reallocation)
   □ Use memory-mapped feature store (Redis cluster with replication)
   □ Implement connection pooling for feature store
   □ Add prediction cache for repeated queries (LRU)

□ 5. Application Optimization
   □ Set client-side timeout (fail fast on degradation)
   □ Implement circuit breaker for downstream services
   □ Add request prioritization (urgent vs. async)
   □ Use async I/O (asyncio, gRPC async)
   □ Profile with CUDA tools (Nsys, Nsight)
```

**Profiling command examples:**

```bash
# Profile Triton with nsys
nsys profile -o triton_profile -t cuda,nvtx,cudnn \
    tritonserver --model-repository=/models

# GPU performance with nvidia-smi
nvidia-smi dmon -d 1 -s pucvmet

# CUDA kernel analysis
nvprof --analysis-metrics -o analysis.nvprof \
    python client.py

# Model throughput with perf_analyzer (Triton)
perf_analyzer -m fraud_model \
    --concurrency-range 1:64:8 \
    --measurement-interval 10000 \
    --latency-report

# ONNX Runtime benchmark
python -m onnxruntime.benchmark \
    --model fraud_model.onnx \
    --input_shape 1,300 \
    --iterations 10000 \
    --warmup 1000
```

---

## 11. Future Outlook

**Near-term (2026–2027):**
- Real-time inference at single-digit microsecond latency via custom silicon
- Streaming feature stores becoming as ubiquitous as traditional databases
- AI-native stream processors (Flink ML, Kafka ML) replacing manual pipelining
- Edge AI maturation: models run on $5 microcontrollers with 10ms latency

**Medium-term (2027–2028):**
- Real-time multimodal AI (video + audio + text inference in single stream)
- Self-adapting models: continuous learning in production with automatic drift handling
- Real-time AI for physical world (autonomous vehicles, robotics, manufacturing)
- Federated real-time learning: models that learn across edge devices without data centralization

**Long-term (2028–2030):**
- Real-time AI at planetary scale: trillions of predictions per day
- Streaming AGI: continuous learning systems that never stop training
- Real-time causal inference: beyond correlation to causal reasoning in streaming data
- AI-managed real-time systems: AI that auto-optimizes inference infrastructure

**Key challenges:**
- **State management** — Maintaining correct state across distributed streaming ML
- **Cold start** — Models must produce good predictions from the first observation
- **Feedback loops** — Real-time predictions that affect future data distributions
- **Monitoring complexity** — Real-time systems generate more alerts than humans can handle
- **Cost of real-time** — Sub-millisecond inference at scale requires significant infrastructure investment
- **Debugging** — Real-time ML failures are hard to reproduce and diagnose

---

*This document is current as of June 2026. Real-time AI infrastructure evolves rapidly; benchmark performance, tool versions, and deployment patterns may change. Cross-reference with 02-AI-Agent-Development.md for agent-based real-time systems, 06-RAG-Retrieval-Systems.md for real-time retrieval patterns, and 08-Edge-AI-Inference.md for edge-specific optimization.*
