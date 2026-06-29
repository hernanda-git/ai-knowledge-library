# 04 — Tools and Frameworks: The 2026 Digital Twin Technology Stack

> **Why this document exists.** Building a digital twin requires choosing from an increasingly complex landscape of platforms, frameworks, databases, and tools. This document provides a comprehensive comparison of every major tool in the 2026 digital twin stack — from physics simulation engines to visualization platforms, from time-series databases to IoT gateways. It includes decision frameworks, integration patterns, and cost comparisons to help you build the right stack for your use case.

---

## Table of Contents

1. [Platform Selection Framework](#1-platform-selection-framework)
2. [Simulation Engines and Physics Libraries](#2-simulation-engines-and-physics-libraries)
3. [AI/ML Frameworks for Twin Intelligence](#3-aiml-frameworks-for-twin-intelligence)
4. [Time-Series Databases](#4-time-series-databases)
5. [IoT Platforms and Gateways](#5-iot-platforms-and-gateways)
6. [3D Visualization and Rendering](#6-3d-visualization-and-rendering)
7. [Data Integration and Streaming](#7-data-integration-and-streaming)
8. [Cloud Platform Comparison](#8-cloud-platform-comparison)
9. [Open Source vs. Commercial: Decision Matrix](#9-open-source-vs-commercial-decision-matrix)
10. [Integration Architecture Patterns](#10-integration-architecture-patterns)

---

## 1. Platform Selection Framework

### 1.1 Decision Tree

```
START: What type of twin?
|
+-- Physical Asset (machine, building, vehicle)
|  +-- High-fidelity physics needed?
|  |  +-- YES -> NVIDIA Omniverse + Modulus
|  |  +-- NO  -> Azure Digital Twins or AWS IoT TwinMaker
|  +-- Manufacturing domain?
|     +-- YES -> Siemens Xcelerator
|     +-- NO  -> Generic platform
|
+-- Process/Workflow (business process, supply chain)
|  +-- Discrete events?
|  |  +-- YES -> AnyLogic or custom agent-based
|  |  +-- NO  -> Digital twin + process mining
|  +-- Real-time optimization needed?
|     +-- YES -> reinforcement learning + simulation
|     +-- NO  -> Analytics dashboard
|
+-- Environment/Space (building, city, terrain)
|  +-- BIM/Facility?
|  |  +-- YES -> Bentley iTwin or Autodesk Tandem
|  |  +-- NO  -> Urban planning?
|  +-- YES -> NVIDIA Omniverse CitySim
|     +-- NO  -> Geographic?
|        +-- YES -> Google Earth Engine + custom
|        +-- NO  -> Generic 3D platform
|
+-- Biological/Chemical (organ, molecule, cell)
   +-- Organ level?
   |  +-- YES -> HeartFlow or custom FEA
   |  +-- NO  -> Molecular level?
   +-- YES -> OpenMM or custom MD simulation
```

### 1.2 Evaluation Criteria Weighting

| Criterion | Weight | Questions to Ask |
|-----------|--------|-----------------|
| **Physics Fidelity** | 20% | How accurate must simulation be? |
| **Real-Time Performance** | 20% | What is the maximum acceptable latency? |
| **Scale** | 15% | How many twins? Sensors per twin? |
| **AI Integration** | 15% | What AI models need integration? |
| **Cost** | 10% | What is the budget? (TCO over 3 years) |
| **Vendor Lock-in** | 10% | How important is vendor independence? |
| **Team Expertise** | 5% | What skills does the team have? |
| **Ecosystem** | 5% | How large is the plugin/integration ecosystem? |

---

## 2. Simulation Engines and Physics Libraries

### 2.1 Comparison Matrix

| Engine | Type | Language | License | Best For | Latency |
|--------|------|----------|---------|----------|---------|
| **NVIDIA Modulus** | Physics-informed ML | Python/C++ | BSD | Hybrid physics-AI | <10ms |
| **Ansys Fluent** | CFD | C++ | Commercial | Fluid dynamics | Hours (GPU: min) |
| **Ansys Mechanical** | FEA | C++ | Commercial | Structural analysis | Hours |
| **OpenFOAM** | CFD | C++ | GPL | Budget CFD | Hours |
| **Siemens Simcenter** | Multi-physics | C++ | Commercial | Industrial multi-physics | Minutes |
| **Dassault SIMULIA** | FEA/CFD | C++ | Commercial | Academic/research | Hours |
| **PyBullet** | Multi-body dynamics | Python/C | Zlib | Robotics, contact | Real-time |
| **MuJoCo** | Multi-body dynamics | C++ | Apache 2.0 | Robotics RL | Real-time |
| **Isaac Sim** | Robotics simulation | Python/C++ | NVIDIA | Robot twins | Real-time |
| **FlexSim** | Discrete event | Custom | Commercial | Manufacturing, logistics | Real-time |
| **AnyLogic** | Multi-method | Java | Commercial | Supply chain, business | Real-time |
| **MATLAB/Simulink** | Multi-physics | MATLAB | Commercial | Control systems | Real-time |

### 2.2 NVIDIA Modulus Deep Dive

NVIDIA Modulus is the emerging standard for physics-informed ML in digital twins:

```python
# Example: Building a thermal digital twin with Modulus
import modulus
from modulus.models.mlp import FullyConnectedArch
from modulus.domain import Domain
from modulus.geometry import Channel2D

# Define the physics-informed neural network
arch = FullyConnectedArch(
    input_keys=['x', 'y', 't'],
    output_keys=['T', 'q_x', 'q_y'],
    layer_size=128,
    nr_layers=6
)

# Define PDE: Heat equation with variable conductivity
@modulus.symbolic
def heat_equation(T, q_x, q_y, x, y, t):
    return {
        'heat_eq': T.diff(t) - (q_x.diff(x) + q_y.diff(y))
    }

# Define domain with boundary conditions
geo = Channel2D((0, 1), (0, 1))
domain = Domain()
domain.add_constraint(
    modulus.constraints.InteriorConstraint(
        equations=[heat_equation],
        geometry=geo,
        batch_size=1000
    )
)

# Result: Real-time thermal prediction in <10ms per timestep
# vs 45 seconds for traditional FEA
```

### 2.3 Hybrid Simulation Stack

```
Layer 4: AI MODELS
  Anomaly Detection | RUL Prediction | Optimization | NLP
  (TensorRT) | (ONNX) | (RLlib) | (vLLM)

Layer 3: NEURAL OPERATORS (Modulus)
  Fourier Neural Operator | DeepONet | PINN
  Physics-constrained ML for PDE solving

Layer 2: TRADITIONAL SIMULATION
  FEA (structural) | CFD (fluid) | MBD (mechanics)
  High fidelity, offline calibration

Layer 1: EMPIRICAL MODELS
  Look-up tables | Curve fitting | Statistical
  Fast, simple, limited extrapolation
```

---

## 3. AI/ML Frameworks for Twin Intelligence

### 3.1 Framework Comparison

| Framework | Focus | Edge Support | ONNX Export | Best For |
|-----------|-------|-------------|-------------|----------|
| **PyTorch** | General ML | Via TorchScript | Yes | Research, custom models |
| **TensorFlow** | General ML | Via TFLite | Yes | Production, serving |
| **TensorRT** | Inference optimization | Native (Jetson) | Yes (input) | Edge inference, <5ms |
| **ONNX Runtime** | Cross-framework inference | Yes | Yes (native) | Model portability |
| **NVIDIA Modulus** | Physics-informed ML | Yes | Yes | Hybrid physics-AI |
| **Ray/RLlib** | Distributed RL | No | No | Reinforcement learning |
| **Hugging Face** | NLP/LLM | Via Optimum | Yes | LLM twin interfaces |
| **Scikit-learn** | Classical ML | Via ONNX | Yes | Simple models, anomaly |

### 3.2 Recommended Model Stack by Twin Type

| Twin Type | Primary Model | Supporting Models | Framework |
|-----------|--------------|-------------------|-----------|
| **Predictive Maintenance** | Temporal Fusion Transformer | Isolation Forest, LSTM | PyTorch + ONNX |
| **Process Optimization** | PPO (Reinforcement Learning) | Bayesian Optimization | Ray/RLlib |
| **Anomaly Detection** | Autoencoder + Mahalanobis | One-Class SVM, LOF | PyTorch + Scikit-learn |
| **Demand Forecasting** | N-BEATS, TFT | Prophet, ARIMA | PyTorch |
| **Quality Prediction** | XGBoost + Neural Network | Random Forest, SVR | XGBoost + PyTorch |
| **NL Interface** | Fine-tuned LLM (7B params) | RAG with twin knowledge | vLLM + Hugging Face |

### 3.3 MLOps for Twin Models

```python
class TwinMLOps:
    """
    End-to-end ML lifecycle management for digital twin AI models.
    Tracks experiments, manages deployments, monitors performance.
    """
    
    def __init__(self, mlflow_uri, model_registry):
        self.mlflow = MlflowClient(mlflow_uri)
        self.registry = model_registry
    
    def train_and_register(self, twin_type, training_data, config):
        """Train a model and register it in the model registry."""
        with mlflow.start_run(run_name=f"{twin_type}_training"):
            mlflow.log_params(config)
            
            model = self.build_model(twin_type, config)
            metrics = model.train(training_data)
            
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")
            
            model_version = self.registry.register(
                name=f"{twin_type}_model",
                artifact_uri=mlflow.get_artifact_uri("model"),
                metrics=metrics
            )
            return model_version
    
    def deploy_with_monitoring(self, twin_id, model_version):
        """Deploy model to a twin with monitoring."""
        self.edge_gateway.deploy_model(twin_id, model_version.id)
        
        monitor = TwinModelMonitor(
            twin_id=twin_id,
            model_version=model_version,
            metrics={
                'prediction_accuracy': {'threshold': 0.90, 'window': '24h'},
                'inference_latency': {'threshold_ms': 50, 'window': '1h'},
                'feature_drift': {'threshold': 0.1, 'window': '7d'}
            }
        )
```

---

## 4. Time-Series Databases

### 4.1 Comprehensive Comparison

| Database | Write (pts/s) | Query | Compression | SQL | Price |
|----------|-------------|-------|-------------|-----|-------|
| **InfluxDB 3.0** | 500K | Fast | 10:1 | InfluxQL/SQL | $$/mo |
| **TimescaleDB** | 200K | Very fast | 8:1 | Full SQL | Free/$ |
| **QuestDB** | 1M+ | Very fast | 12:1 | SQL | Free/$ |
| **Apache IoTDB** | 300K | Fast | 15:1 | SQL | Free |
| **Druid** | 1M+ | Fast | 6:1 | SQL | $$ |
| **ClickHouse** | 500K+ | Very fast | 8:1 | SQL | Free/$ |
| **TDengine** | 1M+ | Fast | 10:1 | SQL | Free/$ |

### 4.2 Selection Guide

| Use Case | Recommended DB | Reason |
|----------|---------------|--------|
| **IoT sensor data (simple)** | InfluxDB 3.0 | Purpose-built, easy setup |
| **IoT + business analytics** | TimescaleDB | Full SQL, joins with business data |
| **High-throughput ingestion** | QuestDB | Fastest writes, lowest latency |
| **Industrial IoT (OPC-UA)** | Apache IoTDB | Industrial protocol support |
| **Real-time analytics + BI** | Druid or ClickHouse | Sub-second OLAP queries |
| **Streaming + state** | ksqlDB | Integrated with Kafka |

### 4.3 Schema Design Patterns

```sql
-- Pattern 1: Wide table (one column per sensor)
CREATE TABLE twin_wide (
    time TIMESTAMPTZ NOT NULL,
    twin_id TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    vibration_x DOUBLE PRECISION,
    vibration_y DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    rpm DOUBLE PRECISION
);

-- Pattern 2: Narrow table (one row per reading)
CREATE TABLE twin_narrow (
    time TIMESTAMPTZ NOT NULL,
    twin_id TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    value DOUBLE PRECISION,
    quality DOUBLE PRECISION
);

-- Pattern 3: Event store (immutable events)
CREATE TABLE twin_events (
    event_id UUID PRIMARY KEY,
    twin_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    sequence_number BIGINT
);
```

---

## 5. IoT Platforms and Gateways

### 5.1 IoT Platform Comparison

| Platform | Protocol Support | Edge Computing | Twin Native | Price |
|----------|-----------------|----------------|-------------|-------|
| **Azure IoT Hub** | MQTT, AMQP, HTTPS | Azure IoT Edge | Azure Digital Twins | $$ |
| **AWS IoT Core** | MQTT, HTTPS, LoRaWAN | Greengrass | IoT TwinMaker | $$ |
| **Google Cloud IoT** | MQTT, HTTP | Edge TPU | Cloud IoT | $$ |
| **Siemens MindSphere** | OPC-UA, MQTT | MindConnect | Native | $$$ |
| **PTC ThingWorx** | OPC-UA, MQTT, REST | Kepware | Native | $$$ |
| **Eclipse Ditto** | MQTT, AMQP | No | Twin-native (open) | Free |
| **HiveMQ** | MQTT only | No | No | Free/$ |

### 5.2 Edge Gateway Architecture

```
+----------------------------------------------------------+
|                  EDGE GATEWAY                             |
|                                                           |
|  SENSOR INTERFACE                                         |
|  OPC-UA | Modbus | MQTT | HTTP | BLE | Zigbee            |
|                         |                                 |
|  DATA PROCESSING                                           |
|  Filter | Aggregate | Transform | Validate                |
|                         |                                 |
|  EDGE AI INFERENCE                                         |
|  Anomaly Detection | Classification | Forecasting         |
|  (TensorRT / ONNX Runtime / TFLite)                      |
|                         |                                 |
|  LOCAL STORAGE                                             |
|  InfluxDB (local) | SQLite | File system                  |
|                         |                                 |
|  CLOUD CONNECTIVITY                                       |
|  MQTT | AMQP | HTTPS | gRPC                              |
+----------------------------------------------------------+
```

### 5.3 Eclipse Ditto: Open-Source Digital Twin Protocol

```json
{
  "thingId": "org.eclipse.ditto:device-001",
  "definition": "org.eclipse.ditto:TemperatureSensor:1.0",
  "attributes": {
    "manufacturer": "Siemens",
    "location": "Factory-A-Floor-3"
  },
  "features": {
    "temperature": {
      "properties": {
        "value": 42.5,
        "unit": "celsius",
        "status": "normal"
      }
    },
    "vibration": {
      "properties": {
        "x": 0.12,
        "y": 0.08,
        "z": 0.15,
        "rms": 0.22
      }
    }
  }
}
```

---

## 6. 3D Visualization and Rendering

### 6.1 Visualization Platform Comparison

| Platform | Rendering | Real-Time | AR/VR | Web | Price |
|----------|-----------|-----------|-------|-----|-------|
| **NVIDIA Omniverse** | RTX path tracing | Yes | Yes | No | $$$$ |
| **Unity** | RT, URP, HDRP | Yes | Yes | WebGL | Free/$$ |
| **Unreal Engine 5** | Nanite, Lumen | Yes | Yes | WebXR | Free/$$ |
| **Three.js** | WebGL | Yes | WebXR | Yes | Free |
| **Babylon.js** | WebGL/WebGPU | Yes | WebXR | Yes | Free |
| **CesiumJS** | 3D geospatial | Yes | No | Yes | Free |
| **Autodesk Viewer** | Custom | Yes | No | Yes | Free/$ |

### 6.2 WebGL Twin Viewer Example

```javascript
// Three.js-based digital twin viewer (simplified)
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class TwinViewer {
  constructor(containerId) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, 
      window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    document.getElementById(containerId)
      .appendChild(this.renderer.domElement);
    this.componentMeshes = new Map();
  }
  
  async loadTwin(twinId) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(`/api/twins/${twinId}/model`);
    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        this.componentMeshes.set(child.name, child);
        child.userData.onClick = () => this.inspectComponent(child.name);
      }
    });
    this.scene.add(gltf.scene);
  }
  
  updateComponentState(componentId, state) {
    const mesh = this.componentMeshes.get(componentId);
    if (!mesh) return;
    const healthColor = this.healthToColor(state.health_score);
    mesh.material.color.setHex(healthColor);
    const vibrationScale = 1 + state.vibration_rms * 0.1;
    mesh.scale.setScalar(vibrationScale);
  }
  
  healthToColor(health) {
    if (health > 0.9) return 0x00ff00;  // Green
    if (health > 0.7) return 0xffff00;  // Yellow
    if (health > 0.5) return 0xff8800;  // Orange
    return 0xff0000;                     // Red
  }
}
```

---

## 7. Data Integration and Streaming

### 7.1 Streaming Platform Comparison

| Platform | Throughput | Latency | Exactly-Once | Schema Registry | Price |
|----------|-----------|---------|-------------|-----------------|-------|
| **Apache Kafka** | 1M+ msg/s | <10ms | Yes | Yes | Free/$$$ |
| **Apache Pulsar** | 1M+ msg/s | <10ms | Yes | Yes | Free/$$ |
| **AWS Kinesis** | 1M+ msg/s | <100ms | Yes | No | $$ |
| **Google Pub/Sub** | 1M+ msg/s | <100ms | At-least-once | No | $$ |
| **Azure Event Hubs** | 1M+ msg/s | <100ms | Yes | No | $$ |
| **Redpanda** | 1M+ msg/s | <5ms | Yes | Yes | Free/$$ |
| **NATS JetStream** | 100K+ msg/s | <1ms | Yes | No | Free |

### 7.2 OPC-UA Integration for Industrial Twins

```python
# OPC-UA to Kafka bridge for industrial digital twins
from opcua import Client
from confluent_kafka import Producer
import json

class OPCUAToKafkaBridge:
    """
    Bridges OPC-UA industrial protocol data to Kafka
    for digital twin processing.
    """
    
    def __init__(self, opcua_endpoint, kafka_brokers):
        self.opcua = Client(opcua_endpoint)
        self.kafka = Producer({
            'bootstrap.servers': kafka_brokers,
            'linger.ms': 5,
            'compression.type': 'lz4'
        })
    
    def subscribe_to_machine(self, machine_node_id, twin_id):
        """Subscribe to OPC-UA data changes for a machine."""
        sensor_map = {
            'ns=2;s=Temperature': f'{twin_id}.temperature',
            'ns=2;s=Vibration_X': f'{twin_id}.vibration_x',
            'ns=2;s=Vibration_Y': f'{twin_id}.vibration_y',
            'ns=2;s=Pressure': f'{twin_id}.pressure',
            'ns=2;s=RPM': f'{twin_id}.rpm',
        }
        
        for opcua_node, sensor_id in sensor_map.items():
            node = self.opcua.get_node(opcua_node)
            handler = OPCUAHandler(sensor_id, self.kafka)
            subscription = self.opcua.create_subscription(
                period=100,  # 100ms
                handler=handler
            )
            subscription.subscribe_data_change(node)

class OPCUAHandler:
    def __init__(self, sensor_id, kafka_producer):
        self.sensor_id = sensor_id
        self.kafka = kafka_producer
    
    def datachange_notification(self, node, val, data):
        event = {
            'sensor_id': self.sensor_id,
            'value': val,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'opcua'
        }
        self.kafka.produce(
            topic=f'dt.sensor.raw.{self.sensor_id}',
            key=self.sensor_id.encode(),
            value=json.dumps(event).encode()
        )
        self.kafka.poll(0)
```

---

## 8. Cloud Platform Comparison

### 8.1 Cloud Digital Twin Services

| Service | AWS | Azure | GCP |
|---------|-----|-------|-----|
| **IoT Ingestion** | IoT Core | IoT Hub | IoT Core |
| **Edge Computing** | Greengrass | IoT Edge | Edge TPU |
| **Twin Platform** | IoT TwinMaker | Azure Digital Twins | Cloud IoT |
| **AI/ML** | SageMaker | Azure ML | Vertex AI |
| **Time-Series** | Timestream | Data Explorer | BigQuery |
| **Visualization** | Custom (Three.js) | Power BI / Mixed Reality | Custom |
| **Stream Processing** | Kinesis Analytics | Stream Analytics | Dataflow |

### 8.2 Cost Comparison (1,000 Twins, 50 Sensors, 10 Hz)

| Component | AWS | Azure | GCP |
|-----------|-----|-------|-----|
| **IoT Ingestion** | $500/mo | $400/mo | $450/mo |
| **Stream Processing** | $2,000/mo | $1,800/mo | $2,200/mo |
| **Twin Platform** | $1,500/mo | $2,000/mo | $1,200/mo |
| **Time-Series DB** | $3,000/mo | $2,500/mo | $3,500/mo |
| **AI/ML (GPU)** | $5,000/mo | $5,500/mo | $4,800/mo |
| **Storage** | $1,500/mo | $1,200/mo | $1,400/mo |
| **Network** | $800/mo | $900/mo | $700/mo |
| **Total** | **$14,300/mo** | **$14,300/mo** | **$14,250/mo** |

---

## 9. Open Source vs. Commercial: Decision Matrix

### 9.1 Open Source Stack

| Component | Tool | License | Total Cost |
|-----------|------|---------|------------|
| **IoT Protocol** | Eclipse Mosquitto (MQTT) | EPL/EDL | Free |
| **Twin Protocol** | Eclipse Ditto | EPL 2.0 | Free |
| **Stream Processing** | Apache Flink | Apache 2.0 | Free (infra cost) |
| **Time-Series DB** | TimescaleDB / QuestDB | Apache 2.0 | Free (infra cost) |
| **AI/ML** | PyTorch + ONNX | BSD/MIT | Free |
| **Edge Runtime** | NVIDIA Jetson + TensorRT | Various | Hardware cost |
| **3D Visualization** | Three.js + React Three Fiber | MIT | Free |
| **Orchestration** | Kubernetes | Apache 2.0 | Free (infra cost) |
| **Total (1000 twins)** | | | **$8,000-15,000/mo** |

### 9.2 Commercial Stack

| Component | Tool | Annual License | Total Cost |
|-----------|------|---------------|------------|
| **IoT Platform** | Azure IoT Hub + Digital Twins | $50K-200K | $200K/yr |
| **Simulation** | Siemens Simcenter | $100K-500K | $300K/yr |
| **AI/ML Platform** | Dataiku / Databricks | $100K-300K | $200K/yr |
| **Visualization** | NVIDIA Omniverse Enterprise | $100K-400K | $300K/yr |
| **Support** | Enterprise support contracts | $50K-200K | $150K/yr |
| **Total (1000 twins)** | | | **$350K-1.15M/yr** |

### 9.3 Decision Factors

| Factor | Favors Open Source | Favors Commercial |
|--------|-------------------|-------------------|
| **Budget** | Limited budget | Large budget available |
| **Team** | Strong engineering team | Small team, need vendor support |
| **Customization** | High customization needs | Standard use cases |
| **Timeline** | Long-term investment | Rapid deployment needed |
| **Domain** | Novel/unique use case | Standard manufacturing/energy |
| **Compliance** | Flexible compliance needs | Strict regulatory requirements |
| **Scale** | Growing scale, need flexibility | Known scale, proven solutions |

---

## 10. Integration Architecture Patterns

### 10.1 Event-Driven Integration

```
+----------+     +----------+     +----------+
|   ERP    |     |   MES    |     |  SCADA   |
| (SAP/    |     | (Siemens |     | (Rockwell|
|  Oracle) |     | Opcenter)|     |    )     |
+----+-----+     +----+-----+     +----+-----+
     |               |               |
     +---------------+---------------+
                     |
               +-----+-----+
               |   KAFKA   |
               |  (Event   |
               |   Bus)    |
               +-----+-----+
                     |
      +--------------+--------------+
      |              |              |
+-----+------+ +----+-----+ +-----+------+
| Twin       | | Twin     | | Twin       |
| Processor  | | AI       | | Alert      |
| (Real-time)| | Models   | | Service    |
+-----+------+ +----+-----+ +-----+------+
      |              |              |
      +--------------+--------------+
                     |
               +-----+-----+
               |  Digital  |
               |  Twin     |
               |  State    |
               |  Store    |
               +-----+-----+
                     |
      +--------------+--------------+
      |              |              |
+-----+------+ +----+-----+ +------+-----+
| Dashboard  | | AR/VR    | | Mobile     |
| (Grafana)  | | Client   | | App        |
+------------+ +----------+ +------------+
```

### 10.2 API Gateway Pattern

```python
# FastAPI-based digital twin API gateway
from fastapi import FastAPI, WebSocket, HTTPException

app = FastAPI(title="Digital Twin API Gateway", version="2.0")

class TwinGateway:
    def __init__(self):
        self.twin_registry = TwinRegistry()
        self.data_service = TwinDataService()
        self.ai_service = TwinAIService()
        self.sim_service = TwinSimulationService()
    
    async def get_twin_state(self, twin_id: str):
        twin = await self.twin_registry.get(twin_id)
        if not twin:
            raise HTTPException(status_code=404, detail="Twin not found")
        state = await self.data_service.get_current_state(twin_id)
        predictions = await self.ai_service.get_predictions(twin_id)
        return {
            'twin_id': twin_id,
            'type': twin.type,
            'status': twin.status,
            'state': state,
            'predictions': predictions,
            'last_updated': state.timestamp
        }
    
    async def simulate(self, twin_id: str, scenario: dict):
        current_state = await self.data_service.get_current_state(twin_id)
        result = await self.sim_service.run_scenario(
            twin_id=twin_id,
            initial_state=current_state,
            scenario=scenario,
            duration_seconds=scenario.get('duration', 3600)
        )
        return {
            'twin_id': twin_id,
            'scenario': scenario,
            'result': result
        }
    
    async def websocket_stream(self, twin_id: str, websocket: WebSocket):
        await websocket.accept()
        subscription = self.data_service.subscribe(twin_id)
        try:
            while True:
                update = await subscription.get_update()
                await websocket.send_json({
                    'type': 'state_update',
                    'data': update
                })
        except Exception:
            await websocket.close()

gateway = TwinGateway()

@app.get("/api/v2/twins/{twin_id}")
async def get_twin(twin_id: str):
    return await gateway.get_twin_state(twin_id)

@app.post("/api/v2/twins/{twin_id}/simulate")
async def simulate_twin(twin_id: str, scenario: dict):
    return await gateway.simulate(twin_id, scenario)

@app.websocket("/ws/twins/{twin_id}")
async def twin_stream(websocket: WebSocket, twin_id: str):
    await gateway.websocket_stream(twin_id, websocket)
```

---

## Cross-References

| Topic | Document |
|-------|----------|
| Overview | `39-Digital-Twins/01-Overview.md` |
| Core Technical Topics | `39-Digital-Twins/02-Core-Technical-Topics.md` |
| Technical Deep Dive | `39-Digital-Twins/03-Technical-Deep-Dive.md` |
| Future Outlook | `39-Digital-Twins/05-Future-Outlook.md` |
| Custom Silicon | `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` |
| Edge AI | `23-Local-AI-Inference-Self-Hosting/` |
| Manufacturing AI | `11-AI-Applications/04-Manufacturing-AI.md` |
| Robotics | `10-Industry/03-AI-for-Robotics.md` |
| Enterprise Deployment | `05-Enterprise/01-Enterprise-AI-Deployment.md` |
| Agent Security | `18-Agent-Security-and-Trust/` |

---

*Last updated: June 2026*
*Category: 39-Digital-Twins*
