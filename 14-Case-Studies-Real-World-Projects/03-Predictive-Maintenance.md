# 03 — Predictive Maintenance for Manufacturing

## Case Study: AI-Driven Predictive Maintenance on Industrial Equipment

| Metadata | Value |
|----------|-------|
| **Industry** | Manufacturing / Industrial IoT |
| **Domain** | Predictive maintenance, anomaly detection |
| **Difficulty** | Advanced |
| **Est. Timeline** | 8-12 weeks |
| **Team Size** | 5-8 engineers (2 IoT, 2 ML, 2 backend, 1 domain expert) |

---

## 🎯 Problem Statement

### Business Context

**Company:** MetalFab Industries (mid-size automotive parts manufacturer, $500M revenue)
**Facility:** 4 factories with 2,500+ machines (CNC, presses, conveyors, robotic arms)
**Maintenance Budget:** $8M/year on reactive + scheduled maintenance

### Pain Points

1. **Unplanned Downtime** — Average 72 hours/month of unplanned downtime across facilities; each hour costs $15,000 in lost production
2. **Reactive Maintenance** — 65% of maintenance is reactive (fix after breakdown); only 20% planned preventive, 15% condition-based
3. **Cascading Failures** — Single bearing failure leads to motor failure → conveyor line downtime → production line stop; average cascade adds 400% to repair cost
4. **Inventory Waste** — $2.3M/year in emergency part ordering (rush shipping + markup)
5. **Safety Incidents** — 12 safety incidents in past year related to unexpected equipment failures
6. **Warranty Loss** — $3.5M in warranty claims denied due to "improper maintenance documentation"

### Success Criteria

| Metric | Target | Baseline |
|--------|--------|---------|
| Unplanned Downtime Reduction | -40% | 72 hrs/month |
| Maintenance Cost Reduction | -30% | $8M/year |
| False Positive Rate | < 5% | N/A |
| Prediction Lead Time | > 48 hours | 0 (reactive) |
| MTBF (Mean Time Between Failures) | +50% | 1,200 hours |
| Spare Parts Inventory Cost | -25% | $2.3M/year |

### Constraints

- Sensors already installed on 70% of critical assets (retrofit remaining)
- Factory network is air-gapped from internet; data must go through OT gateway
- PLC/Siemens S7 controllers; must support Modbus TCP, OPC-UA, MQTT
- ML models must run on-premises (factory cannot have cloud dependency for real-time decisions)
- Maintenance team of 45 people — system must be usable by technicians with limited data literacy

---

## 🏗️ Solution Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FACTORY FLOOR (OT NETWORK)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ASSET TIER (2500+ MACHINES)                     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │   CNC    │ │  Press   │ │ Conveyor │ │   Robot  │ │  Pump    │   │    │
│  │  │ Machine  │ │  #1-#12  │ │  #1-#24  │ │  Arm #1  │ │  #1-#50  │   │    │
│  │  └─────┬────┘ └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │    │
│  │        │           │           │            │           │          │    │
│  │        ▼           ▼           ▼            ▼           ▼          │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │           SENSORS (Temp, Vibration, Current, RPM,           │    │    │
│  │  │            Pressure, Acoustic, Torque, Humidity)            │    │    │
│  │  └──────────────────────────┬──────────────────────────────────┘    │    │
│  └─────────────────────────────┼────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    EDGE GATEWAY TIER                                  │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐    │    │
│  │  │  Siemens IoT2000 │  │   Raspberry Pi  │  │  Advantech Edge   │    │    │
│  │  │  (PLC gateway)   │  │   (sensor hub)  │  │  Server (x86)     │    │    │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘    │    │
│  │           │                    │                     │                │    │
│  │           ▼                    ▼                     ▼                │    │
│  │  ┌──────────────────────────────────────────────────────────────┐    │    │
│  │  │           MQTT Broker (EMQX — on-premise)                   │    │    │
│  │  │           Protocol: OPC-UA / Modbus TCP / MQTT              │    │    │
│  │  └──────────────────────────┬───────────────────────────────────┘    │    │
│  └─────────────────────────────┼──────────────────────────────────────────┘    │
│                                │                                             │
│  ┌─────────────────────────────┼──────────────────────────────────────────┐    │
│  │                    DMZ / DATA INGESTION TIER               │             │    │
│  │                                ▼                           │             │    │
│  │  ┌──────────────────────────────────────────────────────┐  │             │    │
│  │  │         Apache Kafka (Self-hosted, 3-node cluster)   │  │             │    │
│  │  │         Topics: sensor-raw, preprocessed, alerts     │  │             │    │
│  │  └────────────────────────┬─────────────────────────────┘  │             │    │
│  │                           │                                │             │    │
│  └───────────────────────────┼────────────────────────────────┘             │    │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                    FACTORY IT NETWORK (ON-PREMISE DEDICATED)                 │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      STREAM PROCESSING LAYER                          │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐  │   │
│  │  │  Kafka Streams    │  │  Apache Flink     │  │  Feature Store   │  │   │
│  │  │  (aggregations)   │  │  (windowed joins) │  │  (Hopsworks)     │  │   │
│  │  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────┘  │   │
│  │           │                      │                      │            │   │
│  │           ▼                      ▼                      ▼            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │              TIME-SERIES DATABASE (InfluxDB)                  │    │   │
│  │  │              + PostgreSQL (asset metadata + maintenance       │    │   │
│  │  │              history + work orders)                           │    │   │
│  │  └──────────────────────┬───────────────────────────────────────┘    │   │
│  │                         │                                            │   │
│  └─────────────────────────┼────────────────────────────────────────────┘   │
│                           │                                                │
│  ┌─────────────────────────┼────────────────────────────────────────────┐   │
│  │                 MODEL INFERENCE LAYER                                 │   │
│  │                           ▼                                           │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │  MODEL SERVING (Triton Inference Server on 4× NVIDIA T4 GPUs)  │   │   │
│  │  │                                                               │   │   │
│  │  │  ┌──────────────────┐    ┌──────────────────┐                │   │   │
│  │  │  │  Anomaly         │    │  Remaining       │                │   │   │
│  │  │  │  Autoencoder     │    │  Useful Life     │                │   │   │
│  │  │  │  (reconstruction │    │  (XGBoost Regr.) │                │   │   │
│  │  │  │   error score)   │    │                  │                │   │   │
│  │  │  └────────┬─────────┘    └────────┬─────────┘                │   │   │
│  │  │           │                       │                          │   │   │
│  │  │           ▼                       ▼                          │   │   │
│  │  │  ┌──────────────────────────────────────────────────────┐    │   │   │
│  │  │  │  FUSION: Multi-model ensemble decision logic         │    │   │   │
│  │  │  │  anomaly_score > threshold OR RUL < 200h → alert    │    │   │   │
│  │  │  └──────────────────────┬───────────────────────────────┘    │   │   │
│  │  └─────────────────────────┼────────────────────────────────────┘   │   │
│  │                           │                                         │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                             │
│  ┌───────────────────────────┼─────────────────────────────────────────┐   │
│  │                  ALERTING & MAINTENANCE PLATFORM                     │   │
│  │                              ▼                                      │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐ │   │
│  │  │  Alert Manager    │  │  Work Order        │  │  Dashboard      │ │   │
│  │  │  (PagerDuty/SMS)  │  │  Auto-Creation     │  │  (Grafana)      │ │   │
│  │  └───────────────────┘  └───────────────────┘  └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Detail

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sensor Raw   │     │  Windowing   │     │  Feature     │     │  Model       │
│  50Hz data    │────▶│  1min/5min   │────▶│  Engineering  │────▶│  Inference   │
│  (100K msg/s) │     │  sliding     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Alert       │     │  Maintenance │     │  Feedback    │     │  Prediction  │
│  Dispatch    │◀────│  Record      │◀────│  (actual     │◀────│  (alert or   │
│  (PagerDuty) │     │  Database    │     │   failure?)  │     │   normal?)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **IoT Protocol** | OPC-UA + MQTT (Sparkplug B) | 1.04 / 3.1.1 | Industrial standard, bidirectional |
| **Edge Gateway** | Siemens IoT2000 + Raspberry Pi 4 | — | Cost-effective, field-proven |
| **Message Broker** | EMQX (self-hosted) | 5.5 | High-availability MQTT, 100K msg/s |
| **Stream Processing** | Apache Kafka + Kafka Streams | 3.6 | Fault-tolerant event sourcing |
| **Stream Analytics** | Apache Flink | 1.18 | Windowed joins, CEP patterns |
| **Time-Series DB** | InfluxDB | 2.7 | Purpose-built for sensor data |
| **Feature Store** | Hopsworks | 3.4 | Feature sharing, point-in-time correctness |
| **Anomaly Detection** | PyTorch (Autoencoder) | 2.1 | Unsupervised; captures non-linear patterns |
| **RUL Regression** | XGBoost | 2.0 | Accurate, interpretable, fast inference |
| **Model Serving** | Triton Inference Server | 23.12 | GPU-optimized, multi-model |
| **Orchestration** | Airflow + MLflow | 2.8 / 2.11 | Pipeline scheduling & experiment tracking |
| **Dashboard** | Grafana + InfluxDB connector | 10.4 | Real-time monitoring, OT-friendly |
| **Alerting** | Alertmanager + PagerDuty | — | On-call escalation |
| **Metadata DB** | PostgreSQL | 16 | Asset registry, maintenance history |
| **Infrastructure** | Docker + Ansible | — | Reproducible on-prem deployments |

### Hadoop/Spark Ecosystem

```bash
pip install apache-flink==1.18.1
pip install kafka-python==2.0.2
pip install influxdb-client==1.45.0
pip install hsfs==3.4.0  # Hopsworks Feature Store
pip install xgboost==2.0.3
pip install torch==2.1.2
pip install tritonclient==2.41.1
```

---

## ⚙️ Implementation Details

### 1. Feature Engineering Pipeline

```python
# src/features/engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List

class SensorFeatureEngineer:
    """Extract time-domain and frequency-domain features from sensor streams."""

    STATISTICAL_FEATURES = [
        "mean", "std", "min", "max", "range", "rms",
        "skewness", "kurtosis", "crest_factor", "peak_to_peak"
    ]

    @staticmethod
    def extract_statistical(window: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from a sensor window."""
        features = {}
        features["mean"] = np.mean(window)
        features["std"] = np.std(window)
        features["min"] = np.min(window)
        features["max"] = np.max(window)
        features["range"] = features["max"] - features["min"]
        features["rms"] = np.sqrt(np.mean(window ** 2))
        n = len(window)
        features["skewness"] = (np.mean((window - features["mean"]) ** 3) /
                                (features["std"] ** 3 + 1e-10))
        features["kurtosis"] = (np.mean((window - features["mean"]) ** 4) /
                                (features["std"] ** 4 + 1e-10)) - 3
        features["crest_factor"] = features["max"] / (features["rms"] + 1e-10)
        features["peak_to_peak"] = features["max"] - features["min"]
        return features

    @staticmethod
    def extract_frequency(fft_coeffs: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features from FFT."""
        features = {}
        magnitude = np.abs(fft_coeffs)
        power = magnitude ** 2
        features["spectral_energy"] = np.sum(power)
        features["spectral_centroid"] = np.sum(
            np.arange(len(power)) * power
        ) / (np.sum(power) + 1e-10)
        features["spectral_rolloff"] = np.sum(
            power[:int(len(power) * 0.85)]
        ) / (np.sum(power) + 1e-10)
        # Dominant frequencies
        dominant_idx = np.argmax(magnitude[1:]) + 1  # skip DC
        features["dominant_freq"] = dominant_idx
        features["dominant_magnitude"] = magnitude[dominant_idx]
        return features
```

### 2. Autoencoder Anomaly Detector

```python
# src/models/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim

class SensorAutoencoder(nn.Module):
    """Deep autoencoder for unsupervised anomaly detection on sensor data.

    Learns normal operating patterns. High reconstruction error = anomaly.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction error as anomaly score."""
        recon = self.forward(x)
        mse = torch.mean((x - recon) ** 2, dim=1)
        return mse

class AnomalyDetector:
    def __init__(self, input_dim: int, threshold_percentile: float = 95.0):
        self.model = SensorAutoencoder(input_dim)
        self.threshold = 0.0
        self.threshold_percentile = threshold_percentile

    def train(self, train_loader, val_loader, epochs: int = 100):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_scores = []
            with torch.no_grad():
                for batch in val_loader:
                    recon = self.model(batch)
                    loss = criterion(recon, batch)
                    val_loss += loss.item()
                    scores = self.model.compute_anomaly_score(batch)
                    val_scores.extend(scores.cpu().numpy())

            scheduler.step(val_loss)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}")

        # Set threshold based on validation scores
        self.threshold = np.percentile(
            val_scores, self.threshold_percentile
        )
        print(f"Anomaly threshold set at {self.threshold:.4f} "
              f"({self.threshold_percentile}th percentile)")

    def predict(self, x: torch.Tensor) -> tuple[bool, float]:
        """Returns (is_anomaly, anomaly_score)."""
        self.model.eval()
        with torch.no_grad():
            score = self.model.compute_anomaly_score(x).item()
        return score > self.threshold, score
```

### 3. RUL Estimator (XGBoost)

```python
# src/models/rul_estimator.py
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class RULEstimator:
    """Remaining Useful Life estimator using XGBoost.

    Features: sensor statistics, age, maintenance history, operating conditions.
    Target: hours remaining until failure.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost regressor for RUL prediction."""
        self.feature_names = X_train.columns.tolist()

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.1,
            "eval_metric": "mae",
            "seed": 42,
            "n_jobs": -1,
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=50,
        )

        # Validate
        y_pred = self.model.predict(dval)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"RUL Model MAE: {mae:.2f} hours")
        return mae

    def predict(self, features: pd.DataFrame) -> float:
        """Predict remaining useful life in hours."""
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        return self.model.predict(dmatrix)[0]

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance for interpretability."""
        importance = self.model.get_score(importance_type="gain")
        df = pd.DataFrame(
            list(importance.items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
        return df
```

### 4. Inference Pipeline (Triton)

```python
# src/inference/triton_client.py
import tritonclient.http as httpclient
import numpy as np
from typing import Dict

class TritonInferenceClient:
    """Client for NVIDIA Triton Inference Server."""

    def __init__(self, url: str = "localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)

    def predict_anomaly(
        self, features: np.ndarray
    ) -> Dict[str, float]:
        """Run anomaly detection model."""
        input_tensor = httpclient.InferInput(
            "features", features.shape, "FP32"
        )
        input_tensor.set_data_from_numpy(features)

        outputs = [
            httpclient.InferRequestedOutput("anomaly_score"),
            httpclient.InferRequestedOutput("is_anomaly"),
        ]

        result = self.client.infer(
            "anomaly_detector",
            inputs=[input_tensor],
            outputs=outputs,
        )

        return {
            "anomaly_score": result.as_numpy("anomaly_score")[0],
            "is_anomaly": bool(result.as_numpy("is_anomaly")[0]),
        }

    def predict_rul(self, features: np.ndarray) -> float:
        """Run RUL prediction model."""
        input_tensor = httpclient.InferInput(
            "features", features.shape, "FP32"
        )
        input_tensor.set_data_from_numpy(features)

        result = self.client.infer(
            "rul_estimator",
            inputs=[input_tensor],
            outputs=[httpclient.InferRequestedOutput("rul_hours")],
        )

        return float(result.as_numpy("rul_hours")[0])
```

### 5. Alerting Logic

```python
# src/alerts/decision_engine.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class MaintenanceRecommendation:
    asset_id: str
    asset_name: str
    anomaly_score: float
    rul_hours: float
    severity: str  # critical, warning, info
    recommended_action: str
    suggested_window_hours: int

class MaintenanceDecisionEngine:
    """Fuses anomaly detection + RUL into actionable maintenance alerts."""

    CRITICAL_THRESHOLD = 48  # hours
    WARNING_THRESHOLD = 200  # hours

    def evaluate(
        self,
        asset_id: str,
        asset_name: str,
        anomaly_score: float,
        is_anomaly: bool,
        rul_hours: float,
    ) -> Optional[MaintenanceRecommendation]:
        """Generate maintenance recommendation from model outputs."""

        if not is_anomaly and rul_hours > self.WARNING_THRESHOLD:
            return None  # Normal operation

        # Determine severity
        if is_anomaly and rul_hours < self.CRITICAL_THRESHOLD:
            severity = "critical"
            window = 4  # hours
            action = "IMMEDIATE SHUTDOWN & INSPECTION REQUIRED"
        elif is_anomaly or rul_hours < self.CRITICAL_THRESHOLD:
            severity = "critical"
            window = 8
            action = "URGENT: Schedule maintenance within 8 hours"
        elif rul_hours < self.WARNING_THRESHOLD:
            severity = "warning"
            window = 72
            action = "Schedule preventive maintenance within 72 hours"
        else:
            severity = "info"
            window = 168  # 1 week
            action = "Monitor; plan maintenance during next scheduled slot"

        return MaintenanceRecommendation(
            asset_id=asset_id,
            asset_name=asset_name,
            anomaly_score=anomaly_score,
            rul_hours=rul_hours,
            severity=severity,
            recommended_action=action,
            suggested_window_hours=window,
        )
```

### 6. Kafka Stream Processing

```python
# src/streaming/sensor_processor.py
from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np
from src.features.engineering import SensorFeatureEngineer

class SensorStreamProcessor:
    """Process raw sensor data from Kafka, extract features, push to model topics."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.consumer = KafkaConsumer(
            "sensor-raw",
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.feature_engineer = SensorFeatureEngineer()

    def process_stream(self, window_size: int = 300):
        """Read sensor stream, buffer into windows, extract features."""
        buffers = {}  # asset_id -> list of sensor readings

        for message in self.consumer:
            data = message.value
            asset_id = data["asset_id"]
            timestamp = data["timestamp"]
            readings = data["readings"]  # dict of sensor_name: value

            if asset_id not in buffers:
                buffers[asset_id] = []

            buffers[asset_id].append({
                "timestamp": timestamp,
                **readings
            })

            # Process when window is full
            if len(buffers[asset_id]) >= window_size:
                features = self._compute_features(
                    asset_id, buffers[asset_id]
                )
                self.producer.send("sensor-features", features)
                buffers[asset_id] = []  # reset window

    def _compute_features(self, asset_id: str, window: list) -> dict:
        """Compute all features from a sensor window."""
        features = {"asset_id": asset_id, "timestamp": window[-1]["timestamp"]}
        sensor_names = [k for k in window[0].keys() if k != "timestamp"]

        for sensor in sensor_names:
            values = np.array([w[sensor] for w in window], dtype=np.float32)
            stat_feats = self.feature_engineer.extract_statistical(values)
            for feat_name, feat_val in stat_feats.items():
                features[f"{sensor}_{feat_name}"] = feat_val

        return features
```

---

## 📊 Metrics & Results

### Technical Performance

| Metric | Baseline | Post-Deployment | Improvement |
|--------|----------|----------------|-------------|
| **Anomaly Detection AUC** | N/A | 0.946 | — |
| **RUL MAE** | N/A | 42.3 hours | — |
| **False Positive Rate** | N/A | 3.2% | — |
| **Detection Latency** | N/A | < 2 seconds | — |
| **Prediction Lead Time** | 0 (reactive) | 72 hours avg | +72 hours |
| **Model Inference Time** | N/A | 35ms (T4 GPU) | — |
| **Data Pipeline Throughput** | N/A | 85,000 msgs/sec | — |

### Business Impact (12 months post-deployment)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **Unplanned Downtime** | 864 hrs/yr | 562 hrs/yr | -35% |
| **Lost Production Cost** | $12.96M/yr | $8.43M/yr | -$4.53M |
| **Maintenance Spend** | $8.0M/yr | $5.2M/yr | -$2.8M |
| **Spare Parts Inventory** | $2.3M/yr | $1.6M/yr | -$0.7M |
| **MTBF** | 1,200 hrs | 1,920 hrs | +60% |
| **Safety Incidents** | 12/yr | 4/yr | -67% |
| **Warranty Claims Lost** | $3.5M/yr | $1.1M/yr | -$2.4M |

### Annual Savings Breakdown

```
Total Annual Savings: $10,430,000

┌──────────────────────────────────────────────────────────────┐
│  Lost Production Reduction                      $4,530,000   │
│  Maintenance Labor Savings                      $2,800,000   │
│  Spare Parts Inventory Optimization               $700,000   │
│  Emergency Shipping Elimination                   $600,000   │
│  Warranty Claim Recovery                         $2,400,000  │
│  Safety Incident Cost Reduction                   $600,000   │
├──────────────────────────────────────────────────────────────┤
│  Gross Annual Savings                           $11,630,000  │
│  Minus: System Cost (hardware + software)       -$1,200,000  │
│  Net Annual Savings                             $10,430,000  │
│  ROI (Year 1)                                        868%    │
└──────────────────────────────────────────────────────────────┘
```

### Model Performance by Asset Type

```
Asset Type       | Anomaly AUC | RUL MAE | Alerts/Week | Action Rate
─────────────────┼─────────────┼─────────┼─────────────┼────────────
CNC Machines     |    0.968    | 38 hrs  |    12       |    85%
Hydraulic Press  |    0.952    | 45 hrs  |     8       |    88%
Conveyors        |    0.931    | 52 hrs  |    15       |    79%
Robotic Arms     |    0.945    | 40 hrs  |     6       |    92%
Pumps            |    0.959    | 36 hrs  |    22       |    76%
```

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Ensemble models beat single models** — Combining autoencoder (unsupervised) with XGBoost (supervised) gave 23% better F1 than either alone.

2. **On-premise inference was non-negotiable** — Factory floor air-gap meant cloud was impossible. Triton + 4× T4 GPUs handled all workloads.

3. **Feature store was a game-changer** — Hopsworks allowed simple feature reuse between training and serving. Point-in-time correctness caught 3 major data leakage issues during development.

4. **Domain expert integration** — Having a master mechanic on the team prevented us from building obviously wrong features (e.g., vibration patterns that can't exist on certain machines).

### ❌ What Went Wrong

1. **Sensor data quality varied wildly** — 12% of sensors were drift-biased (had 5°C offset). We lost 3 weeks retraining after discovering this. Solution: automated sensor calibration check.

2. **Alert fatigue nearly killed the project** — Week 1 generated 2,000+ alerts/day. Technicians ignored them. We implemented "suppression rules" (same machine, same alert within 48h = auto-snooze).

3. **Cold start problem** — Need 6+ months of normal + failure data. We bootstrapped with synthetic failures (deliberate mis-alignment, controlled bearing damage) on 10 test machines.

4. **Network bandwidth was constrained** — 50Hz × 200 sensors × 2500 machines = 25M data points/sec. Had to implement edge-level downsampling and only send statistical features (not raw).

### ⚠️ Critical Warnings

```
! WARNING: Never auto-stop production equipment without human confirmation.
! WARNING: Alert fatigue is a real project-killer — invest in noise reduction.
! WARNING: Sensor drift is inevitable — build automated health checks.
! WARNING: Model retraining must catch concept drift (new materials, new speeds).
```

### Maintenance Cadence

```
Data Collection:    Continuous (50Hz sampling, 5-min aggregated windows)
Model Retrain:      Weekly (incremental XGBoost), Monthly (full autoencoder)
Feature Validation: Daily automated drift checks
Alert Review:       Shift handoff meeting reviews all 'warning' alerts
Model Audit:       Quarterly (accuracy against actual failures)
```

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-PREDICTIVE-MAINTENANCE/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── sensors.yaml                # Sensor registry & thresholds
│   ├── assets.yaml                 # Asset metadata & hierarchies
│   ├── alerting.yaml               # Alert rules & suppression
│   └── model_configs/
│       ├── autoencoder.yaml
│       └── xgboost_rul.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── mqtt_listener.py
│   │   ├── opcua_client.py
│   │   ├── kafka_producer.py
│   │   └── sensor_validator.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   └── feature_store_client.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── xgboost_rul.py
│   │   ├── ensemble.py
│   │   └── train_pipeline.py
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── kafka_processor.py
│   │   ├── flink_job.py
│   │   └── windowing.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── triton_client.py
│   │   ├── onnx_export.py
│   │   └── batch_inference.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   ├── decision_engine.py
│   │   └── notification.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── time_utils.py
│       └── retry.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_autoencoder.py
│   │   ├── test_xgboost.py
│   │   └── test_alerts.py
│   ├── integration/
│   │   ├── test_kafka_pipeline.py
│   │   ├── test_influxdb.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_sensor_data.json
│       └── simulated_failure_data.csv
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-autoencoder-training.ipynb
│   └── 04-rul-modeling.ipynb
│
├── scripts/
│   ├── simulate_sensors.py
│   ├── seed_feature_store.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── deploy_triton.sh
│
├── edge/
│   ├── gateway_config.yaml
│   ├── sensor_collector.py
│   └── install.sh
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           ├── asset_health.json
│           ├── anomaly_overview.json
│           └── maintenance_optimization.json
│
└── docs/
    ├── architecture.md
    ├── sensor_deployment_guide.md
    ├── model_card.md
    ├── runbook.md
    └── compliance.md
```

### Getting Started

```bash
# 1. Clone template
cp -r TEMPLATE-PREDICTIVE-MAINTENANCE ~/my-predictive-maint
cd ~/my-predictive-maint

# 2. Configure environment
cp .env.example .env
# Edit .env for your infrastructure

# 3. Start infrastructure (Kafka, InfluxDB, PostgreSQL)
make infra-up

# 4. Simulate sensor data for testing
python scripts/simulate_sensors.py --assets 10 --duration 3600

# 5. Train models (on simulated data)
python scripts/train_models.py

# 6. Deploy to Triton
./scripts/deploy_triton.sh

# 7. Run evaluation
python scripts/evaluate_models.py

# 8. Start dashboard
make grafana-up
```

---

## 📚 References & Further Reading

### Academic Papers
- Malhotra et al. (2016) — "LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection" — [arXiv:1607.00148](https://arxiv.org/abs/1607.00148)
- Hundman et al. (2018) — "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" — [KDD 2018](https://arxiv.org/abs/1802.04431)
- Saxena et al. (2008) — "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" — PHM08 Challenge

### Industry Standards
- ISO 55000: Asset Management
- ISA-95: Enterprise-Control System Integration
- IEC 62264: Interface for Manufacturing Operations Management

### Tools & Documentation
- EMQX MQTT Broker: https://www.emqx.io/docs
- Apache Kafka: https://kafka.apache.org/documentation/
- Apache Flink: https://nightlies.apache.org/flink/
- InfluxDB: https://docs.influxdata.com/influxdb/
- Hopsworks Feature Store: https://docs.hopsworks.ai/
- NVIDIA Triton: https://docs.nvidia.com/deeplearning/triton/

---

> **Next**: [04-Healthcare-Diagnosis.md](04-Healthcare-Diagnosis.md) — Medical AI diagnosis for chest X-ray classification with DICOM pipeline and FDA clearance considerations.
