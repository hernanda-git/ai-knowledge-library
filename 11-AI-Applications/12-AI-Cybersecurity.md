# AI for Cybersecurity: Threat Detection, Response, and Operations

> **Last Updated:** June 2026  
> **Category:** 11-AI-Applications — Industry Applications  
> **Cross-References:** 06-Advanced/08-Adversarial-ML.md, 06-Advanced/05-Interpretability.md, 11-AI-Applications/03-Finance-AI.md, 11-AI-Applications/02-Healthcare-AI.md

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Cybersecurity AI Landscape](#2-landscape)
3. [ML-Based Threat Detection — IDS/IPS](#3-threat-detection)
4. [User Entity Behavior Analytics (UEBA)](#4-ueba)
5. [Malware Detection — CNN & Graph-Based](#5-malware-detection)
6. [Phishing Detection — NLP on Email](#6-phishing-detection)
7. [SOAR — Security Orchestration, Automation & Response](#7-soar)
8. [Vulnerability Prioritization](#8-vulnerability)
9. [Adversarial ML in Security](#9-adversarial-ml)
10. [AI for SOC Operations](#10-soc-operations)
11. [SIEM AI — Splunk ML & Elastic ML](#11-siem-ai)
12. [Generative AI for Security — Copilot for SOC](#12-generative-ai)
13. [Real Products — Darktrace, CrowdStrike, SentinelOne](#13-real-products)
14. [Architecture Diagrams](#14-architecture)
15. [Benchmarks & Evaluation](#15-benchmarks)
16. [Code Examples & Hands-On](#16-code-examples)
17. [Cross-References](#17-cross-references)

---

## 1. Introduction

AI has become the central pillar of modern cybersecurity. As attack surfaces expand, threat actors automate their operations, and the volume of security alerts overwhelms human analysts, organizations increasingly depend on machine learning, deep learning, and generative AI to detect, respond to, and prevent cyber threats.

**The scale of the problem:**

| Metric | Value | Year |
|--------|-------|------|
| Security alerts generated per enterprise per day | 200,000+ | 2026 |
| Average time to detect a breach (dwell time) | 10 days (w/o AI) / 2 hours (w/ AI) | 2025 |
| Global cybersecurity unfilled positions | 4.7 million | 2026 |
| Malware samples collected per day | 1.5 million | 2025 |
| Cost of average data breach | $4.88 million | 2025 |

**Key AI security domains covered in this document:**

- **Detection:** ML models identifying malicious activity in real-time
- **Behavioral Analysis:** Understanding normal vs. anomalous entity behavior
- **Malware Analysis:** Binary classification, graph-based family identification
- **Phishing Intelligence:** NLP models detecting social engineering attempts
- **Automation:** SOAR platforms orchestrating incident response
- **Risk Scoring:** AI-driven vulnerability prioritization
- **SOC Augmentation:** AI assistants and co-pilots for security operations

---

## 2. The Cybersecurity AI Landscape

### 2.1 Evolution of AI in Security

```
1990s: Signature-based detection
  └─ Virus definitions, Snort rules, pattern matching

2000s: Statistical anomaly detection
  └─ Baseline profiling, threshold-based alerts, simple ML (naive Bayes)

2010s: Machine learning enters production
  └─ Random Forest for malware, SVM for spam, clustering for anomalies
  └─ Products: Cylance (ML-based AV), Darktrace (Unsupervised learning)

2020: Deep learning for security
  └─ CNN on binary files, RNN on log sequences, GNN for threat graphs
  └─ Products: SentinelOne (Deep learning EDR), CrowdStrike (AI-native)

2023: LLMs and Generative AI
  └─ SOC copilots, automated report writing, natural language query for SIEM
  └─ Products: Splunk AI Assistant, Microsoft Security Copilot

2025+: Autonomous SOC
  └─ Multi-agent systems, automated triage → investigation → remediation
  └─ Products: CrowdStrike Charlotte AI, SentinelOne Purple AI
```

### 2.2 AI Security Categories

| Category | Input Data | Output | Typical Models |
|----------|-----------|--------|----------------|
| Network IDS/IPS | Packet captures, NetFlow | Malicious/benign traffic | Random Forest, 1D-CNN, Autoencoders |
| Endpoint Detection (EDR) | Process events, file ops, registry | Malware/benign binary | CNN, Graph NN, LightGBM |
| User Behavior (UEBA) | Auth logs, access patterns | Anomaly score | Autoencoders, Isolation Forest |
| Email Security | Email headers, body, attachments | Phishing/spam/benign | BERT, DistilBERT, XGBoost |
| Cloud Security | API calls, config changes | Misconfiguration, intrusion | Graph NN, Rule + ML hybrid |
| Identity Security | Auth attempts, MFA logs | Account compromise | LSTM, Transformer |
| Fraud Detection | Transaction data | Fraud/legitimate | XGBoost, TabNet, GNN |

---

## 3. ML-Based Threat Detection — IDS/IPS

### 3.1 Network Intrusion Detection with ML

Traditional signature-based IDS (Snort, Suricata) detect known attack patterns. ML-based IDS extends detection to zero-day attacks and encrypted traffic.

```
ML-Based NIDS Architecture
═════════════════════════════

  Network Traffic (raw PCAP / NetFlow)
     │
  [Feature Extraction]
  ├─ Statistical: packets/sec, bytes/flow, protocol distribution
  ├─ Time-series: inter-arrival times, burst patterns
  ├─ Header-based: IP flags, TCP window size, TTL
  └─ Encrypted traffic: packet size distribution, timing (TLS fingerprinting)
     │
  [Feature Selection]
  ├─ Correlation analysis
  ├─ Mutual information ranking
  └─ Autoencoder-based feature reduction
     │
  [Classification Model]
  ├─ Random Forest / XGBoost (interpretable, fast)
  ├─ 1D-CNN (temporal patterns in raw packets)
  ├─ Autoencoder (reconstruction error = anomaly score)
  └─ Graph Neural Network (flow graphs)
     │
  [Decision Output]
  ├─ Classification: benign / malicious / suspicious
  ├─ Confidence score
  ├─ Attack family (DoS, Probe, R2L, U2R)
  └─ Alert priority (critical, high, medium, low)
```

### 3.2 Model Comparison for NIDS

| Model | Accuracy | F1 Score | False Positive Rate | Training Time | Inference (μs/packet) |
|-------|----------|----------|-------------------|---------------|----------------------|
| Random Forest | 0.96 | 0.95 | 2.1% | 30 min | 45 μs |
| XGBoost | 0.97 | 0.96 | 1.8% | 25 min | 38 μs |
| 1D-CNN | 0.95 | 0.94 | 2.5% | 2 hrs (GPU) | 85 μs |
| Autoencoder (AE) | 0.93 | 0.91 | 3.5% | 1 hr | 55 μs |
| Transformer | 0.97 | 0.96 | 1.5% | 6 hrs (GPU) | 250 μs |
| GraphNN | 0.96 | 0.95 | 2.0% | 4 hrs (GPU) | 180 μs |
| **Hybrid (XGBoost+AE)** | **0.98** | **0.97** | **1.2%** | 2 hrs | 90 μs |

*Evaluation on CIC-IDS2017, CSE-CIC-IDS2018 datasets*

### 3.3 Real-Time NIDS Implementation

```python
"""Real-time ML-based NIDS inference engine."""

import pickle
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import threading
import time

@dataclass
class FlowFeatures:
    """Features extracted from a network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    duration: float
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    tcp_flags: int
    inter_arrival_mean: float
    inter_arrival_std: float
    packet_size_mean: float
    packet_size_std: float

class MLIDS:
    """Machine Learning Intrusion Detection System."""
    
    def __init__(self, model_path: str, threshold: float = 0.85,
                 feature_names: list = None):
        """Load pre-trained model for real-time inference.
        
        Args:
            model_path: path to pickled model (XGBoost/RF)
            threshold: confidence threshold for alert generation
            feature_names: feature ordering expected by model
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.threshold = threshold
        self.feature_names = feature_names
        self.alert_queue = deque(maxlen=10000)
        self.flow_buffer = {}  # (src_ip, dst_ip, sport, dport, proto) → FlowFeatures
        self.running = True
        self.stats = {'processed': 0, 'alerts': 0, 'false_positives': 0}
        
    def extract_features(self, packet: dict) -> Optional[np.ndarray]:
        """Extract feature vector from raw packet/flow data.
        
        In production, this reads from packet capture (scapy, pcap, 
        or network tap). Here we take a pre-parsed dict for clarity.
        """
        features = [
            packet.get('duration', 0),
            packet.get('bytes_sent', 0),
            packet.get('bytes_recv', 0),
            packet.get('packets_sent', 0),
            packet.get('packets_recv', 0),
            packet.get('protocol', 6),
            packet.get('tcp_flags', 0),
            packet.get('inter_arrival_mean', 0),
            packet.get('inter_arrival_std', 0),
            packet.get('packet_size_mean', 0),
            packet.get('packet_size_std', 0),
            np.log1p(packet.get('bytes_sent', 0) + 1),  # log transform
            np.log1p(packet.get('packets_sent', 0) + 1),
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def inference(self, features: np.ndarray) -> tuple:
        """Run model inference.
        
        Returns:
            (is_malicious, confidence, attack_class)
        """
        proba = self.model.predict_proba(features)[0]
        confidence = max(proba)
        prediction = self.model.predict(features)[0]
        
        # Binary: 1 = malicious
        is_malicious = bool(prediction)
        attack_class = self.model.classes_[prediction] if hasattr(
            self.model, 'classes_') else None
        
        return is_malicious, float(confidence), attack_class
    
    def process_packet(self, packet_flow: dict) -> Optional[dict]:
        """Process a single flow record."""
        self.stats['processed'] += 1
        
        features = self.extract_features(packet_flow)
        if features is None:
            return None
        
        is_malicious, confidence, attack_class = self.inference(features)
        
        if is_malicious and confidence >= self.threshold:
            alert = {
                'timestamp': time.time(),
                'src_ip': packet_flow.get('src_ip'),
                'dst_ip': packet_flow.get('dst_ip'),
                'confidence': confidence,
                'attack_type': attack_class,
                'severity': self._severity_from_confidence(confidence),
                'flow_id': packet_flow.get('flow_id')
            }
            self.alert_queue.append(alert)
            self.stats['alerts'] += 1
            return alert
        
        return None
    
    def _severity_from_confidence(self, confidence: float) -> str:
        if confidence >= 0.99:
            return 'critical'
        elif confidence >= 0.95:
            return 'high'
        elif confidence >= 0.90:
            return 'medium'
        return 'low'
    
    def start_capture(self, interface: str = 'eth0'):
        """Start real-time packet capture and analysis thread.
        
        In production, replace with live packet capture library
        (e.g., scapy, nfstream, or AF_PACKET).
        """
        # This is a placeholder for live capture setup
        # capture_thread = threading.Thread(target=self._capture_loop, 
        #                                   args=(interface,))
        # capture_thread.daemon = True
        # capture_thread.start()
        pass

# Example: training an IDS model
def train_nids_model(train_csv: str, test_csv: str) -> dict:
    """Train and evaluate a Random Forest NIDS model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    # Separate features and labels
    feature_cols = [c for c in df_train.columns if c != 'label']
    X_train = df_train[feature_cols]
    y_train = df_train['label']
    X_test = df_test[feature_cols]
    y_test = df_test['label']
    
    # Train
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {'model': model, 'metrics': report, 'features': feature_cols}

"""
Performance on CIC-IDS2017:
Precision: 0.98  Recall: 0.97  F1: 0.97
Training time: ~30M rows in 5 minutes (200 trees, 8 cores)
Inference: ~50 μs per sample
"""
```

### 3.4 Deep Learning NIDS with Autoencoders

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderNIDS(nn.Module):
    """Autoencoder for anomaly-based NIDS.
    
    Trained on benign traffic only. Malicious traffic produces
    high reconstruction error → flagged as anomalous.
    
    Architecture:
      28-dim → 14 → 7 → 14 → 28 (reconstruction)
    """
    
    def __init__(self, input_dim: int = 28):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim),
            nn.Sigmoid()  # normalize output to [0,1]
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
    
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score.
        
        Higher score → more anomalous.
        """
        recon = self.forward(x)
        # Mean squared error per sample
        mse = torch.mean((x - recon) ** 2, dim=1)
        return mse
    
    @torch.no_grad()
    def detect(self, x: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        scores = self.anomaly_score(x)
        return scores > threshold  # True = anomalous

# Training loop
def train_autoencoder_nids(model, train_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.6f}")
    
    return model
```

---

## 4. User Entity Behavior Analytics (UEBA)

### 4.1 Core Concepts

UEBA models normal behavior for users, devices, and applications, then flags deviations as potential security incidents.

```
UEBA Pipeline
══════════════════

  Data Sources:
  ├── Authentication logs (SSO, Kerberos, LDAP)
  ├── VPN connection logs
  ├── Email and collaboration logs
  ├── Endpoint activity (process creation, file access)
  ├── Database access logs
  ├── Cloud API call logs (S3, IAM, etc.)
  └── Physical access logs (badge readers)
      │
      ▼
  [Feature Engineering]
  │
  ├── Behavioral Baselines (per user, per time window)
  │   ├── Login time distribution
  │   ├── Geo-location distribution
  │   ├── Data access patterns
  │   ├── Work hours vs. off-hours
  │   ├── Peer-group comparison (same role, same department)
  │   └── Historical trend (30-day rolling window)
  │
  ├── Feature Vectors:
  │   ├── Time since last login
  │   ├── Number of failed attempts (hourly/daily)
  │   ├── Unusual access hour (Z-score against baseline)
  │   ├── Geo-velocity (impossible travel time)
  │   ├── Data volume deviation
  │   └── Number of distinct resources accessed
  │
      ▼
  [Anomaly Detection Models]
  │
  ├── Unsupervised (no labels required):
  │   ├── Autoencoder (reconstruction error)
  │   ├── Isolation Forest (partition-based anomaly)
  │   ├── Local Outlier Factor (density-based)
  │   └── One-Class SVM (boundary-based)
  │
  ├── Supervised (labeled data):
  │   ├── XGBoost (gradient boosted trees)
  │   ├── LightGBM (faster than XGBoost)
  │   └── TabNet (deep tabular learning)
  │
  └── Time-series:
      ├── LSTM / GRU (sequence of user actions)
      └── Transformer attention (context-aware)
          │
          ▼
  [Risk Scoring Engine]
  │
  ├── Combine multiple signals
  ├── Weight by severity and confidence
  └── Output: user_risk_score (0-100)
```

### 4.2 Autoencoder for UEBA

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from collections import defaultdict

class UEBAAutoencoder(nn.Module):
    """Deep autoencoder for user behavior anomaly detection.
    
    Architecture:
      Input (behavior features, e.g., 64-dim)
      → Encoder: 64 → 32 → 16 → 8 (bottleneck)
      → Decoder: 8 → 16 → 32 → 64
      Output: reconstructed behavior features
    
    Anomaly score = MSE between input and reconstruction.
    """
    
    def __init__(self, n_features: int = 64, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, n_features),
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def anomaly_score(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            # Per-sample MSE
            mse = torch.mean((x - recon) ** 2, dim=1)
        return mse.numpy()

class UEBAEngine:
    """User and Entity Behavior Analytics engine."""
    
    def __init__(self, model: UEBAAutoencoder, threshold_percentile: float = 95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.user_baselines = {}  # user_id → baseline stats
        self.anomaly_threshold = None
        
    def fit_baseline(self, user_data: Dict[str, np.ndarray]):
        """Train autoencoder on normal user behavior."""
        # user_data: {user_id: (n_samples, n_features)}
        all_features = np.vstack(list(user_data.values()))
        
        # Train autoencoder
        tensor_data = torch.FloatTensor(all_features)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(100):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"UEBA Epoch {epoch}: Loss = {total_loss/len(loader):.6f}")
        
        # Set threshold based on training data
        self.model.eval()
        scores = self.model.anomaly_score(tensor_data)
        self.anomaly_threshold = np.percentile(
            scores, self.threshold_percentile
        )
        print(f"Anomaly threshold set at: {self.anomaly_threshold:.4f}")
        
        # Store per-user statistics
        for user_id, features in user_data.items():
            user_scores = self.model.anomaly_score(
                torch.FloatTensor(features)
            )
            self.user_baselines[user_id] = {
                'mean_score': float(np.mean(user_scores)),
                'std_score': float(np.std(user_scores)),
                'n_samples': len(features)
            }
    
    def analyze_user_activity(self, user_id: str, 
                               features: np.ndarray) -> dict:
        """Analyze a user's current activity for anomalies.
        
        Args:
            user_id: user identifier
            features: (n_events, n_features) current activity features
        
        Returns:
            dict with anomaly assessment
        """
        tensor = torch.FloatTensor(features)
        scores = self.model.anomaly_score(tensor)
        
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        
        # Compare to baseline
        is_anomaly = max_score > self.anomaly_threshold
        
        if user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            z_score = (mean_score - baseline['mean_score']) / max(
                baseline['std_score'], 0.001
            )
        else:
            z_score = 0.0  # Unknown user
        
        # Detailed anomaly breakdown
        anomalous_events = []
        for i, score in enumerate(scores):
            if score > self.anomaly_threshold:
                anomalous_events.append({
                    'event_idx': i,
                    'score': float(score),
                    'deviation': f"{((score / self.anomaly_threshold) - 1) * 100:.0f}%"
                })
        
        return {
            'user_id': user_id,
            'is_anomalous': is_anomaly,
            'max_anomaly_score': max_score,
            'mean_anomaly_score': mean_score,
            'z_score_vs_baseline': round(z_score, 2),
            'threshold': self.anomaly_threshold,
            'anomalous_events': anomalous_events,
            'severity': 'high' if max_score > 2 * self.anomaly_threshold 
                       else 'medium' if is_anomaly 
                       else 'low'
        }

"""
UEBA Detection Examples:
┌────────────────────┬─────────────┬──────────────┬────────────────────┐
│ Scenario           │ Normal      │ Anomalous    │ Detection Method   │
├────────────────────┼─────────────┼──────────────┼────────────────────┤
│ Login time         │ 9am-5pm     │ 3am login    │ Time distribution  │
│ Geo-velocity       │ 0-100 km    │ NYC→London   │ Impossible travel  │
│                    │             │ in 1 hour    │                    │
│ Data exfiltration  │ 10-50MB/day │ 5GB in 1hr   │ Volume anomaly     │
│ Lateral movement   │ 3-5 hosts   │ 30 hosts/min │ Beaconing detec.   │
│ Privilege escalation│ Normal user │ sudo to root │ Process chain      │
│ API abuse          │ 100 req/min │ 5000 req/min │ Request rate spike │
└────────────────────┴─────────────┴──────────────┴────────────────────┘
"""
```

### 4.3 Peer-Group Analysis

```python
class PeerGroupAnalyzer:
    """Compare user behavior against peer group (same role/department)."""
    
    def __init__(self):
        self.role_profiles = {}  # role → aggregated feature stats
        
    def build_profiles(self, users: dict):
        """Build peer-group profiles.
        
        users: {user_id: {'role': str, 'features': np.ndarray}}
        """
        role_features = defaultdict(list)
        for uid, info in users.items():
            for features in info['features']:
                role_features[info['role']].append(features)
        
        for role, feats in role_features.items():
            all_feats = np.array(feats)
            self.role_profiles[role] = {
                'mean': np.mean(all_feats, axis=0),
                'std': np.std(all_feats, axis=0) + 1e-8,
                'n_users': len(np.unique([u for u, i in users.items() 
                                          if i['role'] == role]))
            }
    
    def peer_deviation_score(self, user_id: str, role: str, 
                              features: np.ndarray) -> float:
        """How much a user deviates from their peer group."""
        if role not in self.role_profiles:
            return 0.0
        
        profile = self.role_profiles[role]
        # Element-wise z-score, then average
        z_scores = (features - profile['mean']) / profile['std']
        return float(np.mean(np.abs(z_scores)))
```

---

## 5. Malware Detection — CNN & Graph-Based

### 5.1 Malware Detection Approaches

| Approach | Input | Technique | Strengths | Limitations |
|----------|-------|-----------|-----------|-------------|
| Signature-based | File hash or byte sequence | Hash lookup | Instant, zero FP | Only known malware |
| Static ML | Byte sequences, PE headers | CNN on raw bytes, RF on features | Fast, no execution | Packed/obfuscated malware |
| Dynamic ML | API call sequences, syscalls | LSTM, Transformer | Catches runtime behavior | Slow, sandbox evasion |
| Graph-based | Control flow graphs, call graphs | GNN, GraphSAGE | Resilient to obfuscation | Computationally expensive |
| Hybrid | Multiple signals | Ensemble of models | Best coverage | Complex deployment |

### 5.2 CNN on Raw Binary Files

```python
"""
Malware classification using 1D-CNN on raw binary bytes.
Inspired by MalConv (Raff et al., 2018) and subsequent work.
Architecture:
  Raw bytes (2MB max) → Embedding → 1D Conv → Global Max Pool → Dense → Output
"""

import torch
import torch.nn as nn
import numpy as np

class MalConv(nn.Module):
    """1D Convolutional Neural Network for raw binary malware detection.
    
    Processes raw file bytes directly — no feature engineering needed.
    Input: up to 2MB of raw bytes from PE/ELF/Mach-O file.
    """
    
    def __init__(self, max_len: int = 2 * 1024 * 1024,  # 2MB
                 embedding_dim: int = 8,
                 n_filters: int = 256,
                 kernel_size: int = 512):
        super().__init__()
        
        self.max_len = max_len
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        
        # First conv: extract local byte patterns
        self.conv1 = nn.Conv1d(
            embedding_dim, n_filters, 
            kernel_size=kernel_size, stride=1
        )
        
        # Attention-style second conv
        self.conv2 = nn.Conv1d(
            n_filters, n_filters, 
            kernel_size=kernel_size // 4, stride=1
        )
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, 2)  # benign/malware binary
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: (batch, max_len) — byte values 0-255, 0 = padding
        x = self.embedding(x)  # (batch, max_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, max_len)
        
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)  # (batch, n_filters)
        x = self.fc(x)
        return x

def preprocess_binary(file_path: str, max_len: int = 2*1024*1024) -> torch.Tensor:
    """Read binary file and convert to tensor of byte values.
    
    Returns:
        Tensor of shape (1, max_len) with byte values 0-256
        (256 = padding, 0-255 = actual bytes, wrapped to 1-256)
    """
    with open(file_path, 'rb') as f:
        binary = f.read()[:max_len]
    
    # Convert bytes to array (1-256 range, 0 for padding)
    arr = np.frombuffer(binary, dtype=np.uint8).astype(np.int64) + 1
    
    # Pad or truncate to max_len
    if len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)), 'constant')
    
    return torch.LongTensor(arr).unsqueeze(0)

"""
Benchmark on EMBER dataset (2018):
├── MalConv: 0.93 AUC, 100μs inference (GPU)
├── MalConv2 (2020): 0.96 AUC, 120μs inference
├── LightGBM on features: 0.99 AUC, 10μs inference (CPU)
└── GNN on CFG: 0.97 AUC, 500μs inference (GPU)

MalConv advantages:
- No feature extraction needed (works on raw bytes)
- Handles any file format (PE, ELF, Mach-O, PDF)
- Resilient to simple obfuscation
"""
```

### 5.3 Graph-Based Malware Classification

```python
"""
Graph Neural Network for malware classification using
Control Flow Graphs (CFG) or Function Call Graphs (FCG).

Architecture:
  Graph (nodes=basic blocks, edges=control flow)
  → GraphSAGE conv layers
  → Global mean pooling
  → MLP classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

class MalwareGNN(nn.Module):
    """Graph Neural Network for malware classification.
    
    Input: Function Call Graph of a binary file.
    Nodes represent functions, edges represent calls.
    Node features: instruction counts, opcode frequencies, 
                   API call types, basic block properties.
    """
    
    def __init__(self, node_feat_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.conv1 = SAGEConv(node_feat_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7)  # 7 malware families + benign
        )
        
    def forward(self, x, edge_index, batch):
        # x: node features, edge_index: graph connectivity
        # batch: which graph each node belongs to
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling: one vector per graph
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        
        return self.classifier(x)
    
    def classify_file(self, graph_data: Data) -> dict:
        """Classify a single binary file's graph.
        
        Returns:
            dict with predicted family and confidence
        """
        self.eval()
        with torch.no_grad():
            out = self(graph_data.x, graph_data.edge_index, 
                       torch.zeros(graph_data.num_nodes, dtype=torch.long))
            probs = F.softmax(out, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
        
        families = ['benign', 'ransomware', 'trojan', 'worm',
                    'backdoor', 'adware', 'botnet']
        
        return {
            'predicted_family': families[pred],
            'confidence': round(confidence, 4),
            'all_probs': {families[i]: round(float(probs[0, i]), 4)
                         for i in range(len(families))}
        }

"""
Graph-based malware detection results (MalNet-Tiny, 2024):
├── GCN: 0.94 accuracy (4M parameters)
├── GraphSAGE: 0.95 accuracy (3.5M params) ← Best balance
├── GIN: 0.94 accuracy (5M param)
└── GAT: 0.96 accuracy (6M param, slower)

Key advantage: GNNs detect malware families by structural
similarity even when payload bytes are obfuscated.
"""
```

### 5.4 Static Feature-Based Malware Detection (Production)

Most production EDR products use gradient-boosted trees on engineered features:

```python
"""
Production-grade static malware detection features:

Feature Categories (1500+ features):
├── PE Header Features
│   ├── Machine type, subsystem, entry point
│   ├── Section characteristics (.text writable? .data executable?)
│   ├── Import/Export table characteristics
│   └── IAT entropy, TLS callbacks
├── File Statistics
│   ├── File size, entropy histogram
│   ├── Byte frequency distribution (n-grams)
│   ├── String characteristics (printable ratio, suspicious strings)
│   └── Overlay data presence
├── Structural Features
│   ├── Section entropy (mean, min, max, std)
│   ├── Section size ratios
│   ├── Imports: count, suspicious APIs (VirtualAlloc, WriteProcessMemory)
│   └── Resource characteristics
└── Behavioral Indicators (static proxies)
    ├── Packer detection signatures
    ├── Anti-debugging techniques
    └── VM detection attempts
"""

# Example: LightGBM for production malware detection
"""
import lightgbm as lgb

# Training parameters for production malware model
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_data_in_leaf': 20,
    'max_bin': 255,
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

# Production results (typical):
# AUC: 0.995
# True Positive Rate at 0.1% FPR: 0.94
# False Positive Rate: 0.08% → 8 false alarms per 10,000 files
# Inference: ~5 μs per sample (CPU, no GPU needed)
"""
```

---

## 6. Phishing Detection — NLP on Email

### 6.1 The Phishing Challenge

Phishing remains the most common initial attack vector — responsible for over 90% of data breaches. AI-based detection must analyze multiple signals:

```
Phishing Email Analysis Pipeline
══════════════════════════════════

  Email
  ┌──────────┬──────────┬──────────┐
  │  Header   │   Body    │   URLs   │  Attachments
  └────┬─────┴────┬─────┴────┬─────┘     (images, PDFs)
       │          │          │
       ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │SPF/DKIM│ │ NLP    │ │ URL    │
  │DMARC   │ │ Model  │ │ Rep.   │
  │Auth    │ │        │ │Check   │
  │Results │ │        │ │        │
  └────────┘ └───┬────┘ └────────┘
                 │
                 ▼
  ┌────────────────────────┐
  │  Multi-Head Classifier  │
  │  ┌─────┐ ┌──────┐ ┌────┐ │
  │  │Body │ │Intent│ │Risk│ │
  │  │Feat │ │+Soc  │ │Scor│ │
  │  │     │ │Eng.  │ │    │ │
  │  └─────┘ └──────┘ └────┘ │
  └────────────────────────────┘
                 │
                 ▼
         Decision: Spam / Phishing / Benign
```

### 6.2 NLP-Based Phishing Detection

```python
"""
Phishing email detection using transformer-based NLP models.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertModel
)

class PhishingDetector(nn.Module):
    """Transformer-based phishing email detector.
    
    Analyzes multiple email components:
    1. Email body text
    2. Subject line
    3. Sender domain features
    4. URL patterns extracted from body
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_labels: int = 3):  # benign, spam, phishing
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 8, 128),  # +8 for meta features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
        
    def extract_metadata_features(self, email_meta: dict) -> torch.Tensor:
        """Extract hand-crafted features from email metadata.
        
        Features:
          - has_urgent_subject (0/1)
          - num_links_in_body
          - num_attachments
          - has_spoofed_domain (0/1)
          - auth_result (SPF pass/fail/neutral: 2/1/0)
          - contains_financial_keywords (0/1)
          - body_to_link_ratio
          - has_external_image_links (0/1)
        """
        features = [
            email_meta.get('has_urgent_subject', 0),
            min(email_meta.get('num_links', 0) / 10, 1.0),
            min(email_meta.get('num_attachments', 0), 5),
            email_meta.get('has_spoofed_domain', 0),
            {'pass': 2, 'neutral': 1, 'fail': 0}.get(
                email_meta.get('spf_result', 'neutral'), 1),
            1 if any(kw in email_meta.get('body', '').lower() 
                     for kw in ['urgent', 'account', 'password', 
                                'verify', 'suspended']) else 0,
            email_meta.get('body_to_link_ratio', 0),
            email_meta.get('has_external_images', 0)
        ]
        return torch.FloatTensor([features])
    
    def forward(self, body_text: str, metadata: dict) -> dict:
        """Classify email as benign/spam/phishing."""
        # Tokenize body
        inputs = self.tokenizer(
            body_text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        
        # Get BERT embeddings
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Concatenate with metadata features
        meta_feats = self.extract_metadata_features(metadata)
        combined = torch.cat([cls_embedding, meta_feats], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=1)
        
        labels = ['benign', 'spam', 'phishing']
        pred_idx = probs.argmax(dim=1).item()
        
        return {
            'prediction': labels[pred_idx],
            'confidence': probs[0, pred_idx].item(),
            'phishing_probability': probs[0, 2].item(),  # index 2 = phishing
            'all_probs': {l: probs[0, i].item() for i, l in enumerate(labels)},
            'risk_level': 'high' if probs[0, 2] > 0.7 
                         else 'medium' if probs[0, 2] > 0.3 
                         else 'low'
        }

"""
Phishing Detection Performance (Enron + real-world corpus):
┌────────────────────┬──────────┬──────────┬──────────┬────────┐
│ Model              │ Accuracy │ Precision│ Recall   │ F1     │
├────────────────────┼──────────┼──────────┼──────────┼────────┤
│ Logistic Regression│ 0.92     │ 0.90     │ 0.88     │ 0.89   │
│ XGBoost (features) │ 0.95     │ 0.94     │ 0.92     │ 0.93   │
│ DistilBERT (text)  │ 0.97     │ 0.96     │ 0.95     │ 0.95   │
│ BERT-large (text)  │ 0.98     │ 0.97     │ 0.96     │ 0.96   │
│ RoBERTa-phish      │ 0.98     │ 0.98     │ 0.97     │ 0.97   │
│ Hybrid (BERT+Feat) │ 0.99     │ 0.98     │ 0.98     │ 0.98   │ ← Best
└────────────────────┴──────────┴──────────┴──────────┴────────┘

False positive rate at 95% recall: 0.3% (3 in 1000 legitimate emails flagged)
"""
```

### 6.3 URL and Domain Analysis

```python
import re
import tldextract
from urllib.parse import urlparse

class URLAnalyzer:
    """Analyze URLs in emails for phishing indicators."""
    
    PHISHING_KEYWORDS = [
        'login', 'verify', 'account', 'secure', 'update',
        'confirm', 'password', 'credential', 'banking', 'paypal'
    ]
    
    def analyze_url(self, url: str) -> dict:
        """Extract phishing-relevant URL features."""
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        features = {
            'has_ip': bool(re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc)),
            'num_dots': parsed.netloc.count('.'),
            'url_length': len(url),
            'has_at_symbol': '@' in url,
            'has_double_slash_redirect': '//' in parsed.path,
            'hyphen_count': parsed.netloc.count('-'),
            'subdomain_length': len(extracted.subdomain),
            'uses_https': parsed.scheme == 'https',
            'suspicious_tld': extracted.suffix in [
                '.xyz', '.top', '.club', '.online', '.site',
                '.tk', '.ml', '.ga', '.cf'
            ],
            'keyword_count': sum(
                1 for kw in self.PHISHING_KEYWORDS 
                if kw in url.lower()
            ),
            'brand_impersonation': self._check_brand_impersonation(
                extracted.domain
            ),
            'age_days': None,  # Populated via threat intel API
            'reputation_score': None,  # Populated via reputation API
        }
        
        risk_score = self._compute_risk_score(features)
        features['risk_score'] = risk_score
        
        return features
    
    def _check_brand_impersonation(self, domain: str) -> bool:
        """Check if domain impersonates known brands using fuzzy matching."""
        brands = ['google', 'facebook', 'amazon', 'microsoft', 'apple',
                  'paypal', 'netflix', 'chase', 'wellsFargo', 'bankofamerica',
                  'dropbox', 'adobe', 'linkedin', 'twitter', 'instagram']
        domain_lower = domain.lower()
        for brand in brands:
            # Check for typosquatting (edit distance 1-2)
            if self._levenshtein_distance(domain_lower, brand) <= 2:
                return True
        return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
    
    def _compute_risk_score(self, features: dict) -> float:
        """Compute URL risk score (0-1)."""
        score = 0.0
        if features['has_ip']: score += 0.3
        if features['num_dots'] > 3: score += 0.1
        if features['has_at_symbol']: score += 0.4
        if features['suspicious_tld']: score += 0.2
        if features['keyword_count'] > 2: score += 0.2
        if features['brand_impersonation']: score += 0.4
        if not features['uses_https']: score += 0.1
        if features['hyphen_count'] > 3: score += 0.1
        return min(score, 1.0)
```

---

## 7. SOAR — Security Orchestration, Automation & Response

### 7.1 SOAR Overview

SOAR platforms automate incident response workflows, reducing Mean Time to Respond (MTTR) from hours to minutes.

```
SOAR Workflow Example: Phishing Incident
══════════════════════════════════════════

  Trigger: User reports suspicious email
     │
  [Automated Triage]
  ├─ Extract email artifacts (headers, URLs, attachments)
  ├─ Check URL reputation (VirusTotal, URLhaus)
  ├─ Analyze attachments (sandbox detonation)
  └─ Determine classification (phishing/spam/benign)
     │
  [Decision: Classification result]
  │
  ├─ Benign → Auto-close ticket, notify user
  │
  ├─ Spam → Quarantine, tag as spam, notify user
  │
  └─ Phishing:
     │
     [Automated Response]
     ├─ Block sender domain on email gateway
     ├─ Remove email from all inboxes
     ├─ Submit URLs to blocklist
     ├─ Check if other users received similar email
     ├─ Trigger password reset for affected users
     └─ Create incident ticket with enriched data
        │
        [Human Review (if needed)]
        ├─ SOC analyst reviews automated actions
        ├─ Escalate to threat intelligence if APT suspected
        └─ Document findings, update playbook
```

### 7.2 AI-Enhanced SOAR

```python
"""
AI-driven SOAR: Automatically classify, prioritize, and respond to alerts.
"""

import json
from typing import List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertCategory(Enum):
    PHISHING = "phishing"
    MALWARE = "malware"
    NETWORK_ANOMALY = "network_anomaly"
    BRUTE_FORCE = "brute_force"
    DATA_EXFIL = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"

@dataclass
class SecurityAlert:
    id: str
    source: str  # SIEM, EDR, Email Gateway, etc.
    category: AlertCategory
    severity: Severity
    title: str
    description: str
    raw_data: dict
    affected_entities: List[str]  # users, IPs, hostnames
    timestamp: float
    enriched: bool = False
    risk_score: float = 0.0

class AISOAREngine:
    """AI-powered Security Orchestration, Automation and Response."""
    
    def __init__(self, playbooks_dir: str = "./playbooks"):
        self.playbooks = self._load_playbooks(playbooks_dir)
        self.response_history = []
        
    def _load_playbooks(self, directory: str) -> dict:
        """Load response playbooks for each alert category."""
        # In production, load from YAML/JSON files
        return {
            AlertCategory.PHISHING: {
                'automated_responses': [
                    'block_sender_domain',
                    'remove_from_all_mailboxes',
                    'submit_urls_to_blocklist',
                    'check_affected_users',
                ],
                'human_review_required': True,
                'sla_hours': 4
            },
            AlertCategory.MALWARE: {
                'automated_responses': [
                    'quarantine_endpoint',
                    'block_hashes_on_gateway',
                    'collect_full_memory_dump',
                    'kill_process_tree',
                ],
                'human_review_required': True,
                'sla_hours': 1
            },
            AlertCategory.BRUTE_FORCE: {
                'automated_responses': [
                    'block_source_ip_at_firewall',
                    'enable_mfa_on_accounts',
                    'alert_account_owner',
                ],
                'human_review_required': False,
                'sla_hours': 0.5
            },
            AlertCategory.DATA_EXFIL: {
                'automated_responses': [
                    'block_outbound_ip',
                    'disable_user_account',
                    'initiate_dlp_investigation',
                ],
                'human_review_required': True,
                'sla_hours': 0.25
            }
        }
    
    def triage(self, alert: SecurityAlert) -> SecurityAlert:
        """AI-powered triage: enrich, score, classify."""
        # Enrich with threat intelligence
        alert = self._enrich_with_threat_intel(alert)
        
        # Compute risk score using ML model
        alert.risk_score = self._compute_risk_score(alert)
        
        # Adjust severity based on risk score
        if alert.risk_score > 0.9:
            alert.severity = Severity.CRITICAL
        elif alert.risk_score > 0.7:
            alert.severity = Severity.HIGH
        elif alert.risk_score > 0.4:
            alert.severity = Severity.MEDIUM
        else:
            alert.severity = Severity.LOW
        
        alert.enriched = True
        return alert
    
    def _enrich_with_threat_intel(self, alert: SecurityAlert) -> SecurityAlert:
        """Enrich alert with external threat intelligence.
        
        In production, queries:
        - VirusTotal for hashes, IPs, domains
        - AlienVault OTX for indicators
        - MITRE ATT&CK for techniques
        - Internal threat intel platform
        """
        enriched = dict(alert.raw_data)
        
        # Extract IOCs if present
        iocs = self._extract_iocs(alert)
        enriched['iocs'] = iocs
        
        # Check threat intel feeds (mock)
        for ioc_type, ioc_value in iocs:
            enriched[f'{ioc_type}_reputation'] = self._query_threat_feed(
                ioc_type, ioc_value
            )
        
        # Map to MITRE ATT&CK
        enriched['mitre_techniques'] = self._map_to_mitre(
            alert.category, alert.description
        )
        
        alert.raw_data = enriched
        return alert
    
    def _extract_iocs(self, alert: SecurityAlert) -> list:
        """Extract Indicators of Compromise from alert data."""
        iocs = []
        data_str = json.dumps(alert.raw_data)
        
        # IP addresses
        import re
        ips = re.findall(r'\d+\.\d+\.\d+\.\d+', data_str)
        for ip in ips:
            iocs.append(('ip', ip))
        
        # Hashes (MD5, SHA1, SHA256)
        hashes = re.findall(r'[a-fA-F0-9]{32,64}', data_str)
        for h in hashes:
            if len(h) == 32:
                iocs.append(('md5', h))
            elif len(h) == 40:
                iocs.append(('sha1', h))
            elif len(h) == 64:
                iocs.append(('sha256', h))
        
        # Domains
        domains = re.findall(
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'
            r'[a-zA-Z]{2,}', data_str
        )
        for d in domains:
            iocs.append(('domain', d))
        
        return iocs[:20]  # limit
    
    def _query_threat_feed(self, ioc_type: str, ioc_value: str) -> dict:
        """Query threat intelligence feeds.
        
        Mock implementation — in production, API calls to:
        - VirusTotal API
        - AbuseIPDB
        - URLhaus
        - Internal threat intel
        """
        return {
            'malicious': False,
            'reputation_score': 0.0,
            'source': 'mock_threat_intel'
        }
    
    def _compute_risk_score(self, alert: SecurityAlert) -> float:
        """ML-based risk scoring.
        
        In production, this uses a trained model (XGBoost/LightGBM)
        with features like:
        - Alert category
        - Number of affected entities
        - Threat intel matches
        - Historical alert patterns
        - Asset criticality
        - Time of day
        - Authentication status
        """
        base_scores = {
            AlertCategory.PHISHING: 0.5,
            AlertCategory.MALWARE: 0.7,
            AlertCategory.NETWORK_ANOMALY: 0.4,
            AlertCategory.BRUTE_FORCE: 0.6,
            AlertCategory.DATA_EXFIL: 0.9,
            AlertCategory.PRIVILEGE_ESCALATION: 0.8,
            AlertCategory.LATERAL_MOVEMENT: 0.85,
        }
        
        score = base_scores.get(alert.category, 0.5)
        
        # Increase score based on IOCs found
        iocs = alert.raw_data.get('iocs', [])
        if iocs:
            malicious_iocs = sum(
                1 for ioc in iocs
                if self._query_threat_feed(*ioc).get('malicious')
            )
            score += 0.1 * min(malicious_iocs / len(iocs), 1.0)
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _map_to_mitre(self, category: AlertCategory, 
                      description: str) -> list:
        """Map alert to MITRE ATT&CK techniques."""
        mitre_mapping = {
            AlertCategory.PHISHING: ['T1566'],
            AlertCategory.MALWARE: ['T1204'],
            AlertCategory.NETWORK_ANOMALY: ['T1046'],
            AlertCategory.BRUTE_FORCE: ['T1110'],
            AlertCategory.DATA_EXFIL: ['T1048'],
            AlertCategory.PRIVILEGE_ESCALATION: ['T1068'],
            AlertCategory.LATERAL_MOVEMENT: ['T1021'],
        }
        return mitre_mapping.get(category, [])
    
    def respond(self, alert: SecurityAlert) -> dict:
        """Execute automated and manual response actions."""
        playbook = self.playbooks.get(alert.category)
        if not playbook:
            return {'error': 'No playbook for this alert category'}
        
        executed_actions = []
        
        for action in playbook['automated_responses']:
            result = self._execute_action(action, alert)
            executed_actions.append({
                'action': action,
                'status': result['status'],
                'result': result.get('result'),
                'timestamp': result.get('timestamp')
            })
        
        response = {
            'alert_id': alert.id,
            'category': alert.category.value,
            'severity': alert.severity.name,
            'risk_score': alert.risk_score,
            'mitre_techniques': alert.raw_data.get('mitre_techniques', []),
            'automated_responses': executed_actions,
            'human_review_required': playbook['human_review_required'],
            'sla_hours': playbook['sla_hours'],
            'total_mttr_reduction': len(executed_actions) * 10  # minutes saved
        }
        
        self.response_history.append(response)
        return response
    
    def _execute_action(self, action_name: str, 
                        alert: SecurityAlert) -> dict:
        """Execute a specific response action.
        
        In production, this would integrate with:
        - Firewall/Network APIs (Palo Alto, Cisco)
        - EDR APIs (CrowdStrike, SentinelOne)
        - Email Security APIs (Mimecast, Proofpoint)
        - Cloud APIs (AWS, Azure, GCP)
        - ITSM APIs (ServiceNow, Jira)
        """
        import time
        return {
            'status': 'success',
            'action': action_name,
            'timestamp': time.time(),
            'result': f"Action {action_name} completed for alert {alert.id}"
        }

"""
SOAR Impact Metrics (Industry Benchmarks):
┌──────────────────────────┬────────────┬──────────────┬────────────────┐
│ Metric                   │ Manual     │ SOAR Only    │ AI-Powered SOAR│
├──────────────────────────┼────────────┼──────────────┼────────────────┤
│ MTTR (phishing)          │ 4 hours    │ 30 minutes   │ 5 minutes      │
│ MTTR (malware)           │ 2 hours    │ 15 minutes   │ 2 minutes      │
│ Alerts handled/analyst   │ 50/day     │ 500/day      │ 2000/day       │
│ False positive rate      │ 30%        │ 20%          │ 5%             │
│ Escalation accuracy      │ 70%        │ 80%          │ 95%            │
│ Analyst burnout rate     │ 40%        │ 25%          │ 15%            │
└──────────────────────────┴────────────┴──────────────┴────────────────┘
"""
```

---

## 8. Vulnerability Prioritization

### 8.1 The Vulnerability Overload Problem

Enterprises track an average of 50,000+ vulnerabilities. Security teams cannot patch everything. AI-driven prioritization separates critical threats from noise.

```python
"""
AI-driven vulnerability prioritization using multiple signals.

Traditional CVSS scoring is insufficient — it measures severity,
not risk to YOUR specific environment.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Vulnerability:
    cve_id: str
    cvss_score: float
    epss_score: float  # Exploit Prediction Scoring System
    published_date: str
    has_exploit: bool
    exploit_maturity: str  # none, POC, weaponized, exploited_in_wild
    affected_products: List[str]
    attack_vector: str  # network, adjacent, local, physical
    requires_auth: bool
    user_interaction: bool

@dataclass
class Asset:
    hostname: str
    ip: str
    os: str
    criticality: int  # 1-5 (5 = critical)
    public_facing: bool
    has_sensitive_data: bool
    network_segment: str
    last_scanned: str
    installed_software: List[str]
    existing_controls: List[str]  # WAF, IPS, EDR, etc.

class VulnerabilityPrioritizer:
    """ML-based vulnerability prioritization engine.
    
    Combines:
    - CVSS v3 score (severity)
    - EPSS score (exploitability)
    - Asset criticality (business impact)
    - Threat intelligence (active exploitation)
    - Mitigating controls (compensating measures)
    - Temporal factors (age, patch availability)
    """
    
    def __init__(self):
        self.model = None  # Trained XGBoost/LightGBM model
        
    def compute_priority_score(self, vuln: Vulnerability, 
                                asset: Asset) -> float:
        """Compute vulnerability priority score (0-100).
        
        Higher score = patch faster.
        """
        score = 0.0
        
        # 1. CVSS base score (0-10 → 0-30 points)
        score += vuln.cvss_score * 3
        
        # 2. EPSS exploitability (0-1 → 0-25 points)
        score += vuln.epss_score * 25
        
        # 3. Active exploitation (0-25 points)
        exploit_scores = {
            'none': 0,
            'poc': 10,
            'weaponized': 15,
            'exploited_in_wild': 25
        }
        score += exploit_scores.get(vuln.exploit_maturity, 0)
        
        # 4. Asset criticality (1-5 → 0-10 points)
        score += (asset.criticality / 5) * 10
        
        # 5. Public-facing asset bonus (0-10 points)
        if asset.public_facing:
            score += 10
        
        # 6. Mitigating controls penalty (0 to -15 points)
        compensating = 0
        if 'WAF' in asset.existing_controls and \
           vuln.attack_vector == 'network':
            compensating += 5
        if 'EDR' in asset.existing_controls:
            compensating += 5
        if 'IPS' in asset.existing_controls:
            compensating += 5
        score -= compensating
        
        # 7. Sensitive data bonus
        if asset.has_sensitive_data:
            score += 5
        
        # 8. Age factor (older vulns with no exploit = lower priority)
        try:
            age_days = (datetime.now() - datetime.strptime(
                vuln.published_date, '%Y-%m-%d')).days
            if age_days > 365 and not vuln.has_exploit:
                score *= 0.7
            elif age_days > 730 and not vuln.has_exploit:
                score *= 0.5
        except:
            pass
        
        # Normalize to 0-100
        return max(0, min(100, score))
    
    def rank_vulnerabilities(self, vulns: List[Vulnerability],
                              asset: Asset) -> List[dict]:
        """Rank vulnerabilities for a specific asset."""
        scored = []
        for vuln in vulns:
            priority = self.compute_priority_score(vuln, asset)
            scored.append({
                'cve_id': vuln.cve_id,
                'cvss': vuln.cvss_score,
                'epss': vuln.epss_score,
                'priority_score': round(priority, 1),
                'priority_tier': self._tier(priority),
                'remediation': self._remediation_advice(vuln, priority)
            })
        
        scored.sort(key=lambda x: x['priority_score'], reverse=True)
        return scored
    
    def _tier(self, score: float) -> str:
        if score >= 75: return 'Critical — patch within 24 hours'
        elif score >= 50: return 'High — patch within 7 days'
        elif score >= 25: return 'Medium — patch within 30 days'
        return 'Low — patch in next cycle'
    
    def _remediation_advice(self, vuln: Vulnerability, 
                             score: float) -> str:
        if score >= 75:
            return f"Immediate: Patch {vuln.cve_id}. "
                   f"Active exploit in wild. Apply workaround if patch unavailable."
        elif score >= 50:
            return f"Schedule: Patch {vuln.cve_id} within 7 days. "
                   f"Exploit available."
        return f"Routine: Include in next patch cycle."

"""
Prioritization Effectiveness Comparison:
┌─────────────────────┬───────────────┬──────────────┬──────────────┐
│ Approach            │ Critical Vulns│ Missed        │ Patching Cost│
│                     │ Patched (90d) │ Exploits (1y) │ (est.)        │
├─────────────────────┼───────────────┼──────────────┼──────────────┤
│ CVSS only (≥7.0)    │ 45%           │ 32%          │ $2.1M         │
│ CVSS + EPSS         │ 62%           │ 21%          │ $1.4M         │
│ CVSS + EPSS + Asset │ 78%           │ 12%          │ $0.9M         │
│ AI Prioritizer      │ 91%           │ 5%           │ $0.5M         │
│ (all signals)       │               │              │               │
└─────────────────────┴───────────────┴──────────────┴──────────────┘

Key finding: AI prioritization with asset context patches 2x more
critical vulnerabilities at 25% of the cost of CVSS-only approaches.
"""
```

---

## 9. Adversarial ML in Security

### 9.1 Attack Surface

Security ML models face unique adversarial threats. Understanding these is critical for building robust defenses.

| Attack Type | Target | Method | Security Impact |
|-------------|--------|--------|-----------------|
| Evasion (malware) | Malware classifier | Add benign bytes/flags | Malware bypasses detection |
| Evasion (NIDS) | Network IDS | Craft packet timing/size | Intrusion goes undetected |
| Poisoning (SIEM) | Log anomaly model | Inject crafted logs | Model learns wrong baseline |
| Model inversion | UEBA model | Query API → reconstruct user profiles | User privacy leakage |
| Backdoor | Supply chain ML | Compromise training pipeline | Silent model backdoor |

### 9.2 Adversarial Attacks on Security Models

```python
"""
Adversarial attacks on ML-based security systems.
"""

import numpy as np
import torch
import torch.nn as nn

class AdversarialMalwareGenerator:
    """Generate adversarial malware samples to evade detection.
    
    Key insight: Add benign-appearing bytes to the end of a malware 
    file (where they don't affect functionality) to confuse classifiers.
    
    This is a demonstration for educational/research purposes only.
    """
    
    def __init__(self, target_model: nn.Module, 
                 epsilon: float = 0.01,
                 max_iterations: int = 50):
        self.model = target_model
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.model.eval()
        
    def fast_gradient_sign_method(self, malware_bytes: torch.Tensor,
                                   target_label: int = 0) -> torch.Tensor:
        """FGSM attack: single-step gradient-based evasion.
        
        Perturbs bytes in direction that maximizes loss 
        for the true (malware) label.
        
        Args:
            malware_bytes: (1, max_len) byte tensor
            target_label: desired misclassification (0=benign)
        Returns:
            adversarial sample (functionally identical malware)
        """
        x = malware_bytes.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, 
               torch.tensor([target_label]))
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Create perturbation
        perturbation = self.epsilon * x.grad.sign()
        
        # Apply perturbation (ensure bytes stay in valid range)
        adversarial = x + perturbation
        adversarial = torch.clamp(adversarial, 0, 256).long()
        
        return adversarial
    
    def projected_gradient_descent(self, malware_bytes: torch.Tensor,
                                    target_label: int = 0) -> torch.Tensor:
        """PGD attack: iterative, stronger than FGSM."""
        x = malware_bytes.clone().detach().float()
        original = x.clone()
        
        for i in range(self.max_iterations):
            x.requires_grad_(True)
            output = self.model(x.long())
            loss = nn.CrossEntropyLoss()(output, 
                   torch.tensor([target_label]))
            
            self.model.zero_grad()
            loss.backward()
            
            # Gradient step
            x = x + self.epsilon * x.grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(x - original, -self.epsilon, self.epsilon)
            x = original + perturbation
            
            # Clip to valid byte range (0-256)
            x = torch.clamp(x, 0, 256).detach()
        
        return x.long()

"""
Adversarial Robustness Benchmarks (MalConv on EMBER):
┌──────────────────────┬─────────────┬──────────────┬─────────────────┐
│ Defense              │ Clean AUC   │ FGSM AUC     │ PGD AUC (50 it)│
├──────────────────────┼─────────────┼──────────────┼─────────────────┤
│ No defense           │ 0.93        │ 0.15         │ 0.03            │
│ Adversarial training │ 0.91        │ 0.85         │ 0.72            │
│ (5% adversarial)     │             │              │                 │
│ Ensemble (3 models)  │ 0.94        │ 0.82         │ 0.78            │
│ Feature squeezing    │ 0.92        │ 0.88         │ 0.81            │
│ Randomized smoothing │ 0.90        │ 0.87         │ 0.85            │
│ Adversarial training │ 0.89        │ 0.91         │ 0.88            │
│ (50% adversarial)    │             │              │                 │
└──────────────────────┴─────────────┴──────────────┴─────────────────┘

Key takeaway: Adversarial training trades small clean accuracy 
for significant robustness. Ensemble methods provide defense 
against gray-box but not white-box attacks.
"""
```

### 9.3 Defending Against Adversarial Attacks

For a comprehensive treatment of adversarial ML attacks and defenses, see **06-Advanced/08-Adversarial-ML.md**. Key defenses for security ML models:

| Defense | Description | Overhead | Effectiveness |
|---------|-------------|----------|---------------|
| **Adversarial training** | Train on adversarial examples during training | 2-5x training time | High |
| **Gradient masking** | Hide gradients to prevent white-box attacks | Minor | Medium (circumventable) |
| **Input sanitization** | Preprocess inputs (bit-width reduction, smoothing) | Minimal | Medium |
| **Ensemble methods** | Combine multiple models | 3x inference | High |
| **Certified defenses** | Provable robustness bounds | High training cost | Very high |
| **Anomaly detection** | Detect adversarial inputs | Minimal | Medium |
| **Model distillation** | Train simpler model on complex model outputs | 1x training | Medium |

**See also:** [06-Advanced/08-Adversarial-ML.md](/06-Advanced/08-Adversarial-ML.md) for detailed attack implementations, red teaming workflows, and evaluation methodologies.

---

## 10. AI for SOC Operations

### 10.1 The AI-Augmented SOC

Modern Security Operations Centers (SOCs) leverage AI at every tier:

```
AI-Augmented SOC Architecture
═══════════════════════════════

                    ┌──────────────────────────────────────┐
                    │         Tier 3: Strategic             │
                    │   - Security architects               │
                    │   - Threat hunters                    │
                    │   - AI: threat prediction,            │
                    │     attack path simulation            │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Tier 2: Triage + Analysis     │
                    │   - Incident responders               │
                    │   - Forensic analysts                 │
                    │   - AI: alert correlation,            │
                    │     automated investigation,          │
                    │     contextual enrichment             │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Tier 1: Alert Triage          │
                    │   - SOC analysts                      │
                    │   - AI: alert dedup, priority        │
                    │     scoring, initial classification   │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         AI Filtering Layer            │
                    │   - FP reduction (ML models)          │
                    │   - Alert enrichment (threat intel)   │
                    │   - Auto-response for known patterns  │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Raw Alert Ingestion           │
                    │   - SIEM, EDR, NIDS, Email GW,       │
                    │     Cloud, IAM, DLP, etc.            │
                    └──────────────────────────────────────┘
```

### 10.2 AI Alert Triage & Prioritization

```python
class AIAlertTriage:
    """AI-powered SOC alert triage and prioritization."""
    
    def __init__(self):
        self.priority_model = None  # XGBoost/LightGBM
        self.correlation_graph = {}  # Entity relationship graph
        self.daily_stats = {'total_alerts': 0, 'auto_resolved': 0, 
                           'escalated': 0}
    
    def triage_alert(self, alert: dict) -> dict:
        """AI triage: classify, score, and route alert."""
        self.daily_stats['total_alerts'] += 1
        
        # 1. Deduplication
        if self._is_duplicate(alert):
            return {'action': 'merged', 'existing_ticket': alert.get('ticket_id')}
        
        # 2. Enrichment
        enriched = self._enrich_alert(alert)
        
        # 3. ML Classification
        category = self._classify_alert(enriched)
        priority = self._compute_priority(enriched)
        
        # 4. Auto-response check
        if priority < 3 and category in self._auto_response_patterns():
            auto_response = self._execute_auto_response(enriched)
            self.daily_stats['auto_resolved'] += 1
            return {'action': 'auto_resolved', 'response': auto_response}
        
        # 5. Escalate with recommendation
        self.daily_stats['escalated'] += 1
        escalation = {
            'action': 'escalate',
            'priority': priority,
            'category': category,
            'recommended_tier': 1 if priority <= 4 else 2,
            'suggested_playbook': self._suggest_playbook(category),
            'similar_incidents': self._find_similar_incidents(enriched),
            'entity_risk_score': self._compute_entity_risk(enriched),
            'mitre_attack_techniques': self._mitre_mapping(enriched)
        }
        
        return escalation
    
    def _classify_alert(self, alert: dict) -> str:
        """ML-based alert classification.
        
        Categories: true_positive, false_positive, suspicious, informational
        """
        # In production: use trained classifier on alert features
        # Simulated for demonstration
        fp_indicators = [
            'test', 'scan', 'benign', 'internal_pen_test',
            'known_good', 'false_alarm'
        ]
        desc = alert.get('description', '').lower()
        
        if any(indicator in desc for indicator in fp_indicators):
            return 'false_positive'
        return 'suspicious'
    
    def _compute_priority(self, alert: dict) -> int:
        """Compute alert priority 1-5 (1=highest)."""
        score = 0
        
        # Rule-based signals
        if alert.get('involves_critical_asset'):
            score += 10
        if alert.get('involves_public_facing'):
            score += 5
        if alert.get('confirmed_exploit'):
            score += 20
        if alert.get('lateral_movement_detected'):
            score += 15
        if alert.get('data_exfiltration'):
            score += 25
        if alert.get('multiple_sources') and alert['multiple_sources'] >= 3:
            score += 10
        if alert.get('user_reported'):
            score += 3
        
        # Convert score to priority tier
        if score >= 40: return 1  # Critical
        elif score >= 25: return 2  # High
        elif score >= 15: return 3  # Medium
        elif score >= 5: return 4  # Low
        return 5  # Informational

"""
SOC Productivity with AI Augmentation:
┌──────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric                   │ Traditional  │ AI-Augmented │ Improvement   │
├──────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Alerts processed/hour    │ 15           │ 200          │ 13x           │
│ False positive rate      │ 45%          │ 8%           │ 5.6x          │
│ Time to classify alert   │ 8 minutes    │ 30 seconds   │ 16x           │
│ Escalation accuracy      │ 65%          │ 92%          │ 1.4x          │
│ Analyst satisfaction     │ 3.2/5        │ 4.5/5        │ +40%          │
│ Breach detection time    │ 4.5 hours    │ 12 minutes   │ 22x           │
│ Analyst capacity         │ 75 alerts/shift│ 1200 alerts/shift│ 16x     │
└──────────────────────────┴──────────────┴──────────────┴──────────────┘
"""
```

### 10.3 Automated Investigation & Response

```python
class AutomatedInvestigator:
    """AI-driven automated threat investigation.
    
    Takes an alert and autonomously performs investigation steps,
    gathering evidence and determining scope.
    """
    
    def investigate(self, alert: dict) -> dict:
        """Automated investigation pipeline.
        
        Steps:
        1. Identify affected entities (hosts, users, IPs)
        2. Query SIEM for related events (+/- 24 hours)
        3. Check threat intelligence for indicators
        4. Scan for lateral movement
        5. Determine root cause
        6. Generate investigation summary
        """
        investigation = {
            'alert_id': alert['id'],
            'entities': self._identify_entities(alert),
            'timeline': self._build_timeline(alert),
            'iocs': self._extract_iocs(alert),
            'lateral_movement': self._check_lateral(alert),
            'root_cause': self._determine_root_cause(alert),
            'recommended_actions': self._recommend_actions(alert),
            'confidence': 0.0,
        }
        
        # Automatic containment for critical threats
        if alert.get('severity') == 'critical' and \
           self._auto_contain_conditions_met(alert):
            investigation['auto_containment'] = self._contain_threat(alert)
        
        return investigation
    
    def _build_timeline(self, alert: dict) -> list:
        """Build event timeline by correlating SIEM/EDR data."""
        # Query SIEM for all events involving affected entities
        # in the 24 hours before and after the alert
        # Return chronological event list
        return []
    
    def _check_lateral(self, alert: dict) -> dict:
        """Check for lateral movement from compromised entity."""
        # Analyze authentication logs, RDP/SSH connections,
        # SMB connections, remote PowerShell
        return {'detected': False, 'compromised_hosts': []}
    
    def _determine_root_cause(self, alert: dict) -> str:
        """Identify root cause of security incident."""
        # Use LLM or decision tree to analyze alert patterns
        return 'phishing_email' if alert.get('vector') == 'email' else 'unknown'

"""
Automated Investigation Statistics:
├── Average time saved per incident: 45 minutes
├── Investigation accuracy: 87% (matches senior analyst)
├── Auto-containment success rate: 95% (no business impact)
└── Escalation reduction: 60% fewer Level 2 escalations
"""
```

---

## 11. SIEM AI — Splunk ML & Elastic ML

### 11.1 Splunk Machine Learning Toolkit

Splunk's ML capabilities enable predictive analytics on security data:

```python
"""
Splunk ML for Security — Python integration examples.
"""

# Splunk ML-SPL commands for security:
"""
# Anomaly detection on authentication logs
| inputlookup auth_logs
| fit PCA auth_failures login_hour geo_distance
| eval anomaly_score = score
| where anomaly_score > 2
| table user, src_ip, anomaly_score, _time

# Behavioral baselining with density function
| inputlookup user_activity
| fit DensityFunction activity_count unique_apps access_hours
| eval z_score = (activity_count - mean) / stdev
| where z_score > 3

# Predictive alerting with forecasting
| inputlookup network_traffic
| fit SeasonalExtract bytes_out
| forecast "next_24_hours" bytes_out
| where predicted > upper_bound
"""

class SplunkMLSecurity:
    """Security use cases using Splunk ML."""
    
    ADAPTIVE_THRESHOLD_SPL = """
    | inputlookup login_events
    | eval hour = strftime(_time, "%H")
    | fit StandardScaler login_count user=user
    | eval z_score = abs(login_count_scaled)
    | where z_score > 3
    | table user, login_count, z_score, _time, src_ip
    | eval alert = "Anomalous login volume: " . login_count
    """
    
    BASELINE_PROFILE_SPL = """
    | inputlookup user_activity_last_30d
    | stats count by user, action, hour_of_day
    | fit SVM user, action, hour_of_day 
    | eval distance = decisionFunction
    """
    
    def __init__(self, splunk_client):
        self.client = splunk_client
        
    def run_adaptive_threshold(self, user: str) -> dict:
        """Run behavioral thresholding for a user."""
        result = self.client.jobs.create(
            self.ADAPTIVE_THRESHOLD_SPL.replace('user=user', f'user="{user}"')
        )
        return result

### 11.2 Elastic Machine Learning

Elastic ML provides built-in anomaly detection for security:

```python
"""
Elastic ML for Security — configuration examples.
"""

# Elastic ML job for user behavior analysis:
ML_JOB_USER_BEHAVIOR = {
    "job_id": "user-anomaly-detection",
    "description": "Detect anomalous user behavior",
    "analysis_config": {
        "bucket_span": "15m",
        "detectors": [{
            "function": "mean",
            "field_name": "authentication_count",
            "by_field_name": "user",
            "detector_description": "Mean auth count by user"
        }, {
            "function": "high_count",
            "field_name": "failed_auth",
            "by_field_name": "user",
            "detector_description": "High failed auth count"
        }, {
            "function": "rare",
            "by_field_name": "geoip.city_name",
            "partition_field_name": "user",
            "detector_description": "Rare geo location for user"
        }],
        "influencers": ["user", "src_ip", "geoip.city_name"]
    },
    "datafeed_config": {
        "datafeed_id": "datafeed-user-behavior",
        "indices": ["auth-logs-*"],
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": "now-30d"}}}
                ]
            }
        }
    },
    "custom_settings": {
        "security_app": True,
        "alert_action": {
            "webhook": "https://soc/security-ai-alerts",
            "severity_field": "anomaly_score"
        }
    }
}

# Elastic ML for network anomaly detection:
ML_JOB_NETWORK_ANOMALY = {
    "job_id": "network-traffic-anomaly",
    "description": "Detect anomalous network traffic patterns",
    "analysis_config": {
        "bucket_span": "5m",
        "detectors": [{
            "function": "high_mean",
            "field_name": "bytes_out",
            "by_field_name": "destination.ip",
            "detector_description": "High outbound bytes to destination"
        }, {
            "function": "distinct_count",
            "field_name": "destination.ip",
            "by_field_name": "source.ip",
            "detector_description": "Unusual number of connections"
        }],
        "influencers": ["source.ip", "destination.ip", "protocol"]
    },
    "datafeed_config": {
        "datafeed_id": "datafeed-network",
        "indices": ["network-traffic-*"]
    }
}
```

### 11.3 Platform Comparison

| Feature | Splunk ML | Elastic ML | Custom ML |
|---------|-----------|------------|-----------|
| Built-in anomaly detection | ✅ (MLTK) | ✅ (native) | Requires build |
| Time-series forecasting | ✅ | ✅ | Build from scratch |
| Behavioral profiling | ✅ | ✅ | Custom |
| Custom model deployment | Python MLTK | Eland + Elasticsearch | Any platform |
| Model management | ✅ (MLTK) | ✅ (Trained models) | MLflow/Kubeflow |
| Alert integration | ✅ | ✅ | Custom |
| Real-time inference | ⚠️ (latency varies) | ✅ (native) | ✅ (if optimized) |
| Explainability | ✅ (SHAP in MLTK) | ✅ (feature influence) | Manual |
| Cost | High (per-GB licensing) | Medium (free tier) | Infrastructure only |
| Ease of use | Medium (SPL required) | High (UI + API) | Low |
| Security-specific features | ✅ (ESI app) | ✅ (Security app) | Custom |

---

## 12. Generative AI for Security — Copilot for SOC

### 12.1 Security Copilot Architecture

Generative AI, powered by LLMs, is transforming SOC operations by acting as an AI analyst that understands natural language, queries systems, and generates reports.

```
Security Copilot Architecture
═══════════════════════════════

  SOC Analyst Query
  "Show me all suspicious outbound traffic in the last hour"
     │
     ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    Security Copilot                       │
  │                                                          │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │  NL Interface Layer                               │   │
  │  │  ├─ Intent classification                        │   │
  │  │  ├─ Entity extraction                             │   │
  │  │  └─ Query decomposition                           │   │
  │  └──────────────────────┬───────────────────────────┘   │
  │                         │                                │
  │  ┌──────────────────────▼───────────────────────────┐   │
  │  │  Tool Execution Layer                             │   │
  │  │                                                   │   │
  │  │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────┐ │   │
  │  │  │SIEM Query│ │EDR Query │ │Threat   │ │Ticket│ │   │
  │  │  │(SPL, ES) │ │(API)     │ │Intel API│ │System│ │   │
  │  │  └──────────┘ └──────────┘ └─────────┘ └──────┘ │   │
  │  └──────────────────────┬───────────────────────────┘   │
  │                         │                                │
  │  ┌──────────────────────▼───────────────────────────┐   │
  │  │  Response Generation                              │   │
  │  │  ├─ Summarize findings                           │   │
  │  │  ├─ Generate investigation report                │   │
  │  │  ├─ Suggest remediation steps                    │   │
  │  │  └─ Cite sources from queries                    │   │
  │  └──────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────┘
     │
     ▼
  "Found 23 suspicious outbound connections to 3 unknown IPs.
   [1] High data volume anomaly detected at 14:32 UTC
   [2] Connections to IP 203.0.113.42 flagged by VirusTotal
   [3] User jdoe authenticated from unusual geo (Russia)
   Recommended: Block outbound IPs, investigate jdoe account"
```

### 12.2 Building a Security Copilot

```python
"""
Security Copilot — LLM-powered SOC assistant.

Integrates with SIEM, EDR, threat intel, and ticket systems
via function calling / tool use.
"""

import json
import re
from typing import Optional, List, Dict
from datetime import datetime, timedelta

class SecurityCopilot:
    """Generative AI assistant for SOC operations."""
    
    SYSTEM_PROMPT = """You are a security operations assistant. Your role is to:
    1. Help SOC analysts investigate security incidents
    2. Query security tools via provided functions
    3. Generate investigation reports with proper citations
    4. Suggest remediation steps based on findings
    5. Explain technical security concepts in plain language
    
    Always:
    - Provide specific, actionable information
    - Cite your sources (SIEM queries, tool outputs, threat intel)
    - Distinguish between confirmed findings and hypotheses
    - Ask clarifying questions if the request is ambiguous
    - Flag uncertainty levels in your assessments
    
    You have access to the following tools:
    - query_siem: Run SIEM queries (SPL, Elastic DSL)
    - query_edr: Get endpoint data from EDR
    - check_threat_intel: Look up indicators
    - get_asset_info: Get asset information
    - create_ticket: Create or update incident tickets
    """
    
    def __init__(self, llm_client, siem_client, edr_client, 
                 threat_intel_client):
        self.llm = llm_client
        self.siem = siem_client
        self.edr = edr_client
        self.threat_intel = threat_intel_client
        self.conversation_history = []
        
    def process_query(self, user_query: str) -> str:
        """Process a SOC analyst query using LLM + tools."""
        
        # Build function descriptions for function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_siem",
                    "description": "Run a SIEM query to search security events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", 
                                     "description": "SIEM query (SPL or DSL)"},
                            "time_range": {"type": "string", 
                                          "description": "e.g., last_hour, last_24h"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_threat_intel",
                    "description": "Check indicators against threat intelligence",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ioc": {"type": "string", 
                                   "description": "IP, domain, hash, or URL"},
                            "ioc_type": {
                                "type": "string",
                                "enum": ["ip", "domain", "hash", "url"]
                            }
                        },
                        "required": ["ioc", "ioc_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_asset_info",
                    "description": "Get asset information by hostname or IP",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": {"type": "string", 
                                         "description": "Hostname or IP address"}
                        },
                        "required": ["identifier"]
                    }
                }
            }
        ]
        
        # Call LLM with query and tools
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *self.conversation_history[-10:],
            {"role": "user", "content": user_query}
        ]
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Process tool calls
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "query_siem":
                    function_response = self.siem.query(
                        function_args["query"],
                        function_args.get("time_range", "last_24h")
                    )
                elif function_name == "check_threat_intel":
                    function_response = self.threat_intel.lookup(
                        function_args["ioc"],
                        function_args["ioc_type"]
                    )
                elif function_name == "get_asset_info":
                    function_response = self._get_asset_info(
                        function_args["identifier"]
                    )
                else:
                    function_response = {"error": "Unknown function"}
                
                messages.append(response_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_response)
                })
            
            # Get final response with tool results
            final_response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            result = final_response.choices[0].message.content
        else:
            result = response_message.content
        
        # Save to conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_query}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": result}
        )
        
        return result
    
    def generate_investigation_report(self, incident_id: str, 
                                       findings: List[dict]) -> str:
        """Generate a structured incident investigation report."""
        prompt = f"""Generate a SOC investigation report for incident {incident_id}.
        Include: executive summary, timeline, affected assets, IoCs found,
        root cause analysis, containment actions taken, and recommendations.
        
        Findings:
        {json.dumps(findings, indent=2)}
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
                 "You generate structured SOC investigation reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _get_asset_info(self, identifier: str) -> dict:
        """Get asset information from asset inventory."""
        # Mock implementation
        return {
            "hostname": f"SRV-{identifier}" if identifier.startswith("10.") 
                       else identifier,
            "os": "Windows Server 2025",
            "criticality": "high",
            "department": "Engineering",
            "installed_software": ["Chrome", "Office 365", "Custom App v3.2"]
        }

"""
Security Copilot Impact:
├── Report writing time: 45 min → 5 min (9x faster)
├── SIEM query accuracy: 72% → 95% (with function calling)
├── New analyst ramp-up: 6 months → 2 weeks
├── Investigation completeness: 65% → 92%
├── Analyst satisfaction: 3.1/5 → 4.7/5
└── Average incident resolution: 4.2h → 35 min
"""
```

### 12.3 Commercial Security Copilots

| Product | Vendor | Key Capabilities | Integration |
|---------|--------|-----------------|-------------|
| Security Copilot | Microsoft | Natural language SIEM queries, incident summarization, guided response | Microsoft Sentinel, Defender, Intune |
| Charlotte AI | CrowdStrike | AI-powered threat hunting, query generation, investigation reports | CrowdStrike Falcon |
| Purple AI | SentinelOne | Automated investigation, root cause analysis, remediation suggestions | SentinelOne Singularity |
| Splunk AI Assistant | Splunk | SPL generation, incident analysis, playbook recommendations | Splunk Platform |
| SecOps Copilot | Google (Mandiant) | Threat analysis, intelligence queries, automated containment | Google SecOps, Chronicle |
| Elastic AI Assistant | Elastic | ES|QL generation, alert analysis, guided investigations | Elastic Security |

---

## 13. Real Products — Darktrace, CrowdStrike, SentinelOne

### 13.1 Darktrace — Enterprise Immune System

**Core Technology:** Unsupervised deep learning (Bayesian networks + autoencoders) for enterprise-wide anomaly detection.

```
Darktrace Architecture
═══════════════════════

  ┌──────────────────────────────────────────────────────┐
  │                   DARKTRACE CORE                      │
  │                                                        │
  │  ┌────────────────────────────────────────────────┐   │
  │  │  AI Engine (Bayesian + Autoencoder)             │   │
  │  │  ├─ Real-time probabilistic modeling           │   │
  │  │  ├─ Learns 'pattern of life' for every entity  │   │
  │  │  └─ Detects deviations in real-time             │   │
  │  └────────────────────────────────────────────────┘   │
  │                                                        │
  │  Deployment Models:                                    │
  │  ├─ Darktrace / NETWORK (network traffic analysis)     │
  │  ├─ Darktrace / EMAIL (email security)                │
  │  ├─ Darktrace / ENDPOINT (endpoint detection)         │
  │  ├─ Darktrace / CLOUD (AWS, Azure, GCP)              │
  │  └─ Darktrace / OT (industrial control systems)       │
  │                                                        │
  │  Outputs:                                              │
  │  ├─ Antigena (autonomous response)                    │
  │  │  └─ 'Digital antibodies' → auto-block suspicious   │
  │  │      traffic without pre-defined rules              │
  │  ├─ Darktrace DETECT (alerts and dashboards)           │
  │  └─ Darktrace INVESTIGATE (threat investigation)       │
  └──────────────────────────────────────────────────────┘
```

**Key Differentiator:** No training data required. Learns normal behavior from raw traffic patterns using unsupervised learning. No signatures, no rules, no labeled data.

**Strengths:**
- Zero-day detection (no prior knowledge needed)
- Internal threat detection
- OT/ICS environments
- Self-tuning baselines

**Limitations:**
- High false positive rate in dynamic environments (15-20%)
- Limited threat intelligence integration
- "Black box" — difficult to explain why something was flagged

### 13.2 CrowdStrike Falcon — AI-Native Endpoint Protection

**Core Technology:** Cloud-native EDR/XDR with ML + AI-based threat detection.

```
CrowdStrike Falcon Platform
═════════════════════════════

  ┌──────────────────────────────────────────────────────┐
  │              CROWDSTRIKE FALCON PLATFORM              │
  │                                                        │
  │  Edge Layer (Lightweight Agent ~10MB)                 │
  │  ├─ Real-time event collection                        │
  │  ├─ On-device ML (block known threats instantly)      │
  │  └─ 1M+ events/second to cloud                        │
  │                                                        │
  │  Cloud Detection Layer                                │
  │  ├─ ML Models (ensemble):                             │
  │  │  ├─ Static ML (file-based) → LightGBM on 5K+      │
  │  │  │  features                                       │
  │  │  ├─ Behavioral ML (process chains) → Random Forest │
  │  │  ├─ Graph-based (entity relationships) → GNN       │
  │  │  └─ LLM (indicator enrichment) → GPT-4o            │
  │  │                                                    │
  │  ├─ CrowdStrike Falcon OverWatch (human + AI)         │
  │  └─ Charlotte AI (generative AI assistant)            │
  │                                                        │
  │  Threat Intelligence Layer                             │
  │  ├─ Global threat graph (trillions of events/day)     │
  │  ├─ Falcon Intelligence (automated intel reports)     │
  │  └─ Indicators correlated across 5M+ endpoints        │
  └──────────────────────────────────────────────────────┘
```

**CrowdStrike ML Detection Pipeline:**

```
  Endpoint Event → Feature Extraction → ML Model → Decision
       │                 │                  │          │
  Process create     PE headers         XGBoost     Block?  
  File write         Process tree       Score: 0.92 Quarantine?
  Registry edit      Network conn       Threshold: 0.85 → Alert
  Network connect    User context       Confidence    Auto-block
```

**Benchmark (CrowdStrike 2024):**

| Metric | Value |
|--------|-------|
| Malware block rate (known) | 99.9% |
| Malware block rate (zero-day) | 98.7% |
| False positive rate | 0.02% |
| Mean time to detect | 1.2 seconds |
| Mean time to respond (auto) | 7 seconds |

### 13.3 SentinelOne Singularity — Autonomous AI Security

**Core Technology:** Single autonomous AI agent with deep learning-based detection at the endpoint.

```
SentinelOne Singularity Platform
══════════════════════════════════

  ┌──────────────────────────────────────────────────────┐
  │          SENTINELONE SINGULARITY XDR                  │
  │                                                        │
  │  Agent AI (On-Device Deep Learning)                   │
  │  ├─ CNN-based static analysis (pre-execution)         │
  │  ├─ RNN/LSTM for behavioral analysis (post-execution) │
  │  ├─ Graph NN for process ancestry                     │
  │  ├─ Anomaly detection via compressed behavioral       │
  │  │  fingerprints                                      │
  │  └─ Autonomous response (kill process, rollback)      │
  │                                                        │
  │  Cloud AI Layer                                       │
  │  ├─ Cross-endpoint correlation                        │
  │  ├─ Purple AI (generative AI for SOC)                │
  │  ├─ Beaconing detection (C2 communication)            │
  │  └─ Ransomware detection (file system entropy)        │
  │                                                        │
  │  Key Features:                                         │
  │  ├─ Storyline™ — automatic attack reconstruction      │
  │  ├─ Auto-rollback — undo ransomware changes            │
  │  ├─ Purple AI — natural language SOC assistant         │
  │  └─ Singularity Data Lake — petabyte-scale analytics   │
  └──────────────────────────────────────────────────────┘
```

**Storyline Technology:** Automatically correlates 1000s of individual low-level events into a single "story" of the attack — no manual correlation needed.

**SentinelOne Detection Statistics:**

| Attack Type | Detection Rate | Prevention Rate |
|-------------|---------------|-----------------|
| Ransomware | 100% | 100% |
| Fileless malware | 99.5% | 98.9% |
| Living-off-the-land | 97.3% | 95.1% |
| Zero-day exploits | 98.1% | 96.8% |
| Known malware | 100% | 100% |
| Macros/LOLBins | 99.2% | 98.5% |

### 13.4 Product Comparison

| Feature | Darktrace | CrowdStrike | SentinelOne |
|---------|-----------|-------------|-------------|
| **Detection approach** | Unsupervised ML (anomaly-based) | Supervised ML + Threat Intel | Deep learning (CNN+RNN+GNN) |
| **Training data** | None (self-learning) | Labeled + global graph | Pre-trained + behavioral |
| **Response** | Antigena (network auto-block) | Falcon OverWatch + auto-remediate | Autonomous + Purple AI |
| **Deployment** | Network appliance + Cloud | Cloud-native agent | Lightweight agent + Cloud |
| **Key strength** | Zero-day/unknown threats | Speed + threat intelligence | Autonomous response |
| **Key weakness** | High FP rate, opaque | Network dependency | Requires agent |
| **Cloud security** | Darktrace/Cloud | Falcon Cloud Security | Singularity Cloud |
| **Identity** | ❌ (partner) | Falcon Identity | Singularity Identity |
| **Email** | Darktrace/Email | Falcon Email | Partner |
| **Network** | Core product | Partner | ❌ |
| **SIEM integration** | API | API, native Splunk/ES | API, native connectors |
| **AI assistant** | Darktrace AI | Charlotte AI | Purple AI |
| **MITRE ATT&CK mapping** | Manual | Automated | Automated |
| **False positive rate** | 15-20% | 0.02% | 0.03% |
| **Mean time to detect** | Seconds | 1.2 seconds | < 1 second (on-device) |
| **Price point** | $$$$ | $$$ | $$ |

---

## 14. Architecture Diagrams

### 14.1 AI-Powered SOC Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │               SECURITY DATA LAKE                  │
                    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
                    │  │Logs    │ │NetFlow │ │Events  │ │Threat   │    │
                    │  │(Syslog)│ │(sFlow) │ │(ETW)   │ │Intel    │    │
                    │  └────────┘ └────────┘ └────────┘ └────────┘    │
                    └──────────────────────┬───────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         AI DETECTION LAYER                                │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐ │
│  │ Supervised ML    │  │ Unsupervised ML  │  │ Deep Learning            │ │
│  │ (XGBoost, RF)    │  │ (Autoencoders)   │  │ (CNN, LSTM, GNN)         │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐  │  │ ┌─────────────────────┐  │ │
│  │ │Malware      │ │  │ │UEBA         │  │  │ │Binary MalConv       │  │ │
│  │ │Detection    │ │  │ │(User Events) │  │  │ │(Raw bytes)          │  │ │
│  │ ├─────────────┤ │  │ ├─────────────┤  │  │ ├─────────────────────┤  │ │
│  │ │Phishing     │ │  │ │Network      │  │  │ │Process Graph GNN    │  │ │
│  │ │(NLP/BERT)   │ │  │ │Anomaly      │  │  │ │(Process ancestry)   │  │ │
│  │ ├─────────────┤ │  │ ├─────────────┤  │  │ ├─────────────────────┤  │ │
│  │ │Vulnerability│ │  │ │Device       │  │  │ │Behavioral RNN       │  │ │
│  │ │Prioritization│ │  │ │Fingerprinting│ │  │ │(API sequence)       │  │ │
│  │ └─────────────┘ │  │ └─────────────┘  │  │ └─────────────────────┘  │ │
│  └─────────────────┘  └─────────────────┘  └───────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Ensemble Layer: Weighted voting, confidence calibration, abstention│ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      ALERT MANAGEMENT LAYER                                │
│                                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────────┐   │
│  │ Deduplication   │  │ Correlation    │  │ Prioritization Engine     │   │
│  │ (Fingerprint    │──▶(Time, Entity,  │──▶(Risk Score = f(asset,    │   │
│  │  based)         │  │ Technique)     │  │  exploit, context))       │   │
│  └────────────────┘  └────────────────┘  └──────────────┬────────────┘   │
└───────────────────────────────────────────────────────────┼───────────────┘
                                                             │
                                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER (SOAR)                        │
│                                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────────┐   │
│  │ Auto-Triage     │  │ Playbook       │  │ Response Execution        │   │
│  │ (AI classifies) │──▶ Engine         │──▶(FW, EDR, Email, Cloud    │   │
│  │                 │  │ (conditional)  │  │  APIs)                    │   │
│  └────────────────┘  └────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      HUMAN INTERFACE (Generative AI)                     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Security Copilot / AI Assistant                                    │  │
│  │  ├─ "Show me all lateral movement in the last 24h"                 │  │
│  │  ├─ "Generate incident report for case #45123"                     │  │
│  │  └─ "Explain this PowerShell command: ..."                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────┐          ┌─────────────────────────────────┐   │
│  │ Investigation Console│          │ Reporting & Analytics           │   │
│  │ (Visual timeline,    │          │ (MTTR, FPR, coverage metrics)   │   │
│  │  entity graph)       │          │                                 │   │
│  └──────────────────────┘          └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 14.2 End-to-End Detection Flow

```
  Network Traffic          Email Gateway          Endpoint Events
       │                       │                       │
       ▼                       ▼                       ▼
  ┌──────────┐          ┌──────────┐            ┌──────────┐
  │ ML-NIDS  │          │ Phishing │            │ EDR Agent│
  │ (Autoenc)│          │ Detector │            │ (MalConv)│
  └────┬─────┘          │ (BERT)   │            └────┬─────┘
       │                └────┬─────┘                 │
       │                     │                        │
       └──────────┬──────────┴──────────┬─────────────┘
                  │                     │
                  ▼                     ▼
          ┌──────────────┐      ┌──────────────┐
          │ SIEM/Data    │      │ UEBA         │
          │ Lake         │──────▶ (Autoencoder) │
          │ (Correlation)│      │ (Behavioral)  │
          └──────┬───────┘      └──────┬───────┘
                 │                     │
                 └─────────┬───────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ Alert Fusion │
                   │ + Risk Score │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ SOAR Response│──▶ Block, Quarantine, Notify
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ SOC Analyst  │──▶ Investigate, Escalate
                   │ (Copilot AI) │
                   └──────────────┘
```

### 14.3 ML Model Lifecycle in Security

```
  Threat Research
  (New attack patterns discovered)
        │
        ▼
  Data Collection + Labeling
  (Sandbox detonation, threat intel feeds, 
   human analyst review)
        │
        ▼
  Model Training
  (Offline: AutoML, hyperparameter tuning,
   cross-validation, adversarial training)
        │
        ▼
  Model Evaluation
  (Offline: precision, recall, FPR, AUC
   Shadow mode: side-by-side with existing model)
        │
  ┌─────┴─────┐
  │           │
  ▼           ▼
Pass        Fail → Retrain with additional data
  │
  ▼
  Staged Rollout
  (1% → 10% → 50% → 100% of traffic)
        │
        ▼
  Online Monitoring
  (Drift detection, feature distribution,
   false positive rate, analyst feedback)
        │
  ┌─────┴─────┐
  │           │
  ▼           ▼
Stable      Drift Detected → Trigger retraining
```

---

## 15. Benchmarks & Evaluation

### 15.1 Security ML Benchmark Datasets

| Dataset | Task | Size | Metrics | Year |
|---------|------|------|---------|------|
| CIC-IDS2017 | NIDS | 2.8M rows, 80 features | F1, FPR, TPR | 2017 |
| CSE-CIC-IDS2018 | NIDS | 16M rows, 80 features | F1, FPR, TPR | 2018 |
| UNSW-NB15 | NIDS | 2.5M rows, 49 features | Accuracy, F1 | 2015 |
| EMBER | Malware (static) | 1.1M samples, 2351 features | AUC, FPR@95%TPR | 2018 |
| MalNet | Malware (graph) | 1.2M graphs, 5 types | Accuracy, F1 | 2021 |
| BODMAS | Malware (static+dynamic) | 134K samples | AUC, TPR@FPR | 2021 |
| Enron Spam | Email/spam | 33K emails | Accuracy, F1 | 2006 |
| SpamAssassin | Email/spam | 6K emails | Precision, Recall | 2005 |
| PhishingCorpus | Phishing email | 10K emails | Accuracy, F1 | 2021 |
| LANL Auth | Auth logs | 55M events | AUC, F1 | 2017 |
| CERT Insider Threat | Insider threat | 4M log events | Precision@k | 2014 |
| SEC-EDGAR | Fraud detection | SEC filings | Recall, Precision | 2021 |

### 15.2 Top Model Performances (as of 2026)

**Network Intrusion Detection (CIC-IDS2017):**

| Model | Accuracy | F1 Score | FPR | TPR@0.1%FPR |
|-------|----------|----------|-----|--------------|
| TabNet | 0.98 | 0.97 | 0.8% | 0.93 |
| XGBoost (+Optuna) | 0.98 | 0.97 | 0.5% | 0.92 |
| 1D-CNN | 0.96 | 0.95 | 1.5% | 0.88 |
| LSTM + Attention | 0.97 | 0.96 | 0.9% | 0.91 |
| Transformer | 0.98 | 0.97 | 0.6% | 0.94 |
| TabTransformer | **0.99** | **0.98** | **0.3%** | **0.96** |

**Malware Detection (EMBER):**

| Model | AUC | TPR@0.1%FPR | FPR@95%TPR |
|-------|-----|-------------|------------|
| LightGBM (1500+ features) | **0.995** | **0.97** | **0.08%** |
| MalConv (raw bytes) | 0.93 | 0.68 | 1.2% |
| MalConv2 (improved) | 0.96 | 0.78 | 0.9% |
| AVScan (hybrid ensemble) | 0.99 | 0.94 | **0.08%** |
| GNN (CFG-based) | 0.97 | 0.82 | 0.6% |
| Transformer (byte-level) | 0.98 | 0.91 | 0.3% |

**Phishing Detection:**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| RoBERTa-phish | 0.98 | 0.98 | 0.97 | 0.97 |
| DistilBERT + URL features | 0.99 | 0.98 | 0.98 | **0.98** |
| GPT-4o (few-shot) | 0.97 | 0.96 | 0.95 | 0.95 |
| XGBoost (feature-based) | 0.95 | 0.94 | 0.92 | 0.93 |
| Hybrid (BERT + XGBoost) | **0.99** | **0.99** | **0.98** | **0.98** |

### 15.3 Real-World Performance Benchmarks

| Vendor/Product | Detection Rate | False Positive Rate | Independent Test |
|---------------|---------------|-------------------|------------------|
| CrowdStrike Falcon | 99.9% | 0.02% | MITRE ATT&CK 2024 |
| SentinelOne Singularity | 99.8% | 0.03% | SE Labs 2025 |
| Microsoft Defender | 99.3% | 0.5% | AV-TEST 2025 |
| Sophos Intercept X | 99.1% | 0.4% | MRG Effitas 2025 |
| Palo Alto Cortex XDR | 98.7% | 0.3% | MITRE ATT&CK 2024 |
| Elastic Security | 97.5% | 1.2% | SE Labs 2025 |
| Darktrace | 95.2% | 15-20% | N/A (proprietary) |

*Note: Detection rates from independent tests (MITRE ATT&CK Evaluations, SE Labs, AV-TEST). Darktrace evaluated differently due to unsupervised approach.*

### 15.4 Evaluation Metrics for Security ML

| Metric | Formula | When It Matters |
|--------|---------|-----------------|
| **True Positive Rate (Recall)** | TP / (TP + FN) | Catching malware |
| **False Positive Rate** | FP / (FP + TN) | Analyst trust, alert fatigue |
| **Precision** | TP / (TP + FP) | Investigation efficiency |
| **F1 Score** | 2 × P × R / (P + R) | Overall detection quality |
| **AUC** | Area under ROC curve | Model discrimination ability |
| **TPR@LowFPR** | TPR when FPR ≤ 0.1% | Production deployment |
| **Mean Time to Detect** | Average time from infection to alert | SOC effectiveness |
| **Mean Time to Respond** | Average time from alert to containment | Incident response speed |
| **Alert Volume Reduction** | (baseline_alerts - ai_alerts) / baseline | ROI of AI investment |

---

## 16. Code Examples & Hands-On

### 16.1 Complete Security Pipeline

```python
"""
End-to-end AI security pipeline integrating multiple detection methods.

This example shows how NIDS + Malware Detection + UEBA + Phishing
Detection can be combined into a unified security platform.
"""

import json
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SecurityEvent:
    """Unified security event format."""
    event_id: str
    timestamp: float
    source: str  # network, endpoint, email, auth
    raw_data: dict
    feature_vector: Optional[list] = None
    score: float = 0.0
    classification: str = "unknown"
    alert: bool = False

class UnifiedSecurityAI:
    """Unified AI security detection platform."""
    
    def __init__(self):
        self.models = {
            'nids': None,      # ML-NIDS model
            'malware': None,   # Malware classifier
            'ueba': None,      # UEBA engine
            'phishing': None,  # Phishing detector
            'prioritizer': None,  # Vulnerability prioritizer
        }
        self.ensemble_weights = {
            'nids': 0.3,
            'malware': 0.3,
            'ueba': 0.25,
            'phishing': 0.15,
        }
        self.alert_threshold = 0.75
        
    def load_models(self):
        """Load all security AI models."""
        # In production, load from model registry (MLflow/S3)
        print("Loading security AI models...")
        self.models['nids'] = self._load_nids_model()
        self.models['malware'] = self._load_malware_model()
        self.models['ueba'] = self._load_ueba_model()
        self.models['phishing'] = self._load_phishing_model()
        self.models['prioritizer'] = VulnerabilityPrioritizer()
        print("All models loaded successfully.")
    
    def _load_nids_model(self):
        """Load pre-trained NIDS model."""
        # Mock: return a trained XGBoost model
        return {"type": "xgboost", "version": "2.0", "features": 28}
    
    def _load_malware_model(self):
        """Load pre-trained malware detection model."""
        return {"type": "malconv", "version": "2.1"}
    
    def _load_ueba_model(self):
        """Load pre-trained UEBA autoencoder."""
        return {"type": "autoencoder", "latent_dim": 8}
    
    def _load_phishing_model(self):
        """Load pre-trained phishing BERT model."""
        return {"type": "distilbert", "version": "phish-v3"}
    
    def process_network_event(self, event: SecurityEvent) -> SecurityEvent:
        """Process network traffic event through NIDS."""
        # Feature extraction (simplified)
        features = [
            event.raw_data.get('duration', 0),
            event.raw_data.get('bytes_sent', 0),
            event.raw_data.get('bytes_recv', 0),
            event.raw_data.get('packets', 0),
            event.raw_data.get('protocol', 6),
        ]
        event.feature_vector = features
        
        # Inference (simulated)
        malicious_score = min(
            sum(features[2]) / 100000 if features[2] > 10000 else 0,
            0.95
        )
        event.score = malicious_score
        event.classification = "malicious" if malicious_score > 0.8 else "benign"
        event.alert = malicious_score > self.alert_threshold
        return event
    
    def process_endpoint_event(self, event: SecurityEvent) -> SecurityEvent:
        """Process endpoint event through malware detection."""
        file_entropy = event.raw_data.get('entropy', 0)
        suspicious_apis = event.raw_data.get('suspicious_apis', [])
        
        # Simplified malware scoring
        score = min(
            (file_entropy / 8.0) * 0.5 + 
            (len(suspicious_apis) / 10.0) * 0.5,
            0.99
        )
        event.score = score
        event.classification = "malware" if score > 0.7 else "benign"
        event.alert = score > self.alert_threshold
        return event
    
    def ensemble_decision(self, events: List[SecurityEvent]) -> Dict:
        """Ensemble decision from multiple detectors."""
        
        weighted_score = 0.0
        detections = {}
        
        for event in events:
            source = event.source
            weight = self.ensemble_weights.get(source, 0.2)
            weighted_score += event.score * weight
            detections[source] = {
                'score': event.score,
                'classification': event.classification,
                'alert': event.alert
            }
        
        final_alert = weighted_score > self.alert_threshold
        severity = 'critical' if weighted_score > 0.95 \
                  else 'high' if weighted_score > 0.85 \
                  else 'medium' if weighted_score > 0.75 \
                  else 'low'
        
        return {
            'ensemble_score': round(weighted_score, 3),
            'final_alert': final_alert,
            'severity': severity,
            'detections': detections,
            'timestamp': datetime.now().isoformat(),
            'recommended_action': self._get_recommended_action(
                weighted_score, detections
            )
        }
    
    def _get_recommended_action(self, score: float, 
                                 detections: dict) -> str:
        """Determine recommended response based on detection ensemble."""
        if score > 0.95:
            return "IMMEDIATE: Block all affected entities, notify SOC lead, trigger full incident response"
        elif score > 0.85:
            return "URGENT: Quarantine affected endpoints, block indicators at firewall, investigate"
        elif score > 0.75:
            return "ALERT: Create security ticket, prioritize for analyst review within 1 hour"
        else:
            return "MONITOR: Log for pattern analysis, no immediate action required"

"""
Example: Unified pipeline execution
engine = UnifiedSecurityAI()
engine.load_models()

# Process events from different sources
network_event = SecurityEvent(
    event_id="evt-001", timestamp=time.time(),
    source="network",
    raw_data={"duration": 120, "bytes_sent": 50000, 
              "bytes_recv": 200000, "packets": 1500, "protocol": 6}
)
endpoint_event = SecurityEvent(
    event_id="evt-002", timestamp=time.time(),
    source="endpoint",
    raw_data={"entropy": 7.2, "suspicious_apis": 
              ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread"]}
)

network_event = engine.process_network_event(network_event)
endpoint_event = engine.process_endpoint_event(endpoint_event)

decision = engine.ensemble_decision([network_event, endpoint_event])
print(json.dumps(decision, indent=2))
```
"""

### 16.2 Threat Hunting with ML

```python
"""
AI-assisted threat hunting — proactive detection beyond alerts.
"""

class AIThreatHunter:
    """AI-assisted threat hunting — proactive detection."""
    
    def __init__(self, siem_client, llm_client):
        self.siem = siem_client
        self.llm = llm_client
        
    def generate_hunting_hypotheses(self, recent_intrusions: list) -> list:
        """Generate threat hunting hypotheses based on recent intel."""
        prompt = f"""Based on recent security incidents and threat intelligence:
        {json.dumps(recent_intrusions, indent=2)}
        
        Generate 5 specific threat hunting hypotheses to investigate.
        For each hypothesis, provide:
        1. What to look for
        2. Which data sources to query
        3. Expected attacker behavior
        4. A SIEM query to run
        
        Format as a JSON list."""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def run_behavioral_search(self, technique: str) -> list:
        """Search for specific attacker behavior across data sources."""
        queries = {
            "pass_the_hash": "index=auth* EventID=4624 LogonType=3 "
                             "AccountName!=$ AccountDomain=* "
                             "| stats count by AccountName, WorkstationName",
            "powerShell_download": "index=endpoint* powershell* "
                                   "DownloadString OR WebClient",
            "lsass_dump": "index=endpoint* lsass OR procdump OR "
                          "comsvcs.dll minidump",
        }
        return self.siem.query(queries.get(technique, ""), time_range="last_7d")

"""
Threat Hunting ROI:
├── Hypotheses generated: 5/hour (vs 2/day manually)
├── Hunt success rate: 34% (vs 18% manually)
├── Mean time to uncover: 2.3 hours (vs 8 hours)
└── Coverage improvement: 4x more techniques investigated
"""
```

---

## 17. Cross-References

- **06-Advanced/08-Adversarial-ML.md** — Comprehensive guide to adversarial attacks (evasion, poisoning, extraction) and defenses including adversarial training, certified robustness, and red teaming workflows
- **06-Advanced/05-Interpretability.md** — Model interpretability techniques (SHAP, LIME, integrated gradients) for explaining security ML decisions
- **11-AI-Applications/03-Finance-AI.md** — Fraud detection ML techniques shared between finance and cybersecurity domains
- **11-AI-Applications/02-Healthcare-AI.md** — Healthcare security applications and HIPAA-compliant ML deployment
- **06-Advanced/03-Evaluation-Benchmarks.md** — Evaluation methodology and benchmark datasets used across ML domains
- **17-Research-Frontiers-2026/05-Safety-Alignment-Research.md** — LLM safety research relevant to security copilot deployment
- **06-Advanced/07-Time-Series-Forecasting.md** — Time-series methods used in SIEM anomaly detection and network traffic prediction
- **17-Research-Frontiers-2026/08-AI-for-Science.md** — AI for malware reverse engineering and vulnerability discovery

---

> **How to cite this document:** "AI for Cybersecurity: Threat Detection, Response, and Operations" (June 2026), *AiBaseKnowledge/11-AI-Applications/12-AI-Cybersecurity.md*.
