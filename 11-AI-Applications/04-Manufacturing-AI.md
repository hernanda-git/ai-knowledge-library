# AI in Manufacturing & Industry 4.0

## Table of Contents
1. [Introduction](#introduction)
2. [Predictive Maintenance](#predictive-maintenance)
   - [Autoencoders for Anomaly Detection](#autoencoders-for-anomaly-detection)
   - [Vibration Analysis with CNNs](#vibration-analysis-with-cnns)
   - [Remaining Useful Life Prediction](#remaining-useful-life-prediction)
3. [Quality Inspection & Computer Vision](#quality-inspection--computer-vision)
   - [YOLO for Defect Detection](#yolo-for-defect-detection)
   - [Semantic Segmentation for Surface Inspection](#semantic-segmentation-for-surface-inspection)
   - [Hyperspectral Imaging Analysis](#hyperspectral-imaging-analysis)
4. [Digital Twins](#digital-twins)
   - [Physics-Informed Neural Networks](#physics-informed-neural-networks)
   - [Real-Time Simulation and Optimization](#real-time-simulation-and-optimization)
   - [Digital Twin Architectures](#digital-twin-architectures)
5. [Supply Chain Optimization](#supply-chain-optimization)
   - [Demand Forecasting with Transformers](#demand-forecasting-with-transformers)
   - [Inventory Optimization with RL](#inventory-optimization-with-rl)
   - [Supplier Risk Modeling](#supplier-risk-modeling)
6. [Robotics & Cobots](#robotics--cobots)
   - [Robot Path Planning](#robot-path-planning)
   - [Human-Robot Collaboration](#human-robot-collaboration)
   - [Grasping and Manipulation](#grasping-and-manipulation)
7. [Generative Design](#generative-design)
   - [Topology Optimization with GANs](#topology-optimization-with-gans)
   - [Multi-Physics Design Exploration](#multi-physics-design-exploration)
8. [Case Studies](#case-studies)
9. [Cross-References](#cross-references)
10. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

Industry 4.0, the fourth industrial revolution, is characterized by the fusion of digital technologies with physical manufacturing processes. Artificial Intelligence serves as the cognitive layer that transforms connected factories from data-rich but insight-poor environments into adaptive, self-optimizing production systems.

Manufacturing AI presents distinct technical challenges: sensor data is often noisy and high-dimensional, production environments are safety-critical, real-time inference requirements can be as tight as milliseconds, and models must generalize across different product variants and production conditions. The global smart manufacturing market is projected to reach $639 billion by 2028, with AI as a core enabling technology.

## Predictive Maintenance

Predictive maintenance (PdM) uses sensor data to predict equipment failures before they occur, reducing unplanned downtime by 30-50% and maintenance costs by 10-30%.

### Autoencoders for Anomaly Detection

Autoencoders learn a compressed representation of normal operating conditions; deviations from normal operation produce high reconstruction error, signaling an anomaly.

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class IndustrialAutoencoder(nn.Module):
    """Deep convolutional autoencoder for multivariate sensor anomaly detection"""
    def __init__(self, sensor_dim, sequence_length=100):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.sequence_length = sequence_length
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(sensor_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Bottleneck size after convolutions
        self.encoder_out_size = 16 * (sequence_length // 8)
        self.bottleneck = nn.Linear(self.encoder_out_size, 32)
        
        # Decoder
        self.decoder_proj = nn.Linear(32, self.encoder_out_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, sensor_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def forward(self, x):
        # x shape: (batch, sensor_dim, seq_len)
        encoded = self.encoder(x)
        b, c, l = encoded.shape
        encoded_flat = encoded.view(b, -1)
        bottleneck = self.bottleneck(encoded_flat)
        
        decoded = self.decoder_proj(bottleneck).view(b, c, l)
        reconstructed = self.decoder(decoded)
        # Trim to original sequence length
        reconstructed = reconstructed[:, :, :self.sequence_length]
        return reconstructed, bottleneck

class AnomalyDetector:
    """Production anomaly detector using autoencoder"""
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.scaler = StandardScaler()
    
    def train(self, normal_sensor_data):
        """Train autoencoder on normal operation data only"""
        # Normalize
        self.scaler.fit(normal_sensor_data)
        X = self._preprocess(normal_sensor_data)
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(200):
            total_loss = 0
            for batch in self._batch_generator(X, 64):
                recon, _ = self.model(batch)
                loss = nn.MSELoss()(recon, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
        
        # Set threshold based on reconstruction error on validation set
        self._calibrate_threshold(X)
    
    def _calibrate_threshold(self, X):
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()
        self.threshold = np.percentile(errors, self.threshold_percentile)
    
    def predict(self, sensor_data):
        """Return anomaly score and classification"""
        X = self._preprocess(sensor_data)
        self.model.eval()
        with torch.no_grad():
            recon, bottleneck = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()
        
        predictions = []
        for error in errors:
            predictions.append({
                'anomaly_score': float(error),
                'is_anomaly': error > self.threshold,
                'severity': 'CRITICAL' if error > 3 * self.threshold else \
                           'WARNING' if error > 2 * self.threshold else \
                           'ATTENTION' if error > self.threshold else 'NORMAL'
            })
        
        return predictions
    
    def _preprocess(self, data):
        scaled = self.scaler.transform(data)
        # Reshape to (batch, sensor_dim, seq_len)
        return torch.FloatTensor(scaled).permute(0, 2, 1)
    
    def _batch_generator(self, X, batch_size):
        indices = torch.randperm(X.shape[0])
        for i in range(0, len(indices), batch_size):
            yield X[indices[i:i+batch_size]]
```

**Deployment architecture for real-time PdM:**

```yaml
predictive_maintenance_pipeline:
  edge_devices:
    - type: Industrial PC (Siemens IPC427E)
      sensors: [vibration, temperature, current, pressure]
      sampling_rate: 1-10 kHz
      inference: ONNX Runtime with TensorRT
    - type: PLC-integrated (Siemens S7-1500 + AI module)
      latency: < 5ms per inference
    
  cloud_infrastructure:
    - Model training on historical data (weekly)
    - Fleet-wide model aggregation (federated learning)
    - Dashboard and alerting (Grafana + Prometheus)
    
  model_lifecycle:
    - Initial training: 6 months of normal operation data
    - Retraining trigger: concept drift detection (PSI > 0.2)
    - Versioning: MLflow registry
    - A/B testing: 10% of assets on new model version
```

### Vibration Analysis with CNNs

Vibration signals are a rich source of information about rotating machinery health:

```python
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as scipy_signal

class VibrationCNN(nn.Module):
    """
    1D-CNN for bearing fault classification from vibration signals.
    Learns features directly from raw time-domain signals.
    """
    def __init__(self, num_classes=4, input_length=4096):
        super().__init__()
        self.input_length = input_length
        
        # First block: wide kernels to capture low-frequency patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Second block: medium kernels
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Third block: fine-grained features
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Compute size after convolutions
        self._compute_fc_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _compute_fc_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_length)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.fc_size = x.numel()
    
    def forward(self, x):
        # Input: (batch, 1, time_samples)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class VibrationFeatureExtractor:
    """Traditional vibration analysis features + CNN features"""
    
    @staticmethod
    def extract_features(vibration_signal, sampling_rate=25600):
        features = {}
        
        # Time-domain features
        features['rms'] = np.sqrt(np.mean(vibration_signal ** 2))
        features['peak'] = np.max(np.abs(vibration_signal))
        features['crest_factor'] = features['peak'] / features['rms']
        features['skewness'] = scipy.stats.skew(vibration_signal)
        features['kurtosis'] = scipy.stats.kurtosis(vibration_signal)
        features['shape_factor'] = features['rms'] / np.mean(np.abs(vibration_signal))
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(vibration_signal))
        
        # Frequency-domain features
        freq, psd = scipy_signal.welch(vibration_signal, sampling_rate, nperseg=1024)
        features['peak_frequency'] = freq[np.argmax(psd)]
        features['power_band_low'] = np.trapz(psd[(freq >= 0) & (freq < 1000)])
        features['power_band_mid'] = np.trapz(psd[(freq >= 1000) & (freq < 5000)])
        features['power_band_high'] = np.trapz(psd[(freq >= 5000) & (freq < 12800)])
        
        # Bearing fault frequencies (assuming specific bearing type)
        shaft_freq = self._estimate_shaft_frequency(vibration_signal, sampling_rate)
        features['shaft_freq_magnitude'] = self._extract_magnitude_at_frequency(psd, freq, shaft_freq)
        
        # Ball pass frequency outer (BPFO)
        bpfo = shaft_freq * 3.572  # Example for 6205 bearing
        features['bpfo_magnitude'] = self._extract_magnitude_at_frequency(psd, freq, bpfo)
        
        return features
```

**Bearing fault classification accuracy on CWRU dataset:**
- Vibration CNN (raw signal): 99.8% (4-class: normal, inner race, outer race, ball fault)
- Vibration CNN (spectrogram input): 99.5%
- Traditional ML (hand-crafted features): 94-97%

### Remaining Useful Life (RUL) Prediction

RUL prediction estimates the time until equipment failure, enabling just-in-time maintenance scheduling:

```python
class RULPredictor(nn.Module):
    """
    Hybrid CNN-LSTM for Remaining Useful Life prediction.
    
    Architecture from: "A Hybrid Approach for RUL Prediction
    with Deep Learning" (Zhao et al., 2019)
    """
    def __init__(self, n_sensors=14, seq_length=30):
        super().__init__()
        self.n_sensors = n_sensors
        self.seq_length = seq_length
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_sensors, 32, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=100,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(100, num_heads=4, batch_first=True)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # RUL cannot be negative
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_sensors)
        x = x.permute(0, 2, 1)  # -> (batch, n_sensors, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # -> (batch, features, time)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        return self.regressor(pred).squeeze(-1)

# RUL metrics
def compute_ruL_metrics(y_true, y_pred):
    """Compute RUL prediction metrics per PHM08 challenge"""
    errors = y_true - y_pred
    
    # Score function (asymmetric — penalties late predictions more)
    score = np.sum([
        np.exp(-error/13) - 1 if error < 0 else np.exp(error/10) - 1
        for error in errors
    ])
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    return {
        'score': score,
        'mae': mae,
        'rmse': rmse,
        'within_20pct': np.mean(np.abs(errors) < y_true * 0.2)
    }
```

**RUL model performance on NASA C-MAPSS dataset:**

| Subset | RMSE | Score | Description |
|--------|------|-------|-------------|
| FD001 | 12.56 | 263 | 1 operating condition, 1 fault mode |
| FD002 | 22.72 | 1,542 | 6 operating conditions, 1 fault mode |
| FD003 | 12.97 | 317 | 1 operating condition, 2 fault modes |
| FD004 | 23.77 | 1,782 | 6 operating conditions, 2 fault modes |

## Quality Inspection & Computer Vision

Automated visual inspection systems use computer vision to detect defects, measure dimensions, and verify assembly correctness at production-line speeds.

### YOLO for Defect Detection

You Only Look Once (YOLO) provides real-time object detection ideal for production line inspection:

```yaml
yolo_defect_detection:
  model: YOLOv8x
  input_size: 640x640
  fps: 120 (on NVIDIA A100)
  fps: 45 (on Jetson Orin NX at edge)
  
  training:
    dataset: 50,000 labeled defect images
    classes: [
      'scratch', 'dent', 'crack', 'porosity',
      'misalignment', 'color_variation', 'contamination',
      'foreign_material', 'incomplete', 'flash'
    ]
    augmentation:
      - mosaic (scale 0.5-1.5)
      - mixup (alpha=0.5)
      - random_perspective (degrees=10, translate=0.1, scale=0.5)
      - hsv_augmentation (h=0.015, s=0.7, v=0.4)
    
    hyperparameters:
      lr0: 0.01
      lrf: 0.01
      momentum: 0.937
      weight_decay: 0.0005
      warmup_epochs: 3
      epochs: 300
  
  performance:
    mAP@0.5: 0.965
    mAP@0.5:0.95: 0.712
    precision: 0.94
    recall: 0.93
```

Using the YOLO API for custom defect detection:

```python
from ultralytics import YOLO

class DefectInspector:
    def __init__(self, model_path='yolov8x_defect.pt'):
        self.model = YOLO(model_path)
        
    def inspect(self, image, confidence_threshold=0.25):
        results = self.model(image, conf=confidence_threshold)[0]
        
        defects = []
        for box in results.boxes:
            defect = {
                'type': results.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'area': self._bbox_area(box.xyxy[0].tolist()),
                'location': self._location_classification(box.xyxy[0].tolist())
            }
            defects.append(defect)
        
        # Pass/fail decision
        critical_defects = [d for d in defects 
                           if d['type'] in ['crack', 'dent', 'contamination']
                           and d['confidence'] > 0.7]
        
        return {
            'defects': defects,
            'defect_count': len(defects),
            'pass_inspection': len(critical_defects) == 0,
            'image_quality_score': self._compute_image_quality(results)
        }
    
    def _location_classification(self, bbox):
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        if y_center < 0.33: zone = "top"
        elif y_center < 0.67: zone = "middle"
        else: zone = "bottom"
        
        if x_center < 0.33: side = "left"
        elif x_center < 0.67: side = "center"
        else: side = "right"
        
        return f"{zone}_{side}"
```

### Semantic Segmentation for Surface Inspection

Pixel-level defect segmentation enables precise defect geometry measurement:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceSegmentationNet(nn.Module):
    """
    Semantic segmentation for surface defect detection.
    Uses DeepLabV3+ architecture with MobileNetV2 backbone
    for edge deployment.
    """
    def __init__(self, num_classes=5, backbone='mobilenetv2'):
        super().__init__()
        
        # Backbone
        if backbone == 'mobilenetv2':
            from torchvision.models import mobilenet_v2
            base = mobilenet_v2(pretrained=True)
            self.backbone = base.features
            self.backbone_channels = 1280
        else:
            from torchvision.models import resnet50
            base = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.backbone_channels = 2048
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(self.backbone_channels, [6, 12, 18])
        
        # Decoder
        self.decoder = Decoder(256, num_classes)
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Encoder
        features = self.backbone(x)
        
        # ASPP
        aspp_out = self.aspp(features)
        
        # Decoder
        output = self.decoder(aspp_out)
        
        # Upsample to original size
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=True)
        
        return output

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ))
        
        # Image-level features
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(256 * (len(atrous_rates + 2)), 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        # Upsample image-level features
        res[-1] = F.interpolate(res[-1], size=res[0].shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat(res, dim=1)
        return self.project(x)
```

## Digital Twins

A digital twin is a virtual replica of a physical system that mirrors its real-time state and enables simulation, prediction, and optimization.

### Physics-Informed Neural Networks (PINNs)

PINNs incorporate physical laws (governed by PDEs) into the neural network training process, enabling accurate simulation with sparse data:

```python
import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    """
    PINN for thermal dynamics in a manufacturing process.
    
    Governing PDE: Heat equation
    ∂u/∂t = α ∇²u + Q(x,t)
    
    where u = temperature, α = thermal diffusivity, Q = heat source
    """
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super().__init__()
        self.net = self._build_network(layers)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learned diffusivity
    
    def _build_network(self, layers):
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*modules)
    
    def forward(self, x, t):
        # x: spatial coordinate, t: time
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    
    def compute_pde_residual(self, x, t):
        """Compute the PDE residual for physics-based loss"""
        u = self.forward(x, t)
        
        # Compute gradients using autograd
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        # PDE residual: u_t - α * u_xx = 0
        residual = u_t - self.alpha * u_xx
        return residual
    
    def pinn_loss(self, x_data, t_data, u_data, x_pde, t_pde):
        # Data loss (supervised)
        u_pred = self.forward(x_data, t_data)
        data_loss = nn.MSELoss()(u_pred, u_data)
        
        # Physics loss (unsupervised PDE residual)
        pde_res = self.compute_pde_residual(x_pde, t_pde)
        physics_loss = torch.mean(pde_res ** 2)
        
        return data_loss + 0.1 * physics_loss  # Weighted combination
```

### Digital Twin Architecture

```yaml
digital_twin_system:
  layers:
    physical_twin:
      sensors: [temperature, pressure, vibration, current, position]
      actuators: [motor drives, valves, heaters, robots]
      controller: Siemens S7-1500 PLC
      communication: OPC UA (IEC 62541)
    
    virtual_twin:
      simulation_engine: Simulink Real-Time / Ansys Twin Builder
      physics_model: FEA (Finite Element) + CFD (Computational Fluid Dynamics)
      ML_enhancements:
        - Reduced Order Models (ROM) for real-time simulation
        - Hybrid physics-ML models (PINNs)
      
    digital_shadow:
      synchronization: real-time (10ms cycle time)
      state_estimation: Extended Kalman Filter + ML correction
      data_storage: Time-series DB (InfluxDB) + Event store (Kafka)
    
    digital_thread:
      traceability: Blockchain-based part genealogy
      material_flow: RFID + Computer vision tracking
      quality_history: ML-annotated inspection results
    
  use_cases:
    predictive_maintenance:
      - Model: Gradient boosting + LSTM ensemble
      - Horizon: 7-day future predictions
      - Action: Maintenance scheduling optimization
    
    production_optimization:
      - Model: Bayesian optimization
      - Objective: Minimize cycle time + energy + waste
      - Update: Every hour
    
    what_if_simulation:
      - Model: Surrogate neural network (trained on simulation data)
      - Parameters: [material batch, feed rate, temperature, pressure]
      - Response time: < 1 second (vs. hours for full FEM)
```

## Supply Chain Optimization

### Demand Forecasting with Transformers

```python
class SupplyChainTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon demand forecasting.
    
    Processes static features (product category, supplier),
    known future inputs (promotions, holidays), and
    observed past inputs (sales history, inventory levels).
    """
    def __init__(self, 
                 num_static=10,
                 num_known_future=5,
                 num_observed=8,
                 hidden_size=128,
                 num_heads=4,
                 num_layers=3,
                 forecast_horizon=28):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
        # Feature encoders
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.observed_encoder = nn.Linear(num_observed, hidden_size)
        self.known_future_encoder = nn.Linear(num_known_future, hidden_size)
        
        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            hidden_size * 2,  # observed + static
            hidden_size,
            num_layers,
            dropout=0.1,
            batch_first=True
        )
        
        # Multi-head attention decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Position encoding for decoder
        self.pos_encoding = nn.Embedding(forecast_horizon, hidden_size)
        
        # Output heads
        self.output_layer = nn.Linear(hidden_size, 1)
        self.quantile_outputs = nn.ModuleDict({
            'p10': nn.Linear(hidden_size, 1),
            'p50': nn.Linear(hidden_size, 1),
            'p90': nn.Linear(hidden_size, 1)
        })
    
    def forward(self, static_features, observed_inputs, known_future_inputs):
        batch_size = observed_inputs.size(0)
        
        # Encode inputs
        static_enc = self.static_encoder(static_features).unsqueeze(1)
        
        # Combine observed with static (tile over time)
        seq_len = observed_inputs.size(1)
        static_tiled = static_enc.expand(-1, seq_len, -1)
        observed_enc = self.observed_encoder(observed_inputs)
        encoder_input = torch.cat([observed_enc, static_tiled], dim=-1)
        
        # LSTM encoding
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_input)
        
        # Decoder with known future inputs
        future_enc = self.known_future_encoder(known_future_inputs)
        positions = torch.arange(self.forecast_horizon, device=observed_inputs.device)
        pos_enc = self.pos_encoding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = future_enc + pos_enc
        
        # Self-attention and cross-attention
        decoder_output = self.decoder(
            decoder_input,
            encoder_output
        )
        
        # Point forecast
        point_forecast = self.output_layer(decoder_output).squeeze(-1)
        
        # Quantile forecasts for uncertainty estimation
        quantiles = {}
        for name, layer in self.quantile_outputs.items():
            quantiles[name] = layer(decoder_output).squeeze(-1)
        
        return point_forecast, quantiles

# Training with quantile loss
def quantile_loss(y_true, y_pred, tau):
    """Pinball loss for quantile regression"""
    error = y_true - y_pred
    return torch.mean(torch.max(tau * error, (tau - 1) * error))
```

## Robotics & Cobots

### Robot Path Planning with RL

```python
import numpy as np
from collections import deque

class RLPathPlanner:
    """
    Deep RL for robot arm path planning in cluttered environments.
    Uses PPO with motion constraints.
    """
    def __init__(self, n_joints=6):
        self.n_joints = n_joints
        self.state_dim = n_joints * 2 + 3  # joint angles + velocities + target position
        self.action_dim = n_joints  # joint velocity commands
        
        self.policy = self._build_policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
    
    def _build_policy(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
    
    def train_environment(self, env, n_episodes=10000):
        """Train using PPO in simulation environment"""
        memory = deque(maxlen=100000)
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(200):  # max steps per episode
                action, log_prob, value = self._select_action(state)
                next_state, reward, done = env.step(action)
                
                memory.append((state, action, reward, next_state, log_prob, value, done))
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # PPO update every 2000 steps
            if len(memory) >= 2000:
                self._ppo_update(memory)
        
        return episode_rewards
    
    def _select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            features = self.policy(state_tensor)
            
            # Action head
            mean = nn.Linear(256, self.action_dim)(features)
            std = nn.Parameter(torch.ones(self.action_dim) * 0.2)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # Value head
            value = nn.Linear(256, 1)(features)
            
        return action.squeeze().numpy(), log_prob.item(), value.item()
```

### Human-Robot Collaboration

```yaml
collaborative_robot_system:
  safety:
    standard: ISO 10218 + ISO/TS 15066
    speed_monitoring: < 250mm/s in collaborative mode
    force_limiting: < 150N static, < 50N quasi-static
    separation_distance: Dynamic (ML-predicted human trajectory)
    
  perception:
    human_detection:
      - 3D LiDAR (Velodyne VLP-16) for human tracking
      - RGB-D camera (Intel RealSense D455) for gesture recognition
      - Model: MediaPipe + custom RNN for motion prediction
    
    workspace_monitoring:
      - Model: Mask R-CNN for object segmentation
      - Safety zones: [restricted, warning, safe] with dynamic boundaries
      - Update rate: 30Hz
    
  task_allocation:
    model: Mixed Integer Linear Programming + RL refinement
    optimization:
      - Minimize: cycle time + idle time + energy
      - Constraints: human fatigue, robot reachability, safety zones
    
    dynamic_reassignment:
      triggers:
        - Human fatigue detected (posture analysis)
        - Skill mismatch observed (error rate increase)
        - Safety zone violations
```

## Generative Design

Generative design uses AI to explore vast design spaces, producing optimized geometries that would be impossible to conceive manually.

### Topology Optimization with GANs

```python
class TopologyOptimizationGAN:
    """
    Generative adversarial network for structural topology optimization.
    
    Generator produces optimized material distributions given
    boundary conditions and performance targets.
    """
    def __init__(self, resolution=64):
        self.resolution = resolution
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
    
    def _build_generator(self):
        """U-Net style generator for image-to-image translation"""
        # Input: Load conditions (pressure maps, fixed constraints)
        # Output: Material distribution (density field)
        
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._block(4, 64, 4, 2)  # Input: load conditions (4 channels)
                self.enc2 = self._block(64, 128, 4, 2)
                self.enc3 = self._block(128, 256, 4, 2)
                self.enc4 = self._block(256, 512, 4, 2)
                self.enc5 = self._block(512, 512, 4, 2)
                self.enc6 = self._block(512, 512, 4, 2)
                self.enc7 = self._block(512, 512, 4, 2)
                
                # Bottleneck
                self.bottleneck = self._block(512, 512, 4, 2)
                
                # Decoder
                self.dec7 = self._up_block(512, 512, 4, 2, dropout=True)
                self.dec6 = self._up_block(1024, 512, 4, 2, dropout=True)
                self.dec5 = self._up_block(1024, 512, 4, 2, dropout=True)
                self.dec4 = self._up_block(1024, 256, 4, 2)
                self.dec3 = self._up_block(512, 128, 4, 2)
                self.dec2 = self._up_block(256, 64, 4, 2)
                self.dec1 = nn.Sequential(
                    nn.ConvTranspose2d(128, 1, 4, 2, 1),
                    nn.Tanh()  # Output normalized density [-1, 1]
                )
            
            def _block(self, in_c, out_c, kernel, stride):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel, stride, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(0.2)
                )
            
            def _up_block(self, in_c, out_c, kernel, stride, dropout=False):
                layers = [
                    nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU()
                ]
                if dropout:
                    layers.append(nn.Dropout(0.5))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                e3 = self.enc3(e2)
                e4 = self.enc4(e3)
                e5 = self.enc5(e4)
                e6 = self.enc6(e5)
                e7 = self.enc7(e6)
                
                b = self.bottleneck(e7)
                
                d7 = self.dec7(b)
                d7 = torch.cat([d7, e7], dim=1)
                d6 = self.dec6(d7)
                d6 = torch.cat([d6, e6], dim=1)
                d5 = self.dec5(d6)
                d5 = torch.cat([d5, e5], dim=1)
                d4 = self.dec4(d5)
                d4 = torch.cat([d4, e4], dim=1)
                d3 = self.dec3(d4)
                d3 = torch.cat([d3, e3], dim=1)
                d2 = self.dec2(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d1 = self.dec1(d2)
                return d1
        
        return Generator()
```

## Case Studies

### Case Study 1: Siemens' Neural Network for Gas Turbines

**Background**: Siemens deployed AI for predictive maintenance and optimization of their SGT-800 gas turbine fleet.

**Technical details:**
- Sensors: 200+ (temperature, pressure, vibration, emissions, fuel flow)
- Data: 1TB per turbine per year
- Models:
  1. Combustion dynamics anomaly detection (autoencoder)
  2. NOx emissions prediction (XGBoost, R²=0.94)
  3. Remaining useful life for hot gas path components (LSTM ensemble)
  4. Compressor fouling detection (unsupervised clustering)

**Results:**
- 25% reduction in unplanned downtime
- 10% reduction in NOx emissions through optimized combustion tuning
- 2x extension of maintenance intervals for certain components
- Deployed across 300+ turbines globally

### Case Study 2: BMW's Computer Vision for Paint Inspection

**Background**: BMW implemented deep learning for automated paint defect detection in their Dingolfing plant.

**Architecture:**
```yaml
bmw_paint_inspection:
  image_acquisition:
    lighting: LED dome (12 distinct illumination patterns)
    cameras: 16x Basler ace (5MP, 50fps)
    resolution: 10μm/pixel
  
  model:
    architecture: Custom CNN (ResNet-50 backbone + FPN)
    classes: [dust, crater, sagging, orange_peel, solvent_pop, scratch]
    training_data: 5M labeled patches (from 250K car bodies)
    
  performance:
    detection_rate: 98.7%
    false_positive_rate: 0.3 per car body
    inspection_time: 45 seconds (vs. 5 minutes manual)
    
  savings:
    - 30% reduction in rework costs
    - 100% inspection coverage (vs. 20% sample-based manual)
    - 15% improvement in first-pass yield
```

## Cross-References

This document relates to other categories in the AI Knowledge Base:

- **[02-Healthcare-AI.md](02-Healthcare-AI.md)** — Medical imaging CNN architectures transfer to quality inspection; predictive maintenance of hospital equipment
- **[03-Finance-AI.md](03-Finance-AI.md)** — Supply chain models share transformer architectures with financial forecasting
- **[08-Agriculture-AI.md](08-Agriculture-AI.md)** — Computer vision for crop health parallels surface defect detection
- **[09-Transportation-AI.md](09-Transportation-AI.md)** — Predictive maintenance for vehicle fleets uses similar sensor analysis
- **[10-Energy-AI.md](10-Energy-AI.md)** — Turbine monitoring and smart grid optimization share industrial IoT patterns

## Summary & Conclusion

AI in manufacturing is transforming traditional factories into intelligent, connected production systems. The field draws on a wide range of AI techniques:

- **Anomaly Detection**: Autoencoders and isolation forests for identifying equipment degradation before failure
- **Computer Vision**: YOLO, semantic segmentation, and defect classification for automated quality inspection
- **Predictive Models**: LSTM, CNN, and hybrid architectures for remaining useful life estimation
- **Physics-Informed ML**: PINNs and hybrid models that combine data-driven learning with physical constraints
- **Digital Twins**: Comprehensive virtual replicas enabling simulation, prediction, and optimization across the product lifecycle
- **Generative Design**: GANs and optimization algorithms for exploring novel, high-performance geometries

The trajectory of manufacturing AI is toward increasingly autonomous operations — self-optimizing production lines, collaborative robots that adapt to human workers, and digital twins that span the entire supply chain. Success requires not just sophisticated models but robust integration with industrial control systems, real-time data pipelines, and a workforce trained to collaborate with intelligent machines.
