# AI in Agriculture & Food Tech

## Table of Contents
1. [Introduction](#introduction)
2. [Precision Agriculture](#precision-agriculture)
   - [Drone Imagery & Multispectral Analysis](#drone-imagery--multispectral-analysis)
   - [Satellite Remote Sensing](#satellite-remote-sensing)
   - [Variable Rate Technology](#variable-rate-technology)
3. [Crop Yield Prediction](#crop-yield-prediction)
   - [LSTM for Time-Series Forecasting](#lstm-for-time-series-forecasting)
   - [Transformer Architectures for Yield Modeling](#transformer-architectures-for-yield-modeling)
   - [Hybrid CNN-RNN Models](#hybrid-cnn-rnn-models)
4. [Plant Disease Detection](#plant-disease-detection)
   - [CNN-Based Classification (ResNet, EfficientNet)](#cnn-based-classification-resnet-efficientnet)
   - [Segmentation for Disease Severity Assessment](#segmentation-for-disease-severity-assessment)
   - [Mobile Deployment for Field Use](#mobile-deployment-for-field-use)
5. [Automated Irrigation Systems](#automated-irrigation-systems)
   - [Reinforcement Learning for Water Optimization](#reinforcement-learning-for-water-optimization)
   - [IoT Sensor Fusion](#iot-sensor-fusion)
   - [Soil Moisture Modeling](#soil-moisture-modeling)
6. [Livestock Monitoring](#livestock-monitoring)
   - [Computer Vision for Health Assessment](#computer-vision-for-health-assessment)
   - [RFID & Sensor-Based Tracking](#rfid--sensor-based-tracking)
   - [Behavioral Analysis with ML](#behavioral-analysis-with-ml)
7. [Soil Analysis & Nutrient Management](#soil-analysis--nutrient-management)
   - [Hyperspectral Soil Sensing](#hyperspectral-soil-sensing)
   - [Nutrient Recommendation Engines](#nutrient-recommendation-engines)
   - [Soil Organic Carbon Prediction](#soil-organic-carbon-prediction)
8. [Supply Chain Optimization](#supply-chain-optimization)
   - [Farm-to-Table Traceability](#farm-to-table-traceability)
   - [Demand Forecasting for Perishables](#demand-forecasting-for-perishables)
   - [Cold Chain Monitoring](#cold-chain-monitoring)
9. [Case Studies](#case-studies)
   - [John Deere's AI Ecosystem](#john-deeres-ai-ecosystem)
   - [Blue River Technology & See & Spray](#blue-river-technology--see--spray)
   - [CropX Soil Intelligence](#cropx-soil-intelligence)
   - [Climate FieldView Platform](#climate-fieldview-platform)
10. [Cross-References](#cross-references)
11. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

Artificial Intelligence is revolutionizing agriculture, an industry that must feed a global population projected to reach 10 billion by 2050. The intersection of AI, sensor technology, and robotics is giving rise to "Agriculture 4.0" — a paradigm where every plant, animal, and soil particle can be monitored, analyzed, and optimized with data-driven precision.

The agricultural AI market was valued at approximately $1.7 billion in 2023 and is projected to exceed $4.7 billion by 2028, growing at a CAGR of over 22%. This growth is driven by several converging factors: declining costs of drone and satellite imagery, the proliferation of IoT sensors in fields, advances in computer vision for disease detection, and the pressing need to increase crop yields while reducing water, fertilizer, and pesticide usage.

Agriculture AI presents unique technical challenges that distinguish it from other AI domains:

1. **Spatio-Temporal Complexity**: Agricultural data spans both space (fields, regions, global) and time (growth cycles, seasons, years). Models must capture complex interactions across both dimensions simultaneously.

2. **Data Scarcity in Edge Cases**: While satellite data is abundant, labeled data for specific diseases, pest infestations, or local soil conditions is often scarce, requiring techniques like few-shot learning and transfer learning.

3. **Environmental Variability**: Weather, soil type, topography, and microclimates create non-stationary distributions that challenge model generalization.

4. **Real-Time Edge Inference**: Many agricultural AI systems must run on drones, tractors, or mobile devices in the field with limited connectivity, requiring compressed models and on-device inference.

5. **Causal Understanding**: Correlation-based models can fail dramatically when environmental conditions shift — a disease detection model trained on dry-season data may fail in the wet season if it learned spurious correlations.

This document provides a deep technical exploration of the architectures, algorithms, and deployment patterns that power modern AI systems across the agricultural value chain — from precision farming and crop monitoring to livestock management and supply chain optimization.

---

## Precision Agriculture

Precision agriculture (PA) is the data-driven management of agricultural inputs — water, fertilizer, pesticides, seeds — at the sub-field level. Instead of treating a field uniformly, PA applies the right input, in the right amount, at the right place, at the right time. AI is the engine that converts raw sensor data into actionable prescriptions.

### Drone Imagery & Multispectral Analysis

Unmanned Aerial Vehicles (UAVs) equipped with multispectral and hyperspectral cameras are the workhorses of precision agriculture. A typical agricultural drone survey captures five or more spectral bands — red, green, blue, near-infrared (NIR), red-edge — at spatial resolutions of 2-10 cm per pixel.

**Multispectral Indices**: Raw spectral bands are combined into vegetation indices that correlate with crop health:

```python
import numpy as np
import rasterio
from typing import Tuple

class VegetationIndices:
    """Compute standard vegetation indices from multispectral imagery."""
    
    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Vegetation Index.
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1. Healthy vegetation: 0.6-0.9.
        """
        denominator = nir + red + 1e-10  # avoid division by zero
        return (nir - red) / denominator
    
    @staticmethod
    def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Red Edge.
        More sensitive to chlorophyll content than NDVI.
        """
        return (nir - red_edge) / (nir + red_edge + 1e-10)
    
    @staticmethod
    def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Enhanced Vegetation Index.
        Corrects for atmospheric scattering and soil background.
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        """
        return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-10)
    
    @staticmethod
    def gci(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
        """
        Green Chlorophyll Index.
        GCI = (NIR / Green) - 1
        Correlates with leaf chlorophyll concentration.
        """
        return (nir / (green + 1e-10)) - 1
    
    @staticmethod
    def ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Moisture Index.
        NDMI = (NIR - SWIR) / (NIR + SWIR)
        Indicates water stress in vegetation.
        """
        return (nir - swir) / (nir + swir + 1e-10)


class OrthomosaicProcessor:
    """Process drone orthomosaics into field-level analytics."""
    
    def __init__(self, image_path: str):
        with rasterio.open(image_path) as src:
            self.bands = src.read()
            self.profile = src.profile
            self.transform = src.transform
    
    def compute_health_map(self) -> np.ndarray:
        """Generate a crop health map using NDVI + texture features."""
        nir = self.bands[3].astype(np.float32)  # Band 4: NIR
        red = self.bands[2].astype(np.float32)  # Band 3: Red
        green = self.bands[1].astype(np.float32)
        
        ndvi_map = VegetationIndices.ndvi(nir, red)
        
        # Texture analysis using local variance
        from scipy.ndimage import uniform_filter
        local_var = uniform_filter(ndvi_map**2, size=5) - uniform_filter(ndvi_map, size=5)**2
        
        # Combine NDVI with texture for health classification
        health_score = ndvi_map * 0.7 + (1 - np.sqrt(local_var)) * 0.3
        return np.clip(health_score, 0, 1)
```

**Deep Learning for Weed Detection from Drone Imagery**: Beyond vegetation indices, deep learning models classify every pixel in drone orthomosaics to distinguish crops from weeds:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class WeedDetectionUNet(nn.Module):
    """
    U-Net architecture for pixel-wise weed/crop segmentation
    from multispectral drone imagery.
    """
    def __init__(self, in_channels=5, num_classes=3):  # crop, weed, soil
        super().__init__()
        
        # Encoder: EfficientNet-B0 as backbone
        encoder = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Modify first conv for multispectral input
        original_conv = encoder.features[0][0]
        self.encoder_conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        # Copy RGB weights and average for extra channels
        with torch.no_grad():
            rgb_weights = original_conv.weight.mean(dim=1, keepdim=True)
            self.encoder_conv.weight[:, :3] = original_conv.weight
            self.encoder_conv.weight[:, 3:] = rgb_weights.repeat(1, in_channels-3, 1, 1)
        
        # Extract encoder stages
        self.stage1 = nn.Sequential(self.encoder_conv, encoder.features[0][1:])
        self.stage2 = encoder.features[1]
        self.stage3 = encoder.features[2]
        self.stage4 = encoder.features[3]
        self.stage5 = encoder.features[4]
        self.stage6 = encoder.features[5]
        
        # Decoder with skip connections
        self.up6 = nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2)
        self.dec6 = self._decoder_block(256 + 112, 256)  # 112 from stage5
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec7 = self._decoder_block(128 + 40, 128)   # 40 from stage4
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec8 = self._decoder_block(64 + 24, 64)      # 24 from stage3
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec9 = self._decoder_block(32 + 16, 32)      # 16 from stage2
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        s6 = self.stage6(s5)
        
        # Decoder with skip connections
        d6 = self.up6(s6)
        d6 = torch.cat([d6, s5], dim=1)
        d6 = self.dec6(d6)
        
        d7 = self.up7(d6)
        d7 = torch.cat([d7, s4], dim=1)
        d7 = self.dec7(d7)
        
        d8 = self.up8(d7)
        d8 = torch.cat([d8, s3], dim=1)
        d8 = self.dec8(d8)
        
        d9 = self.up9(d8)
        d9 = torch.cat([d9, s2], dim=1)
        d9 = self.dec9(d9)
        
        return self.final_conv(d9)
```

### Satellite Remote Sensing

Satellite imagery provides the broad-scale perspective that complements drone-based close-ups. Major satellite platforms for agriculture include:

- **Sentinel-2** (ESA): 10m resolution, 13 spectral bands, 5-day revisit time. The workhorse for operational agricultural monitoring.
- **Landsat 8/9** (NASA/USGS): 30m resolution, 11 bands, 16-day revisit. Essential for historical analysis (40+ year archive).
- **Planet Labs** (Commercial): 3m resolution, 4 bands (R,G,B,NIR), daily revisit. Used for high-frequency crop monitoring.
- **WorldView-3** (Commercial): 0.3m panchromatic, 1.24m multispectral, 8 bands. Used for detailed field analysis.

**Cloud Masking & Atmospheric Correction**: Raw satellite data requires significant preprocessing:

```python
import numpy as np
from typing import Dict

class SatellitePreprocessor:
    """Preprocess satellite imagery for agricultural analysis."""
    
    @staticmethod
    def sentinel2_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
        """
        Generate cloud mask from Sentinel-2 Scene Classification Layer (SCL).
        SCL values:
        - 0: No data
        - 1: Saturated/Defective
        - 2: Dark area pixels
        - 3: Cloud shadows
        - 4: Vegetation
        - 5: Bare soils
        - 6: Water
        - 7: Low probability clouds (unclassified)
        - 8+9: Medium/high probability clouds
        - 10: Thin cirrus
        - 11: Snow
        """
        cloud_mask = np.isin(scl_band, [3, 7, 8, 9, 10])
        shadow_mask = scl_band == 3
        return cloud_mask | shadow_mask
    
    @staticmethod
    def landsat_toa_to_sr(toa_bands: Dict[str, np.ndarray],
                          metadata: Dict) -> Dict[str, np.ndarray]:
        """
        Simplified Landsat Top-of-Atmosphere to Surface Reflectance
        using the COST (Chavez) method for atmospheric correction.
        """
        sr_bands = {}
        for band_name, radiance in toa_bands.items():
            # Dark object subtraction
            dark_pixel_value = np.percentile(radiance[radiance > 0], 1)
            sr = radiance - dark_pixel_value
            sr = np.clip(sr / (sr.max() + 1e-10), 0, 1)
            sr_bands[band_name] = sr
        return sr_bands
    
    @staticmethod
    def interpolate_nodata(band: np.ndarray, nodata_value: int = 0) -> np.ndarray:
        """
        Interpolate missing data (cloud shadows, gaps) using
        inverse distance weighting.
        """
        from scipy.interpolate import griddata
        
        mask = band != nodata_value
        if mask.sum() < 100:
            return band  # Not enough valid pixels
        
        y, x = np.where(mask)
        values = band[mask]
        
        yi, xi = np.where(~mask)
        interpolated = griddata(
            (y, x), values, (yi, xi), method='nearest'
        )
        
        result = band.copy()
        result[~mask] = interpolated
        return result


class TimeSeriesExtractor:
    """Extract pixel-level time series from satellite image stacks."""
    
    def __init__(self, image_collection: np.ndarray, dates: list):
        """
        image_collection: (T, H, W, B) tensor of geocoded images
        dates: list of datetime objects
        """
        self.images = image_collection
        self.dates = dates
    
    def pixel_ts(self, row: int, col: int) -> np.ndarray:
        """Extract time series for a single pixel."""
        return self.images[:, row, col, :]
    
    def field_ts(self, field_mask: np.ndarray) -> np.ndarray:
        """
        Extract mean time series for an entire field.
        field_mask: boolean array (H, W)
        """
        masked = self.images[:, field_mask, :]
        return masked.mean(axis=1)
    
    def savitzky_golay_smooth(self, ts: np.ndarray,
                               window_length: int = 7,
                               polyorder: int = 2) -> np.ndarray:
        """
        Smooth NDVI time series using Savitzky-Golay filter.
        Removes cloud contamination artifacts while preserving
        phenological transitions.
        """
        from scipy.signal import savgol_filter
        return savgol_filter(ts, window_length, polyorder, axis=0)
```

### Variable Rate Technology

Variable Rate Technology (VRT) is the actuation layer of precision agriculture — applying inputs at varying rates across a field based on prescription maps generated by AI models.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

class PrescriptionMapGenerator:
    """
    Generate variable-rate application maps for fertilizer, seed, or irrigation.
    """
    
    def __init__(self, field_data: pd.DataFrame):
        """
        field_data columns: x, y, elevation, soil_type, 
        organic_matter, CEC, electrical_conductivity, 
        historical_ndvi_mean, historical_yield
        """
        self.data = field_data
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42
        )
    
    def generate_management_zones(self, n_zones: int = 5) -> pd.DataFrame:
        """
        Cluster field into management zones using soil and historical data.
        Zones represent regions with similar yield potential.
        """
        features = ['elevation', 'organic_matter', 'CEC', 
                    'electrical_conductivity', 'historical_ndvi_mean']
        
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        self.data['zone'] = kmeans.fit_predict(self.data[features])
        
        # Characterize each zone
        zone_summary = self.data.groupby('zone')[features].mean()
        zone_summary['yield_potential'] = self.data.groupby('zone')['historical_yield'].mean()
        zone_summary = zone_summary.sort_values('yield_potential')
        zone_summary['application_rate_multiplier'] = (
            zone_summary['yield_potential'] / zone_summary['yield_potential'].max()
        )
        
        return zone_summary
    
    def generate_nitrogen_prescription(self, 
                                        target_yield: float = 12.0) -> pd.DataFrame:
        """
        Generate variable-rate N application map.
        Uses SUSTAIN algorithm adapted for ML.
        """
        features = ['zone', 'organic_matter', 'soil_nitrogen', 
                    'expected_mineralization', 'previous_crop_factor']
        
        # Train yield prediction model
        y = self.data['historical_yield'].values
        X = self.data[features].values
        self.model.fit(X, y)
        
        # Predict yield potential at each location
        self.data['predicted_yield'] = self.model.predict(X)
        
        # Calculate N requirement using yield goal approach
        # N_rate (kg/ha) = (Yield_goal - Predicted_yield) * N_uptake_factor
        n_uptake = 22  # kg N per ton of grain
        self.data['n_prescription'] = np.maximum(
            0, (target_yield - self.data['predicted_yield']) * n_uptake
        )
        
        # Apply minimum and maximum bounds
        self.data['n_prescription'] = self.data['n_prescription'].clip(40, 200)
        
        return self.data[['x', 'y', 'zone', 'n_prescription', 'predicted_yield']]
```

---

## Crop Yield Prediction

Accurate crop yield prediction at field, regional, and national scales is critical for food security planning, commodity trading, and agricultural insurance. AI has dramatically improved upon traditional process-based crop models (like DSSAT and APSIM) by learning directly from data.

### LSTM for Time-Series Forecasting

Long Short-Term Memory networks excel at the sequential nature of crop growth, where conditions at each growth stage influence final yield.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class CropGrowthSequenceDataset(Dataset):
    """Dataset of sequential growth-stage data for yield prediction."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        sequences: (n_samples, n_timesteps, n_features)
        Features per timestep: temp_min, temp_max, precipitation, 
        solar_radiation, soil_moisture, VPD, growth_stage
        targets: (n_samples,) — final yield in tons/ha
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class CropYieldLSTM(nn.Module):
    """
    Multi-layer LSTM with attention for crop yield prediction.
    
    Architecture:
    - 3-layer stacked LSTM with dropout
    - Temporal attention mechanism
    - Skip connections from early growth stages
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.3,
                 seq_length: int = 120):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Capture past and future context
        )
        
        # Temporal attention over hidden states
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Feature extractors for different growth phases
        self.early_features = nn.Linear(hidden_dim * 2, 64)
        self.mid_features = nn.Linear(hidden_dim * 2, 64)
        self.late_features = nn.Linear(hidden_dim * 2, 64)
        
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(192 + hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden*2)
        
        # Temporal attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        
        # Growth-phase features
        seq_len = x.size(1)
        early_end = seq_len // 4
        mid_end = seq_len // 2
        
        early_pool = lstm_out[:, :early_end, :].mean(dim=1)
        mid_pool = lstm_out[:, early_end:mid_end, :].mean(dim=1)
        late_pool = lstm_out[:, mid_end:, :].mean(dim=1)
        
        early_feat = self.early_features(early_pool)
        mid_feat = self.mid_features(mid_pool)
        late_feat = self.late_features(late_pool)
        
        # Combine all features
        combined = torch.cat([context, early_feat, mid_feat, late_feat], dim=1)
        
        return self.regressor(combined).squeeze()
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    epochs: int = 100, lr: float = 1e-3):
        """Training loop with early stopping and learning rate scheduling."""
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    predictions = self(batch_x)
                    val_loss += criterion(predictions, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_yield_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

### Transformer Architectures for Yield Modeling

Transformers have emerged as powerful alternatives to LSTMs for yield prediction, particularly when modeling long-range dependencies across the growing season.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for growth-stage awareness."""
    
    def __init__(self, d_model: int, max_len: int = 366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class CropYieldTransformer(nn.Module):
    """
    Transformer encoder for crop yield prediction.
    
    Input: Daily weather and soil data across the growing season.
    Output: Final yield prediction.
    
    Key innovations:
    - Learned growth-stage embeddings
    - Multi-scale temporal attention
    - Auxiliary crop phenology prediction
    """
    
    def __init__(self, input_dim: int = 7, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Growth stage embeddings (learned)
        self.growth_stage_embed = nn.Embedding(7, d_model)  # 7 growth stages
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Multi-scale pooling
        self.weekly_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.monthly_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Auxiliary phenology prediction head
        self.phenology_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Predict growth stage probabilities
        )
        
        # Yield regression head
        self.yield_head = nn.Sequential(
            nn.Linear(d_model * 3, 256),  # daily + weekly + monthly features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, growth_stages=None):
        """
        x: (batch, seq_len, input_dim) — daily weather/soil features
        growth_stages: (batch, seq_len) — optional stage labels
        """
        # Project and add position encoding
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Add growth stage information
        if growth_stages is not None:
            stage_emb = self.growth_stage_embed(growth_stages)  # (batch, seq_len, d_model)
            x = x + stage_emb * 0.1
        
        # Transformer encoder
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # Daily-level features (global average pooling)
        daily_feat = encoded.mean(dim=1)  # (batch, d_model)
        
        # Weekly-level features
        batch_size, seq_len, d_model = encoded.shape
        n_weeks = seq_len // 7
        if n_weeks > 0:
            weekly = encoded[:, :n_weeks*7, :].reshape(batch_size, n_weeks, 7, d_model)
            weekly = weekly.mean(dim=2)  # (batch, n_weeks, d_model)
            weekly_feat, _ = self.weekly_attn(weekly, weekly, weekly)
            weekly_feat = weekly_feat.mean(dim=1)  # (batch, d_model)
        else:
            weekly_feat = daily_feat
        
        # Monthly-level features
        n_months = seq_len // 30
        if n_months > 0:
            monthly = encoded[:, :n_months*30, :].reshape(batch_size, n_months, 30, d_model)
            monthly = monthly.mean(dim=2)
            monthly_feat, _ = self.monthly_attn(monthly, monthly, monthly)
            monthly_feat = monthly_feat.mean(dim=1)
        else:
            monthly_feat = daily_feat
        
        # Concatenate multi-scale features
        combined = torch.cat([daily_feat, weekly_feat, monthly_feat], dim=1)
        
        # Auxiliary prediction
        phenology_pred = self.phenology_head(encoded.mean(dim=1))
        
        # Final yield prediction
        yield_pred = self.yield_head(combined).squeeze()
        
        return yield_pred, phenology_pred


class YieldEnsemble(nn.Module):
    """
    Ensemble of LSTM, Transformer, and CNN-based models for robust yield prediction.
    Uses learned weights to combine predictions based on input data quality.
    """
    
    def __init__(self, input_dim: int = 7, seq_length: int = 120):
        super().__init__()
        self.lstm = CropYieldLSTM(input_dim, seq_length=seq_length)
        self.transformer = CropYieldTransformer(input_dim)
        
        # Meta-learner that weights ensemble members
        self.meta_learner = nn.Sequential(
            nn.Linear(3, 16),  # 3 model predictions
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, growth_stages=None):
        lstm_pred = self.lstm(x)
        transformer_pred, _ = self.transformer(x, growth_stages)
        
        # Stack predictions
        preds = torch.stack([lstm_pred, transformer_pred, 
                             (lstm_pred + transformer_pred) / 2], dim=1)
        
        # Learn optimal weighting
        weights = self.meta_learner(preds)
        
        return (weights * preds).sum(dim=1)
```

### Hybrid CNN-RNN Models

For spatial-temporal yield prediction (predicting yield at every pixel in a field), hybrid CNN-RNN architectures combine the spatial feature extraction of CNNs with the temporal modeling of RNNs:

```python
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatio-temporal modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding
        )
    
    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat([x, h_prev], dim=1)
        
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c


class SpatialTemporalYieldModel(nn.Module):
    """
    ConvLSTM for pixel-wise yield prediction across an entire field.
    
    Input: (T, H, W, C) — time series of satellite images
    Output: (H, W) — yield map
    """
    
    def __init__(self, input_channels: int = 5, hidden_dim: int = 64,
                 num_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stacked ConvLSTM layers
        self.cell_list = nn.ModuleList()
        self.cell_list.append(
            ConvLSTMCell(input_channels, hidden_dim, kernel_size)
        )
        for _ in range(1, num_layers):
            self.cell_list.append(
                ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
            )
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        x: (batch, T, H, W, C)
        """
        batch, T, H, W, C = x.shape
        
        # Initialize states
        h_t = [torch.zeros(batch, self.hidden_dim, H, W, device=x.device)
               for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch, self.hidden_dim, H, W, device=x.device)
               for _ in range(self.num_layers)]
        
        # Process each timestep
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (batch, H, W, C)
            x_t = x_t.permute(0, 3, 1, 2)  # (batch, C, H, W)
            
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.cell_list[layer](
                    x_t if layer == 0 else h_t[layer-1],
                    (h_t[layer], c_t[layer])
                )
        
        # Final output from top layer
        return self.output_conv(h_t[-1]).squeeze(1)  # (batch, H, W)
```

---

## Plant Disease Detection

Plant diseases cause 20-40% of global crop losses annually. AI-powered disease detection, particularly using deep learning on leaf images, has achieved accuracy exceeding that of human agronomists for many diseases.

### CNN-Based Classification (ResNet, EfficientNet)

The PlantVillage dataset (54,000+ images of 38 crop-disease pairs) has become the standard benchmark for plant disease classification:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b4
from PIL import Image

class PlantDiseaseClassifier:
    """
    Production-ready plant disease classifier with model ensembling.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = 38  # PlantVillage classes
        
        # Model 1: ResNet50
        self.model1 = resnet50(weights='IMAGENET1K_V2')
        in_features = self.model1.fc.in_features
        self.model1.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
        
        # Model 2: EfficientNet-B4
        self.model2 = efficientnet_b4(weights='IMAGENET1K_V1')
        in_features = self.model2.classifier[1].in_features
        self.model2.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
        
        # Freeze batch norm stats for deployment
        self.model1.eval()
        self.model2.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Disease label mapping
        self.label_map = {
            0: 'Apple___Apple_scab', 1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
            4: 'Blueberry___healthy', 5: 'Cherry___Powdery_mildew',
            6: 'Cherry___healthy', 7: 'Corn___Cercospora_leaf_spot',
            8: 'Corn___Common_rust', 9: 'Corn___Northern_Leaf_Blight',
            10: 'Corn___healthy', 11: 'Grape___Black_rot',
            12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            14: 'Grape___healthy', 15: 'Orange___Haunglongbing_(Citrus_greening)',
            16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
            18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy',
            20: 'Potato___Early_blight', 21: 'Potato___Late_blight',
            22: 'Potato___healthy', 23: 'Raspberry___healthy',
            24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew',
            26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy',
            28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight',
            30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold',
            32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites_(Two-spotted_spider_mite)',
            34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'
        }
    
    def predict(self, image: Image.Image, temperature: float = 1.0) -> dict:
        """
        Ensemble prediction with temperature scaling for confidence calibration.
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out1 = self.model1(img_tensor) / temperature
            out2 = self.model2(img_tensor) / temperature
            
            # Soft ensemble (average logits)
            ensemble_logits = (out1 + out2) / 2
            probabilities = torch.softmax(ensemble_logits, dim=1)
            
            confidence, pred_idx = torch.max(probabilities, dim=1)
            pred_idx = pred_idx.item()
            confidence = confidence.item()
        
        # Top-5 predictions
        top5_probs, top5_idx = torch.topk(probabilities, 5, dim=1)
        
        results = {
            'top_prediction': {
                'disease': self.label_map[pred_idx],
                'confidence': confidence,
                'is_healthy': 'healthy' in self.label_map[pred_idx]
            },
            'top5': [
                {
                    'disease': self.label_map[idx.item()],
                    'confidence': prob.item()
                }
                for idx, prob in zip(top5_idx[0], top5_probs[0])
            ]
        }
        
        return results
    
    def uncertainty_aware_prediction(self, image: Image.Image,
                                      n_passes: int = 10) -> dict:
        """
        Monte Carlo Dropout for uncertainty estimation.
        Enable dropout at inference for uncertainty quantification.
        """
        self.model1.train()  # Enable dropout
        self.model2.train()
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        predictions = []
        
        for _ in range(n_passes):
            with torch.no_grad():
                out1 = self.model1(img_tensor)
                out2 = self.model2(img_tensor)
                ensemble = (out1 + out2) / 2
                predictions.append(torch.softmax(ensemble, dim=1))
        
        predictions = torch.stack(predictions)  # (n_passes, 1, n_classes)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Predictive entropy as uncertainty metric
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        
        return {
            'mean_prediction': torch.argmax(mean_pred, dim=1).item(),
            'confidence': mean_pred.max().item(),
            'uncertainty': entropy.item(),
            'epistemic_uncertainty': std_pred.max().item()
        }
```

### Segmentation for Disease Severity Assessment

Classification tells us *what* disease is present, but segmentation tells us *how severe* it is — critical for treatment decisions:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseSeverityUNet(nn.Module):
    """
    U-Net with deep supervision for disease severity segmentation.
    Outputs disease area as percentage of leaf area.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = self._conv_block(512, 1024)
        
        # Decoder with deep supervision
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Deep supervision outputs
        self.out4 = nn.Conv2d(512, num_classes, 1)
        self.out3 = nn.Conv2d(256, num_classes, 1)
        self.out2 = nn.Conv2d(128, num_classes, 1)
        self.out1 = nn.Conv2d(64, num_classes, 1)
        
        # Severity regression head (from bottleneck)
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 severity score
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bridge
        b = self.bridge(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Deep supervision outputs
        out4 = self.out4(d4)
        out3 = F.interpolate(self.out3(d3), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.out2(d2), scale_factor=4, mode='bilinear', align_corners=True)
        out1 = self.out1(d1)
        
        # Disease severity (0-100%)
        severity = self.severity_head(b)
        
        return {
            'segmentation': out1,
            'deep_supervision': [out4, out3, out2, out1],
            'severity': severity
        }
    
    def compute_disease_percentage(self, segmentation_logits: torch.Tensor,
                                    leaf_mask: torch.Tensor) -> float:
        """Calculate disease area as percentage of total leaf area."""
        disease_mask = torch.sigmoid(segmentation_logits) > 0.5
        disease_pixels = (disease_mask & leaf_mask).sum().item()
        total_leaf_pixels = leaf_mask.sum().item()
        
        if total_leaf_pixels == 0:
            return 0.0
        
        return (disease_pixels / total_leaf_pixels) * 100
```

### Mobile Deployment for Field Use

Deploying disease detection models on mobile devices requires model compression techniques:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobilePlantDiseaseNet(nn.Module):
    """
    Lightweight CNN for on-device plant disease classification.
    Uses depthwise separable convolutions and hard-swish activations.
    ~1.2M parameters, suitable for mobile CPU inference.
    """
    
    def __init__(self, num_classes: int = 38, input_size: int = 224):
        super().__init__()
        
        def dwise_conv(ch_in, ch_out, stride):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_in, 3, stride=stride, padding=1,
                          groups=ch_in, bias=False),
                nn.BatchNorm2d(ch_in),
                nn.ReLU6(inplace=True),
                nn.Conv2d(ch_in, ch_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU6(inplace=True)
            )
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            dwise_conv(32, 64, 1),
            dwise_conv(64, 128, 2),
            dwise_conv(128, 128, 1),
            dwise_conv(128, 256, 2),
            dwise_conv(256, 256, 1),
            dwise_conv(256, 512, 2),
            dwise_conv(512, 512, 1),
            dwise_conv(512, 512, 1),
            dwise_conv(512, 512, 1),
            dwise_conv(512, 512, 1),
            dwise_conv(512, 1024, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def quantize_for_mobile(model: nn.Module, 
                         representative_dataset: torch.utils.data.DataLoader,
                         device: str = 'cpu') -> torch.nn.Module:
    """
    Post-training quantization to INT8 for mobile deployment.
    Reduces model size ~4x and improves inference speed ~2-3x.
    """
    model.eval()
    model.to(device)
    
    # Fuse Conv+BN+ReLU
    model_fused = torch.quantization.fuse_modules(model, [
        ['stem.0', 'stem.1', 'stem.2'],
        ['blocks.0.0', 'blocks.0.1', 'blocks.0.2'],
        ['blocks.0.3', 'blocks.0.4', 'blocks.0.5'],
    ])
    
    # Quantization configuration
    model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_fused)
    
    # Calibrate with representative data
    with torch.no_grad():
        for images, _ in representative_dataset:
            model_prepared(images.to(device))
            if len(representative_dataset) > 10:  # Limit calibration batches
                break
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized
```

---

## Automated Irrigation Systems

Agriculture accounts for 70% of global freshwater withdrawals. AI-powered irrigation can reduce water usage by 20-50% while maintaining or increasing yields.

### Reinforcement Learning for Water Optimization

Reinforcement Learning (RL) treats irrigation scheduling as a sequential decision problem where an agent learns optimal watering policies through interaction with the environment:

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

class IrrigationEnv(gym.Env):
    """
    Custom Gym environment for irrigation optimization.
    
    State: soil moisture, temperature, humidity, wind speed,
           solar radiation, precipitation forecast, growth stage,
           days since last irrigation
    Action: irrigation amount (continuous: 0-50mm)
    Reward: -water_usage + yield_gain - penalty for water stress
    """
    
    def __init__(self, 
                 soil_params: dict = None,
                 crop_params: dict = None,
                 max_episode_steps: int = 365):
        super().__init__()
        
        # Default soil parameters (silty loam)
        self.soil = soil_params or {
            'field_capacity': 0.32,     # m³/m³
            'wilting_point': 0.12,      # m³/m³
            'saturation': 0.45,         # m³/m³
            'max_root_depth': 1.0,      # meters
            'hydraulic_conductivity': 0.2  # mm/hr
        }
        
        # Crop parameters (corn)
        self.crop = crop_params or {
            'kc_init': 0.4,    # Crop coefficient initial
            'kc_mid': 1.2,     # Crop coefficient mid-season
            'kc_end': 0.6,     # Crop coefficient end
            'root_depth_init': 0.3,  # Initial root depth (m)
            'root_depth_max': 1.0,   # Maximum root depth (m)
            'critical_depletion': 0.55,  # MAD fraction
            'growth_stages': 4
        }
        
        self.max_steps = max_episode_steps
        self.current_step = 0
        
        # Observation space: [soil_moisture, temp, humidity, wind,
        #                      solar_rad, precip_forecast, growth_stage,
        #                      days_since_irrigation, root_depth, ET0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10, 0, 0, 0, 0, 0, 0, 0.1, 0]),
            high=np.array([1.0, 50, 100, 50, 1000, 100, 4, 30, 2.0, 15]),
            dtype=np.float32
        )
        
        # Action: irrigation amount in mm (0-50)
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([50.0]), dtype=np.float32
        )
        
        # Weather data (would be loaded from historical records)
        self.weather = self._generate_weather()
        
        self.state = None
    
    def _generate_weather(self):
        """Generate synthetic weather data for training."""
        np.random.seed(42)
        n_days = 365
        
        # Seasonal patterns
        doy = np.arange(n_days)
        temp = 15 + 15 * np.sin(2 * np.pi * (doy - 90) / 365)  # Celsius
        temp += np.random.normal(0, 3, n_days)  # Random variation
        
        solar = 200 + 300 * np.sin(2 * np.pi * (doy - 80) / 365)  # W/m²
        solar = np.maximum(solar, 0)
        
        # Rainfall with seasonality
        rain_prob = 0.3 + 0.2 * np.sin(2 * np.pi * (doy - 120) / 365)
        rain_amount = np.random.exponential(10, n_days) * (np.random.random(n_days) < rain_prob)
        
        return {
            'temp': temp,
            'solar': solar,
            'rain': rain_amount,
            'humidity': 60 + 20 * np.sin(2 * np.pi * (doy - 180) / 365),
            'wind': 3 + 2 * np.random.random(n_days)
        }
    
    def _compute_et0(self, temp, solar, humidity, wind):
        """Reference evapotranspiration (FAO Penman-Monteith simplified)."""
        # Simplified Hargreaves equation
        return 0.0023 * solar * (temp + 17.8) * np.sqrt(temp + 20) / 1000
    
    def _compute_crop_coefficient(self, growth_stage):
        """Interpolate crop coefficient based on growth stage."""
        kc_values = [self.crop['kc_init'], 
                     (self.crop['kc_init'] + self.crop['kc_mid']) / 2,
                     self.crop['kc_mid'],
                     self.crop['kc_end']]
        return kc_values[int(growth_stage)]
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = {
            'soil_moisture': self.soil['field_capacity'] * 0.8,
            'growth_stage': 0,
            'days_since_irrigation': 0,
            'root_depth': self.crop['root_depth_init'],
            'cumulative_stress': 0.0,
            'cumulative_yield': 0.0
        }
        return self._get_obs(), {}
    
    def _get_obs(self):
        w = self.weather
        i = self.current_step
        et0 = self._compute_et0(w['temp'][i], w['solar'][i], 
                                 w['humidity'][i], w['wind'][i])
        return np.array([
            self.state['soil_moisture'],
            w['temp'][i],
            w['humidity'][i],
            w['wind'][i],
            w['solar'][i],
            w['rain'][i],
            self.state['growth_stage'],
            self.state['days_since_irrigation'],
            self.state['root_depth'],
            et0
        ], dtype=np.float32)
    
    def step(self, action):
        irrigation = action[0]
        i = self.current_step
        w = self.weather
        
        # Apply irrigation
        if irrigation > 0:
            self.state['soil_moisture'] = min(
                self.soil['saturation'],
                self.state['soil_moisture'] + irrigation / (self.soil['max_root_depth'] * 1000)
            )
            self.state['days_since_irrigation'] = 0
        else:
            self.state['days_since_irrigation'] += 1
        
        # Compute evapotranspiration
        et0 = self._compute_et0(w['temp'][i], w['solar'][i],
                                w['humidity'][i], w['wind'][i])
        kc = self._compute_crop_coefficient(self.state['growth_stage'])
        etc = et0 * kc  # Crop evapotranspiration
        
        # Soil water balance
        rainfall = w['rain'][i] / (self.soil['max_root_depth'] * 1000)
        deep_percolation = max(0, self.state['soil_moisture'] - self.soil['field_capacity']) * 0.1
        
        self.state['soil_moisture'] += rainfall
        self.state['soil_moisture'] -= etc / (self.soil['max_root_depth'] * 1000)
        self.state['soil_moisture'] -= deep_percolation
        self.state['soil_moisture'] = np.clip(
            self.state['soil_moisture'],
            self.soil['wilting_point'],
            self.soil['saturation']
        )
        
        # Water stress calculation
        soil_deficit = (self.soil['field_capacity'] - self.state['soil_moisture']) / \
                       (self.soil['field_capacity'] - self.soil['wilting_point'])
        water_stress = max(0, soil_deficit - self.crop['critical_depletion'])
        
        if water_stress > 0:
            self.state['cumulative_stress'] += water_stress
        
        # Growth stage progression
        self.state['growth_stage'] = min(
            self.crop['growth_stages'] - 1,
            int(self.current_step / (self.max_steps / self.crop['growth_stages']))
        )
        
        # Root depth growth
        self.state['root_depth'] = min(
            self.crop['root_depth_max'],
            self.crop['root_depth_init'] + 
            (self.crop['root_depth_max'] - self.crop['root_depth_init']) * 
            (self.current_step / self.max_steps)
        )
        
        self.current_step += 1
        
        # Reward calculation
        water_usage_penalty = -0.01 * irrigation
        stress_penalty = -10 * water_stress
        yield_reward = -0.1 * water_stress  # Yield loss proxy
        
        reward = water_usage_penalty + stress_penalty + yield_reward
        
        # Terminal conditions
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'et_crop': etc,
            'water_stress': water_stress,
            'cumulative_stress': self.state['cumulative_stress'],
            'total_irrigation': self.state['days_since_irrigation']
        }
        
        return self._get_obs(), reward, terminated, truncated, info


def train_irrigation_policy():
    """Train PPO agent for irrigation scheduling."""
    env = IrrigationEnv()
    eval_env = IrrigationEnv()
    
    # PPO with MLP policy
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs={
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),
            'activation_fn': nn.ReLU
        }
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./irrigation_models/',
        log_path='./irrigation_logs/',
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    model.learn(total_timesteps=500000, callback=eval_callback)
    model.save('irrigation_ppo_final')
    
    return model


class RuleBasedIrrigation:
    """
    Interpretable rule-based irrigation scheduler for comparison.
    Used as baseline and fallback for RL policies.
    """
    
    def __init__(self, soil_type: str = 'silty_loam'):
        self.thresholds = {
            'silty_loam': {'refill_point': 0.50, 'target': 0.85},
            'sandy_loam': {'refill_point': 0.40, 'target': 0.75},
            'clay_loam': {'refill_point': 0.60, 'target': 0.90}
        }
        self.params = self.thresholds.get(soil_type, self.thresholds['silty_loam'])
    
    def get_irrigation_amount(self, soil_moisture: float,
                               field_capacity: float) -> float:
        """
        Managed Allowable Depletion (MAD) based scheduling.
        Irrigate when soil moisture drops below refill point.
        Apply enough water to bring to target.
        """
        if soil_moisture / field_capacity < self.params['refill_point']:
            target_moisture = field_capacity * self.params['target']
            deficit = target_moisture - soil_moisture
            return max(0, deficit * 100)  # Convert to mm
        return 0.0
```

### IoT Sensor Fusion

Modern irrigation systems integrate multiple sensor types:

```python
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SensorReading:
    """Unified sensor reading from any IoT device."""
    timestamp: datetime
    sensor_id: str
    sensor_type: str  # 'soil_moisture', 'temperature', 'flow', 'pressure'
    value: float
    unit: str
    location: tuple  # (lat, lon)
    quality_score: float  # 0-1, signal quality


class SensorFusion:
    """
    Kalman filter-based sensor fusion for robust soil moisture estimation.
    Combines multiple sensor sources with uncertainty weighting.
    """
    
    def __init__(self, n_sensors: int = 5):
        self.n_sensors = n_sensors
        self.state_mean = np.array([0.25])  # Initial soil moisture estimate
        self.state_cov = np.array([[0.01]])   # Initial uncertainty
        self.process_noise = 0.001
        self.measurement_noise = 0.01
    
    def predict(self, dt: float = 1.0):
        """Predict step: evolve state forward in time."""
        # Process model: soil moisture evolves slowly with drainage
        F = np.array([[1.0]])  # State transition
        Q = np.array([[self.process_noise * dt]])
        
        self.state_mean = F @ self.state_mean
        self.state_cov = F @ self.state_cov @ F.T + Q
    
    def update(self, measurements: List[SensorReading]):
        """
        Update step: incorporate new sensor readings.
        Uses Mahalanobis distance for outlier rejection.
        """
        if not measurements:
            return
        
        values = np.array([m.value for m in measurements])
        weights = np.array([m.quality_score for m in measurements])
        
        # Outlier rejection using z-score
        if len(values) > 1:
            z_scores = np.abs(values - np.median(values)) / (np.std(values) + 1e-10)
            weights[z_scores > 2.0] = 0.0
        
        if weights.sum() == 0:
            return
        
        # Weighted average measurement
        weights = weights / weights.sum()
        z = np.sum(values * weights)
        R = self.measurement_noise / (weights.sum() + 1e-10)
        
        # Kalman gain
        H = np.array([[1.0]])
        S = H @ self.state_cov @ H.T + R
        K = self.state_cov @ H.T / S
        
        # Update
        innovation = z - H @ self.state_mean
        self.state_mean = self.state_mean + K * innovation
        self.state_cov = (np.eye(1) - K @ H) @ self.state_cov
    
    def get_estimate(self) -> Dict:
        """Return fused estimate with uncertainty bounds."""
        std = np.sqrt(self.state_cov[0, 0])
        return {
            'soil_moisture': float(self.state_mean[0]),
            'std': float(std),
            'ci_95_lower': float(self.state_mean[0] - 1.96 * std),
            'ci_95_upper': float(self.state_mean[0] + 1.96 * std)
        }


class IrrigationController:
    """
    Production irrigation controller with RL policy fallback.
    """
    
    def __init__(self, rl_policy_path: str = 'irrigation_ppo_final.zip'):
        self.sensor_fusion = SensorFusion()
        self.rule_based = RuleBasedIrrigation()
        self.rl_model = None  # Load from path in production
        
        # Safety limits
        self.max_daily_irrigation = 50  # mm/day
        self.min_irrigation_interval = 24  # hours
        self.last_irrigation_time = None
    
    def decide_irrigation(self, 
                           sensor_readings: List[SensorReading],
                           weather_forecast: Dict,
                           use_rl: bool = True) -> Dict:
        """
        Decide irrigation amount using RL policy with rule-based constraints.
        """
        # Fuse sensor data
        self.sensor_fusion.predict()
        self.sensor_fusion.update(sensor_readings)
        estimate = self.sensor_fusion.get_estimate()
        
        # Check safety constraints
        if self.last_irrigation_time:
            hours_since = (datetime.now() - self.last_irrigation_time).total_seconds() / 3600
            if hours_since < self.min_irrigation_interval:
                return {'irrigation_amount': 0, 'reason': 'Too soon since last irrigation'}
        
        # Check rain forecast
        if weather_forecast.get('precipitation_probability', 0) > 0.7:
            return {'irrigation_amount': 0, 'reason': 'Rain expected'}
        
        if use_rl and self.rl_model is not None:
            # RL-based decision with safety constraints
            obs = self._prepare_observation(estimate, weather_forecast)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            amount = float(np.clip(action[0], 0, self.max_daily_irrigation))
        else:
            # Rule-based fallback
            amount = self.rule_based.get_irrigation_amount(
                estimate['soil_moisture'], 0.32  # field capacity
            )
            amount = min(amount, self.max_daily_irrigation)
        
        if amount > 0:
            self.last_irrigation_time = datetime.now()
        
        return {
            'irrigation_amount': amount,
            'soil_moisture': estimate['soil_moisture'],
            'uncertainty': estimate['std'],
            'reason': 'RL policy' if use_rl else 'Rule-based'
        }
```

---

## Livestock Monitoring

AI-powered livestock monitoring systems track individual animal health, behavior, and productivity, enabling early disease detection and precision management.

### Computer Vision for Health Assessment

```python
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class LivestockHealthMonitor(nn.Module):
    """
    Multi-task model for livestock health assessment from camera imagery.
    
    Tasks:
    1. Body condition scoring (BCS 1-5)
    2. Lameness detection (binary)
    3. Weight estimation (regression)
    4. Individual identification
    """
    
    def __init__(self, n_individuals: int = 500):
        super().__init__()
        
        # Shared backbone
        backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Task-specific heads
        self.bcs_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)  # BCS 1-5
        )
        
        self.lameness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.weight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Individual identification (re-identification)
        self.id_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Embedding vector
        )
        self.id_classifier = nn.Linear(256, n_individuals)
    
    def forward(self, x):
        features = self.backbone(x)
        
        bcs = self.bcs_head(features)
        lameness = self.lameness_head(features).squeeze()
        weight = self.weight_head(features).squeeze()
        embedding = self.id_head(features)
        id_logits = self.id_classifier(embedding)
        
        return {
            'body_condition_score': bcs,
            'lameness': lameness,
            'estimated_weight': weight,
            'identity_embedding': embedding,
            'identity_logits': id_logits
        }


class ActivityMonitor:
    """
    Monitor livestock activity patterns using accelerometer and GPS data.
    Detects deviations from normal behavior that may indicate illness.
    """
    
    def __init__(self, window_size: int = 1440):  # 24 hours at 1-min resolution
        self.window_size = window_size
        self.behavior_model = self._build_behavior_model()
    
    def _build_behavior_model(self):
        """LSTM autoencoder for behavioral pattern learning."""
        return nn.Sequential(
            nn.LSTM(input_size=6, hidden_size=64, batch_first=True, bidirectional=True),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
    
    def detect_anomaly(self, accelerometer_data: np.ndarray,
                        gps_data: np.ndarray) -> Dict:
        """
        Detect anomalous behavior patterns.
        
        accelerometer_data: (T, 3) — x, y, z acceleration
        gps_data: (T, 3) — lat, lon, speed
        """
        # Combine sensors
        features = np.concatenate([accelerometer_data, gps_data], axis=1)
        
        # Compute activity metrics
        activity_level = np.linalg.norm(accelerometer_data, axis=1)
        resting_time = (activity_level < 0.1).sum() / len(activity_level)
        moving_speed = gps_data[:, 2].mean()
        total_distance = np.sum(np.sqrt(
            np.diff(gps_data[:, 0])**2 + np.diff(gps_data[:, 1])**2
        ))
        
        # Behavior classification
        # 0: Resting, 1: Grazing, 2: Walking, 3: Running, 4: Drinking
        behavior = self._classify_behavior(activity_level, gps_data[:, 2])
        
        # Anomaly score based on deviation from historical patterns
        anomaly_score = self._compute_anomaly_score(features)
        
        return {
            'activity_level': float(activity_level.mean()),
            'resting_percentage': float(resting_time * 100),
            'average_speed': float(moving_speed),
            'total_distance': float(total_distance),
            'dominant_behavior': int(np.argmax(np.bincount(behavior))),
            'behavior_distribution': {
                'resting': float((behavior == 0).sum() / len(behavior)),
                'grazing': float((behavior == 1).sum() / len(behavior)),
                'walking': float((behavior == 2).sum() / len(behavior)),
                'active': float((behavior == 3).sum() / len(behavior)),
                'drinking': float((behavior == 4).sum() / len(behavior))
            },
            'anomaly_score': float(anomaly_score),
            'health_alert': anomaly_score > 0.8
        }
    
    def _classify_behavior(self, activity: np.ndarray, 
                            speed: np.ndarray) -> np.ndarray:
        """Simple threshold-based behavior classification."""
        behavior = np.zeros_like(activity, dtype=int)
        behavior[(activity >= 0.1) & (activity < 0.3) & (speed < 0.5)] = 1  # Grazing
        behavior[(activity >= 0.3) & (activity < 0.7) & (speed >= 0.5) & (speed < 2.0)] = 2  # Walking
        behavior[activity >= 0.7] = 3  # Running
        behavior[(activity >= 0.1) & (activity < 0.3) & (speed < 0.1)] = 4  # Drinking
        return behavior
    
    def _compute_anomaly_score(self, features: np.ndarray) -> float:
        """Compute reconstruction error as anomaly score."""
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            reconstructed = self.behavior_model(features_tensor)
            error = nn.MSELoss()(features_tensor, reconstructed)
        return float(error)
```

---

## Soil Analysis & Nutrient Management

### Hyperspectral Soil Sensing

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import joblib

class SoilSpectroscopyAnalyzer:
    """
    Predict soil properties from visible-near-infrared (Vis-NIR) spectroscopy.
    """
    
    def __init__(self):
        self.pca = PCA(n_components=20)
        self.models = {
            'organic_carbon': RandomForestRegressor(n_estimators=200, max_depth=12),
            'total_nitrogen': RandomForestRegressor(n_estimators=200, max_depth=12),
            'soil_ph': RandomForestRegressor(n_estimators=200, max_depth=10),
            'clay_content': RandomForestRegressor(n_estimators=200, max_depth=10),
            'cec': RandomForestRegressor(n_estimators=200, max_depth=12)
        }
    
    def preprocess_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Preprocess raw reflectance spectra.
        
        Steps:
        1. Remove noisy edges (350-399nm and 2451-2500nm)
        2. Continuum removal (baseline correction)
        3. Savitzky-Golay smoothing
        4. Standard normal variate (SNV) normalization
        """
        from scipy.signal import savgol_filter
        
        # Trim edges
        spectra = spectra[:, 50:-50]
        
        # SNV normalization
        mean = spectra.mean(axis=1, keepdims=True)
        std = spectra.std(axis=1, keepdims=True)
        spectra = (spectra - mean) / (std + 1e-10)
        
        # Savitzky-Golay smoothing (1st derivative)
        spectra = savgol_filter(spectra, window_length=11, polyorder=2, deriv=1, axis=1)
        
        return spectra
    
    def predict(self, spectra: np.ndarray) -> dict:
        """Predict multiple soil properties from spectra."""
        processed = self.preprocess_spectra(spectra)
        features = self.pca.transform(processed)
        
        predictions = {}
        for property_name, model in self.models.items():
            predictions[property_name] = model.predict(features)
        
        return predictions
    
    def get_nutrient_recommendation(self, 
                                      soil_properties: dict,
                                      crop_type: str = 'corn') -> dict:
        """
        Generate fertilizer recommendations based on soil analysis.
        Uses crop-specific algorithms (adapted from university extension guidelines).
        """
        recommendations = {}
        
        if crop_type == 'corn':
            # Nitrogen recommendation (kg/ha)
            n_rate = max(0, (soil_properties.get('yield_goal', 12) * 22 
                           - soil_properties.get('soil_nitrogen', 50) * 0.5
                           - soil_properties.get('organic_matter', 2) * 5))
            recommendations['nitrogen'] = n_rate
            
            # Phosphorus recommendation
            p_level = soil_properties.get('phosphorus', 30)
            if p_level < 15:
                recommendations['phosphorus'] = 60
            elif p_level < 30:
                recommendations['phosphorus'] = 30
            else:
                recommendations['phosphorus'] = 0
            
            # Potassium recommendation
            k_level = soil_properties.get('potassium', 150)
            if k_level < 100:
                recommendations['potassium'] = 80
            elif k_level < 200:
                recommendations['potassium'] = 40
            else:
                recommendations['potassium'] = 0
        
        return recommendations
```

---

## Supply Chain Optimization

### Farm-to-Table Traceability with Blockchain + AI

```python
from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np

class SupplyChainOptimizer:
    """
    AI-powered agricultural supply chain optimization.
    Integrates blockchain verification with ML-based demand forecasting.
    """
    
    def __init__(self):
        self.demand_model = None  # Trained demand forecasting model
        self.quality_model = None  # Quality degradation prediction
    
    def optimize_routing(self, 
                          orders: List[Dict],
                          inventory: Dict,
                          fleet: Dict,
                          traffic_data: Dict) -> Dict:
        """
        Optimize delivery routing for perishable goods.
        Uses a variant of the Vehicle Routing Problem with Time Windows (VRPTW)
        with perishability constraints.
        """
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        n_orders = len(orders)
        n_vehicles = len(fleet)
        
        # Distance matrix (could use real road distances)
        locations = [o['location'] for o in orders]
        distance_matrix = self._compute_distance_matrix(locations)
        
        # Time windows considering perishability
        time_windows = []
        for o in orders:
            latest_delivery = o['harvest_time'] + timedelta(hours=o['shelf_life_hours'])
            time_windows.append((o['earliest_delivery'], latest_delivery))
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            n_orders, n_vehicles, 0  # 0 = depot
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_idx, to_idx):
            return distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time window constraints
        def time_callback(from_idx, to_idx):
            return distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30,  # Slack
            24 * 60,  # Max time per vehicle
            True,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Set time windows
        for order_idx in range(1, n_orders):
            start, end = time_windows[order_idx - 1]
            time_dimension.CumulVar(order_idx).SetRange(
                int(start.timestamp()), int(end.timestamp())
            )
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            routes = []
            for vehicle_id in range(n_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    order_idx = manager.IndexToNode(index)
                    if order_idx > 0:  # Skip depot
                        route.append(orders[order_idx - 1])
                    index = solution.Value(routing.NextVar(index))
                if route:
                    routes.append({
                        'vehicle_id': vehicle_id,
                        'orders': route,
                        'total_distance': solution.ObjectiveValue()
                    })
            
            return {'routes': routes, 'total_distance': solution.ObjectiveValue()}
        
        return {'routes': [], 'total_distance': float('inf')}
    
    def _compute_distance_matrix(self, locations: List[tuple]) -> np.ndarray:
        """Haversine distance matrix between locations."""
        n = len(locations)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                lat1, lon1 = np.radians(locations[i])
                lat2, lon2 = np.radians(locations[j])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                matrix[i][j] = 6371 * c  # Earth radius in km
        return matrix
```

---

## Case Studies

### John Deere's AI Ecosystem

John Deere has transformed from a traditional farm equipment manufacturer into an AI-powered agricultural technology company. Key AI implementations:

1. **See & Spray Technology**: Acquired through Blue River Technology (2017). Uses computer vision to distinguish crops from weeds and apply herbicide only where needed, reducing herbicide use by 90%.

2. **AutoTrac Guidance**: GPS-based automated steering with sub-2.5cm accuracy using RTK (Real-Time Kinematic) correction. Uses Kalman filtering for sensor fusion of GPS, IMU, and wheel encoders.

3. **HarvestLab 3000**: Near-infrared (NIR) sensor mounted on harvesters that measures grain moisture, protein, starch, and oil content in real time. Uses partial least squares regression (PLSR) models trained on thousands of samples.

4. **Operations Center**: Cloud platform that aggregates data from 500,000+ connected machines. Uses ML for predictive maintenance, fuel optimization, and field analytics.

**Technical Architecture**:

```python
class SeeAndSpraySystem:
    """
    Simplified architecture of John Deere's See & Spray system.
    """
    
    def __init__(self):
        # 36 cameras (17 front, 19 rear) capturing at 20 fps
        self.camera_count = 36
        self.inference_fps = 20
        
        # Neural network: custom lightweight CNN (MobileNet-derived)
        # Runs on NVIDIA Jetson AGX Orin (275 TOPS)
        self.model = self._load_weed_cnn()
    
    def _load_weed_cnn(self):
        """Load optimized weed detection model."""
        class WeedCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Depthwise separable convolutions for speed
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(),
                    self._ds_conv(32, 64, 1),
                    self._ds_conv(64, 128, 2),
                    self._ds_conv(128, 128, 1),
                    self._ds_conv(128, 256, 2),
                )
                self.classifier = nn.Conv2d(256, 2, 1)  # Crop vs Weed
                
            def _ds_conv(self, ch_in, ch_out, stride):
                return nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, 3, stride, 1, groups=ch_in),
                    nn.BatchNorm2d(ch_in),
                    nn.ReLU6(),
                    nn.Conv2d(ch_in, ch_out, 1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU6(),
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)
        
        model = WeedCNN()
        model.load_state_dict(torch.load('see_and_spray_weights.pth'))
        model.eval()
        return model
    
    def process_frame(self, frame):
        """Process a single camera frame and trigger spray nozzles."""
        # Preprocess
        # ... frame normalization and resizing
        
        # Inference
        with torch.no_grad():
            output = self.model(frame)
        
        # Generate nozzle activation map
        # Each nozzle covers ~4cm width; 36 nozzles across 12m boom
        spray_map = torch.argmax(output, dim=0)
        
        # Send to PLC for nozzle actuation (latency < 100ms)
        return spray_map
```

### Blue River Technology / See & Spray

Acquired by John Deere in 2017 for $305M, Blue River Technology developed the See & Spray system that uses computer vision to detect individual plants and spray only weeds.

**Key Technical Details**:
- **Camera System**: 36 cameras mounted on a 120-foot spray boom, capturing images at 20 Hz
- **Edge Compute**: NVIDIA Jetson AGX Orin (275 TOPS) for real-time inference
- **Model Architecture**: Custom MobileNet-variant with depthwise separable convolutions
- **Training Data**: 5+ million labeled plant images
- **Inference Latency**: < 100ms from image capture to nozzle actuation
- **Herbicide Reduction**: 90% reduction in herbicide use
- **Precision**: Individual nozzle control at 4-inch resolution

### CropX Soil Intelligence

CropX provides soil sensing and analytics for precision irrigation. Their system:

1. **Soil Sensors**: Wireless soil moisture, temperature, electrical conductivity sensors at multiple depths
2. **Cloud ML**: Proprietary algorithms for irrigation recommendations
3. **Integration**: API integration with major irrigation controllers (Valmont, Lindsay, Netafim)

**Technical Approach**: CropX uses a hybrid physics-ML model that combines:
- **Richards Equation** (physics-based soil water movement)
- **Random Forest** for local calibration
- **Transfer Learning** to adapt to new fields with minimal data

```python
class CropXIrrigationModel:
    """
    Simplified representation of CropX's hybrid modeling approach.
    """
    
    def __init__(self):
        self.physics_model = self._build_richards_solver()
        self.ml_corrector = RandomForestRegressor(
            n_estimators=100, max_depth=8
        )
    
    def _build_richards_solver(self):
        """Simplified 1D Richards equation solver for soil water movement."""
        # Uses van Genuchten-Mualem soil hydraulic model
        pass
    
    def predict_irrigation(self, 
                           soil_data: dict,
                           weather_data: dict,
                           historical_accuracy: dict = None) -> dict:
        """
        Combine physics model with ML correction.
        """
        # Physics-based prediction
        physics_prediction = self.physics_model.solve(
            soil_data, weather_data
        )
        
        # ML correction (trained on sensor feedback)
        features = np.array([
            soil_data['soil_moisture'],
            soil_data['electrical_conductivity'],
            weather_data['temperature_avg'],
            weather_data['precipitation_7day'],
            physics_prediction['drainage_rate']
        ]).reshape(1, -1)
        
        correction = self.ml_corrector.predict(features)[0]
        
        corrected_prediction = physics_prediction['soil_moisture'] + correction
        
        return {
            'recommended_irrigation': max(0, corrected_prediction),
            'physics_baseline': physics_prediction['soil_moisture'],
            'ml_correction': correction,
            'confidence': self._estimate_confidence(historical_accuracy)
        }
```

### Climate FieldView Platform

Bayer's Climate FieldView is the most widely adopted digital agriculture platform, covering 200+ million acres across 23 countries.

**AI Capabilities**:
1. **Field Health Imagery**: Automated NDVI analysis from satellite imagery
2. **Variable Rate Seeding**: ML models optimize seed populations based on soil type, historical yield, and topography
3. **Nitrogen Advisor**: AI recommends split nitrogen applications based on weather forecasts and crop growth stage
4. **Harvest Monitoring**: Real-time yield monitoring with spatial analytics

---

## Cross-References

This document intersects with several other domains in the AI Applications series:

- **[04-Manufacturing-AI.md](04-Manufacturing-AI.md)**: Predictive maintenance techniques for agricultural machinery share architectures with industrial equipment monitoring. The autoencoder-based anomaly detection approach in manufacturing is directly applicable to irrigation pump monitoring and harvester health tracking.

- **[06-Retail-AI.md](06-Retail-AI.md)**: Demand forecasting for agricultural commodities uses similar time-series architectures (LSTM, Transformer) as retail demand prediction. Supply chain optimization for perishables shares Vehicle Routing Problem formulations with retail logistics.

- **[07-Media-Entertainment-AI.md](07-Media-Entertainment-AI.md)**: Computer vision architectures for plant disease detection (ResNet, EfficientNet, U-Net) are the same model families used in content moderation and image generation. Style transfer techniques can be adapted for synthetic data generation in agricultural training datasets.

- **[05-Education-AI.md](05-Education-AI.md)**: Bayesian Knowledge Tracing used in intelligent tutoring systems is conceptually similar to soil nutrient depletion modeling — both track a latent state (knowledge/nutrients) that evolves with interventions (lessons/fertilizer).

- **[03-Finance-AI.md](03-Finance-AI.md)**: Crop yield prediction models are used in agricultural insurance (parametric insurance products). The risk management frameworks from finance apply to crop price hedging and weather derivative pricing.

---

## Summary & Conclusion

This document has provided a deep technical exploration of AI applications in agriculture and food technology, covering:

1. **Precision Agriculture**: Drone and satellite imagery analysis, vegetation indices, variable rate technology for targeted input application.

2. **Crop Yield Prediction**: LSTM, Transformer, and hybrid CNN-RNN architectures that model the complex spatio-temporal dynamics of crop growth.

3. **Plant Disease Detection**: CNN-based classification (ResNet, EfficientNet) and U-Net segmentation for disease severity assessment, plus mobile deployment techniques.

4. **Automated Irrigation**: Reinforcement learning (PPO, SAC) for water optimization, IoT sensor fusion with Kalman filtering, and rule-based fallback systems.

5. **Livestock Monitoring**: Multi-task computer vision models for health assessment, accelerometer-based behavior analysis, and anomaly detection.

6. **Soil Analysis**: Hyperspectral sensing for rapid soil property prediction and AI-driven nutrient management recommendations.

7. **Supply Chain Optimization**: Route optimization for perishables, cold chain monitoring, and blockchain-integrated traceability.

The convergence of cheap sensors, edge AI hardware, and advanced deep learning architectures is transforming agriculture from a traditional, intuition-based practice into a data-driven science. The most successful systems combine physics-based models (soil water dynamics, crop growth models) with machine learning corrections, leveraging the interpretability of the former and the pattern-recognition power of the latter.

Key challenges that remain include:
- **Causality**: Moving beyond correlation-based predictions to causal models that can generalize across changing climates
- **Data Integration**: Fusing heterogeneous data sources (satellite, drone, IoT, weather, soil) into unified representations
- **Edge Deployment**: Running complex models on resource-constrained devices in remote fields
- **Accessibility**: Making AI tools affordable and usable for smallholder farmers who produce most of the world's food
