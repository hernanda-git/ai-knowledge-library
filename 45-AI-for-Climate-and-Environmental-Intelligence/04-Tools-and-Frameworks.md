# 04 - AI for Climate & Environmental Intelligence: Tools and Frameworks

> **Category:** 45-AI-for-Climate-and-Environmental-Intelligence
> **Last updated:** July 1, 2026
> **Cross-references:** `03-Agents/03-Agentic-Frameworks.md`, `13-Top-Demand/06-RAG-Retrieval-Systems.md`, `33-AI-Native-Software-Development/02-Coding-Agents-and-AI-Pair-Programming.md`

---

## Table of Contents

1. [AI Weather Forecasting Frameworks](#1-ai-weather-forecasting-frameworks)
2. [Climate Data Platforms](#2-climate-data-platforms)
3. [Satellite Data Processing](#3-satellite-data-processing)
4. [Carbon and Emissions Tools](#4-carbon-and-emissions-tools)
5. [Agriculture and Food Systems](#5-agriculture-and-food-systems)
6. [Climate Risk Analytics](#6-climate-risk-analytics)
7. [Open-Source Models and Weights](#7-open-source-models-and-weights)
8. [Cloud Platforms for Climate AI](#8-cloud-platforms-for-climate-ai)
9. [Visualization and Decision Support](#9-visualization-and-decision-support)
10. [Getting Started Guide](#10-getting-started-guide)

---

## 1. AI Weather Forecasting Frameworks

### 1.1 NVIDIA Earth-2

NVIDIA's comprehensive platform for AI weather and climate simulation:
- **FourCastNet**: AFNO-based weather prediction model
- **CorrDiff**: AI super-resolution for weather data
- **Diffusion-based ensemble**: Generative ensemble forecasting
- **Digital twin platform**: Real-time Earth system monitoring

```bash
pip install earth2-studio
```

### 1.2 GraphCast (DeepMind)

GraphCast is available through the DeepMind Weather API and as open weights:
- **Input**: 37 atmospheric variables at 13 pressure levels
- **Output**: Same variables at T+6h
- **Resolution**: 0.25 degree (1440x721 global grid)
- **Speed**: <1 minute on a single TPU

### 1.3 Pangu-Weather (Huawei)

Available as open-source on GitHub:
- **Architecture**: 3D Earth-Specific Transformer
- **Variables**: Geopotential, temperature, humidity, wind (u, v)
- **Pressure levels**: 13 levels from 1000hPa to 100hPa

### 1.4 NeuralGCM (Google Research)

Hybrid physics-AI climate model:
- **Component**: Replaces parameterized physics with neural networks
- **Capability**: Can simulate decades of climate
- **Resolution**: ~1.4 degree (configurable)

---

## 2. Climate Data Platforms

### 2.1 Reanalysis Data Access

| Platform | Dataset | Access Method | Size |
|----------|---------|--------------|------|
| **Copernicus CDS** | ERA5, ERA5-Land | API, web interface | ~5TB |
| **NASA GES DISC** | MERRA-2, GPM | API, OPeNDAP | ~2TB |
| **NCAR RDA** | Various reanalysis | Web, FTP | Varies |
| **Google Earth Engine** | ERA5, MODIS, Landsat | JS/Python API | Cloud-hosted |

### 2.2 Python Data Access

```python
import cdsapi

client = cdsapi.Client()
client.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['2m_temperature', '10m_u_component_of_wind'],
        'year': '2025',
        'month': '06',
        'day': ['01', '15'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
    },
    'era5_data.nc'
)
```

### 2.3 Satellite Data Platforms

| Platform | Data Source | Use Case |
|----------|-----------|----------|
| **Google Earth Engine** | Landsat, Sentinel, MODIS | Global scale analysis |
| **Copernicus Open Access Hub** | Sentinel-1/2/3 | European focus |
| **USGS EarthExplorer** | Landsat | Global, historical |
| **NASA Earthdata** | MODIS, VIIRS, SMAP | US/global |
| **Sentinel Hub** | All Sentinel data | Real-time access |

---

## 3. Satellite Data Processing

### 3.1 Geospatial ML Libraries

| Library | Focus | Key Features |
|---------|-------|-------------|
| **Rasterio** | Raster I/O | Read/write geospatial rasters |
| **GDAL/OGR** | Geospatial data | Format translation |
| **xarray** | N-dimensional arrays | Labeled dimensions, lazy loading |
| **GeoPandas** | Vector data | Spatial operations |
| **TorchGeo** | Geospatial ML | PyTorch datasets for satellite imagery |

### 3.2 TorchGeo for Climate Applications

```python
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader

dataset = Sentinel2(root='/data/sentinel2', bands=['B04', 'B03', 'B02', 'B08'], res=10)
sampler = RandomGeoSampler(dataset, size=256, length=10000)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

for batch in dataloader:
    images = batch['image']
    masks = batch['mask']
```

---

## 4. Carbon and Emissions Tools

### 4.1 Emissions Monitoring Platforms

| Tool | Organization | Focus | Access |
|------|-------------|-------|--------|
| **Climate TRACE** | Coalition | Global emissions from satellite | Free API |
| **GHGSat** | GHGSat Inc. | Methane point sources | Commercial |
| **Global Forest Watch** | WRI | Deforestation monitoring | Free |
| **Carbon Mapper** | Nonprofit | CO2 and methane from satellites | Research |

### 4.2 Carbon Footprint Calculator

```python
class CarbonFootprintCalculator:
    def __init__(self):
        self.emission_factors = {
            'electricity_kwh': 0.4,    # kg CO2 per kWh (US average)
            'natural_gas_therm': 5.3,
            'gasoline_gallon': 8.9,
            'flight_mile': 0.255,
            'beef_kg': 27.0,
        }

    def calculate(self, activities):
        total = 0
        for activity, amount in activities.items():
            if activity in self.emission_factors:
                total += amount * self.emission_factors[activity]
        return total
```

### 4.3 MRV Tools

| Tool | Type | Application |
|------|------|-------------|
| **Verra VCS** | Standard | Voluntary carbon credit verification |
| **Gold Standard** | Standard | Sustainable development credits |
| **Pachama** | AI platform | Forest carbon credit verification |
| **Sylvera** | AI platform | Carbon credit ratings |

---

## 5. Agriculture and Food Systems

### 5.1 Precision Agriculture Platforms

| Platform | Focus | Access |
|----------|-------|--------|
| **Climate FieldView** | Crop analytics | Commercial |
| **Planet Labs** | Satellite imagery (daily 3m) | Commercial |
| **OpenET** | Evapotranspiration | Free |
| **Sentinel Hub** | EO data access | Free tier |

### 5.2 Vegetation Index Computation

```python
import numpy as np

def compute_vegetation_indices(bands):
    red = bands['B04'].astype(float)
    nir = bands['B08'].astype(float)
    swir = bands['B11'].astype(float)
    return {
        'NDVI': (nir - red) / (nir + red + 1e-10),
        'EVI': 2.5 * (nir - red) / (nir + 6*red - 7.5*red + 1),
        'SAVI': 1.5 * (nir - red) / (nir + red + 0.5),
        'NDWI': (nir - swir) / (nir + swir + 1e-10),
    }
```

---

## 6. Climate Risk Analytics

### 6.1 Risk Assessment Platforms

| Platform | Focus | Customers |
|----------|-------|-----------|
| **Jupiter Intelligence** | Physical climate risk | Financial institutions |
| **Cervest** | Climate risk scoring | Investors, insurers |
| **RMS (Moody's)** | Catastrophe modeling | Insurance/reinsurance |
| **Four Twenty Seven** | Transition + physical risk | Governments, investors |

### 6.2 Risk Calculation

```python
class ClimateRiskCalculator:
    def __init__(self, hazard_model, exposure_data, vulnerability_functions):
        self.hazard = hazard_model
        self.exposure = exposure_data
        self.vulnerability = vulnerability_functions

    def calculate_expected_loss(self, asset, scenario='SSP2-4.5'):
        hazard_intensity = self.hazard.predict(asset.location, scenario, lead_year=2050)
        asset_value = self.exposure.get_value(asset.id)
        damage_ratio = self.vulnerability[asset.type](hazard_intensity)
        expected_loss = asset_value * damage_ratio * hazard_intensity.annual_probability
        return {'expected_annual_loss': expected_loss, 'damage_ratio': damage_ratio}
```

---

## 7. Open-Source Models and Weights

| Model | Repository | License | Framework |
|-------|-----------|---------|-----------|
| **GraphCast** | github.com/google-deepmind/graphcast | Apache 2.0 | JAX |
| **Pangu-Weather** | github.com/huawei-noah/WeatherPredictor | Apache 2.0 | PyTorch |
| **FourCastNet** | github.com/NVlabs/FourCastNet | BSD 3-Clause | PyTorch |
| **ClimaX** | github.com/microsoft/climax | MIT | PyTorch |
| **Aurora** | github.com/microsoft/aurora | MIT | PyTorch |
| **NeuralGCM** | github.com/google-research/neuralgcm | Apache 2.0 | JAX |
| **Prithvi** | github.com/NASA-IMPACT/Prithvi-EO | Apache 2.0 | PyTorch |

---

## 8. Cloud Platforms for Climate AI

| Platform | Key Features | Climate Tools | Free Tier |
|----------|-------------|---------------|-----------|
| **Google Earth Engine** | Cloud geospatial analysis | ERA5, MODIS, Landsat | Generous |
| **AWS (SageMaker)** | ML training/inference | Open Data on S3 | Limited |
| **Azure Planetary Computer** | Geospatial ML | STAC catalog | Generous |
| **NVIDIA DGX Cloud** | GPU clusters | Earth-2 access | Limited |

### 8.1 Google Earth Engine Example

```javascript
var modis = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterDate('2025-06-01', '2025-06-30')
    .select('NDVI');
var ndvi = modis.mean().multiply(0.0001);
var visParams = {min: 0.0, max: 0.9, palette: ['red', 'yellow', 'green']};
Map.centerObject(ndvi, 3);
Map.addLayer(ndvi, visParams, 'June 2025 NDVI');
```

---

## 9. Visualization and Decision Support

| Library | Type | Best For |
|---------|------|----------|
| **Folium** | Interactive maps | Web-based exploration |
| **Cartopy** | Static maps | Publication-quality figures |
| **Kepler.gl** | Large-scale viz | Millions of data points |
| **deck.gl** | WebGL maps | Browser-based rendering |

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_global_temperature(data, title='Global Temperature Anomaly'):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))
    im = ax.pcolormesh(data.longitude, data.latitude, data.values,
                       transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-3, vmax=3)
    ax.coastlines()
    plt.colorbar(im, ax=ax, label='Temperature Anomaly (K)', shrink=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    plt.savefig('temperature_anomaly.png', dpi=150, bbox_inches='tight')
```

---

## 10. Getting Started Guide

### 10.1 Recommended Learning Path

**Week 1-2: Foundations**
- Set up Python with xarray, netCDF4, cartopy
- Download ERA5 sample data via CDS API
- Reproduce a basic analysis

**Week 3-4: Satellite Data**
- Access Sentinel-2 via Copernicus or Google Earth Engine
- Compute vegetation indices
- Build a simple crop type classifier

**Week 5-6: AI Weather Models**
- Install and run FourCastNet
- Compare predictions against ERA5
- Understand evaluation metrics

**Week 7-8: Applications**
- Choose a focus area (agriculture, disaster, climate risk)
- Build an end-to-end pipeline

### 10.2 Essential Python Stack

```bash
pip install numpy scipy pandas xarray netCDF4 h5py
pip install rasterio geopandas cartopy shapely pyproj
pip install scikit-learn torch torchvision
pip install planetary-computer pystac-client earthengine-api
pip install matplotlib seaborn plotly folium
pip install climpred xesmf esmpy
```

### 10.3 Key Datasets

| Dataset | Size | Resolution | Use Case |
|---------|------|-----------|----------|
| **ERA5 (monthly)** | ~10GB | 0.25 deg | Climate analysis |
| **ERA5-Land (hourly)** | ~50GB | 0.1 deg | Land surface processes |
| **MODIS NDVI** | ~100GB | 250m | Vegetation monitoring |
| **Sentinel-2 L2A** | TB scale | 10m | Land cover classification |

---

## Related Library Documents

- `03-Agents/03-Agentic-Frameworks.md` -- Framework patterns for climate monitoring
- `13-Top-Demand/06-RAG-Retrieval-Systems.md` -- RAG for climate knowledge
- `37-AI-Native-Databases/01-Overview.md` -- Data infrastructure for environmental data

---

*This document is part of the AI Base Knowledge Library -- an open, structured collection of AI knowledge for practitioners, researchers, and learners.*
