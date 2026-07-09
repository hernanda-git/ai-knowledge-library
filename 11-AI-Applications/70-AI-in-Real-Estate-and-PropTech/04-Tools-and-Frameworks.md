# Tools and Frameworks — AI in Real Estate and PropTech

> A practical catalog of the vendors, open-source libraries, and platforms used to build AI for real estate. Organized by capability (valuation, computer vision, NLP, recommendation, geospatial, agent infra) with selection guidance and integration notes. Cross-referenced to sibling library categories.

---

## Table of Contents

1. [Capability → Tool Map](#capability--tool-map)
2. [Valuation and AVM Tooling](#valuation-and-avm-tooling)
3. [Computer-Vision Libraries and Vendors](#computer-vision-libraries-and-vendors)
4. [NLP / LLM Frameworks](#nlp--llm-frameworks)
5. [Recommendation and Search](#recommendation-and-search)
6. [Geospatial and Mapping](#geospatial-and-mapping)
7. [Data Providers and Enrichment APIs](#data-providers-and-enrichment-apis)
8. [Agent and Workflow Infrastructure](#agent-and-workflow-infrastructure)
9. [MLOps and Observability](#mlops-and-observability)
10. [Databases and Feature Stores](#databases-and-feature-stores)
11. [Build vs. Buy Decision Tree](#build-vs-buy-decision-tree)
12. [Integration Reference Architecture](#integration-reference-architecture)
13. [Cross-Category References](#cross-category-references)

---

## Capability → Tool Map

| Capability | Build with (OSS) | Buy (vendors) |
|------------|------------------|---------------|
| AVM / tabular ML | LightGBM, XGBoost, scikit-learn, PyTorch Tabular | HouseCanary, CoreLogic, Quantarium |
| Computer vision | torchvision, YOLO, SegFormer, transformers | Restb.ai, Cape Analytics, HOVER |
| NLP / lease abstraction | LangChain, LlamaIndex, Haystack, spaCy | LeasePilot, Latch, custom |
| Recommendation | implicit, TensorFlow Recommenders, FAISS | portal-native |
| Geospatial | GDAL, geopandas, Leaflet, Kepler.gl | Mapbox, Esri, UrbanLogiq |
| Data enrichment | — | Cherre, ATTOM, Realtor Property API |
| Agents / workflows | LangGraph, Temporal, Prefect | Rechat, Structurely |
| MLOps | MLflow, Evidently, Metaflow | cloud ML platforms |
| Vector search | FAISS, pgvector, Weaviate, Qdrant | managed vector DBs |

---

## Valuation and AVM Tooling

### Open-source
- **LightGBM / XGBoost / CatBoost** — gradient-boosted trees; the production AVM backbone.
- **scikit-learn** — hedonic regression, baselines, pipelines.
- **statsmodels** — interpretable econometric models (useful for explainability).
- **Prophet / NeuralProphet / Darts / PyTorch Forecasting** — time-series rent/price forecasting.
- **skforecast / gluonts** — probabilistic forecasting.

### Commercial AVM vendors
| Vendor | Focus |
|--------|-------|
| HouseCanary | AVM + market analytics API |
| CoreLogic | ClearAVM, property data |
| Quantarium | QVM, computer vision |
| Red Bell Real Estate | AVM + hybrid appraisal |
| Zillow Zestimate | Consumer-facing AVM |

### When to build vs. buy
- Build when you have proprietary inspection/transaction data and need differentiated accuracy.
- Buy when you need broad coverage fast and lack labeled data.

---

## Computer-Vision Libraries and Vendors

### Open-source
- **torchvision** — detection/segmentation models and transforms.
- **Ultralytics YOLOv8/v9** — fast detection; good for edge (see `../62-Edge-AI-and-On-Device-Inference/`).
- **Hugging Face transformers** — DETR, SegFormer, Mask2Former, vision foundation models.
- **MMDetection / MMSegmentation** — research-grade detection/segmentation.
- **Detectron2** — Facebook's detection/segmentation framework.

### Vendors
| Vendor | Capability |
|--------|-----------|
| Restb.ai | Property photo CV APIs (room, condition, features) |
| Cape Analytics | Exterior/roof CV from aerial |
| HOVER | 3D exterior + measurements |
| GeoPhy | CRE valuation + CV |
| Nearmap / Ecopia | High-res aerial + vector maps |

---

## NLP / LLM Frameworks

- **LangChain / LlamaIndex** — RAG orchestration for lease/title Q&A (see `../04-RAG/`).
- **Haystack** — production RAG pipelines.
- **spaCy / GLiNER** — fast NER and clause extraction.
- **LayoutLMv3 / Donut (Hugging Face)** — layout-aware document understanding for scanned leases.
- **Pydantic** — structured extraction schemas.
- **Guidance / Outlines** — constrained LLM generation.

```python
# Minimal RAG over a lease corpus with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
docs = SimpleDirectoryReader("./leases/").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(response_mode="compact")
print(query_engine.query("Summarize renewal and break-lease terms."))
```

---

## Recommendation and Search

- **implicit / Cornac** — collaborative filtering.
- **TensorFlow Recommenders (TFRS)** — two-tower DNN recommenders.
- **FAISS / annoy / ScaNN** — ANN vector search for listing matching.
- **pgvector / Weaviate / Qdrant** — vector DBs with metadata filtering (see `../37-AI-Native-Databases/`).
- **OpenSearch / Elasticsearch** — hybrid keyword+vector search.

---

## Geospatial and Mapping

- **GDAL / rasterio** — raster (imagery) processing.
- **geopandas / shapely** — vector GIS operations.
- **Leaflet / Mapbox GL / Kepler.gl** — interactive maps and visualizations.
- **H3 / S2** — geospatial indexing for aggregation.
- **Esri ArcGIS / UrbanLogiq** — enterprise GIS analytics.

```python
import geopandas as gpd, h3

gdf = gpd.read_file("parcels.geojson")
# assign H3 cell (resolution 8) to each parcel for aggregation
gdf["h3"] = gdf.geometry.centroid.apply(lambda p: h3.latlng_to_cell(p.y, p.x, 8))
agg = gdf.groupby("h3")["sale_price"].mean()
```

---

## Data Providers and Enrichment APIs

| Provider | What |
|----------|------|
| Cherre | Real-estate data connectivity + enrichment |
| ATTOM Data | Property attributes, AVM, flood, environmental |
| Realtor.com / Zillow APIs | Listing feeds (licensing-restricted) |
| Redfin / MLS | Listing data (broker agreements) |
| County open data | Deeds, permits, tax assessor |
| Freddie Mac / FHFA | Macro indices (HPI) |
| NOAA / FEMA | Flood zones, climate |

> Data licensing is the single biggest practical constraint in PropTech. Verify MLS/portal terms before building on feeds (see `../43-AI-Data-Provenance-and-Content-Authenticity/`).

---

## Agent and Workflow Infrastructure

- **LangGraph** — stateful agent graphs (lead routing, offer bots).
- **Temporal / Prefect / Airflow** — durable, long-running transaction workflows (see `../31-AI-Workflow-Orchestration-and-Durable-Execution/`).
- **CrewAI / AutoGen** — multi-agent coordination.
- **Rechat / Structurely** — turnkey agent copilots (lead nurture, CRM).

```python
# LangGraph sketch: lead -> prequal -> offer
from langgraph.graph import StateGraph
sg = StateGraph(LeadState)
sg.add_node("prequal", assess_prequal)
sg.add_node("offer", price_offer)
sg.add_edge("prequal", "offer")
sg.set_entry_point("prequal")
app = sg.compile()
```

---

## MLOps and Observability

- **MLflow** — experiment tracking, model registry.
- **Evidently / WhyLogs** — data/prediction drift monitoring.
- **Metaflow / Kubeflow** — pipelines.
- **Prometheus / Grafana / OpenTelemetry** — service + LLM tracing (see `../20-Agent-Infrastructure-and-Observability/`).
- **Arize / LangSmith** — LLM observability.

---

## Databases and Feature Stores

- **Postgres + pgvector** — relational + vector in one (see `../37-AI-Native-Databases/`).
- **Neo4j** — ownership/comp graphs (see `../69-GraphRAG-and-Knowledge-Graph-Retrieval/`).
- **DuckDB / ClickHouse** — analytics over feature tables.
- **Feast** — feature store for shared AVM/CV/recsys features.
- **Redis / DynamoDB** — online low-latency feature serving.

---

## Build vs. Buy Decision Tree

```
                 Do you have proprietary data (inspections/transactions)?
                 /                                          \
              YES                                           NO
               |                                             |
   Is accuracy a differentiator?                       Buy a vendor API
        /              \                                  (HouseCanary, Restb.ai)
      YES              NO
       |                |
  Build core +      Use vendor for core,
  buy commodity      build UX/agent layer
  (CV, AVM)          on top
```

General guidance: buy commodity capabilities (broad AVM, standard CV), build where you have unique data or workflow IP.

---

## Integration Reference Architecture

```
  Data (MLS, imagery, deeds, IoT)
        │  (ETL + contracts)
        ▼
  Feature store (Feast) ──▶ AVM (LightGBM) + CV (YOLO) + Recsys (TFRS)
        │                        │
        │                  Vector DB (pgvector) + KG (Neo4j)
        │                        │
        ▼                        ▼
  Agent/Workflow layer (LangGraph + Temporal)
        │
        ▼
  Portal / API / Copilot  (observability + guardrails)
```

Cross-cutting: MLOps (MLflow/Evidently), security (see `../18-Agent-Security-and-Trust/`), cost control (see `../59-AI-Agent-Financial-Governance-and-Cost-Control/`).

---

## Cross-Category References

- `../04-RAG/`, `../69-GraphRAG-and-Knowledge-Graph-Retrieval/` — lease/title + comp graphs.
- `../03-Agents/`, `../31-AI-Workflow-Orchestration-and-Durable-Execution/` — transaction agents.
- `../50-Multimodal-AI/`, `../62-Edge-AI-and-On-Device-Inference/` — CV, edge.
- `../37-AI-Native-Databases/` — vector/relational storage.
- `../56-MLOps-and-AI-Platform-Engineering/`, `../20-Agent-Infrastructure-and-Observability/` — ops.
- `../53-AI-Model-Cascading-and-Multi-Model-Orchestration/` — routing LLM calls.
- `../43-AI-Data-Provenance-and-Content-Authenticity/` — data licensing/lineage.
- `../67-AI-in-Finance-and-Financial-Services/` — mortgage tooling overlap.
