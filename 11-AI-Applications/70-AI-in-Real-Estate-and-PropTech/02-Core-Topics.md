# Core Topics — AI in Real Estate and PropTech

> Deep treatment of the core AI capabilities that power modern real estate: automated valuation models (AVMs), computer-vision property inspection, NLP for leases and titles, recommendation/matching, and predictive analytics. Each topic includes the problem, the data, the model, and production considerations, with cross-references to sibling library categories.

---

## Table of Contents

1. [Automated Valuation Models (AVMs)](#automated-valuation-models-avms)
2. [Computer-Vision Property Inspection](#computer-vision-property-inspection)
3. [NLP for Leases, Titles, and Contracts](#nlp-for-leases-titles-and-contracts)
4. [Recommendation and Matching Engines](#recommendation-and-matching-engines)
5. [Predictive Analytics: Pricing, Demand, Risk](#predictive-analytics-pricing-demand-risk)
6. [Generative AI for Listings and Marketing](#generative-ai-for-listings-and-marketing)
7. [Geospatial and Graph Methods](#geospatial-and-graph-methods)
8. [Data Foundations](#data-foundations)
9. [Evaluation and Metrics](#evaluation-and-metrics)
10. [Cross-Category References](#cross-category-references)

---

## Automated Valuation Models (AVMs)

### The problem
Accurate, instant, and cheap property valuation at scale. Traditional appraisals cost $300–$600 and take days; an AVM returns an estimate in milliseconds.

### Data inputs
| Input | Source |
|-------|--------|
| Comparable sales | MLS, county records, portals |
| Property attributes | Tax assessor, listing feeds, imagery |
| Location features | School ratings, transit, crime, noise |
| Macro signals | Mortgage rates, employment, inventory |
| Imagery | Exterior/interior photos, aerial |

### Modeling approaches
- **Hedonic regression** — price as function of attributes (bedrooms, sqft, location).
- **Gradient-boosted trees** (XGBoost, LightGBM, CatBoost) — the production workhorse for AVMs; handle non-linearities and missingness.
- **Neural nets / embeddings** — for image-aware valuation (photo quality, condition).
- **Ensemble / stacked models** — blend multiple estimators; reduce error.
- **Geospatial priors** — spatial autocorrelation via kriging or graph neighbors.

### Key metric
- **Median absolute percentage error (MAPE)** and **weighted error by segment**. Zestimate targets low-single-digit % MAPE in liquid markets; error rises in thin/unique markets.

### Production concerns
- **Coverage gaps** — rural/unique homes have few comps; fall back to hedonic + human review.
- **Drift** — rate shocks and inventory swings require frequent retraining (see `../56-MLOps-and-AI-Platform-Engineering/`).
- **Explainability** — regulators and users want "why this price" (see `../64-AI-Model-Explainability-and-XAI/`).
- **Bias** — avoid proxies for protected classes (see `../55-AI-Ethics-and-Responsible-AI/`).

```python
# Sketch: gradient-boosted AVM (features -> price)
import pandas as pd
import lightgbm as lgb

df = pd.read_parquet("listings.parquet")  # sqft, beds, baths, age, zip, ...
X = df[["sqft","beds","baths","age_yrs","median_income_zip","miles_transit"]]
y = df["sale_price"]

train, val = df.iloc[:80_000], df.iloc[80_000:]
dtrain = lgb.Dataset(X.loc[train.index], y.loc[train.index])
dval    = lgb.Dataset(X.loc[val.index],   y.loc[val.index])

params = {"objective":"regression_l1","n_estimators":2000,
          "learning_rate":0.02,"num_leaves":63,"subsample":0.8}
booster = lgb.train(params, dtrain, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(100)])

pred = booster.predict(X.loc[val.index])
mape = ( (pred - y.loc[val.index]).abs() / y.loc[val.index] ).median()
print(f"Validation MAPE: {mape:.3%}")
```

---

## Computer-Vision Property Inspection

### The problem
Turn photos (exterior/interior, drone, street-view) into structured condition and feature data used for valuation, underwriting, and maintenance.

### Tasks
| Task | Output |
|------|--------|
| Room type classification | bedroom / kitchen / bath … |
| Condition scoring | excellent → poor per surface |
| Feature detection | pool, HVAC, solar, garage, fireplace |
| Damage detection | roof wear, cracks, water stains, mold |
| Exterior segmentation | roof, siding, lawn, driveway |
| Floor-plan reconstruction | walls, openings, room polygons |

### Models
- **Classification/Detection** — EfficientNet, YOLOv8/v9, DETR.
- **Segmentation** — SegFormer, Mask2Former (roof, facade, parcel).
- **Vision Transformers / foundation models** — general encoders fine-tuned per task (see `../50-Multimodal-AI/`).
- **Generative** — diffusion inpainting to simulate renovations; floor-plan generation.

### Example: detect a pool and roof condition
```python
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

img = Image.open("aerial.jpg")
inputs = processor(images=img, return_tensors="pt")
outs = model(**inputs)
# Map scores/boxes; in production, fine-tune on labeled property imagery
# and add custom classes: pool, solar_panel, hvac_unit, roof_damage
```

### Production concerns
- **Label cost** — need labeled property datasets; active learning helps.
- **Edge deployment** — on-site inspection apps run models on-device (see `../62-Edge-AI-and-On-Device-Inference/`).
- **Generalization** — models trained in one region fail in others (architectural styles); domain adaptation needed.

---

## NLP for Leases, Titles, and Contracts

### The problem
Leases, titles, and purchase agreements are long, free-text, and jurisdiction-specific. Extracting obligations, dates, parties, and clauses is slow manual work.

### Tasks
| Task | Example |
|------|---------|
| Clause extraction | "no-pet", "break-lease fee", "renewal" |
| Party & date extraction | landlord, tenant, start/end |
| Obligation summarization | "tenant pays utilities" |
| Risk flagging | unusual penalties, non-standard terms |
| Q&A over a lease | "What is the deposit?" |

### Approaches
- **LLM + RAG** — embed lease corpus, retrieve relevant spans, answer with citations (see `../04-RAG/` and `../69-GraphRAG-and-Knowledge-Graph-Retrieval/`).
- **Fine-tuned extractors** — span classification for clause types.
- **Layout-aware models** — for scanned PDFs (LayoutLMv3, Donut).

```python
# Lease Q&A with RAG
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

docs = load_lease_pdfs("./leases/")            # split + embed
vs = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vs.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(ChatOpenAI(model="gpt-4o-mini"),
                                 retriever=retriever)
print(qa.invoke("What is the security deposit and return window?"))
```

### Production concerns
- **Hallucination** — legal text must be verified (see `../52-AI-Hallucination-Detection-and-Mitigation/`).
- **Jurisdiction** — clause meaning varies by state/country; route by geography.
- **Privacy** — tenant PII must be redacted (see `../40-AI-Data-Sovereignty-and-Privacy/`).

---

## Recommendation and Matching Engines

### The problem
Match buyers/renters to homes, and match listings to likely buyers; personalize feeds and alerts.

### Approaches
| Approach | Use |
|----------|-----|
| Collaborative filtering | "users like you viewed X" |
| Content-based | match attributes to preferences |
| Two-tower DNN | user & listing embeddings; scalable ANN search |
| LLM preference extraction | parse natural-language wants ("walkable, 2bd, <$600k") |

### Example intent parse
```python
prefs = llm.extract(
    "I want a 2-bed near transit under $600k with a yard",
    schema={"beds":int,"max_price":int,"transit":bool,"yard":bool}
)
# -> {"beds":2,"max_price":600000,"transit":True,"yard":True}
```

---

## Predictive Analytics: Pricing, Demand, Risk

- **Rent/price forecasting** — time-series (Prophet, DeepAR, TFT) per geography.
- **Demand / absorption** — inventory and days-on-market forecasting.
- **Default / credit risk** — for mortgage (overlaps `../67-AI-in-Finance-and-Financial-Services/`).
- **Tenant churn** — for multifamily retention.
- **Yield / cap-rate models** — for CRE investors.

---

## Generative AI for Listings and Marketing

- **Listing copy** — generate descriptions from attributes + photos (see `../33-AI-Native-Software-Development/` for product patterns).
- **Image enhancement** — declutter, virtually stage, sky replacement.
- **Virtual tours** — NeRF / Gaussian-splat 3D from photos.
- **Ad creative** — variations for channels; must avoid fair-housing violations in targeting copy.

---

## Geospatial and Graph Methods

- **Spatial features** — distance to transit/schools; noise; flood zone.
- **Neighborhood graphs** — comp propagation; comparable selection.
- **Ownership networks** — fraud detection, entity resolution (see `../69-GraphRAG-and-Knowledge-Graph-Retrieval/`).
- **Market graphs** — correlate submarkets for forecasting.

---

## Data Foundations

| Data | Owner / source | Notes |
|------|----------------|-------|
| MLS listings | Brokerages / portals | Licensing restricted |
| County records | Assessor / recorder | Public, messy |
| Imagery | Drone, street-view, satellites | Cost, privacy |
| Economic | BLS, Census, Freddie Mac | Macro |
| IoT | Buildings | Occupancy, energy |
| Proprietary | iBuyer inspections | Competitive moat |

Data contracts and lineage matter for auditability (see `../43-AI-Data-Provenance-and-Content-Authenticity/`).

---

## Evaluation and Metrics

| Capability | Primary metric |
|------------|----------------|
| AVM | MAPE, segment error, bias by zip |
| CV inspection | F1 per class, agreement vs appraiser |
| Lease NLP | Extraction precision/recall |
| Recommender | CTR, save-rate, match quality |
| Forecasting | sMAPE, coverage |

Always report **segmented** error — overall MAPE hides disparate performance that creates fair-housing risk (see `../58-AI-Evaluation-and-Benchmarking-at-Scale/`).

---

## Cross-Category References

- `../02-LLMs/`, `../04-RAG/` — lease/title Q&A.
- `../50-Multimodal-AI/` — vision inspection, floor plans.
- `../69-GraphRAG-and-Knowledge-Graph-Retrieval/` — comp/ownership graphs.
- `../29-Reasoning-and-Inference-Scaling/` — forecasting.
- `../55-AI-Ethics-and-Responsible-AI/`, `../64-AI-Model-Explainability-and-XAI/` — fairness, transparency.
- `../62-Edge-AI-and-On-Device-Inference/` — edge inspection.
- `../56-MLOps-and-AI-Platform-Engineering/` — retraining, monitoring.
- `../67-AI-in-Finance-and-Financial-Services/` — mortgage overlap.
