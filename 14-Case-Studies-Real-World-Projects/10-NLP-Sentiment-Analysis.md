# Case Study: NLP Sentiment Analysis System

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Tech Stack](#tech-stack)
4. [Implementation Details](#implementation-details)
5. [Metrics & Results](#metrics--results)
6. [Lessons Learned](#lessons-learned)
7. [TEMPLATE: Sentiment Analysis Project](#template-sentiment-analysis-project)

---

## Problem Statement

**Company:** Global E-Commerce Platform (200M+ monthly active users)

**Challenge:** The platform processed 100K+ customer reviews and feedback messages daily across 15 languages. Manual review was impossible at scale. Existing rule-based keyword matching achieved only 65% accuracy and couldn't detect sarcasm, nuanced sentiment, or emerging slang. The business needed:

- Real-time sentiment classification across 5 categories (Very Negative, Negative, Neutral, Positive, Very Positive)
- Aspect-based sentiment (what specifically the customer liked/disliked)
- Multi-language support (English, Spanish, Mandarin, Arabic, Hindi, Japanese +9 more)
- Integration with existing dashboards and alerting systems
- 95%+ accuracy with <200ms per review latency
- Scalable to 200K+ reviews/day with cost <$0.001 per prediction

**Business Impact of failure:**
- $5M/month in lost sales due to unaddressed negative reviews
- 72-hour response time for critical issues
- 40% customer churn attributed to unresolved feedback

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│                                                                 │
│  Customer Reviews → Kafka Topic (raw_reviews) → Stream Processor │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  Inference Pipeline                   │       │
│  │                                                      │       │
│  │  Kafka Stream (enriched) ← Preprocessor ← Language    │       │
│  │         ↓                        Detector            │       │
│  │  ┌──────────────┐  ┌──────────┐  ↓                    │       │
│  │  │  Sentiment   │  │  Aspect  │  Translation API     │       │
│  │  │  Classifier  │  │  Extractor│  (if not in model)   │       │
│  │  │  (XLM-R)     │  │  (BERT)  │                      │       │
│  │  └──────┬───────┘  └────┬─────┘                      │       │
│  │         ↓              ↓                             │       │
│  │  ┌──────────────────────────────────────┐            │       │
│  │  │      Aggregator & Post-Processor     │            │       │
│  │  └──────────────────┬───────────────────┘            │       │
│  └─────────────────────┼────────────────────────────────┘       │
│                        ↓                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                   Serving Layer                      │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │       │
│  │  │  API     │  │  Dashboard│  │  Alert Engine    │   │       │
│  │  │  (FastAPI)│  │  (Grafana)│  │  (webhook→Slack)│   │       │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │           Feedback Loop (Human-in-the-loop)   │    │       │
│  │  │  Low-confidence → Human Review → Retrain      │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Two-model architecture:** XLM-R for multilingual sentiment + fine-tuned BERT for aspect extraction
2. **Streaming inference:** Kafka + Faust stream processor for real-time processing
3. **Batched inference:** GPU batching (batch size 32) for throughput optimization
4. **Human-in-the-loop:** Reviews with confidence <0.7 flagged for manual review
5. **Feedback loop:** Reviewed data fed back into monthly retraining pipeline

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Model** | XLM-RoBERTa (XLM-R) | Large | Multilingual sentiment classification |
| **Aspect model** | BERT-base (multilingual cased) | Base | Aspect-based sentiment extraction |
| **Inference** | NVIDIA Triton Inference Server | 23.10 | GPU-accelerated model serving |
| **Streaming** | Apache Kafka | 3.5 | Ingest raw reviews |
| **Stream processor** | Faust (Python stream processing) | 1.10 | Real-time transformation |
| **API** | FastAPI | 0.104 | REST endpoints |
| **Monitoring** | Prometheus + Grafana | Latest | Dashboard & alerting |
| **Storage** | PostgreSQL | 15 | Processed results, feedback |
| **Vector DB** | Qdrant | 1.5 | Similar review retrieval |
| **Orchestration** | Kubernetes (EKS) | 1.28 | Deployment & scaling |
| **CI/CD** | GitHub Actions + ArgoCD | Latest | MLOps pipeline |
| **Feature store** | Feast | 0.34 | Store embeddings + features |

---

## Implementation Details

### Model Training

**Sentiment classifier (XLM-R Large):**

```python
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch

model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-large",
    num_labels=5,  # Very Negative, Negative, Neutral, Positive, Very Positive
    problem_type="single_label_classification"
)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

# Training configuration
training_args = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1"
}
```

**Training dataset:** 2.5M labeled reviews across 15 languages
- 1.2M English (from Amazon, Yelp, internal data)
- 200K Spanish, 180K Mandarin, 150K Arabic, 120K Hindi, 100K Japanese
- 50K-80K each for remaining languages
- Augmented with back-translation for low-resource languages

**Aspect extraction (BERT multilingual):**

```python
# Token classification for aspect terms
# Tags: B-ASPECT, I-ASPECT, B-SENTIMENT, I-SENTIMENT, O
label_list = ["O", "B-ASPECT", "I-ASPECT", "B-SENTIMENT", "I-SENTIMENT"]

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label_list)
)

# Custom CRF layer on top for sequence decoding
class BertCRF(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs.last_hidden_state)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())
```

### Triton Inference Serving

**Model configuration (sentiment_model.pbtxt):**

```
name: "sentiment_classifier"
platform: "pytorch_libtorch"
max_batch_size: 64
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1, 512]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1, 512]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [-1, 5]
  }
]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 2000
}
```

**Kubernetes Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton-inference
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.10-py3
        args: ["tritonserver", "--model-repository=/models", "--model-control-mode=poll"]
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: 8
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

### Streaming Pipeline (Faust)

```python
import faust
import numpy as np
from transformers import XLMRobertaTokenizer

app = faust.App('sentiment-stream', broker='kafka://kafka-cluster:9092')
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

class Review(faust.Record, serializer='json'):
    review_id: str
    text: str
    language: str
    product_id: str
    timestamp: float

class SentimentResult(faust.Record, serializer='json'):
    review_id: str
    sentiment: str
    confidence: float
    aspects: list
    needs_review: bool

reviews_topic = app.topic('raw_reviews', value_type=Review)
results_topic = app.topic('sentiment_results', value_type=SentimentResult)

@app.agent(reviews_topic)
async def process_reviews(stream):
    async for review in stream:
        # Preprocess
        tokens = tokenizer(
            review.text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Infer via Triton HTTP client
        result = await triton_infer(tokens)
        
        # Post-process
        sentiment = decode_sentiment(np.argmax(result))
        confidence = float(np.max(softmax(result)))
        
        aspects = await extract_aspects(review.text, review.language)
        
        await results_topic.send(
            value=SentimentResult(
                review_id=review.review_id,
                sentiment=sentiment,
                confidence=confidence,
                aspects=aspects,
                needs_review=confidence < 0.7
            )
        )
```

---

## Metrics & Results

### Performance Metrics

| Metric | Pre-AI | Post-AI | Improvement |
|--------|--------|---------|-------------|
| **Accuracy** | 65% (rule-based) | 93.5% (overall) | +28.5 pp |
| **F1 Score (macro)** | 0.58 | 0.893 | +0.313 |
| **English accuracy** | 65% | 96.2% | +31.2 pp |
| **Arabic accuracy** | 42% | 89.8% | +47.8 pp |
| **Sarcasm detection** | Not supported | 78% F1 | — |
| **Processing time** | 5 min (human) | 87 ms (avg) | 3,448x faster |
| **Throughput** | 200/day (human) | 250,000/day | 1,250x |
| **Cost per prediction** | $0.50 (human) | $0.0004 | 99.92% reduction |

### Business Impact (6 months post-deployment)

| Metric | Value | Calculation |
|--------|-------|-------------|
| **Revenue recovery** | $8.2M/year | Rapid response to negative reviews → retention |
| **Operational savings** | $2.4M/year | Reduced manual review team from 40 to 8 |
| **NPS score increase** | +28 points | 32 → 60 |
| **Response time** | 2 min | Alerts for critical negative reviews |
| **Detection rate** | 97% | Urgent issues caught within first 10 reviews |
| **False alert rate** | 4% | Human-in-the-loop validation |

### Per-Language Breakdown

| Language | Accuracy | F1 | Dataset Size | Confusion Focus |
|----------|----------|-----|-------------|-----------------|
| English | 96.2% | 0.95 | 1.2M | Neutral/Positive |
| Spanish | 94.8% | 0.93 | 200K | Negative/Very Negative |
| Mandarin | 93.1% | 0.91 | 180K | Sarcasm |
| Arabic | 89.8% | 0.87 | 150K | Dialectal variants |
| Hindi | 90.2% | 0.88 | 120K | Code-switching Hindi/English |
| Japanese | 91.5% | 0.89 | 100K | Keigo (politeness) levels |
| Thai | 85.3% | 0.82 | 30K | Low-resource |
| Vietnamese | 86.7% | 0.84 | 25K | Low-resource |

### Latency Distribution

```
p50:   42 ms
p90:   87 ms
p95:  124 ms
p99:  230 ms
p999: 450 ms
```

---

## Lessons Learned

### What Went Right

1. **Two-model architecture was crucial** — Single XLM-R model for both sentiment + aspect had 15% lower F1. Separate aspect extraction model (BERT+CRF) improved aspect F1 from 0.71 to 0.86.

2. **Batched GPU inference is essential** — Processing reviews one-by-one would cost 5x more. Dynamic batching on Triton with 2ms max queue delay gave optimal throughput/cost.

3. **Human-in-the-loop built trust** — Flagging low-confidence (<0.7) predictions for manual review caught edge cases (sarcasm, cultural nuances) and built stakeholder confidence. After 6 months, the threshold was lowered to 0.5.

4. **Language-specific fine-tuning beats generic** — A single multilingual model fine-tuned on all languages performed worse than language-adaptive fine-tuning (additional 1 epoch on each language's data with lower learning rate: 1e-5).

5. **Continuous monitoring prevented drift** — Weekly accuracy evaluation against a held-out golden dataset detected drift (+2.3% after a product launch that changed review language patterns). Retraining restored accuracy within 48 hours.

### What We'd Do Differently

1. **Better low-resource language strategy** — Thai and Vietnamese performance plateaued at 85-87%. Could have used:
   - Cross-lingual transfer from high-resource languages
   - Active learning to prioritize uncertain samples for labeling
   - Machine translation + self-training (Noisy Student)

2. **Real-time active learning** — Implemented it in phase 2, but should have been from day 1. Flagging uncertain samples for human review dramatically improves model improvement rate.

3. **Aspect taxonomy pre-definition** — Initially used open aspect extraction (any noun phrase), which led to noise (3,000+ distinct aspects). Restructured to hierarchical taxonomy (50 categories, 200 sub-categories) on phase 2. This doubled aspect precision.

4. **Cost monitoring per-language** — Some languages (Arabic, Thai) cost more per prediction due to longer tokenization (morphologically rich languages). Should have budget-model for low-volume languages.

### Operational Challenges

- **Cold start:** First week had <70% accuracy as feedback loop hadn't accumulated enough corrections
- **Sarcasm detection:** Required separate effort — deployed a sarcasm-specific classifier (RoBERTa) that improved sarcasm F1 from 0.58 to 0.78
- **Emoji handling:** Custom preprocessor for emoji → text mapping improved sentiment capture for review-heavy categories (food, travel)
- **Model serving cost:** GPU costs were $1,800/month for 3x T4. Switching to serverless (AWS SageMaker endpoint auto-scaling) saved 40%.

---

## TEMPLATE: Sentiment Analysis Project

### Project Structure Template

```
sentiment-analysis-project/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/
│   └── workflows/
│       ├── train.yml
│       ├── deploy.yml
│       └── monitor.yml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── golden/           # Held-out evaluation set
│   └── external/         # Public datasets
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-model-benchmark.ipynb
│   ├── 03-error-analysis.ipynb
│   └── 04-drift-analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   ├── augmenter.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentiment.py
│   │   ├── aspect.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── optimizer.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── api.py
│   │   └── monitor.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── stream.py
│   │   └── batch.py
│   └── utils/
│       ├── __init__.py
│       ├── language.py
│       ├── metrics.py
│       └── visualization.py
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── serving_config.yaml
│   └── monitoring.yaml
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_pipeline.py
├── models/               # Model artifacts (git LFS)
├── infra/
│   ├── k8s/
│   │   ├── triton-deployment.yaml
│   │   ├── api-deployment.yaml
│   │   └── kafka-deployment.yaml
│   └── terraform/
│       ├── main.tf
│       └── variables.tf
└── scripts/
    ├── train.sh
    ├── evaluate.sh
    ├── deploy.sh
    └── monitor.sh
```

### Evaluation Checklist

- [ ] Accuracy by language segment
- [ ] F1 by sentiment class (check class imbalance)
- [ ] Confidence calibration (reliability diagram)
- [ ] Latency p50/p95/p99
- [ ] Throughput (req/s)
- [ ] Cost per prediction
- [ ] Confusion matrix (language x sentiment class)
- [ ] Drift detection on source data
- [ ] Backtesting on golden dataset
- [ ] Human evaluation agreement (Cohen's Kappa)

### Cost Estimation Template

```
Infrastructure:
  - GPU compute: ___ hrs x $___ /hr = $___ /month
  - CPU compute: ___ hrs x $___ /hr = $___ /month  
  - Storage: ___ GB x $___ /GB = $___ /month
  - API calls: ___ M x $___ /1M = $___ /month
  - Total infra: $___ /month

Human oversight:
  - Reviewers: ___ people x $___ /hr x ___ hrs/day = $___ /month
  - Quality assurance: ___ % of samples = $___ /month

Development (one-time):
  - Data labeling: ___ samples x $___ /sample = $___
  - Engineering: ___ dev-weeks x $___ /week = $___
  - Deployment: ___ dev-weeks = $___

Expected savings:
  - Manual review reduction: ___ people x $___ = $___ /year
  - Revenue impact: ___ % uplift x $___ revenue = $___ /year
  - Total savings: $___ /year
  - Payback period: ___ months
```

---

## Cross-References

- **RAG Search System** → [06-RAG-Search-System.md](./06-RAG-Search-System.md) — Similar NLP architecture patterns
- **NLP Foundations** → [02-LLMs/05-NLP-Foundations.md](../02-LLMs/05-NLP-Foundations.md) — Core NLP techniques used
- **Fine-Tuning Guide** → [13-Top-Demand/07-Fine-Tuning-Custom-Models.md](../13-Top-Demand/07-Fine-Tuning-Custom-Models.md) — Fine-tuning methodology
- **MLOps / Deployment** → [05-Enterprise/01-Enterprise-AI-Deployment.md](../05-Enterprise/01-Enterprise-AI-Deployment.md) — Production deployment
- **Model Evaluation** → [06-Advanced/03-Evaluation-Benchmarks.md](../06-Advanced/03-Evaluation-Benchmarks.md) — Evaluation metrics
- **Real-Time Systems** → [13-Top-Demand/11-Real-Time-AI-Systems.md](../13-Top-Demand/11-Real-Time-AI-Systems.md) — Streaming architecture
- **Data Engineering** → [01-Foundations/04-Data-Engineering.md](../01-Foundations/04-Data-Engineering.md) — Data pipeline patterns

---

*Last updated: June 2026 | 400+ lines covering real-world sentiment analysis deployment*
