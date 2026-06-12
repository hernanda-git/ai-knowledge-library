# 06 — AI Services Playbook: Fine-Tuning, Training, MLOps & Managed AI Operations

## 1. Executive Summary

The AI services market is rapidly growing as enterprises seek specialized expertise that off-the-shelf models can't provide. Unlike AI consulting (which is strategic/advisory), AI services are hands-on technical engagements: building custom models, fine-tuning existing ones, deploying MLOps pipelines, and managing AI operations.

This playbook covers five core service lines with pricing, delivery methodology, case studies, and scaling playbooks.

### Market Overview

| Service Line | Market Size (2025) | Growth Rate | Typical Engagement |
| --- | --- | --- | --- |
| Fine-tuning services | $3.8B | 45% | $20K–$200K per project |
| Custom model development | $6.2B | 35% | $50K–$500K per project |
| MLOps / AI infrastructure | $4.5B | 40% | $30K–$150K initial + retainer |
| AI training & workshops | $2.1B | 30% | $5K–$50K per engagement |
| Managed AI operations | $3.5B | 50% | $10K–$100K/month retainer |
| **Total AI Services** | **$20.1B** | **38%** | |

## 2. Service Line 1: Fine-Tuning Services

### 2.1 Service Description

Customizing pre-trained models (LLMs, vision models, etc.) for specific domains, tasks, or data distributions.

### 2.2 Typical Engagements

| Engagement | Scope | Timeline | Price Range |
| --- | --- | --- | --- |
| Domain adaptation (e.g., legal LLM) | Fine-tune on 10K–100K domain documents | 4–8 weeks | $30K–$100K |
| Instruction tuning | Create instruction dataset, fine-tune for specific task | 3–6 weeks | $20K–$75K |
| RLHF / preference tuning | Collect preferences, reward model training, RLHF | 6–12 weeks | $50K–$200K |
| LoRA adapters | Lightweight fine-tuning for specific tasks | 2–4 weeks | $10K–$40K |
| Continued pre-training | Domain-specific pre-training on top of base model | 8–16 weeks | $100K–$500K |
| Knowledge distillation | Student model from larger teacher (for deployment) | 4–8 weeks | $30K–$100K |

### 2.3 Fine-Tuning Delivery Methodology

**Phase 1: Assessment (Week 1)**
- [ ] Evaluate base model options (Llama, Mistral, GPT, Claude)
- [ ] Assess data availability, quality, and licensing
- [ ] Define evaluation metrics and success criteria
- [ ] Build data pipeline for processing

**Phase 2: Data Preparation (Week 2–3)**
- [ ] Data cleaning and deduplication
- [ ] Format conversion (ChatML, Alpaca, ShareGPT)
- [ ] Train/val/test split (80/10/10)
- [ ] Data quality validation (diversity, coverage)

**Phase 3: Training (Week 3–5)**
- [ ] Hyperparameter search (LR, batch size, epochs)
- [ ] Training with experiment tracking (Weights & Biases, MLflow)
- [ ] Regular evaluation checkpoints
- [ ] Ablation studies (what data matters most?)

**Phase 4: Evaluation (Week 5–6)**
- [ ] Standard benchmark evaluation
- [ ] Custom task evaluation
- [ ] Human evaluation (preference testing)
- [ ] Safety/red-teaming evaluation
- [ ] Comparison with baseline model

**Phase 5: Delivery (Week 6–8)**
- [ ] Model packaging (GGUF, ONNX, vLLM format)
- [ ] Deployment guide and inference code
- [ ] Documentation and model card
- [ ] Client handoff and training

### 2.4 Fine-Tuning Pricing Calculator

```python
# Fine-Tuning Pricing Estimator
# Input parameters
base_model = "Llama 3 70B"  # or "Mistral 7B", "Llama 3 8B"
gpu_hourly_cost = 3.50  # A100-80GB spot pricing
num_gpus = 8            # For 70B model
training_hours = 48     # Training time
num_tokens = 10_000_000 # Training data tokens

# Pricing components
compute_cost = gpu_hourly_cost * num_gpus * training_hours
data_prep_hours = 40 * 200  # 40 hrs prep * $200/hr
training_engineer_hours = 80 * 250  # 80 hrs eng * $250/hr
eval_hours = 40 * 200  # 40 hrs eval * $200/hr
project_management = 20 * 150  # 20 hrs PM * $150/hr

total_cost = compute_cost + data_prep_hours + training_engineer_hours + eval_hours + project_management
margin = 1.35  # 35% margin
price = total_cost * margin

print(f"Compute cost: ${compute_cost:,.0f}")
print(f"Data prep: ${data_prep_hours:,.0f}")
print(f"Training eng: ${training_engineer_hours:,.0f}")
print(f"Evaluation: ${eval_hours:,.0f}")
print(f"PM: ${project_management:,.0f}")
print(f"Total cost: ${total_cost:,.0f}")
print(f"Price (35% margin): ${price:,.0f}")
```

### 2.5 Case Study: Domain-Specific Fine-Tuning

| Detail | Value |
| --- | --- |
| Client | Mid-size law firm (200 attorneys) |
| Problem | Generic LLMs don't understand legal terminology, cite accurate cases, or follow legal document structure |
| Solution | Fine-tuned Llama 3 70B on 50K legal documents (opinions, briefs, contracts) + instruction-tuned for legal tasks |
| Timeline | 6 weeks |
| Cost to client | $85K |
| Results | 92% accuracy on legal citation vs. 68% for base model; 3x faster document drafting |
| Retention | Client signed 12-month managed services ($12K/mo) |
| Lesson | The fine-tuned model was good, but the ongoing support (monitoring, updates, retraining) was where the recurring revenue was |

## 3. Service Line 2: Custom Model Development

### 3.1 Service Description

Building custom models from scratch or combining multiple models for specific use cases. Different from fine-tuning — this involves novel architectures, multi-model systems, or specialized models.

### 3.2 Typical Engagements

| Engagement | Scope | Timeline | Price Range |
| --- | --- | --- | --- |
| Custom classifier | Binary/multi-class for domain-specific content | 4–8 weeks | $30K–$100K |
| Recommendation system | Personalized recommendations for specific domain | 8–16 weeks | $50K–$200K |
| Multi-model orchestration | Router + multiple specialized models | 6–12 weeks | $50K–$150K |
| Custom vision model | Object detection, segmentation for specific use case | 8–16 weeks | $50K–$200K |
| Anomaly detection | Time-series or log anomaly detection | 6–12 weeks | $40K–$150K |
| Forecasting model | Demand, inventory, financial forecasting | 8–16 weeks | $50K–$200K |

### 3.3 Development Methodology (TDSP Adapted)

```
Week 1-2: Business Understanding
  └─ Define problem, success criteria, data requirements
  └─ Feasibility study (can AI solve this?)
  
Week 3-6: Data Engineering
  └─ Data collection, integration, cleaning
  └─ Feature engineering and selection
  
Week 7-12: Model Development
  └─ Baseline model → iterate on architecture
  └─ Hyperparameter optimization
  └─ Cross-validation and performance evaluation
  
Week 13-14: Evaluation & Validation
  └─ A/B testing framework
  └─ Edge case analysis
  └─ Fairness and bias evaluation
  
Week 15-16: Deployment
  └─ Model serving infrastructure
  └─ Monitoring and alerting
  └─ Documentation and handoff
```

## 4. Service Line 3: MLOps / AI Infrastructure Consulting

### 4.1 Service Description

Setting up and optimizing the infrastructure, pipelines, and processes that enable ML teams to build, deploy, and monitor models efficiently.

### 4.2 Core Service Offerings

| Service | Description | Timeline | Price |
| --- | --- | --- | --- |
| MLOps audit | Assess current ML infrastructure, identify gaps, recommend improvements | 2 weeks | $15K–$30K |
| ML pipeline setup | CI/CD for ML, experiment tracking, model registry | 4–6 weeks | $30K–$80K |
| Model serving infra | Deploy models with API endpoints, autoscaling, monitoring | 3–8 weeks | $40K–$120K |
| Data pipeline architecture | Feature store, data versioning, transformation pipelines | 4–8 weeks | $50K–$100K |
| GPU infrastructure | GPU cluster setup, scheduling, cost optimization | 2–4 weeks | $20K–$60K |
| Full MLOps platform | End-to-end ML platform (pipeline + serving + monitoring) | 8–16 weeks | $100K–$300K |
| Emergency optimization | Fix broken pipelines, reduce costs, improve latency | 1–4 weeks | $15K–$50K |

### 4.3 MLOps Tech Stack Recommendation Framework

| Category | Startup (1–10 ML eng) | Growth (10–50) | Enterprise (50+) |
| --- | --- | --- | --- |
| **Experiment tracking** | MLflow, Weights & Biases | Weights & Biases, Neptune | Comet, MLflow Enterprise |
| **Feature store** | Feast (open-source) | Tecton, Feast | Tecton, SageMaker Feature Store |
| **Pipeline orchestration** | Airflow, Prefect | Prefect, Dagster | Airflow, Kubeflow |
| **Model registry** | MLflow Model Registry | MLflow, DVC | MLflow, Sagemaker, Vertex AI |
| **Model serving** | FastAPI, BentoML | BentoML, Seldon Core | Seldon, KServe, SageMaker |
| **Monitoring** | Evidently, WhyLabs | Arize, WhyLabs | Arize, Fiddler, WhyLabs |
| **GPU orchestration** | RunPod, Vast.ai | K8s + GPU operator | K8s + Volcano + custom |

### 4.4 MLOps Pricing Models

The MLOps market is bifurcating: open-source tools are free but require expertise to deploy. Our service bridges that gap.

| Pricing Model | Description | Example Price |
| --- | --- | --- |
| **Project-based** | Fixed price for setup | $30K–$150K one-time |
| **Time & materials** | $200–$400/hour for ongoing work | $20K–$60K/mo for team |
| **Managed MLOps** | We run your ML infrastructure | $10K–$50K/mo retainer |
| **Emergency support** | Premium rate for urgent fixes | $400–$600/hour |

## 5. Service Line 4: AI Training & Workshops

### 5.1 Workshop Formats

| Format | Duration | Price | Audience | Content |
| --- | --- | --- | --- | --- |
| **Executive AI briefing** | 2–4 hours | $5K–$10K | C-suite, VP | AI strategy, opportunities, risks |
| **AI foundations** | 1–2 days | $10K–$20K | Product managers, engineers | Hands-on with LLMs, prompting, APIs |
| **Advanced LLM engineering** | 2–3 days | $15K–$30K | ML engineers, data scientists | Fine-tuning, RAG, agent frameworks |
| **AI product management** | 1–2 days | $10K–$20K | Product managers | Roadmap, evaluation, AI UX |
| **MLOps bootcamp** | 3–5 days | $20K–$40K | DevOps, ML engineers | Production ML, monitoring, pipelines |
| **Custom curriculum** | Custom | $5K–$15K per day | Mixed | Tailored to client needs |

### 5.2 Workshop Pricing Calculator

```
Workshop price =
  (Preparation hours × Blended rate)
  + (Delivery hours × Blended rate)
  + (Materials development × rate)
  + Travel expenses (if on-site)

Example: 2-day Advanced LLM Engineering
Prep: 20h × $200 = $4,000
Delivery: 16h × $250 = $4,000
Materials: 10h × $150 = $1,500
Travel: $1,500
Total: $11,000
Price: $15,000 (36% margin)

Per-person price for open enrollment:
$15,000 / 15 attendees = $1,000/person
Recommended: $1,500–$2,500/person (includes materials + lunch)
```

### 5.3 Workshop Delivery Checklist

**Pre-Workshop (2–4 weeks before)**:
- [ ] Client needs assessment call
- [ ] Participant skill survey
- [ ] Customize curriculum to client's domain (examples, data)
- [ ] Set up cloud environment (Jupyter notebooks, API keys)
- [ ] Send pre-reading materials
- [ ] Confirm logistics (venue, AV, catering)

**During Workshop**:
- [ ] 50/50 split: lecture and hands-on exercises
- [ ] All exercises tested and working
- [ ] Have backup plans for API outages
- [ ] Collect feedback at end of each day
- [ ] Real-time Q&A throughout

**Post-Workshop**:
- [ ] Send slides, recordings, and materials
- [ ] Provide 30-day post-workshop support via email
- [ ] Follow-up survey (Net Promoter Score)
- [ ] Offer discounted follow-up consulting
- [ ] Upsell to longer engagement

### 5.4 Case Study: Enterprise AI Training Program

| Detail | Value |
| --- | --- |
| Client | Fortune 500 financial services company |
| Need | Upskilling 500 engineers in GenAI over 6 months |
| Program | 2-day AI Foundations (cohorts of 20) + 3-day Advanced LLM Engineering (cohorts of 15) |
| Pricing | $12K per Foundations workshop (20 cohorts = $240K) + $25K per Advanced (12 cohorts = $300K) |
| Total revenue | $540K |
| Timeline | 6 months |
| Results | 92% NPS, 40% of attendees built AI prototypes within 3 months |
| Upsell | 3 attendees' prototypes led to $150K in consulting projects |

## 6. Service Line 5: Managed AI Operations

### 6.1 Service Description

Ongoing management of AI systems in production — the "AI as a managed service" model. Client owns the IP, we run the operations.

### 6.2 Service Tiers

| Tier | Services Included | Response Time | Price |
| --- | --- | --- | --- |
| **Monitor** | Model monitoring, drift detection, basic alerts | 4 hours business hours | $5K–$10K/mo |
| **Maintain** | Monitor + bug fixes, minor updates, retraining | 2 hours business hours | $10K–$25K/mo |
| **Manage** | Maintain + full operations, cost optimization, enhancements | 1 hour, 24/7 | $25K–$50K/mo |
| **Partner** | Manage + strategic roadmap, quarterly reviews, dedicated team | 30 min, 24/7 | $50K–$100K/mo |

### 6.3 Managed AI Operations Scope

**In-scope (typical)**:
- Model serving infrastructure management
- GPU/CPU resource monitoring and optimization
- Model performance monitoring (latency, throughput, accuracy)
- Data drift detection and alerting
- Cost optimization (right-sizing instances, spot usage)
- Security updates and patches
- Model retraining (quarterly or as needed)
- 24/7 incident response (for higher tiers)
- Monthly performance reports

**Out-of-scope (typical)**:
- New model development (separate SOW)
- Training data acquisition
- Business logic changes
- Integration with new systems (separate project)

### 6.4 Managed Operations Pricing Model

```
Monthly retainer =
  (Infrastructure management hours × Rate)
  + (Monitoring infrastructure cost × Margin)
  + (On-call premium)
  + (Emergency response fee)

Example — "Maintain" tier for mid-market client:
Infra management: 40h/mo × $200 = $8,000
Monitoring tools: $500/mo × 1.2 = $600
On-call rotation: $3,000/mo
Total: $11,600/mo → Round to $12K/mo

Additional: $200/hour for work beyond included hours
```

## 7. Services Business Model Comparison

| Aspect | Fine-Tuning | Custom Models | MLOps | Training/Workshops | Managed Operations |
| --- | --- | --- | --- | --- | --- |
| **Recurring revenue** | Low | Low | Medium | Low | High |
| **Margins** | 35–55% | 35–50% | 40–55% | 50–70% | 45–60% |
| **Scalability** | Medium | Low | Medium | Low (capacity constrained) | Medium |
| **IP creation** | High | High | Medium | Medium | Low |
| **Client dependency** | Low | Medium | Medium | Low | High |
| **Sales cycle** | 4–12 weeks | 8–24 weeks | 4–16 weeks | 2–8 weeks | 8–24 weeks |
| **Typical ACV** | $50K | $100K | $75K | $20K | $120K |
| **Key skill** | Data + training | Research + engineering | DevOps + ML | Teaching + comm | Operations |

## 8. From Project to Retainer: The Services Ladder

The key to building a sustainable services business is converting one-time projects into recurring retainers.

### 8.1 The Conversion Funnel

```
┌────────────────────────┐
│  Workshop / Assessment  │  $10K–$40K one-time
│  (Low risk entry point) │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  POC / Pilot Project    │  $30K–$100K one-time
│  (Show value)           │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Full Implementation    │  $50K–$300K one-time
│  (Prove reliability)    │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Managed Operations     │  $10K–$50K/month
│  (Recurring revenue)    │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Strategic Partnership  │  $50K–$100K+/month
│  (Deep integration)     │
└────────────────────────┘
```

### 8.2 Conversion Strategies

| From → To | Strategy | Success Rate |
| --- | --- | --- |
| Workshop → POC | Identify 1 high-value use case during workshop, offer POC at 50% standard rate | 30–50% |
| POC → Implementation | Show measurable ROI from POC, propose full implementation | 50–70% |
| Implementation → Managed ops | "Your system is live now. Who's monitoring it at 2 AM?" | 40–60% |
| Managed ops → Strategic | Quarterly business reviews, propose roadmap expansion | 25–40% |

## 9. Services Marketing & Sales

### 9.1 Lead Generation for AI Services

| Channel | Effectiveness | Cost per Lead | Notes |
| --- | --- | --- | --- |
| **Content marketing (technical blog)** | Very high | $100–$500 | Tutorials, benchmarks, case studies |
| **Speaking at conferences** | High | $500–$2,000 | Submit talks, get on stage |
| **Open-source contributions** | Very high | $0 (time cost) | Build tools that showcase your expertise |
| **Partnerships (cloud providers)** | High | 15–20% referral fee | AWS, GCP, Azure referral programs |
| **Cold outbound (email)** | Low-medium | $500–$1,000 | Requires excellent targeting |
| **Consulting marketplaces** | Medium | 20–30% commission | Upwork, Toptal, Catalant |
| **Referrals** | Very high | 10–20% referral fee | Best source, nurture this |
| **Webinars** | Medium | $200–$500 | High effort for medium return |

### 9.2 Services Proposal Template (Abbreviated)

```
PROJECT: [Project Name]
CLIENT: [Client Name]

EXECUTIVE SUMMARY
[2–3 sentences on problem, solution, value]

APPROACH
[Methodology we'll use, why it works]

SCOPE OF WORK
1. Phase 1 — Assessment: [Activities, duration]
2. Phase 2 — Build: [Activities, duration]
3. Phase 3 — Deploy: [Activities, duration]

TEAM
[Names, roles, expertise]

INVESTMENT
Phase 1: $[X]
Phase 2: $[Y]
Phase 3: $[Z]
Total: $[X+Y+Z]

TIMELINE: [X] weeks

WHY US
[Differentiators: experience, IP, case studies]

NEXT STEPS
[Acceptance, start date, kickoff call]
```

## 10. Services Financial Model

### 10.1 Service Line P&L Template

| Category | Fine-Tuning | Custom Models | MLOps | Workshops | Managed Ops | Total |
| --- | --- | --- | --- | --- | --- | --- |
| **Revenue** | $300K | $500K | $250K | $150K | $400K | **$1.6M** |
| **Direct Costs** | | | | | | |
| Engineer salaries | $100K | $200K | $80K | $40K | $120K | $540K |
| Contractor costs | $50K | $50K | $30K | $10K | $40K | $180K |
| Compute infra | $40K | $60K | $20K | $5K | $20K | $145K |
| Travel & expenses | $5K | $10K | $5K | $20K | $5K | $45K |
| **Total direct** | **$195K** | **$320K** | **$135K** | **$75K** | **$185K** | **$910K** |
| **Gross margin** | **35%** | **36%** | **46%** | **50%** | **54%** | **43%** |
| **G&A (allocated)** | $60K | $100K | $50K | $30K | $80K | $320K |
| **Net profit** | $45K | $80K | $65K | $45K | $135K | **$370K** |
| **Net margin** | **15%** | **16%** | **26%** | **30%** | **34%** | **23%** |

### 10.2 Scaling Benchmarks

| Metric | Solo | 10-person | 25-person | 50-person |
| --- | --- | --- | --- | --- |
| Revenue | $200K–$500K | $2M–$4M | $5M–$12M | $10M–$25M |
| Rev per consultant | $200K–$250K | $200K–$400K | $200K–$480K | $200K–$500K |
| Utilization | 65–75% | 70–75% | 70–80% | 70–80% |
| Gross margin | 50–60% | 40–55% | 40–50% | 35–45% |
| Net margin | 15–25% | 10–20% | 10–15% | 8–12% |
| Recurring revenue % | 10–20% | 20–30% | 25–40% | 30–50% |

## 11. Risks & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Model provider releases free version of our service | Entire service line obsolete | Specialize in vertical expertise, data, integration |
| Client hires their own AI team | Lose managed ops contracts | Build relationships with client leadership, be indispensable |
| GPU hardware shortage | Can't deliver fine-tuning | Multi-cloud strategy, alternative architectures |
| Open-source model quality matches fine-tuned model | Clients question value of fine-tuning | Focus on data flywheel (client's proprietary data = moat) |
| Consulting commoditization | Price compression | Develop IP, productize services, focus on outcomes |
| Difficulty retaining talent | Service quality drops | Clear career paths, above-market pay, interesting work |
| Client doesn't maintain model post-deployment | Model decays → blame us | Clear SLAs, ongoing monitoring in contract |

---
*Next: Playbook 07 — Freemium to Enterprise Monetization Ladder.*
