# Multi-Cloud AI Cost Governance

## Overview

Managing AI costs across multiple cloud providers is complex due to different pricing models, commitment tiers, data transfer costs, and rapidly changing model availability. Effective multi-cloud AI cost governance requires visibility, allocation, optimization, and automation.

## Cloud AI Pricing Comparison (June 2026)

| Model Class | AWS Bedrock | Azure OpenAI | GCP Vertex AI |
|------------|-------------|-------------|---------------|
| GPT-4o class | $2.50/$10.00 | $2.50/$10.00 | $2.50/$10.00 |
| GPT-4o-mini class | $0.15/$0.60 | $0.15/$0.60 | $0.15/$0.60 |
| Claude 4 Opus | $15.00/$75.00 | — | — |
| Llama 4 (70B) | $0.99/$0.99 | — | $0.99/$0.99 |
| Gemini 2.0 Flash | — | — | $0.10/$0.40 |
| Embeddings (ADA) | $0.10/1M | $0.10/1M | — |

*Format: $/1M input tokens / $/1M output tokens. Prices subject to change.*

## Cost Optimization Strategies by Cloud

### AWS
- **Provisioned Throughput:** Commit to hourly throughput for 1mo/6mo discount
- **SageMaker Savings Plans:** 30-60% discount for 1yr/3yr commitment
- **Spot Instances:** 60-90% discount for training (be aware of interruptions)
- **Bedrock Batch:** 50% discount for batch inference

### Azure
- **OpenAI Provisioned:** Reserved throughput units for predictable workloads
- **Azure Reservations:** 1yr/3yr for ML compute
- **Azure Hybrid Benefit:** Use existing Windows Server/SA licenses

### GCP
- **Vertex AI Commitments:** 1yr/3yr for training/prediction
- **Preemptible VMs:** 60-80% discount for training
- **TPU Reservations:** Reserved capacity for ML workloads

## Cost Allocation Model

```
Provider → Service → Project → Team → Cost Center
    ↓
Resource tagging (required for allocation)
    ↓
Daily cost ingestion → BigQuery/Athena/Log Analytics
    ↓
Cost dashboard (Grafana/Power BI/Looker)
    ↓
Anomaly detection → Alerting → auto-remediation

## Unit Economics Framework

Measure cost per:
- Inference request (by model, by latency tier)
- Training run (by model size, by dataset size)
- Token processed (input vs output, cached vs fresh)
- User (by tier, by feature usage)
- API call (by endpoint, by authentication method)

Track: Cost per unit → Trend over time → Benchmark vs on-premise

## Cost Optimization Checklist

- [ ] Implement tagging for all AI resources
- [ ] Set up budget alerts at 50%/80%/100%
- [ ] Use reserved capacity for predictable workloads
- [ ] Enable spot/preemptible instances for training
- [ ] Cache common prompts (30-70% savings)
- [ ] Route simple queries to cheaper models
- [ ] Monitor data transfer costs between clouds
- [ ] Review unused committed capacity monthly
- [ ] Auto-stop idle training instances
- [ ] Use batch inference for non-real-time workloads
