# 02 — AI Customer Support Agent

## Case Study: Production AI Support Agent with RAG + LLM + Human Handoff

| Metadata | Value |
|----------|-------|
| **Industry** | Customer Service / SaaS |
| **Domain** | AI-powered ticket resolution |
| **Difficulty** | Intermediate |
| **Est. Timeline** | 4-6 weeks |
| **Team Size** | 4-6 engineers (2 ML, 2 backend, 1 frontend, 1 PM) |

---

## 🎯 Problem Statement

### Business Context

**Company:** AcmeCloud SaaS (B2B, 50K+ customers, $200M ARR)
**Support Volume:** 15,000+ tickets per week across email, chat, and phone
**Support Team:** 120 agents across 3 shifts (24/5, limited weekend coverage)

### Pain Points

1. **High Ticket Volume** — 15K+ weekly tickets, growing 8-10% month-over-month
2. **Slow Response Time** — Average first response: 15 minutes; peak hours exceed 45 minutes
3. **Agent Churn** — 35% annual agent turnover due to repetitive Tier-1 issues
4. **Inconsistent Quality** — 40% of tickets require escalation; resolution quality varies widely by agent
5. **24/7 Gap** — No support on weekends; Monday morning backlog of 3,000+ tickets
6. **Cost Pressure** — Each ticket costs $15-25 to resolve; annual support cost exceeding $12M

### Success Criteria

| Metric | Target | Current Baseline |
|--------|--------|-----------------|
| Deflection Rate | 75%+ (auto-resolve without human) | 0% |
| First Response Time | < 3 minutes | 15 minutes |
| Customer Satisfaction (CSAT) | 90%+ | 78% |
| Cost per Ticket | < $8 | $18 |
| Escalation Rate | < 20% | 40% |
| Agent Productivity | 2x tickets/agent/day | 35 tickets/day |

### Constraints

- Must integrate with existing Zendesk instance
- Cannot expose raw LLM outputs (hallucination risk)
- Must support 10 languages (English, Spanish, French, German, Japanese, etc.)
- GDPR / SOC2 compliance required
- P99 latency < 5 seconds end-to-end

---

## 🏗️ Solution Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER CHANNELS                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐                │
│  │  Email  │  │  Chat   │  │  Phone  │  │  API   │                │
│  └────┬────┘  └────┬────┘  └────┬────┘  └───┬────┘                │
│       │            │            │            │                     │
└───────┼────────────┼────────────┼────────────┼─────────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER (Zendesk API)                  │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐    │
│  │   Ticket Ingestion    │  │  Conversation History Fetcher    │    │
│  │   (Webhook + Poll)    │  │  (past 90 days of interactions)  │    │
│  └──────────┬───────────┘  └──────────────┬───────────────────┘    │
│             │                             │                        │
└─────────────┼─────────────────────────────┼────────────────────────┘
              │                             │
              ▼                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER (LangGraph)                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    SUPERVISOR GRAPH                          │    │
│  │                                                             │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │    │
│  │   │   Classify    │───▶│  Retrieve    │───▶│  Generate    │  │    │
│  │   │   Intent      │    │  Context     │    │  Response    │  │    │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │    │
│  │          │                   │                   │           │    │
│  │          ▼                   ▼                   ▼           │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │    │
│  │   │  Sentiment   │    │  Knowledge   │    │  Fact-Check  │  │    │
│  │   │  Analysis    │    │  Base Query   │    │  Verifier    │  │    │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │    │
│  │          │                   │                   │           │    │
│  │          └───────────────────┴───────────────────┘           │    │
│  │                              │                               │    │
│  │                              ▼                               │    │
│  │                   ┌──────────────────────┐                   │    │
│  │                   │  Confidence Check    │                   │    │
│  │                   │  (> 0.85 ? resolve   │                   │    │
│  │                   │   : human handoff)   │                   │    │
│  │                   └──────────┬───────────┘                   │    │
│  └──────────────────────────────┼───────────────────────────────┘    │
│                                 │                                    │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │   AUTO-RESOLVE    │       │  HUMAN HANDOFF    │
        │   (CSAT survey)   │       │  (agent takeover)  │
        └───────────────────┘       └───────────────────┘
```

### RAG Pipeline Detail

```
                         RAG PIPELINE FLOW
                         
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ticket Text  │────▶│  Query       │────▶│  Retrieve    │
│  + History    │     │  Enrichment  │     │  Top-5 Docs  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM         │◀────│  Context     │◀────│  Re-ranking  │
│  Generate    │     │  Assembly    │     │  (CohereRerank)
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  Response    │────▶│  Escalation  │
│  Draft       │     │  Check       │
└──────────────┘     └──────────────┘
```

### Knowledge Base Ingestion Pipeline

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Source    │    │  Chunking  │    │  Embedding │    │  Vector    │
│  Docs      │───▶│  (500-     │───▶│  (text-    │───▶│  Store     │
│  (HTML/PDF)│    │  1000 tok) │    │  embedding-│    │  ChromaDB  │
└────────────┘    └────────────┘    │  ada-002)  │    └────────────┘
                                    └────────────┘
┌────────────┐    ┌────────────┐    ┌────────────┐
│  GitHub    │    │  Code      │    │  Metadata  │
│  Wiki/FAQ  │───▶│  Splitter  │───▶│  Extractor │
└────────────┘    └────────────┘    └────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **LLM** | OpenAI GPT-4 / GPT-4-turbo | gpt-4-0125-preview | Best instruction following, low hallucination |
| **Orchestration** | LangGraph | 0.1.x | Graph-based agent state machine |
| **LLM Framework** | LangChain | 0.2.x | Ecosystem maturity, tool integration |
| **Vector Store** | ChromaDB | 0.5.x | Self-hosted, embedded, no external infra |
| **Embeddings** | text-embedding-3-large | OpenAI | 1536-d, state-of-the-art retrieval quality |
| **Reranker** | Cohere Rerank v3 | cohere.2024 | Improves retrieval precision by 15-20% |
| **Ticket System** | Zendesk API | v2 | Existing customer infrastructure |
| **Backend API** | FastAPI | 0.111.x | Async-native, high throughput |
| **Monitoring** | LangSmith + Prometheus | 2024 | LLM traceability + system metrics |
| **Queue** | Redis + Celery | 5.x / 5.x | Async task processing |
| **Database** | PostgreSQL | 16 | Ticket metadata, user profiles |
| **Cache** | Redis | 7.x | Session state, rate limiting |
| **CI/CD** | GitHub Actions | N/A | Automated testing + deployment |
| **Container** | Docker + Kubernetes | 24.x / 1.29 | Scalable microservices deployment |

### Dependency Installation

```bash
# Core dependencies
pip install langchain==0.2.12 langgraph==0.1.18 langchain-openai==0.1.21
pip install chromadb==0.5.5 fastapi==0.111.1 uvicorn[standard]==0.30.1
pip install openai==1.40.6 pydantic==2.8.2 tiktoken==0.7.0
pip install redis==5.0.8 celery==5.4.0 psycopg2-binary==2.9.9
pip install cohere==5.6.4

# Infrastructure
pip install docker-compose==1.29.2
```

---

## ⚙️ Implementation Details

### 1. Intent Classification

```python
# src/pipelines/classifier.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

INTENT_CLASSES = [
    "account_management", "billing", "technical_issue",
    "feature_request", "general_inquiry", "bug_report",
    "security_concern", "escalation_request"
]

classifier_prompt = ChatPromptTemplate.from_template("""
Classify the following support ticket into exactly one intent category.
Categories: {categories}

Ticket Subject: {subject}
Ticket Description: {description}

Respond with ONLY the category name, nothing else.
""")

class IntentClassifier:
    def __init__(self, model="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.chain = classifier_prompt | self.llm

    async def classify(self, subject: str, description: str) -> str:
        result = await self.chain.ainvoke({
            "categories": ", ".join(INTENT_CLASSES),
            "subject": subject,
            "description": description[:2000]  # truncate
        })
        return result.content.strip()
```

### 2. RAG Retrieval

```python
# src/pipelines/retriever.py
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class KnowledgeRetriever:
    def __init__(self, collection_name="support_docs"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.client = chromadb.PersistentClient(
            path="./data/chromadb"
        )
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    async def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """Retrieve top-k relevant documents."""
        docs = self.vectorstore.similarity_search_with_score(
            query, k=k
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in docs
        ]

    async def rerank(self, query: str, docs: list[dict], top_n: int = 5):
        """Re-rank using Cohere Rerank for precision."""
        import cohere
        co = cohere.Client()  # API key from env
        reranked = co.rerank(
            query=query,
            documents=[d["content"] for d in docs],
            model="rerank-english-v3.0",
            top_n=top_n
        )
        return [
            docs[r.index] for r in reranked.results
        ]
```

### 3. LangGraph Supervisor

```python
# src/pipelines/supervisor_graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class AgentState(TypedDict):
    ticket_id: str
    subject: str
    description: str
    customer_id: str
    intent: str
    sentiment: str
    retrieved_contexts: list
    response: str
    confidence: float
    should_escalate: bool

def create_support_graph():
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("check_confidence", check_confidence)
    workflow.add_node("human_handoff", human_handoff)
    workflow.add_node("resolve_ticket", resolve_ticket)

    # Define edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_response")
    workflow.add_edge("generate_response", "check_confidence")

    # Conditional edge based on confidence threshold
    workflow.add_conditional_edges(
        "check_confidence",
        lambda state: "resolve" if state["confidence"] > 0.85 else "escalate",
        {
            "resolve": "resolve_ticket",
            "escalate": "human_handoff"
        }
    )
    workflow.add_edge("resolve_ticket", END)
    workflow.add_edge("human_handoff", END)

    return workflow.compile()

# Compiled graph ready for invocation
support_graph = create_support_graph()
```

### 4. Response Generation with Guardrails

```python
# src/pipelines/response_generator.py
RESPONSE_TEMPLATE = """
You are a helpful customer support agent for AcmeCloud SaaS platform.
Use the following context from our knowledge base to answer the customer's question.

Context:
{context}

Customer Question: {question}

Instructions:
1. Be concise and professional
2. Cite specific documentation when possible
3. If unsure, state "I'll connect you with a specialist"
4. Do NOT make up API endpoints, pricing, or features
5. Format steps as numbered lists for clarity

Response:
"""

async def generate_safe_response(
    question: str,
    context: list[str],
    llm
) -> tuple[str, float]:
    formatted_context = "\n\n".join(context)
    prompt = RESPONSE_TEMPLATE.format(
        context=formatted_context,
        question=question
    )

    response = await llm.ainvoke(prompt)

    # Confidence check using LLM self-evaluation
    confidence_check = await llm.ainvoke(
        f"On a scale of 0-1, how confident are you that this response "
        f"correctly answers the question using only the provided context?\n"
        f"Question: {question}\nResponse: {response}\nConfidence (0-1):"
    )

    confidence = float(confidence_check.content.strip())
    return response.content, min(confidence, 1.0)
```

### 5. Human Handoff

```python
# src/pipelines/handoff.py
class HumanHandoff:
    """Handles transfer to human agent with full context."""

    async def transfer_to_agent(
        self,
        state: AgentState,
        zendesk_client
    ):
        # Create handoff summary
        summary = f"""
        TICKET #{state['ticket_id']}
        Intent: {state['intent']}
        Sentiment: {state['sentiment']}
        Customer: {state['customer_id']}

        AI Attempt:
        {state['response']}

        Confidence: {state['confidence']:.2f}
        Retrieved Contexts: {len(state['retrieved_contexts'])}
        """

        # Update Zendesk ticket with AI summary
        await zendesk_client.update_ticket(
            ticket_id=state['ticket_id'],
            comment=summary,
            priority="high" if state['sentiment'] == "negative" else "normal"
        )

        # Assign to appropriate agent based on intent
        if state['intent'] == "billing":
            await zendesk_client.assign_to_group(
                ticket_id=state['ticket_id'],
                group_id="billing_team"
            )
        elif state['intent'] in ("technical_issue", "bug_report"):
            await zendesk_client.assign_to_group(
                ticket_id=state['ticket_id'],
                group_id="tier2_support"
            )

        return {"status": "handed_off", "ticket_id": state['ticket_id']}
```

---

## 📊 Metrics & Results

### Technical Metrics

| Metric | Pre-Deployment | Post-Deployment | Improvement |
|--------|---------------|----------------|-------------|
| **First Response Time** | 15 minutes | 2 minutes | 87% faster |
| **Resolution Time** | 45 minutes avg | 12 minutes avg | 73% faster |
| **Deflection Rate** | 0% | 80% | 80 p.p. |
| **CSAT Score** | 78% | 91% | +13 p.p. |
| **Escalation Rate** | 40% | 18% | -22 p.p. |
| **P95 Latency** | N/A | 2.3 seconds | N/A |
| **P99 Latency** | N/A | 4.1 seconds | N/A |

### Business ROI

| Item | Before | After | Savings |
|------|--------|-------|---------|
| **Cost per Ticket** | $18.50 | $7.20 | 61% reduction |
| **Monthly Ticket Volume** | 65,000 | 65,000 (same) | — |
| **Agents Required** | 120 FTE | 75 FTE | 45 FTE |
| **Annual Support Cost** | $14.4M | $5.6M | **$8.8M/year** |
| **Weekend Coverage** | None | Full auto (deflection) | New capability |
| **Agent Attrition** | 35% | 18% | -17 p.p. |

### A/B Test Results (4-week experiment)

```
Metric              Control (Human-only)  Treatment (AI-assisted)  Delta
─────────────────  ────────────────────  ───────────────────────  ─────
CSAT               78.2%                 91.5%                   +13.3 pp
Resolution Rate    82.1%                 94.3%                   +12.2 pp  
FCR (First Contact) 65.4%                89.7%                   +24.3 pp
Avg Handle Time    22.4 min              8.6 min                 -61.6%
Reopen Rate        12.3%                 5.1%                    -7.2 pp
```

### Cost Savings Breakdown

```
Total Annual Savings: $8,800,000

┌────────────────────────────────────────────────────────────┐
│  Labor Cost Reduction (45 FTE × $90K avg)      $4,050,000  │
│  Infrastructure (servers, OpenAI API cost)     -$650,000   │
│  Training & Ramp-up Savings                     $380,000   │
│  Overtime Reduction (peak hours)                $420,000   │
│  Escalation Cost Reduction                      $2,100,000 │
│  Agent Attrition Savings                        $1,200,000 │
│  Weekend Auto-Resolution (new coverage)          $1,300,000 │
├────────────────────────────────────────────────────────────┤
│  Net Annual Savings                              $8,800,000 │
└────────────────────────────────────────────────────────────┘
```

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Progressive escalation was key** — Starting with simple, high-confidence responses built trust. We launched with 40% deflection and gradually raised thresholds.

2. **Re-ranking is essential** — Cohere Rerank improved top-1 accuracy from 0.72 to 0.88. Raw vector similarity alone wasn't sufficient for support context.

3. **Structured handoff summaries** — Agents consistently reported they could resolve handed-off tickets 40% faster because the AI provided a complete context summary.

4. **Feedback loop integration** — CSAT feedback on AI-resolved tickets was fed back into fine-tuning prompts. This loop improved deflection rate by 15% over 3 months.

### ❌ What Went Wrong

1. **Over-confident on edge cases** — Initial confidence threshold was 0.7, leading to 12% bad auto-resolutions. We raised to 0.85 and saw error rate drop to 2%.

2. **Underestimated language support complexity** — Japanese and Arabic needed separate embedding models. We initially used a single model for all languages.

3. **Hallucination in billing responses** — The model occasionally invented pricing tiers. We added a strict "forbidden knowledge" constraint in the system prompt.

4. **Latency spikes during burst traffic** — Kafka queue helped buffer, but we needed to add auto-scaling on the LLM inference endpoints.

### ⚠️ Critical Warnings

```
! WARNING: NEVER auto-resolve security-related tickets.
! WARNING: Always log and audit ALL AI responses for compliance.
! WARNING: Implement kill-switch to revert to human-only mode.
! WARNING: Monitor for drift in customer sentiment over time.
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph over simple chains** | Needed branching logic for different intents + conditional escalation |
| **Self-hosted ChromaDB** (not Pinecone) | Cost savings of ~$50K/year; privacy compliance |
| **GPT-4 over GPT-3.5** | 23% better accuracy on response quality; worth the 3x cost |
| **Async FastAPI** (not Flask) | Needed to handle 50+ concurrent requests during peak hours |
| **Human-in-the-loop** (not full auto) | Legal/compliance requirements; customers expected human option |

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-CUSTOMER-SUPPORT-AGENT/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── logging.yaml
│   ├── model_config.yaml
│   └── prompts/
│       ├── classifier.yaml
│       ├── response_generator.yaml
│       └── handoff_summary.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── zendesk_client.py
│   │   ├── knowledge_base_loader.py
│   │   └── chunking.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── sentiment.py
│   │   ├── retriever.py
│   │   ├── response_generator.py
│   │   ├── supervisor_graph.py
│   │   ├── confidence_check.py
│   │   └── handoff.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── webhooks.py
│   │   └── middleware.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── tracing.py
│   │   └── alerts.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── rate_limiter.py
│       └── secrets_manager.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_classifier.py
│   │   ├── test_retriever.py
│   │   ├── test_response_generator.py
│   │   └── test_graph.py
│   ├── integration/
│   │   ├── test_zendesk_client.py
│   │   ├── test_chromadb.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_tickets.json
│       └── mock_knowledge_base.json
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-retrieval-evaluation.ipynb
│   └── 03-response-quality-analysis.ipynb
│
├── scripts/
│   ├── seed_knowledge_base.py
│   ├── simulate_traffic.py
│   ├── evaluate_retrieval.py
│   └── deploy.sh
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment-api.yaml
│   ├── deployment-worker.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   └── hpa.yaml
│
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── modules/
│   │   ├── eks/
│   │   ├── rds/
│   │   └── redis/
│   └── environments/
│       ├── dev.tfvars
│       ├── staging.tfvars
│       └── prod.tfvars
│
└── docs/
    ├── architecture.md
    ├── api.md
    ├── deployment.md
    ├── runbook.md
    └── compliance.md
```

### Key Configuration Files

**configs/config.yaml:**
```yaml
project:
  name: customer-support-agent
  environment: production

llm:
  provider: openai
  model: gpt-4-turbo
  temperature: 0.2
  max_tokens: 1024

retrieval:
  chunk_size: 800
  chunk_overlap: 200
  top_k_initial: 10
  top_k_final: 5
  embedding_model: text-embedding-3-large
  vector_store: chromadb
  chromadb_path: /data/chromadb

zendesk:
  subdomain: your-subdomain
  api_version: v2
  rate_limit: 100  # requests per minute

thresholds:
  confidence_auto_resolve: 0.85
  confidence_escalate: 0.5
  sentiment_negative_escalate: true

monitoring:
  langsmith_tracing: true
  prometheus_metrics: true
  alert_email: ops@company.com

languages:
  supported:
    - en
    - es
    - fr
    - de
    - ja
  default: en
```

**docker-compose.yml:**
```yaml
version: "3.9"

services:
  api:
    build: .
    command: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
      - chromadb
    volumes:
      - ./data:/data

  worker:
    build: .
    command: celery -A src.worker worker --loglevel=info
    env_file: .env
    depends_on:
      - redis
      - chromadb
    volumes:
      - ./data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:0.5.5
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  chroma_data:
```

### API Endpoints (FastAPI)

```python
# src/serving/api.py
from fastapi import FastAPI, HTTPException
from src.serving.schemas import (
    TicketRequest, TicketResponse, HealthResponse
)
from src.pipelines.supervisor_graph import support_graph

app = FastAPI(title="AI Support Agent API", version="2.0.0")

@app.post("/tickets/process", response_model=TicketResponse)
async def process_ticket(ticket: TicketRequest):
    """Process an incoming support ticket end-to-end."""
    initial_state = {
        "ticket_id": ticket.ticket_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "customer_id": ticket.customer_id,
    }
    result = await support_graph.ainvoke(initial_state)
    return TicketResponse(
        ticket_id=result["ticket_id"],
        response=result["response"],
        confidence=result["confidence"],
        escalated=result["should_escalate"],
        resolved=not result["should_escalate"]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime_seconds=12345
    )
```

### Makefile

```makefile
.PHONY: install test run clean deploy

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env

test:
	pytest tests/unit/ --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

run:
	docker-compose up --build

run-dev:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

seed-db:
	python scripts/seed_knowledge_base.py

evaluate:
	python scripts/evaluate_retrieval.py

deploy:
	cd terraform && terraform apply -auto-approve
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/

clean:
	rm -rf venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	docker-compose down -v
```

### Getting Started

```bash
# 1. Clone the template
cp -r TEMPLATE-CUSTOMER-SUPPORT-AGENT ~/my-support-bot
cd ~/my-support-bot

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (OpenAI, Zendesk, Cohere)

# 3. Install dependencies
make install

# 4. Seed the knowledge base
python scripts/seed_knowledge_base.py \
  --source ./data/sample_kb/ \
  --collection support_docs

# 5. Run tests
make test

# 6. Start the service
make run-dev

# 7. Test with a sample ticket
curl -X POST http://localhost:8000/tickets/process \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "test-001",
    "subject": "Cannot login to dashboard",
    "description": "Getting 403 error when trying to access admin dashboard",
    "customer_id": "cust-12345"
  }'
```

---

## 📚 References & Further Reading

### Academic Papers
- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Komeili et al. (2022) — "Internet-Augmented Dialogue Generation" — [arXiv:2107.07566](https://arxiv.org/abs/2107.07566)
- Shinn et al. (2023) — "Reflexion: Language Agents with Verbal Reinforcement Learning" — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

### Official Documentation
- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain: https://python.langchain.com/docs
- ChromaDB: https://docs.trychroma.com/
- Zendesk API: https://developer.zendesk.com/api-reference/
- OpenAI API: https://platform.openai.com/docs

### Related Blog Posts
- "Building a Production-Grade AI Support Agent" — Intercom Engineering Blog
- "How Zendesk Uses AI to Deflect 80% of Tier-1 Tickets" — Zendesk Blog
- "Our Journey Building an LLM-Based Support System" — Shopify Engineering

### Monitoring & Observability
- LangSmith: https://smith.langchain.com/
- Prometheus + Grafana: https://prometheus.io/docs/visualization/grafana/
- OpenTelemetry: https://opentelemetry.io/docs/

---

> **Next**: [03-Predictive-Maintenance.md](03-Predictive-Maintenance.md) — Predictive maintenance for manufacturing with IoT sensor pipelines and anomaly detection.
