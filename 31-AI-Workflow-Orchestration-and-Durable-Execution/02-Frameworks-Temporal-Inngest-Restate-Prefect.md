# 02 — Frameworks: Temporal, Inngest, Restate, Prefect 3.0

> Last updated: June 19, 2026

This document provides a deep, code-level comparison of the four major production-grade durable execution engines. Each one solves the same core problem — surviving crashes, persisting state, supporting long-running workflows — but the developer experience, language support, deployment model, and AI-specific affordances differ substantially. If you're picking a framework today, this is the document you need.

---

## 1. The Four Contenders

| Framework | Founded | Language focus | License | AI-specific features | Hosted option |
|-----------|---------|----------------|---------|----------------------|---------------|
| **Temporal** | 2019 (Cadence fork) | Go, Java, TypeScript, Python, .NET, PHP, Ruby | MIT (server) + Business (cloud) | Workflow versioning, time-skipping tests, OpenTelemetry built-in | Temporal Cloud |
| **Inngest** | 2022 | TypeScript, Python (beta) | Apache 2.0 (SDK) + SaaS | Built-in LLM rate limit handling, step-level memoization, replay-safe LLM calls | Inngest Cloud (free tier) |
| **Restate** | 2022 | TypeScript, Java, Python, Go, Rust | Apache 2.0 | Durable RPC, idempotency built-in, suspend/resume via Kafka | Restate Cloud |
| **Prefect 3.0** | 2018 | Python | Apache 2.0 | Dynamic workflows, autonomous mode, "AI workflows as a primitive" | Prefect Cloud |

### 1.1 When to pick which (decision matrix)

| If you need... | Pick |
|----------------|------|
| **Most mature, largest community** | Temporal |
| **Pure serverless (Lambda, Vercel, Cloudflare Workers)** | Inngest |
| **Strongest type safety (TypeScript-first)** | Restate |
| **Python-first with familiar DAG syntax** | Prefect 3.0 |
| **Kafka-native event sourcing** | Restate |
| **Multi-language teams (Go backend, TS frontend, Python data)** | Temporal |
| **Quick prototype, hosted free tier** | Inngest or Prefect |
| **On-prem, no vendor cloud** | Temporal (self-host) or Restate (self-host) |
| **AI-native primitives (LLM-specific affordances)** | Inngest or Prefect 3.0 |

---

## 2. Temporal — The Default Choice

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Temporal Server                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Frontend   │  │ History    │  │ Matching   │            │
│  │ Service    │  │ Service    │  │ Service    │            │
│  │ (gRPC)     │  │ (Cassandra/│  │ (Task      │            │
│  │            │  │  Postgres) │  │  Queues)   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
        ↑              ↑              ↑
        │              │              │
   ┌────┴───┐    ┌────┴───┐    ┌────┴───┐
   │Worker A│    │Worker B│    │Worker C│
   │Python  │    │TypeScr │    │Go      │
   └────────┘    └────────┘    └────────┘
```

- **Frontend Service**: gRPC endpoint that workers and clients connect to
- **History Service**: persists the event log (Cassandra, Postgres, MySQL)
- **Matching Service**: polls for tasks, dispatches to workers
- **Workers**: stateless, run your workflow + activity code, poll the server for tasks

### 2.2 Hello World (Python)

```python
# workflow.py
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

@activity.defn
async def greet(name: str) -> str:
    return f"Hello, {name}!"

@workflow.defn
class GreetingWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        result = await workflow.execute_activity(
            greet, name,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=5,
                backoff_coefficient=2.0,
            ),
        )
        return result
```

```python
# worker.py
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from workflow import GreetingWorkflow, greet

async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="greeting-queue",
        workflows=[GreetingWorkflow],
        activities=[greet],
    )
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# client.py
import asyncio
from temporalio.client import Client
from workflow import GreetingWorkflow

async def main():
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        GreetingWorkflow.run,
        "Alice",
        id="greeting-alice-001",
        task_queue="greeting-queue",
    )
    print(f"Result: {result}")  # "Hello, Alice!"

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.3 AI agent example (Python) — the killer use case

```python
from datetime import timedelta
from dataclasses import dataclass
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
import httpx

@dataclass
class ResearchRequest:
    query: str
    user_id: str

@dataclass
class ResearchResult:
    summary: str
    sources: list[str]
    cost_usd: float

# === ACTIVITIES (the I/O) ===

@activity.defn
async def call_llm(prompt: str, model: str = "gpt-4o") -> dict:
    """Calls an LLM with retry on 429/503."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {activity.info().workflow_run_id}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        r.raise_for_status()
        return r.json()

@activity.defn
async def web_search(query: str) -> list[dict]:
    """Web search with retry on transient failures."""
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.tavily.com/search", json={"query": query})
        r.raise_for_status()
        return r.json()["results"]

@activity.defn
async def send_to_human_review(result: ResearchResult) -> str:
    """Notifies human reviewers and returns a review ID."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://review.example.com/queue",
            json={"result": result.__dict__}
        )
        return r.json()["review_id"]

# === WORKFLOW (the orchestration) ===

@workflow.defn
class ResearchAgentWorkflow:
    def __init__(self):
        self._status = "starting"
        self._approved = False
        self._reviewer = None

    @workflow.run
    async def run(self, req: ResearchRequest) -> ResearchResult:
        self._status = "planning"
        plan_resp = await workflow.execute_activity(
            call_llm,
            f"Create a research plan for: {req.query}. Return 5 search queries as JSON.",
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
                non_retryable_error_types=["ValueError"],
            ),
        )
        queries = plan_resp["choices"][0]["message"]["content"]

        # Parallel web searches — partial failure handling
        self._status = "searching"
        search_tasks = [
            workflow.execute_activity(
                web_search, q,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            for q in queries
        ]
        # gather with return_exceptions=True — survive partial failures
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        successful = [r for r in all_results if not isinstance(r, Exception)]
        failed = [r for r in all_results if isinstance(r, Exception)]

        if len(successful) < 2:
            raise workflow.ApplicationError(
                f"Only {len(successful)}/5 searches succeeded. Failed: {failed}"
            )

        self._status = "synthesizing"
        summary_resp = await workflow.execute_activity(
            call_llm,
            f"Synthesize these results into a 500-word summary:\n{successful}",
            start_to_close_timeout=timedelta(minutes=5),
        )
        summary = summary_resp["choices"][0]["message"]["content"]

        result = ResearchResult(
            summary=summary,
            sources=[r["url"] for r in successful],
            cost_usd=0.15,  # in production, sum token costs from responses
        )

        # High-cost action — require human approval
        self._status = "awaiting_approval"
        await workflow.execute_activity(
            send_to_human_review, result,
            start_to_close_timeout=timedelta(minutes=2),
        )
        await workflow.wait_condition(lambda: self._approved, timeout=timedelta(days=7))

        if not self._approved:
            self._status = "rejected"
            return result  # Caller will see the result, but the workflow ended in "rejected" state

        self._status = "completed"
        return result

    @workflow.signal
    async def approve(self, reviewer: str) -> None:
        self._approved = True
        self._reviewer = reviewer

    @workflow.query
    def status(self) -> dict:
        return {
            "status": self._status,
            "approved": self._approved,
            "reviewer": self._reviewer,
        }
```

### 2.4 Temporal's strengths

- **Multi-language**: Go, Java, TS, Python, .NET, PHP, Ruby — the broadest SDK coverage
- **Mature**: Battle-tested at Uber, Netflix, Datadog, Snap, Airbnb, Coinbase
- **Cloud or self-host**: Temporal Cloud (managed) or self-host on Kubernetes
- **Versioning**: First-class workflow versioning (`workflow.get_version(...)`)
- **OpenTelemetry**: Built-in distributed tracing
- **Test framework**: `temporalio.testing.WorkflowEnvironment` with time-skipping

### 2.5 Temporal's weaknesses

- **Operational overhead**: Self-hosting Temporal is hard (Cassandra + multiple services)
- **Python SDK lag**: Python SDK is less mature than Go/TS — some advanced features land later
- **Cost**: Temporal Cloud is expensive at scale ($0.000025 per workflow transition + storage)
- **No built-in LLM primitives**: You wrap LLM calls in activities manually

---

## 3. Inngest — The Serverless-First Choice

### 3.1 Architecture

Inngest is designed for **serverless-first** deployment (Vercel, Cloudflare Workers, AWS Lambda, Netlify):

```
┌──────────────────────────────────────────┐
│            Inngest Dev Server             │
│         (single binary, local dev)        │
│                                            │
│   ┌──────────┐    ┌──────────┐           │
│   │ Event    │    │ Function │           │
│   │ Log      │    │ Runner  │           │
│   │ (Postgres)│    │ (stateless)│         │
│   └──────────┘    └──────────┘           │
└──────────────────────────────────────────┘
              ↑ polling / HTTP
              │
       ┌──────┴──────┐
       │ Serverless  │
       │ Functions   │
       │ (Vercel,    │
       │  Lambda,    │
       │  Cloudflare)│
       └─────────────┘
```

### 3.2 Hello World (TypeScript)

```typescript
// inngest/functions/onboarding.ts
import { inngest } from "./client";

export const onboardingWorkflow = inngest.createFunction(
  {
    id: "onboarding-workflow",
    concurrency: { limit: 10 },
    retries: 3,
  },
  { event: "user/signed.up" },
  async ({ event, step }) => {
    // step.run wraps an I/O operation
    const profile = await step.run("create-profile", async () => {
      return await db.profiles.create({
        userId: event.data.userId,
        email: event.data.email,
      });
    });

    // step.sleep is a "workflow sleep" — no worker occupied
    await step.sleep("wait-for-warmup", "1d");

    // step.waitForEvent blocks until an external event arrives
    const approval = await step.waitForEvent("wait-for-approval", {
      event: "user/approved",
      timeout: "7d",
      if: `async.data.userId == "${event.data.userId}"`,
    });

    if (!approval) {
      return { status: "approval_timeout" };
    }

    await step.run("send-welcome-email", async () => {
      return await email.send(profile.email, "Welcome!");
    });

    return { status: "onboarded", approver: approval.data.approver };
  },
);
```

### 3.3 AI agent example (TypeScript)

```typescript
// inngest/functions/research-agent.ts
import { inngest } from "./client";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

export const researchAgent = inngest.createFunction(
  {
    id: "research-agent",
    // LLM-specific: rate limit to 10 concurrent calls
    concurrency: {
      limit: 10,
      key: "event.data.userId",
    },
    retries: 5,
  },
  { event: "research/requested" },
  async ({ event, step }) => {
    // step.ai.wrap is a built-in LLM helper that handles 429/503 retries
    const planResp = await step.ai.wrap("plan-research", openai("gpt-4o"), {
      messages: [{
        role: "user",
        content: `Create a research plan for: ${event.data.query}. Return 5 search queries.`,
      }],
    });

    const queries = JSON.parse(planResp.text).queries;

    // Parallel search with built-in LLM rate-limit handling
    const searchResults = await Promise.all(
      queries.map((q: string) =>
        step.run(`search-${q}`, async () => {
          return await tavily.search(q);
        })
      )
    );

    const summary = await step.ai.wrap("synthesize", openai("gpt-4o"), {
      messages: [{
        role: "user",
        content: `Synthesize: ${JSON.stringify(searchResults)}`,
      }],
    });

    // High-cost action — human approval via step.waitForEvent
    await step.run("submit-for-review", async () => {
      return await reviewQueue.submit({ summary: summary.text });
    });

    const approval = await step.waitForEvent("wait-for-approval", {
      event: "research/approved",
      timeout: "7d",
      if: `async.data.userId == "${event.data.userId}"`,
    });

    if (!approval) {
      return { status: "approval_timeout", summary: summary.text };
    }

    return { status: "approved", summary: summary.text, approver: approval.data.approver };
  },
);
```

### 3.4 Inngest's strengths

- **Serverless-native**: Runs on Vercel, Cloudflare, Lambda without infrastructure
- **Built-in LLM helpers**: `step.ai.wrap` handles 429/503 retries, timeouts, cost tracking
- **Free local dev**: Single binary, no Postgres/Cassandra needed
- **TypeScript-first**: Best TS DX of any durable execution framework
- **Step-level memoization**: Automatic — same step call with same input returns cached result
- **Free tier**: Generous (1M function runs/month, 100K step runs/month)

### 3.5 Inngest's weaknesses

- **Newer**: Less battle-tested than Temporal (founded 2022)
- **Python in beta**: TypeScript is first-class; Python SDK is still maturing
- **Vendor-leaning**: Free tier exists, but the production-grade features are paid
- **Limited multi-language**: TS + Python only — no Go, Java, etc.

---

## 4. Restate — The Type-Safe, Kafka-Native Choice

### 4.1 Architecture

Restate is built on top of **Kafka for event sourcing**, with a unique "durable RPC" model:

```
┌──────────────────────────────────────────┐
│             Restate Server                │
│                                            │
│   ┌──────────┐    ┌──────────┐           │
│   │ Invoker  │    │ Workflow │           │
│   │ (handles │    │ Service  │           │
│   │  services)│   │ (handles │           │
│   │           │   │  workflows)│          │
│   └──────────┘    └──────────┘           │
│         ↑                                  │
│         │ Event log                        │
│         ↓                                  │
│   ┌──────────┐                            │
│   │  Kafka   │ (or RocksDB / SQLite)      │
│   └──────────┘                            │
└──────────────────────────────────────────┘
```

### 4.2 Hello World (TypeScript)

```typescript
// restate/services/greeter.ts
import * as restate from "@restatedev/restate-sdk";
import { greeter } from "./greeter-service";

const greeterService = restate.service({
  name: "Greeter",
  handlers: {
    greet: async (ctx: restate.RpcContext, name: string) => {
      // ctx.call is durable — will retry on failure
      const result = await ctx.call(greeter, "formatGreeting", name);
      return result;
    },
  },
});

// Stateful workflow
const onboardingWorkflow = restate.workflow({
  name: "OnboardingWorkflow",
  handlers: {
    run: async (ctx: restate.WorkflowContext, employeeId: string) => {
      const profile = await ctx.call(profileService, "create", employeeId);
      await ctx.sleep(86400_000); // 1 day, in ms
      const approved = await ctx.promise<string>("approval");
      if (!approved) {
        return { status: "rejected" };
      }
      await ctx.call(emailService, "sendWelcome", profile);
      return { status: "onboarded" };
    },
    approve: async (ctx: restate.WorkflowSharedContext, approver: string) => {
      ctx.resolvePromise("approval", approver);
    },
  },
});
```

### 4.3 Restate's strengths

- **Strongest type safety**: TypeScript types are preserved end-to-end (no serialization hacks)
- **Idempotency built-in**: Every service call is automatically idempotent (request keys)
- **Kafka-native**: Can use existing Kafka infrastructure for the event log
- **Lightweight deployment**: Single binary + Kafka (or just RocksDB for dev)
- **Multi-language**: TypeScript, Java, Python, Go, Rust

### 4.4 Restate's weaknesses

- **Smaller community** than Temporal or Inngest
- **Less AI-specific tooling**: No `step.ai.wrap` equivalent
- **Newer pattern**: "Durable RPC" is novel — requires mental model shift

---

## 5. Prefect 3.0 — The Python-First Choice

### 5.1 Architecture

Prefect 3.0 introduced **"dynamic workflows"** and **"autonomous mode"** — workflows that can spawn sub-workflows at runtime, designed for AI agents:

```python
from prefect import flow, task
from prefect.tasks import exponential_backoff

@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=2))
async def call_llm(prompt: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

@task(retries=5)
async def web_search(query: str) -> list[dict]:
    return await tavily.search(query)

@flow(log_prints=True)
async def research_agent(query: str, user_id: str) -> dict:
    # Prefect 3.0 dynamic workflows
    plan = await call_llm(f"Plan: {query}")

    # Parallel with partial failure handling
    results = await asyncio.gather(
        *[web_search(q) for q in parse_queries(plan)],
        return_exceptions=True,
    )

    successful = [r for r in results if not isinstance(r, Exception)]
    if len(successful) < 2:
        raise RuntimeError(f"Only {len(successful)} searches succeeded")

    summary = await call_llm(f"Summarize: {successful}")
    return {"summary": summary, "sources": successful}
```

### 5.2 Prefect 3.0's strengths

- **Python-first**: Best DX for Python data/ML teams
- **Familiar DAG syntax**: Decorator-based, looks like Airflow
- **Free tier**: Generous hosted Prefect Cloud
- **Dynamic workflows**: 3.0 specifically added support for AI agents that mutate their own structure

### 5.3 Prefect 3.0's weaknesses

- **Python only**: No Go/TS SDKs
- **Less rigorous durable execution**: Prefect 3.0 improved here, but still not as rock-solid as Temporal
- **Smaller community** than Temporal for production AI use cases

---

## 6. Head-to-Head Selection Matrix

| Dimension | Temporal | Inngest | Restate | Prefect 3.0 |
|-----------|----------|---------|---------|-------------|
| **Maturity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Python DX** | ⭐⭐⭐⭐ | ⭐⭐ (beta) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TypeScript DX** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| **Serverless-native** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **AI-specific features** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Self-hosting ease** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Free tier** | N/A (cloud paid) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Community size** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Long-running workflows** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Human-in-the-loop** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Recommendation by use case

- **Large enterprise, Python or Go, multi-team** → Temporal
- **Serverless startup, TypeScript, Vercel/Cloudflare** → Inngest
- **Type-safety obsessive, Kafka shop, TS or Go** → Restate
- **Python data/ML team, familiar with Airflow** → Prefect 3.0
- **Mixed team, complex orchestration, vendor-flexible** → Temporal
- **AI agent framework (LangGraph/CrewAI) + workflow backend** → Inngest or Temporal

---

## 7. The Hidden Cost: Lock-in

A workflow written for Temporal looks very different from one written for Inngest:

- Temporal: `@workflow.defn`, `workflow.execute_activity`, `workflow.wait_condition`
- Inngest: `inngest.createFunction`, `step.run`, `step.waitForEvent`
- Restate: `restate.workflow`, `ctx.call`, `ctx.promise`
- Prefect: `@flow`, `@task`, `exponential_backoff`

**There is no portable "durable workflow" abstraction.** If you pick Temporal, migrating to Inngest later means rewriting every workflow. The "right" choice is to pick a framework you'll still want in 3 years, given your team's growth trajectory.

### 7.1 Reducing lock-in

A few strategies:

1. **Wrap your business logic in plain functions**, and only the orchestration in the framework SDK. The plain functions are then testable without the framework.

2. **Use the framework's "test environment"** so you can run workflows without a real server (e.g., `temporalio.testing.WorkflowEnvironment`, `inngest/test`).

3. **Avoid framework-specific features** (e.g., Temporal's `workflow.get_version`) — they don't migrate.

4. **Keep workflows small** and delegate complex logic to plain services. This is a healthy architecture anyway.

---

## 8. Cross-References

- `01-Overview-and-Durable-Execution-Primitives.md` — the 5 primitives these frameworks expose
- `03-Agent-Native-Orchestration.md` — agent-specific frameworks that sit on top of these
- `04-Patterns-Sagas-Retries-HITL-Compensation.md` — patterns that work across all 4 frameworks
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md` — observability integration
- `30-Small-Language-Models/` — SLM-based agents have different cost/latency profiles affecting framework choice

---

## 9. Resources

- [Temporal documentation](https://docs.temporal.io/)
- [Inngest documentation](https://www.inngest.com/docs)
- [Restate documentation](https://docs.restate.dev/)
- [Prefect 3.0 release notes](https://www.prefect.io/blog/prefect-3-0-is-generally-available)
- [Comparison: Temporal vs Inngest vs Restate](https://www.inngest.com/blog/comparison-temporal-vs-inngest)
- [Mistral Workflows announcement](https://mistral.ai/news/workflows)

---

*This document is part of the AI Knowledge Library — 31-AI-Workflow-Orchestration-and-Durable-Execution directory. Generated by Auto-Enricher cycle 2026-06-19.*
