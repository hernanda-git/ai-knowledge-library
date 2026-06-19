# 01 — AI Workflow Orchestration & Durable Execution: Overview

> Last updated: June 19, 2026

In 2024, an AI agent was a Python loop. In 2025, it was a LangGraph state machine. In 2026, it is a **durable workflow** — a function that may run for hours, spawn hundreds of activities, wait for human approval, retry on transient LLM errors, and resume exactly where it left off after a crash. This document explains why this shift happened, what the five core primitives are, and how durable execution differs from "just use Celery".

---

## 1. The Problem: Agents Outgrew the Request/Response Model

### 1.1 The 30-second agent (2024)

In 2024, the typical "agent" was a synchronous function that ran in a single request/response cycle:

```python
# 2024 — naive agent
def research_agent(question: str) -> str:
    plan = llm.invoke(f"Plan how to answer: {question}")
    results = [search_tool(p) for p in plan.steps]
    answer = llm.invoke(f"Synthesize: {plan} {results}")
    return answer
```

This worked when:
- Total runtime < 30 seconds
- No external state to track
- No human approval needed
- Failures were either fatal (return error) or retryable in the same request

### 1.2 The 6-hour agent (2026)

By 2026, the same agent has grown into something else entirely:

```python
# 2026 — durable workflow
@workflow.defn
class ResearchWorkflow:
    @workflow.run
    async def run(self, question: str) -> Report:
        # Step 1: Planning (1 minute, 3 LLM calls)
        plan = await workflow.execute_activity(
            plan_research, question,
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )

        # Step 2: Parallel data collection (5-30 minutes, 10-50 web searches)
        search_results = await asyncio.gather(*[
            workflow.execute_activity(
                web_search, query,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=5, backoff=2.0)
            )
            for query in plan.search_queries
        ])

        # Step 3: Human approval (hours or days — workflow persists)
        await workflow.execute_activity(
            submit_for_review, plan,
            start_to_close_timeout=timedelta(days=7)
        )
        # Wait for human to click "approve" in the dashboard
        approval = await workflow.wait_for_signal("approval")

        if not approval.approved:
            return Report(status="rejected", reason=approval.reason)

        # Step 4: Synthesis (5 minutes, expensive LLM call)
        report = await workflow.execute_activity(
            synthesize_report, plan, search_results,
            start_to_close_timeout=timedelta(minutes=10)
        )

        # Step 5: Persist to durable storage
        await workflow.execute_activity(
            save_report, report,
            start_to_close_timeout=timedelta(minutes=1)
        )

        return report
```

**This workflow may run for hours. It may be killed at any moment. It may be replayed from the beginning. It must produce the same result.**

### 1.3 The wall every agent hits

Every team that ships an agent eventually hits the same wall:

| Wall | Symptom | Naive solution | Why it fails |
|------|---------|---------------|--------------|
| **The crash wall** | Worker dies mid-workflow, all in-memory state lost | Save state to Redis every step | Race conditions, partial saves, hard to reason about |
| **The retry wall** | LLM API returns 503, agent crashes | Wrap in try/except + retry | Retry storm, no idempotency, exponential cost blowup |
| **The long-running wall** | Workflow needs to wait 3 days for a human response | `time.sleep(3 * 86400)` | Worker is occupied, can't process other workflows |
| **The parallel wall** | 50 web searches needed | `asyncio.gather` | One failure cancels all, no partial result handling |
| **The version wall** | Update the workflow code, old workflows in-flight break | None | New code path runs for old workflow, inconsistent state |
| **The observability wall** | "Why did this workflow fail yesterday at 3am?" | Add print statements | Lost on next run, no trace, no history |
| **The state-query wall** | "What's the status of workflow #abc-123?" | Query the database directly | Database is denormalized, out of sync with workflow |

Every one of these walls has the same answer: **durable execution**.

---

## 2. The Five Core Primitives

Durable execution frameworks expose the same five primitives, regardless of vendor (Temporal, Inngest, Restate, Prefect all use these terms or close variants):

### 2.1 Workflow

The **orchestrating function**. The only function allowed to coordinate other steps. Must be **deterministic** — given the same inputs, it must produce the same sequence of activity calls.

```python
# Temporal — Python SDK
@workflow.defn
class OnboardingWorkflow:
    @workflow.run
    async def run(self, employee_id: str) -> str:
        # ✅ Deterministic — only calls activities, no I/O
        profile = await workflow.execute_activity(
            create_profile, employee_id,
            start_to_close_timeout=timedelta(seconds=30)
        )
        await workflow.execute_activity(
            send_welcome_email, profile,
            start_to_close_timeout=timedelta(seconds=10)
        )
        return "onboarded"
```

```typescript
// Inngest — TypeScript SDK
export const onboardingWorkflow = inngest.createFunction(
  { id: "onboarding-workflow" },
  { event: "employee/created" },
  async ({ event, step }) => {
    const profile = await step.run("create-profile", () =>
      createProfile(event.data.employeeId)
    );
    await step.run("send-welcome-email", () =>
      sendWelcomeEmail(profile)
    );
    return "onboarded";
  }
);
```

### 2.2 Activity

The **I/O function**. The only function allowed to talk to the outside world. Activities can be retried, have timeouts, and are the unit of failure isolation.

```python
# Activity — may fail, will be retried
@activity.defn
async def create_profile(employee_id: str) -> Profile:
    response = await httpx.post(
        "https://hr.example.com/profiles",
        json={"employee_id": employee_id}
    )
    response.raise_for_status()
    return Profile(**response.json())
```

**Key insight:** workflows are *cheap to retry* (they just replay from the event log), but activities are *expensive to retry* (they hit external systems). The activity is where idempotency matters most.

### 2.3 Event (Trigger)

The **entry point** for a new workflow instance. Every workflow must be started by an event (Temporal's "schedules" and "signals" can also start workflows).

```python
# Start a workflow
client = await Client.connect("localhost:7233")
handle = await client.start_workflow(
    OnboardingWorkflow.run,
    "emp-12345",
    id=f"onboarding-emp-12345",
    task_queue="onboarding"
)
```

### 2.4 Signal (External Input)

An **event delivered to a running workflow**. Used for human approval, cancellation, parameter updates mid-run, etc.

```python
# Send a signal from outside
async def approve_onboarding(workflow_id: str, approver: str):
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(OnboardingWorkflow.approve, approver)

# Wait for signal inside workflow
@workflow.defn
class OnboardingWorkflow:
    def __init__(self):
        self._approved = False
        self._approver = None

    @workflow.signal
    async def approve(self, approver: str) -> None:
        self._approved = True
        self._approver = approver

    @workflow.run
    async def run(self, employee_id: str) -> str:
        # ... previous steps ...
        await workflow.wait_condition(lambda: self._approved)
        return f"onboarded by {self._approver}"
```

### 2.5 Query (Read State)

A **read-only request for the workflow's current state**. Used to render dashboards, debug "where is workflow X", etc.

```python
@workflow.defn
class OnboardingWorkflow:
    @workflow.query
    def status(self) -> dict:
        return {
            "current_step": self._current_step,
            "approved": self._approved,
            "approver": self._approver,
        }

# Query from outside
handle = client.get_workflow_handle(workflow_id)
status = await handle.query(OnboardingWorkflow.status)
```

---

## 3. How Durable Execution Works (The Event Sourcing Model)

### 3.1 The event log

The framework persists every state-changing decision to an **event log** (typically in a database like Cassandra, Postgres, or MySQL):

```
Workflow ID: onboarding-emp-12345
Event 1: WorkflowStarted { input: "emp-12345" }
Event 2: ActivityScheduled { name: "create_profile", input: "emp-12345" }
Event 3: ActivityCompleted { result: { "id": "p-001", "email": "alice@..." } }
Event 4: ActivityScheduled { name: "send_welcome_email", input: { ... } }
Event 5: ActivityCompleted { result: { "delivered": true } }
Event 6: WorkflowCompleted { result: "onboarded" }
```

### 3.2 Replay = determinism + event log

When the worker crashes and is restarted, the framework **replays the workflow from the beginning**. But on each `execute_activity` call, it checks the event log first:

```python
# Replay logic (simplified)
async def execute_activity(self, name, input, ...):
    if self.event_log.has_completed(name, input):
        # Replay path — return cached result, don't call the activity
        return self.event_log.get_result(name, input)
    else:
        # Fresh path — call the activity
        result = await self.activities[name](input)
        self.event_log.append(ActivityCompleted(name, input, result))
        return result
```

**This is why workflows must be deterministic**: if the workflow produced a different sequence of activity calls on replay, the event log would be wrong. No `time.sleep`, no `random.random()`, no `datetime.now()` — anything non-deterministic must be wrapped in an activity.

### 3.3 Trade-offs

| Trade-off | What it means |
|-----------|---------------|
| **Replay cost** | First few minutes of a long workflow are replayed on every worker restart. Workflows with 1000+ activities can take 5-30 seconds to replay. Mitigations: versioning, deterministic fast-forward. |
| **Event log size** | A workflow with 1000 activities produces 3000+ events (scheduled, started, completed). Long-running workflows can have MB of log. Mitigations: archival, compaction, namespace retention. |
| **Activity must be idempotent** | If the activity fails after the side effect happened (e.g., payment charged but response lost), the retry will charge again. Use idempotency keys. |
| **Vendor lock-in** | Each framework has its own SDK and runtime. Migrating from Temporal to Inngest is a rewrite, not a config change. |
| **Local testing** | Determinism makes local testing easy (use the time-skipping test environment), but debugging replay issues requires learning the framework's debugger. |

---

## 4. The 2024-2026 Shift: From Generic to AI-Native

### 4.1 Phase 1 (2018-2023): Generic durable execution

| Framework | Founded | Original use case | AI adoption |
|-----------|---------|-------------------|-------------|
| **AWS Step Functions** | 2016 | Generic AWS orchestration | Some |
| **Azure Durable Functions** | 2017 | Azure serverless orchestration | Some |
| **Cadence** (Uber) | 2016 | Ride-sharing dispatch | Limited |
| **Temporal** (Cadence fork) | 2019 | Microservices orchestration | Strong |
| **Prefect** | 2018 | Data pipelines | Some |
| **Apache Airflow** | 2014 | Cron replacement / ETL | Some |

These were designed for **deterministic business processes** (order fulfillment, KYC checks, data ETL). They had no awareness of LLM-specific failure modes (rate limits, token exhaustion, hallucination cascades).

### 4.2 Phase 2 (2024-2025): AI workflows on generic engines

Teams started building AI workflows on Temporal, Inngest, etc. They discovered the gaps:
- LLM rate limits (429) needed special retry policies
- LLM non-determinism broke workflow determinism assumptions
- Human approval was needed for high-cost actions, not low-cost ones
- LLM cost tracking had to be plumbed through the workflow

### 4.3 Phase 3 (2025-2026): Agent-native orchestration

The new generation was built specifically for LLM agents:

| Tool | Released | Key innovation |
|------|----------|---------------|
| **LangGraph** | Oct 2024 | State machine primitives designed for agent reasoning |
| **Inngest** (AI features) | 2024-2025 | Built-in LLM rate-limit handling, step-level memoization |
| **Mistral Workflows** | Apr 2026 | First major vendor to ship durable execution as a managed service with native LLM support |
| **Conductor** | 2026 | Open-source "deterministic orchestration for multi-agent AI workflows" |
| **Mcp-Agent** | 2025-2026 | Build effective agents with Model Context Protocol + durable execution |
| **Mistral Workflows** (managed) | Apr 2026 | Mistral's own entry — signals durable execution has gone mainstream |
| **Konductor** | 2026 | AI Orchestration Agent Framework for every dev |
| **Open-artisan** | Mar 2026 | OpenCode plugin for structured AI workflow orchestration |
| **Durable Endpoints** | Feb 2026 | "Make any API endpoint unbreakable" — durable execution as HTTP middleware |
| **Graph-flow** | Apr 2026 | LangGraph-inspired AI agent workflows in Rust |

The trend: **durable execution is moving from infrastructure-team tooling to developer-default**. Mistral's own entry (Apr 2026) is the watershed moment — a frontier model vendor now ships workflow orchestration as a first-class product, alongside the model API.

---

## 5. Why Not Just Use Celery / Task Queues / asyncio.gather?

Common objections, and the answers:

### 5.1 "Celery is good enough"

Celery is a **task queue**, not a **durable execution engine**. The difference:

| Aspect | Celery | Durable execution |
|--------|--------|-------------------|
| State | Stateless tasks | Stateful workflows |
| Crash recovery | Re-run the whole task | Resume from the last completed activity |
| Long waits | Holds a worker slot for hours | Workflow suspends, worker is free |
| Visibility | "Task started / completed" | "Currently on step 47 of 200, last result: X" |
| Signals | Polling or callback | First-class |
| Versioning | Manual | Built-in (worker-version vs workflow-version) |
| Determinism guarantee | None | Required by the framework |

Celery is fine for stateless fire-and-forget work. It is not fine for "schedule a workflow that may run for 3 days, must be observable, and may be updated mid-run."

### 5.2 "asyncio.gather is good enough"

```python
# This LOOKS durable. It is not.
async def run(self):
    results = await asyncio.gather(*[call_llm(q) for q in queries])
    return results
```

Problems:
- One task failure cancels all siblings (without `return_exceptions=True`)
- No retry per task
- No timeout per task
- No state checkpointing — if the process dies, all in-flight tasks are lost
- No way to inspect "which tasks are still running"
- No way to send a signal mid-flight

`asyncio.gather` is a parallelism primitive. Durable execution is an **error-recovery and state-management primitive**. They solve orthogonal problems.

### 5.3 "I'll just save state to Postgres"

```python
# This LOOKS durable. It is not safe.
async def run(self):
    state = await db.load(workflow_id)
    state.step += 1
    await db.save(workflow_id, state)
    result = await call_llm(...)
    state.result = result
    await db.save(workflow_id, state)
```

Problems:
- Concurrent updates to the same workflow (race conditions)
- Partial failures (state saved, activity never ran, or vice versa)
- Compensating actions are ad-hoc
- No query/signal primitives
- You re-invent the wheel badly

You will eventually end up building a worse version of Temporal. Use Temporal.

---

## 6. The 2026 Decision Tree

**You are building an AI agent. What orchestration do you need?**

```
Is the workflow stateless and < 30 seconds?
├── Yes → asyncio.gather or ThreadPoolExecutor
└── No → Does the workflow need to wait for external events (human approval, webhook)?
    ├── No → Is parallel execution with partial-failure handling needed?
    │   ├── No → asyncio.create_task with try/except
    │   └── Yes → Inngest or Restate (lightweight)
    └── Yes → Do you need fine-grained control over state and signals?
        ├── No → AWS Step Functions or Azure Durable Functions (managed, vendor-locked)
        └── Yes → Choose based on requirements:
            ├── LLM-native, OpenAI-compatible → Mistral Workflows
            ├── Python-first, large community → Temporal
            ├── TypeScript-first, event-driven → Inngest
            ├── Need to run on edge/serverless → Restate
            ├── Data-pipeline-flavored → Prefect 3.0
            ├── Agent reasoning state machines → LangGraph + Temporal
            ├── Multi-agent coordination specifically → Conductor or Mcp-Agent
            └── Quick prototype, no infra → LangGraph standalone (in-memory)
```

---

## 7. What This Category Covers

The rest of this category goes deep on:

1. **[02-Frameworks](./02-Frameworks-Temporal-Inngest-Restate-Prefect.md)** — Temporal, Inngest, Restate, Prefect 3.0: architecture, code examples, pricing, selection matrix
2. **[03-Agent-Native-Orchestration](./03-Agent-Native-Orchestration.md)** — LangGraph, Conductor, Mistral Workflows, Mcp-Agent, Open-artisan, Graph-flow, Konductor, Durable Endpoints
3. **[04-Patterns](./04-Patterns-Sagas-Retries-HITL-Compensation.md)** — Sagas, retries, human-in-loop, compensation, idempotency, versioning
4. **[05-Production](./05-Production-Deployment-and-Case-Studies.md)** — Real deployments, observability, migration patterns, common pitfalls

---

## 8. Cross-References

- `03-Agents/01-Agent-Architectures.md` — agent reasoning patterns that must be supported by orchestration
- `03-Agents/02-Multi-Agent-Systems.md` — multi-agent topologies (blackboard, hierarchical) that need orchestration
- `03-Agents/03-Agentic-Frameworks.md` — LangGraph, CrewAI, AutoGen — the layer that often sits on top of orchestration
- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL approval steps are first-class workflow signals
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md` — observability for long-running workflows
- `20-Agent-Infrastructure-and-Observability/07-Agent-Reliability-and-Resilience.md` — reliability patterns that complement durable execution

---

## 9. Resources

- [Temporal: Durable Execution Explained](https://temporal.io/blog/temporal-101-the-go-sdk-temporal-s-deterministic-execution-model)
- [Inngest: Durable Functions in 2025](https://www.inngest.com/blog/announcing-durable-workflows)
- [Restate: The Durable Execution Manifesto](https://restate.dev/blog/durable-execution-explained/)
- [Mistral Workflows announcement (Apr 2026)](https://mistral.ai/news/workflows)
- [AWS Step Functions: When to Use](https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html)
- [Cadence (Temporal's predecessor) — Uber engineering blog](https://www.uber.com/blog/cadence/)

---

*This document is part of the AI Knowledge Library — 31-AI-Workflow-Orchestration-and-Durable-Execution directory. Generated by Auto-Enricher cycle 2026-06-19.*
