# 11 — Agentic Git: Evaluation and Benchmarking

> How do you know if your agent's commits are good? The benchmarks that measure agentic-git performance, the eval patterns for custom repos, and the metrics the 2026 ecosystem has converged on.

## 1. Why "agentic git" needs its own benchmarks

Standard LLM benchmarks (MMLU, HumanEval, GSM8K) measure the model's general capability, not the agent's ability to operate inside a repository. SWE-bench gets closer — it asks an agent to fix a real GitHub issue — but stops at the diff. It doesn't measure:

- Commit hygiene (message quality, conventional commits, Lore trailers)
- Multi-file consistency across a feature
- Branch and PR hygiene
- Behavior under conflict with other agents
- Test discipline
- Documentation/discoverability of decisions
- Cost efficiency

A new benchmark class is needed.

## 2. The 2026 benchmarks

### 2.1 SWE-bench (`SWE-bench/SWE-bench`, 5.2k★)

The foundational benchmark. From the README:

> SWE-bench: Can Language Models Resolve Real-world Github Issues?

It works by:

1. Collecting real GitHub issues and their resolving PRs from popular Python repos.
2. Checking out the codebase at the pre-PR state.
3. Giving an agent the issue text and a test patch (the new tests the PR added).
4. Asking the agent to produce a patch.
5. Applying the agent's patch and running the test patch.
6. Scoring: percentage of tests that pass.

Variants:

- **SWE-bench Verified** (500 hand-verified instances) — the de-facto standard
- **SWE-bench Lite** (smaller subset)
- **SWE-bench Multimodal** (with image attachments)

What it measures: *can the agent fix this real-world issue?*

What it doesn't measure: commit message quality, branch strategy, multi-agent coordination.

### 2.2 mini-SWE-agent (`SWE-agent/mini-swe-agent`, 5.2k★)

> The 100 line AI agent that solves GitHub issues or helps you in your command line

A minimal agent harness designed to be a strong baseline for SWE-bench. Runs entirely in a bash sandbox; uses `git apply` to apply patches. The constraint to ~100 lines forces focus.

### 2.3 SWELancer-Benchmark (`openai/SWELancer-Benchmark`, 1.4k★)

> Can Frontier LLMs Earn $1M from Real-World Freelance Software Engineering?

OpenAI's benchmark that takes the SWE-bench idea further: real freelance tasks from real companies, real money attached. As of mid-2026, frontier models earn ~$100K-$400K from this benchmark, vs. $1M for a perfect human.

What it adds: real economic signal, real-world messiness (multiple files, integration with real services, multi-day tasks).

### 2.4 AutoCodeRover (`AutoCodeRoverSG/auto-code-rover`, 3.1k★)

> A project structure aware autonomous software engineer aiming at autonomous program improvement

A multi-stage agent that performs context retrieval, then code modification, then test generation. Reports SWE-bench numbers as the primary metric but also tracks intermediate stages.

### 2.5 AgentBench (`THUDM/AgentBench`, 3.5k★, ICLR'24)

> A Comprehensive Benchmark to Evaluate LLMs as Agents

The broader agent benchmark suite. Includes code tasks (the closest to agentic-git) plus web navigation, game playing, and operating system tasks.

### 2.6 Terminal-Bench (`harbor-framework/terminal-bench`, 2.4k★)

> A benchmark for LLMs on complicated tasks in the terminal

From the same team as `harbor-framework/harbor`. Tests whether agents can navigate, edit, and operate inside a shell environment. Directly relevant to git CLI fluency.

### 2.7 Chronos (`Kodezi/Chronos`, 4.9k★)

> Kodezi Chronos is a debugging-first language model that achieves state-of-the-art...

A model fine-tuned specifically for debugging. Claims SOTA on SWE-bench.

### 2.8 Harbor (`harbor-framework/harbor`, 2.5k★)

> Harbor is a framework for running agent evaluations and creating and using RL environments

The infrastructure for running all the above. If you want to build your own benchmark, Harbor is the starting point.

### 2.9 Refact (`smallcloudai/refact`, 3.6k★)

> AI Agent that handles engineering tasks end-to-end: integrates with developers'...

Commercial agent with a public benchmark dashboard. Useful for tracking SOTA across the field.

## 3. The metrics the field has converged on

### 3.1 Pass rate

The dominant metric. For a benchmark of N tasks, the score is `tasks_with_passing_tests / N`. Variations:

- **pass@1** — single attempt, single chance
- **pass@k** — k attempts, scored if any one passes (estimates capability ceiling)
- **best-of-n** — n attempts, scored by some selection criterion

### 3.2 Modified-line precision

For agentic-git specifically: of the lines the agent changed, what fraction were necessary?

```python
def modified_line_precision(agent_diff: str, gold_diff: str) -> float:
    agent_lines = set(extract_changed_lines(agent_diff))
    gold_lines = set(extract_changed_lines(gold_diff))
    if not agent_lines:
        return 0.0
    return len(agent_lines & gold_lines) / len(agent_lines)
```

Measures: does the agent touch only what it should?

### 3.3 Commit hygiene score

Composite metric:

| Sub-metric | What it measures | Weight |
|---|---|---|
| Subject matches conventional commit pattern | Format compliance | 10% |
| Body present when rationale is non-obvious | Communication quality | 15% |
| Trailers present (Co-authored-by, Refs) | Attribution | 10% |
| No debug output (`console.log`, `print("DEBUG")`) | Discipline | 20% |
| No secrets in commit | Safety | 25% |
| Tests included with the change | Quality | 20% |

A 2026 custom benchmark for agentic-git would score this for every PR.

### 3.4 Cost-to-resolve

Total tokens × $/token + compute time. The metric that decides which agent a team can actually afford to deploy.

```python
def cost_per_resolved_task(total_tokens: int, model: str, wall_time_s: float) -> float:
    cost = total_tokens * MODEL_PRICING[model] / 1_000_000
    # Add compute cost (rough)
    cost += wall_time_s * 0.001  # $0.001/sec for typical CPU
    return cost
```

SOTA on SWE-bench means nothing if it costs $50 per issue.

### 3.5 Wall-clock-to-merge

How long from "agent starts task" to "PR merged to main"? Includes:

- Agent reasoning time
- CI time
- Human review time
- Rebase/conflict resolution time

A useful KPI for production teams.

## 4. Custom repo evals

For your own repo, the 2026 pattern is a custom eval set:

```python
# evals/coding_tasks.yaml
tasks:
  - id: fix-login-race
    description: "Fix the race condition in login refresh"
    seed_commit: abc123def
    test_files:
      - tests/auth/login.test.ts
    expected_files_modified:
      - src/auth/login.ts
      - tests/auth/login.test.ts
    expected_test_outcome: 15/15 passing
    max_tokens: 30000
    max_wall_time_s: 600

  - id: add-rate-limit
    description: "Add per-IP rate limiting to /api/login"
    seed_commit: def456
    test_files:
      - tests/api/rate-limit.test.ts
    expected_files_modified:
      - src/api/middleware/rate-limit.ts
      - tests/api/rate-limit.test.ts
      - config/ratelimit.yaml
    expected_test_outcome: 8/8 passing
    max_tokens: 25000
```

Then a runner:

```python
# evals/run.py
import yaml, subprocess, json

def run_eval(task: dict, agent_cmd: str) -> dict:
    # 1. Reset repo to seed commit
    subprocess.run(["git", "reset", "--hard", task["seed_commit"]], check=True)
    subprocess.run(["git", "clean", "-fd"], check=True)

    # 2. Run the agent
    result = subprocess.run(agent_cmd + " " + task["description"],
                          capture_output=True, text=True, timeout=task["max_wall_time_s"])

    # 3. Compute metrics
    diff = subprocess.run(["git", "diff", task["seed_commit"]], capture_output=True, text=True).stdout
    files_modified = parse_diff_files(diff)

    # 4. Run tests
    test_result = subprocess.run(["pnpm", "test"], capture_output=True, text=True)

    return {
        "task_id": task["id"],
        "files_modified": files_modified,
        "files_expected": task["expected_files_modified"],
        "precision": precision(files_modified, task["expected_files_modified"]),
        "recall": recall(files_modified, task["expected_files_modified"]),
        "test_pass": test_outcome(test_result),
        "tokens_used": parse_token_count(result.stdout),
        "wall_time_s": ...,
        "commit_hygiene_score": score_commit_hygiene(diff),
    }
```

This is what every serious agentic-git team runs before promoting an agent from staging to production.

## 5. Evaluation-as-CI

The 2026 pattern: a separate CI pipeline that runs eval tasks against the latest version of the agent:

```yaml
# .github/workflows/agent-eval.yml
name: Agent Eval
on:
  schedule:
    - cron: '0 6 * * *'  # daily at 6 AM
  workflow_dispatch:

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: pnpm install
      - name: Run agent evals
        run: pnpm tsx evals/run.ts --report results.json
      - name: Compare to last run
        run: pnpm tsx evals/compare.ts results.json
      - name: Comment on regression
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚨 Agent eval regression: see artifacts for details.'
            })
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: agent-eval-results
          path: results.json
```

When a new model version drops, when you change the agent harness, when you change the prompt — you know immediately whether the agent got better or worse on your actual repo.

## 6. Cross-agent evaluation

How does your agent compare to others on the same tasks? The 2026 pattern:

```python
# Run the same eval against multiple agents
agents = {
    "claude-opus-4.5": "claude code --model claude-opus-4.5",
    "gpt-5": "codex --model gpt-5",
    "cursor-cloud": "cursor-agent --cloud",
    "your-fine-tune": "your-agent --path ./your-finetuned-model",
}

results = {}
for name, cmd in agents.items():
    results[name] = run_eval_suite(tasks, cmd)
```

Output:

```text
                    pass@1   avg-tokens   $/task   commit-hygiene   wall-time-min
claude-opus-4.5     72.3%    18,400       $0.55    8.4/10           6.2
gpt-5               69.1%    22,100       $0.62    7.9/10           5.8
cursor-cloud        65.4%    31,200       $1.10    8.1/10           12.4
your-fine-tune      74.8%    12,800       $0.18    6.2/10           4.1
```

Your fine-tuned agent is best on pass rate and cost — but worst on commit hygiene. Decide what's important to you.

## 7. The field-specific eval

Different domains need different evals:

### 7.1 Frontend eval

- Pixel-diff tests (does the rendered DOM match the snapshot?)
- Lighthouse score (does the change regress performance?)
- Accessibility score (axe-core)
- Visual regression (Percy, Chromatic)

### 7.2 Backend eval

- API contract tests (does the endpoint still respond per OpenAPI spec?)
- Load tests (k6, Locust, Artillery)
- Database migration validity
- Type-check + integration tests

### 7.3 Infrastructure eval

- Terraform plan output (does `terraform plan` show the expected changes?)
- Cost estimate (does the change stay within budget?)
- Security scan (Checkov, tfsec)
- Drift detection (does the deployed state match the manifest?)

### 7.4 Data/ML eval

- Model performance (accuracy, F1, etc.)
- Data validation (Great Expectations)
- Pipeline DAG validity (does the new node connect correctly?)
- Reproducibility (does the pipeline produce the same output on rerun?)

## 8. The eval set as a forcing function

The best way to make your agent better: build an eval set.

Once you have 50+ eval tasks that measure the things you care about (commit hygiene, test discipline, file scope, security), you can:

- A/B test prompt changes
- Compare model versions objectively
- Fine-tune against the failure modes
- Track improvement over time

The eval set becomes the single most valuable artifact in your agentic-git setup.

## 9. The 12-month horizon for agent evals

What to expect by Q2 2027:

- **Standardized agent-trace replay** (entireio + h5i schema merge). Benchmarks will be reproducible bit-for-bit.
- **Per-commit evals as a service.** "Did this agent commit improve the codebase?" becomes a queryable score.
- **Multi-agent coordination benchmarks.** Tests like "two agents work the same repo without conflicts."
- **Lore-coverage benchmarks.** What percentage of files have meaningful Lore atoms? Higher = better.
- **Reputation-aware benchmarks.** Benchmarks that weight scores by the agent's track record.

## 10. See also

- Cat 23 — *Local AI Inference & Self-Hosting* (running your own evals locally)
- Cat 12 — *AI Agent Architecture* (what makes a good agent)
- **09-Replay-Debug-and-Observability** — using eval results for debugging
- **07-Prompt-and-Commit-Patterns** — what "good commits" look like (the thing evals measure)
