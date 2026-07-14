# Diffusion & Masked Generative LMs — Tools and Frameworks

> Concrete ecosystem: open-source repos, hosted APIs (Mercury, Gemini Diffusion), runnable examples, deployment snippets, and a buyer's guide for picking a tool. Companion to `03-Technical-Deep-Dive.md`.

---

## 1. Open-source repositories

| Repo | What it provides | License | Use it for |
|------|------------------|---------|-----------|
| **LLaDA (StepFun / PKU)** `github.com/stepfun-ai/LLaDA` | 8B masked diffusion model + inference + training code | Apache-2.0 | Self-host, fine-tune, research |
| **MDLM** `github.com/kuleshov-group/mdlm` | Reference masked diffusion training/sampling | MIT | Learning the objective |
| **SEDD** `github.com/louisluu3010/score_entanglement_diffusion` (and UT Austin originals) | Score-entropy discrete diffusion | MIT | ODE-sampling research |
| **Diffusion-LM** `github.com/XiangLi1999/Diffusion-LM` | Continuous diffusion + controllable gen | MIT | Controllable generation study |
| **torchdiffeq / flow-matching libs** | ODE solvers for SEDD-style sampling | MIT | Custom samplers |
| **Inception Labs Mercury** | Commercial API + some open code/models | Proprietary + partial open | Production low-latency gen |

> Note: repo URLs above follow the canonical project namespaces; verify the exact path on the host forge before cloning (names/forks vary). The **concepts** (masked diffusion, SEDD, MDLM) are stable regardless of the exact repo.

---

## 2. Hosted APIs (2025–2026)

### 2.1 Inception Labs — Mercury
- Positioned as "the first commercial-scale diffusion LLM."
- **Mercury Coder** targets code completion/generation with claimed **5–10x faster** token generation vs comparable AR models at similar quality.
- Access: REST/OpenAI-compatible chat endpoint with a `diffusion_steps` parameter.
- Best for: agent tool-call generation, autocomplete, structured output.

```python
# Mercury (OpenAI-compatible shape, illustrative)
from openai import OpenAI
client = OpenAI(base_url="https://api.inceptionlabs.ai/v1", api_key=KEY)
r = client.chat.completions.create(
    model="mercury-coder-small",
    messages=[{"role":"user","content":"write a python quicksort"}],
    extra_body={"diffusion_steps": 24},   # step budget = latency knob
)
print(r.choices[0].message.content)
```

### 2.2 Google — Gemini Diffusion
- Integrated into the Gemini family; diffusion variant emphasized for **code and long-form speed**.
- Access via the standard Gemini API; model id `gemini-diffusion-*`.
- Best for: high-volume generation inside Google Cloud pipelines.

### 2.3 Self-hosted LLaDA
- 8B weights, runs on a single 24GB GPU (INT4) or 40GB+ (FP16).
- Compatible with `vLLM`-style serving after a small adaptation (step-batched scheduler).

---

## 3. Runnable example: LLaDA-style local inference

```python
# pip install transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaDA is loaded as a causal-LM container but uses BIDIRECTIONAL attention internally
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct",
                                             torch_dtype="auto", device_map="auto")

MASK = tok.mask_token_id

def ask(prompt, steps=32, resp_len=128):
    # build: <prompt> [MASK]*resp_len  (LLaDA uses special chat/role tokens)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    mask = torch.full((1, resp_len), MASK, dtype=ids.dtype, device=ids.device)
    x = torch.cat([ids, mask], dim=1)
    sched = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    for i, t in enumerate(sched):
        logits = model(x, timestep=t).logits          # model accepts timestep
        out = logits.argmax(-1)
        conf = logits.softmax(-1).max(-1).values
        cur = (x[0] == MASK)
        k = max(1, int((cur.sum().item()) / (steps - i)))
        tk = conf.masked_fill(~cur, -1e9).topk(k).indices
        x[0, tk] = out[0, tk]
    return tok.decode(x[0], skip_special_tokens=True)
```

> LLaDA's exact chat template and `timestep` argument differ by release; consult the repo's `generate.py` for the canonical wrapper. The loop above captures the **algorithm**, which is stable.

---

## 4. Serving with vLLM / TGI (adaptation notes)

Standard AR servers assume a KV-cache and token streaming. For diffusion:

1. **Disable KV-cache assumptions** — the model re-encodes the full (small) sequence each step.
2. **Scheduler = step-batched**: collect `N` requests, run `S` steps where each step is a full forward over all `N × T` positions.
3. **Expose `diffusion_steps` as a request param** (latency SLA knob).
4. **Streaming**: emit tokens as their confidence crosses a threshold (confidence-ordered streaming), not strictly left-to-right.

```yaml
# illustrative service config
model: LLaDA-8B-Instruct
engine: step-batch            # not continuous-batch
default_steps: 24
max_steps: 64
stream_mode: confidence_ordered
constraints: json_schema       # native constrained unmasking
```

---

## 5. Tooling by task

| Task | Recommended tool | Why |
|------|------------------|-----|
| Research / learning | MDLM, SEDD repos | clean reference impls |
| Self-host 8B | LLaDA + transformers | Apache-2.0, documented |
| Production low-latency | Mercury API | 5–10x speed, managed |
| Code gen @ scale | Mercury Coder / Gemini Diffusion | tuned for code |
| Structured output | any masked LM + constrained unmasking | valid-by-construction JSON/SQL |
| Long-form | block-diffusion fork | causality + speed |
| Fine-tune | LLaDA + PEFT LoRA | SFT recipe published |

---

## 6. Constrained JSON example (agent tool-call)

Diffusion produces valid JSON natively via constrained unmasking — ideal for `03-Agents/05-Tool-Implementations.md`.

```python
schema = {"type":"object","properties":{
            "action":{"type":"string"},
            "args":{"type":"object"}}, "required":["action","args"]}

def gen_tool_call(prompt):
    # 1) propose a masked response
    # 2) during unmasking, forbid tokens that break the partial JSON
    # 3) return validated dict
    raw = diffusion_sample(model, prompt, resp_len=96, steps=24,
                           constraint=json_constraint(schema))
    return json.loads(raw)      # guaranteed parseable
```

No external grammar engine required — the mask itself enforces structure.

---

## 7. Integration with agent frameworks

Diffusion LMs slot into `03-Agents/` stacks as a **fast generator**:

```
User → Planner(AR, hard reasoning) → DiffLM(generate tool call / draft) → Executor
                                   → DiffLM(generate report) → Critic(AR)
```

Use AR for planning/reasoning, diffusion for the high-volume generation (tool-call JSON, summaries, code). This hybrid is the 2026 best practice.

---

## 8. Edge / on-device

Small masked LMs (1–3B) quantize well (`62-Edge-AI-and-On-Device-Inference/`, `30-Small-Language-Models/`) and are excellent for **on-device autocomplete** where AR latency is painful. Because there's no KV-cache memory growth, diffusion is memory-friendly on edge NPUs for fixed-length outputs.

---

## 9. Monitoring stack

Wire into `20-Agent-Infrastructure-and-Observability/`:

- **Metrics:** `diffusion_steps_used`, `remask_ratio`, `p50/p95_latency`, `tokens_per_sec_equiv`, `constraint_violations`.
- **Traces:** one trace per request spanning `S` steps; annotate step confidence.
- **Golden tests:** a fixed JSON/SQL set regenerated after every model swap.

---

## 10. Buyer's guide

| If you need… | Pick |
|--------------|------|
| Max quality on reasoning | AR (GPT-class) — `02-LLMs/02-Model-Families.md` |
| Lowest latency, high QPS | Mercury / Gemini Diffusion / self-host LLaDA |
| Valid structured output | Masked LM + constrained unmasking |
| Full control / fine-tune | LLaDA-8B (open) |
| On-device autocomplete | Small quantized masked LM |
| Long-form quality | Block-diffusion hybrid |

---

## 11. Common pitfalls when adopting

1. **Treating it like AR** — forgetting the `t` embedding and bidirectional attention → garbage.
2. **Too few steps** — 4 steps is only for autocomplete; chat needs ≥16.
3. **No length signal** — add a length token or fixed response span.
4. **Skipping remasking** — enables repetition.
5. **Measuring only quality** — must also measure speed/cost (`41-AI-Cost-Optimization-and-Enterprise-ROI/`).
6. **AR verifier mismatch** — if you keep an AR judge, calibrate it for diffusion's stylistic differences.

---

## 12. Quick start checklist

- [ ] Pick a system (Mercury API for speed, LLaDA for control).
- [ ] Set a step budget SLA (start 24).
- [ ] Add length handling.
- [ ] Enable remasking.
- [ ] Add constrained unmasking for any structured output.
- [ ] Benchmark quality + latency vs your current AR model.
- [ ] Wire metrics (steps, remask ratio, latency).
- [ ] Hybridize: AR for reasoning, diffusion for generation.

---

## 13. Cross-references

- Agents: `03-Agents/05-Tool-Implementations.md`
- Cost/ROI: `41-AI-Cost-Optimization-and-Enterprise-ROI/`
- Inference perf: `63-GPU-Kernel-and-Inference-Performance-Engineering/`
- Quantization: `02-LLMs/04-Quantization.md`
- Evaluation: `69-AI-Evaluation-and-LLM-Testing/`
- Observability: `20-Agent-Infrastructure-and-Observability/`
- Edge: `62-Edge-AI-and-On-Device-Inference/`, `30-Small-Language-Models/`
- Reasoning (where AR leads): `29-Reasoning-and-Inference-Scaling/`

---

## 15. CI/CD and testing for diffusion models

Treat diffusion generation like any model in `33-AI-Native-Software-Development/` and `69-AI-Evaluation-and-LLM-Testing/`:

- **Golden set**: fixed prompts + expected structured outputs, regenerated on every model/step-budget change.
- **Latency gate**: fail the build if `p95` exceeds the SLA at the chosen step budget.
- **Quality gate**: fail if accuracy drops > X% vs baseline on the golden set.
- **Constraint gate**: 100% of JSON/SQL outputs must parse (constrained unmasking guarantees this, but verify).
- **Drift check**: monitor remask ratio and confidence distribution in production (`20-Agent-Infrastructure-and-Observability/`).

```yaml
# .github/workflows/diffusion-check.yml (illustrative)
jobs:
  eval:
    steps:
      - run: python eval.py --golden golden.jsonl --steps 24 --max_p95 0.2
      - run: python eval.py --constraint json --expect-valid 1.0
```

---

## 16. Cost comparison worked example

Assume a service doing 10M generations/day, avg 200 tokens, A100-hour $2.

| Setup | tok/s/model | time/gen | GPUs | $/day |
|-------|-------------|----------|------|-------|
| AR 7B (60 tok/s) | 60 | 3.3 s | 20 | ~$1,100 |
| Masked 8B (32 steps, 2.0 s) | — | 2.0 s | 12 | ~$660 |
| Mercury-class (5x) | — | 0.7 s | 5 | ~$275 |

Diffusion's better utilization cuts GPU count ~2–4x at parity quality — directly relevant to `41-AI-Cost-Optimization-and-Enterprise-ROI/`.

---

## 17. Cloud deployment snippet (Docker + step-batch server)

```dockerfile
FROM pytorch/pytorch:2.3-cuda12.1
COPY . /app
RUN pip install transformers torch vllm-adaptor
CMD ["python","serve.py","--engine","step-batch","--default-steps","24"]
```

```python
# serve.py (sketch)
app.post("/generate")
def gen(req):
    steps = req.get("diffusion_steps", 24)
    out = diffusion_sample(model, req.prompt, resp_len=req.len, steps=steps,
                           constraint=req.get("schema"))
    return {"text": out, "steps_used": steps}
```

Deploy behind an autoscaler keyed on **request queue depth**, not token rate (since each request does `S` full passes).

---

## 18. Migration guide: AR → diffusion

1. **Inventory** generation calls; tag by latency-sensitivity and structure.
2. **Pilot** on the top latency-sensitive / structured calls (tool-calls, summaries).
3. **Wrap** the diffusion model behind the same chat/completions interface your AR model uses (OpenAI-compatible), exposing `diffusion_steps`.
4. **A/B** quality + latency + cost for 2 weeks.
5. **Expand** to more call types; keep AR for reasoning cores.
6. **Retire** AR only where diffusion meets quality parity at lower cost.

---

## 19. Vendor matrix (mid-2026)

| Vendor | Model | Access | Best for | Notes |
|--------|-------|--------|----------|-------|
| Inception Labs | Mercury / Mercury Coder | API | Code, low-latency | 5–10x speed claim |
| Google | Gemini Diffusion | API (Gemini) | Cloud pipelines | Long-form speed |
| StepFun / PKU | LLaDA 8B | Open weights | Self-host, fine-tune | Apache-2.0 |
| Academic | MDLM / SEDD | Code | Research | Reference impls |
| (your infra) | distilled LLaDA | Self | Edge/隐私 | see `23-Local-AI-Inference-Self-Hosting/` |

---

## 20. Frequently asked adoption questions

**Q: Will my existing RAG/agent code break?**
A: No — wrap diffusion behind the same completions interface; only add `diffusion_steps`. RAG context goes in the prompt like always (`04-RAG/`).

**Q: Do I lose function-calling?**
A: No — emit the tool-call as constrained JSON via masked unmasking (`03-Agents/05-Tool-Implementations.md`).

**Q: Is it safe to use for customer-facing chat?**
A: Yes, with the same guardrails as AR (`18-Agent-Security-and-Trust/`, `55-AI-Ethics-and-Responsible-AI/`). Add output validation.

**Q: How do I debug bad outputs?**
A: Inspect step count, remask ratio, and final confidence (see `20-Agent-Infrastructure-and-Observability/`).

---

## 21. Cross-references

- Agents: `03-Agents/05-Tool-Implementations.md`
- Cost/ROI: `41-AI-Cost-Optimization-and-Enterprise-ROI/`
- Inference perf: `63-GPU-Kernel-and-Inference-Performance-Engineering/`
- Quantization: `02-LLMs/04-Quantization.md`
- Evaluation: `69-AI-Evaluation-and-LLM-Testing/`
- Observability: `20-Agent-Infrastructure-and-Observability/`
- Edge: `62-Edge-AI-and-On-Device-Inference/`, `30-Small-Language-Models/`
- Reasoning (where AR leads): `29-Reasoning-and-Inference-Scaling/`
- RAG: `04-RAG/`
- Self-hosting: `23-Local-AI-Inference-Self-Hosting/`
- Dev/CI: `33-AI-Native-Software-Development/`

---

## 22. Summary

The diffusion/masked-LM toolkit is production-grade in 2026: **LLaDA** (open, 8B, fine-tunable), **Mercury** (managed, 5–10x speed), **Gemini Diffusion** (Google Cloud), plus reference code for **MDLM/SEDD**. Serve **step-batched** (no KV-cache), expose `diffusion_steps` as an SLA knob, and use **constrained unmasking** for valid structured output. Hybridize with AR for reasoning. `05-Future-Outlook.md` covers where this goes next.
