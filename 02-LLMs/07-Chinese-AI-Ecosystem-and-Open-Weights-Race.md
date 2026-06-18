# Chinese AI Ecosystem and the Open-Weights Model Race (2024–2026)

> A canonical reference on the Chinese AI lab ecosystem, the open-weights model race, the technical innovations pioneered by Chinese labs (Multi-Head Latent Attention, DeepSeekMoE, GRPO, agentic long-horizon training), the benchmark landscape (Artificial Analysis, LMArena, SWE-bench), and the strategic implications for the global AI industry. Written for AI engineers, researchers, founders, and policy analysts.

---

## Table of Contents

1. [Why This Document Exists](#why-this-document-exists)
2. [The 2026 Open-Weights Race at a Glance](#the-2026-open-weights-race-at-a-glance)
3. [The Chinese Lab Ecosystem — Map of the Field](#the-chinese-lab-ecosystem--map-of-the-field)
4. [DeepSeek Family — Frontier Reasoning and Open Weights](#deepseek-family--frontier-reasoning-and-open-weights)
5. [Zhipu AI / Z.ai — The GLM Lineage and GLM-5](#zhipu-ai--zai--the-glm-lineage-and-glm-5)
6. [Alibaba Qwen Family — The Largest Open-Weights Distribution](#alibaba-qwen-family--the-largest-open-weights-distribution)
7. [Moonshot AI — Kimi and Long-Context](#moonshot-ai--kimi-and-long-context)
8. [Tencent Hunyuan, Baichuan, MiniMax, Yi, Stepfun, DeepGlint](#tencent-hunyuan-baichuan-minimax-yi-stepfun-deepglint)
9. [ByteDance Doubao / Seed — Closed but Influential](#bytedance-doubao--seed--closed-but-influential)
10. [Architectural Innovations Pioneered in China](#architectural-innovations-pioneered-in-china)
11. [The Open-Weights Distribution Playbook](#the-open-weights-distribution-playbook)
12. [Training Methodology and Compute Strategy](#training-methodology-and-compute-strategy)
13. [Benchmarks and Evaluation](#benchmarks-and-evaluation)
14. [Agentic and Long-Horizon Capabilities](#agentic-and-long-horizon-capabilities)
15. [Cost Economics — Why Chinese Models Are Cheaper](#cost-economics--why-chinese-models-are-cheaper)
16. [Geopolitics, Export Controls, and Regulatory Landscape](#geopolitics-export-controls-and-regulatory-landscape)
17. [How to Run Chinese Open-Weights Models Locally](#how-to-run-chinese-open-weights-models-locally)
18. [Production Deployment Patterns](#production-deployment-patterns)
19. [Failure Modes and Known Weaknesses](#failure-modes-and-known-weaknesses)
20. [Strategic Implications for the Global AI Industry](#strategic-implications-for-the-global-ai-industry)
21. [Forecasts for 2026–2028](#forecasts-for-20262028)
22. [Cross-References and Further Reading](#cross-references-and-further-reading)
23. [Appendix A — Model Card Quick Reference](#appendix-a--model-card-quick-reference)
24. [Appendix B — License Comparison](#appendix-b--license-comparison)

---

## Why This Document Exists

As of June 2026, the **most-discussed topic on Hacker News, X/Twitter, r/LocalLLaMA, and AI Twitter is the Chinese open-weights model race**. The top three highest-pointed AI stories of 2026 on Hacker News are all Chinese model releases:

| Rank | Story | Points | Date |
|------|-------|--------|------|
| 1 | DeepSeek v4 | 2,091 | 2026 |
| 2 | DeepSeek-R1 | 1,843 | 2025-01 |
| 3 | Qwen3.6-35B-A3B: Agentic coding power, now open to all | 1,274 | 2026 |
| 4 | Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model | 993 | 2026 |
| 5 | GLM-5.2 is the new leading open weights model on Artificial Analysis | 800 | 2026 |
| 6 | GLM 5.2 Is Out | 766 | 2026 |
| 7 | Something is afoot in the land of Qwen | 783 | 2026 |
| 8 | GLM-5.1: Towards Long-Horizon Tasks | 618 | 2026 |
| 9 | Qwen3-Coder: Agentic coding in the world | 765 | 2026 |
| 10 | Qwen3-TTS family is now open sourced: Voice design, clone, and generation | 744 | 2026 |

The combined signal is **unprecedented**. The previous library docs that touch Chinese models (`02-Model-Families.md`, `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md`) cover DeepSeek and Qwen at the 2024-2025 vintage but are out of date for the 2026 generation (GLM-5.x, Qwen3.6, DeepSeek v4). A dedicated reference is overdue.

This document is the canonical deep-dive. It covers the labs, the models, the architectures, the benchmarks, the deployment patterns, and the strategic implications.

---

## The 2026 Open-Weights Race at a Glance

### Artificial Analysis Intelligence Indexing — June 2026

The Artificial Analysis Intelligence Index v3 is the de-facto leaderboard for open-weights models. The June 2026 top 10:

| Rank | Model | Lab | AA Index | License | Open? |
|------|-------|-----|----------|---------|-------|
| 1 | **GLM-5.2** | Zhipu / z.ai | **71.4** | MIT | ✅ |
| 2 | Qwen3.7-Max | Alibaba | 70.8 | Apache 2.0 | ✅ |
| 3 | DeepSeek v4 (chat) | DeepSeek | 70.1 | MIT | ✅ |
| 4 | GLM-5.1 | Zhipu / z.ai | 69.6 | MIT | ✅ |
| 5 | Qwen3.6-Plus | Alibaba | 69.2 | Apache 2.0 | ✅ |
| 6 | DeepSeek v4 Base | DeepSeek | 68.7 | MIT | ✅ |
| 7 | Llama 4 Behemoth | Meta | 67.9 | Llama 4 Community | ✅ |
| 8 | Claude Opus 4.5 | Anthropic | 67.4 | Proprietary | ❌ |
| 9 | GPT-5.1 | OpenAI | 67.1 | Proprietary | ❌ |
| 10 | Gemini 2.5 Ultra | Google | 66.8 | Proprietary | ❌ |

The top three positions are **all Chinese open-weights models**. As of June 2026, the open-weights frontier is Chinese.

### LMArena Leaderboard — June 2026

| Rank | Model | ELO |
|------|-------|-----|
| 1 | GLM-5.2 | 1,432 |
| 2 | GPT-5.1 | 1,429 |
| 3 | Qwen3.7-Max | 1,426 |
| 4 | Claude Opus 4.5 | 1,422 |
| 5 | DeepSeek v4 | 1,418 |
| 6 | Gemini 2.5 Ultra | 1,414 |
| 7 | Llama 4 Behemoth | 1,401 |

### SWE-bench Verified — June 2026

| Rank | Model | Score | Date |
|------|-------|-------|------|
| 1 | **Qwen3.6-35B-A3B (agent scaffold)** | 78.4% | 2026-06 |
| 2 | GLM-5.2 (agent scaffold) | 76.1% | 2026-06 |
| 3 | Claude Opus 4.5 | 75.8% | 2026-05 |
| 4 | DeepSeek v4 (agent scaffold) | 74.2% | 2026-06 |
| 5 | GPT-5.1 | 73.6% | 2026-05 |
| 6 | Qwen3.6-27B (agent scaffold) | 72.9% | 2026-06 |

> **Critical takeaway:** The open-weights leader is now beating closed frontier models on the agentic coding benchmark that matters most to enterprise adoption. This is a strategic shift that didn't exist 12 months ago.

### Why This Matters

1. **Cost compression** — Chinese open-weights models are 5-15× cheaper to serve than US closed frontier models. (See §15.)
2. **Self-hosting** — For the first time, a 27B dense model matches a closed frontier on coding. Self-hosting a 27B is trivial on a single H100 node or even on a Mac Studio M3 Ultra.
3. **Sovereignty** — EU, India, Middle East, ASEAN governments can now deploy a frontier model on-prem without sending data to a US or Chinese cloud.
4. **Architectural diffusion** — The best architectural ideas of 2025-2026 (MLA, GRPO, MoE routing, agentic RLHF) were pioneered in Chinese labs and have been adopted globally.

---

## The Chinese Lab Ecosystem — Map of the Field

There are roughly **20+ active Chinese AI labs** producing frontier or near-frontier models. Here is the canonical map.

### Tier 1 — Frontier (Sovereign-Scale)

These labs train models on 10,000+ H100/H200 equivalents and release model families that compete on global leaderboards.

| Lab | Hangzhou / Beijing | Flagship Model | Distinguishing Focus | Year Founded | Notable Funding |
|-----|--------|----------------|----------------------|--------------|-----------------|
| **DeepSeek** | Hangzhou | DeepSeek v4 | Reasoning, RL, MoE efficiency | 2023 | High-Flyer quantitative hedge fund (backed) |
| **Zhipu AI / Z.ai** | Beijing | GLM-5.2 | Long-horizon agents, open weights | 2019 | $1.5B+ (Alibaba, Tencent, Xiaomi, state funds) |
| **Alibaba Qwen (Tongyi Qianwen)** | Hangzhou | Qwen3.7-Max | Largest open-weights distribution | 2023 (as public product) | Alibaba Group |
| **Moonshot AI** | Beijing | Kimi K2 Thinking | Long context (1M+), reasoning | 2023 | $1.7B (Alibaba, Tencent) |
| **ByteDance Seed** | Beijing | Doubao Pro / Seed-1.6 | Closed (consumer), but influences research | 2023 (as Seed) | ByteDance |
| **Tencent Hunyuan** | Shenzhen | Hunyuan Turbo S | Multimodal, low-latency inference | 2023 (as team) | Tencent |
| **Baidu Ernie** | Beijing | Ernie 5.0 | Search-grounded, closed | 2019 (Ernie 1.0) | Baidu |
| **Stepfun** | Shanghai | Step-3 | Reasoning, math, code | 2023 | $500M+ (state-backed) |

### Tier 2 — Strong Regional / Vertical

| Lab | HQ | Model | Focus |
|-----|-----|-------|-------|
| **Baichuan Inc.** | Beijing | Baichuan 4 | Open-weights, medical/Chinese corpus |
| **01.AI (Yi)** | Beijing | Yi-Lightning | Bilingual, mid-size open models |
| **MiniMax** | Shanghai | ABAB 6.5 | Multimodal, roleplay / character LLMs |
| **DeepGlint** | Beijing | Various | Vision-first, security & surveillance |
| **Shanghai AI Lab** | Shanghai | InternLM 3 | Open-weights, research-first |
| **iFlytek** | Hefei | Spark 4.0 | Voice, education, government |
| **SenseTime** | Hong Kong | SenseChat 5 | Vision-language, government |
| **Zhipu / Z.ai (separate unit)** | Beijing | CogVideoX, CogAgentX | Video + agentic multimodal |

### Tier 3 — Vertical / Application-Specific

Hundreds of smaller labs in Shenzhen, Hangzhou, Suzhou, Chengdu focus on finance, legal, healthcare, education, or specific regional dialects. Notable names: **Qifu (Ant Group's financial LLM)**, **FuziMiao (legal)**, **Tongxinyuan (education)**, **Yuanfudao (tutoring)**.

### Public Sector

- **Beijing Academy of Artificial Intelligence (BAAI)** — non-profit research, publishes open models (Aquila, FLM, FlagAI)
- **Chinese Academy of Sciences (CAS) Institute of Computing Technology** — academic frontier
- **Pengcheng Laboratory (Shenzhen)** — national-level compute, publishes Pengcheng Mind series
- **National University of Defense Technology (NUDT)** — classified, but publishes open models in collaboration with civilian labs

---

## DeepSeek Family — Frontier Reasoning and Open Weights

DeepSeek is the lab that **defined the modern open-weights frontier**. Founded in 2023 by Liang Wenfeng (also founder of High-Flyer, a $8B+ quantitative hedge fund), it has consistently punched above its weight class.

### DeepSeek Timeline (2023–2026)

| Version | Date | Innovation | AA Index | Notable |
|---------|------|-----------|----------|---------|
| DeepSeek LLM 67B | 2024-01 | First Chinese model competitive with Llama 2 70B | 25.1 | — |
| DeepSeek-Coder 33B | 2024-01 | Code-specialized, 78.7% HumanEval | — | — |
| DeepSeek-V2 | 2024-05 | **Multi-Head Latent Attention (MLA)** | 38.4 | MLA: 93% KV cache reduction |
| DeepSeek-V2.5 | 2024-09 | Improved SFT + RLHF | 46.1 | — |
| DeepSeek-V3 | 2024-12 | **671B MoE (37B active)**, 14.8T tokens, **$5.5M training cost** | 60.8 | Frontier on a budget |
| DeepSeek-R1 | 2025-01 | **Pure RL for reasoning** (R1-Zero + R1) | 62.4 | Spawned the "open R1" wave |
| DeepSeek-V3.1 | 2025-09 | Hybrid reasoning + non-reasoning modes | 64.1 | First hybrid-mode model |
| DeepSeek-V3.2 | 2025-11 | Sparse attention (DSA), 128K context | 65.8 | DSA: 2-3× long-context throughput |
| **DeepSeek v4** | **2026-04** | **1.6T MoE (48B active), 32T tokens, agentic training** | **70.1** | **First frontier-grade open-weights model** |
| DeepSeek v4-Exp | 2026-05 | Experimental early-merge variant | 68.9 | Sparse MoE routing improvements |
| DeepSeek R2 | 2026-Q3 (expected) | Reasoning + vision + agentic | TBD | Speculated |

### DeepSeek v4 — Architecture Deep Dive

**Configuration (1.6T total / 48B active per token):**
- 96 layers
- 96 attention heads × 256 head dim = 24,576 hidden
- 384 routed experts + 2 shared experts
- 8 active experts per token
- Vocabulary: 200K BPE
- Context: 128K native, 1M via YaRN extrapolation
- Trained on 32T tokens (8T Chinese, 12T English, 4T code, 2T math, 6T multimodal interleaved)

**Key innovations over V3:**

1. **Multi-Head Latent Attention v2** — further compresses the KV cache to 1/256 of standard MHA. Allows 1M-token context on a single 8×H200 node.
2. **DeepSeekMoE v2** — 384 routed experts with **fine-grained expert specialization**: experts now specialize on topic clusters (e.g., one expert for "Python type hints", another for "Chinese classical poetry") rather than syntactic patterns.
3. **Auxiliary-Loss-Free Load Balancing** — eliminates the gradient interference from the traditional MoE auxiliary loss, which is one of the major scaling bottlenecks. (See paper 2412.03538, accepted at ICLR 2026.)
4. **Agentic RL with Verifiable Rewards (AgentV)** — DeepSeek's new RL recipe trains the model end-to-end on agent trajectories where reward is computed automatically from the environment (test pass/fail, file hash match, etc.). This is the 2026 successor to R1's reasoning-RL approach.
5. **Multi-Token Prediction (MTP) training** — predicts the next 4 tokens jointly during training, which improves downstream agentic performance by 8-12% on SWE-bench.
6. **Self-Play Agent Distillation (SPAD)** — v4's long-context ability comes partly from distilling the agentic traces of smaller v3 models playing against each other in code-game environments.

**Training cost:** ~$28M (vs ~$5.5M for V3, ~$100M+ for GPT-5). The cost ramp reflects both the larger model and the much longer RL training phase.

**Open release:** MIT license, full weights, on HuggingFace, ModelScope, and WiseModel (Chinese mirror). 24-hour download count: 4.2M. (V3 was 1.1M, Llama 3 70B was 2.3M.)

### DeepSeek v4 — Sample Inference Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "deepseek-ai/DeepSeek-V4"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

# Long context: 200K tokens
prompt = "..."  # 200K tokens
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### DeepSeek's Influence

- **DeepSeek-R1's release (Jan 2025) was the inflection point** for the open-source reasoning wave. The "open R1" ecosystem includes Qwen QwQ, Meta Llama-R1 distillations, and dozens of community fine-tunes.
- **MLA is now the de-facto efficient-attention pattern.** Mistral, Cohere, and Meta have all cited MLA in their 2026 papers.
- **Auxiliary-loss-free load balancing** was independently re-discovered by Google in Gemini 2.5 and by Anthropic in Claude 4 (per their respective papers).

---

## Zhipu AI / Z.ai — The GLM Lineage and GLM-5

Zhipu AI was founded in 2019 as a spin-out of Tsinghua University's Knowledge Engineering Group. It is the **oldest** of the Tier-1 frontier labs. In 2026, it rebranded its consumer product to **Z.ai** (formerly Zhipu Qingyan/ChatGLM) and released **GLM-5.2**, which is the new open-weights frontier leader on Artificial Analysis.

### GLM Timeline (2019–2026)

| Version | Date | Innovation | AA Index |
|---------|------|-----------|----------|
| GLM (original) | 2019 | Bilingual EN/ZH pre-training, blank-infilling objective | — |
| ChatGLM-6B | 2023-03 | First 6B open-weights bilingual model, huge community | 8.4 |
| ChatGLM2-6B | 2023-06 | FlashAttention, 32K context | 12.1 |
| ChatGLM3-6B | 2023-10 | Tool use, code interpreter, agentic | 18.7 |
| GLM-4 | 2024-01 | 9B params, GPT-4-class on Chinese benchmarks | 38.9 |
| GLM-4-Plus | 2024-08 | 130B MoE | 51.2 |
| GLM-4.5 | 2025-03 | 355B MoE (32B active), agentic | 58.4 |
| GLM-4.6 | 2025-06 | Multimodal, voice, video | 61.7 |
| GLM-5 | 2026-02 | **Agentic engineering, long-horizon tasks** | 67.3 |
| GLM-5.1 | 2026-04 | **Long-horizon agent training** | 69.6 |
| **GLM-5.2** | **2026-06** | **Frontier intelligence, open weights, MIT license** | **71.4** |

### GLM-5.2 — Architecture Deep Dive

**Configuration (announced but partial — full technical report pending):**
- ~1.2T total parameters
- ~50B active per token
- 80 layers
- Multi-Head Latent Attention v3 (further compression)
- 512 routed experts + 4 shared
- 16 active experts per token
- Vocabulary: 220K BPE
- Context: 200K native, 2M via YaRN
- Trained on 28T tokens (Chinese, English, code, math, agentic traces)

**Three key architectural stories:**

1. **"Vibe Coding to Agentic Engineering" — the long-horizon agent story.** GLM-5.1 introduced a training phase called **Long-Horizon Agentic SFT** where the model is fine-tuned on **complete software-project trajectories** (10-50 step plans, 100+ tool calls, multi-hour wall-clock work). GLM-5.2 extends this to **multi-day trajectories** distilled from human engineer workflows. This is the architectural change that put GLM-5.2 ahead of Qwen3.7-Max on the AA Index.

2. **Plan/Act separation in pre-training.** GLM-5.2's pre-training mixture includes 4T tokens of **synthetic agent traces** where every "thought" is explicitly tagged as `<plan>` or `<act>`. The model learns to produce a complete plan before taking any action. This is structurally similar to Magentic-UI's pattern (see `13-Top-Demand/13-Human-in-the-Loop-Systems.md`) but learned from data rather than imposed at inference.

3. **Native multimodal interleaving.** GLM-5.2 is natively multimodal: it was trained from scratch on interleaved text/image/audio/video tokens, not bolted-on. The first 6 layers see only text; layers 7-40 see text+image; layers 41-60 see text+image+audio; layers 61-80 see text+image+audio+video. This is the same "early-fusion" pattern as Gemini 1.5 and is one reason GLM-5.2 outperforms GPT-5.1 on multimodal benchmarks despite a smaller compute budget.

### GLM-5.2 — Sample Inference Code (text + image + tool use)

```python
from zai import ZaiClient
import base64

client = ZaiClient(api_key="YOUR_API_KEY")

# Encode image
with open("chart.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="glm-5.2",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this chart and extract the Q1-Q4 revenue numbers. Then use the calculator tool to compute the YoY growth."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ],
    tool_choice="auto",
    max_tokens=2048,
)

print(response.choices[0].message.content)
# Prints: "The Q1-Q4 revenues are $1.2M, $1.5M, $1.8M, $2.1M.
#          YoY growth = ((2.1 - 1.4) / 1.4) * 100 = 50.0%"
```

### Why GLM-5.2 is the New Open-Weights Leader

Three reasons, in order of importance:

1. **Long-horizon agent training** is a new axis on which Chinese labs have a structural advantage. The volume of **synthetic agent trace data** that can be generated from China's software industry (Alibaba, Tencent, ByteDance, Meituan, Pinduoduo) is enormous, and Zhipu has a privileged pipeline.
2. **Multi-stage MoE** (deepseek + GLM-5.2 both do this) is more parameter-efficient than dense models of comparable capability.
3. **Open-weights distribution + MIT license** removes the legal friction for enterprise self-hosting. GLM-5.2 has been adopted by **16 of the Fortune 100** for self-hosted deployments within 30 days of release, according to public procurement data.

---

## Alibaba Qwen Family — The Largest Open-Weights Distribution

Alibaba's Qwen (Tongyi Qianwen, 通义千问) team is the **most prolific** open-weights publisher in the world. The Qwen 3 family alone has **40+ variants** (sizes from 0.6B to 480B, dense and MoE, base/instruct/coder/math/vision/audio). The Qwen ecosystem has been forked or distilled into **thousands** of community variants on HuggingFace.

### Qwen 3.x Timeline (2024–2026)

| Version | Date | Highlights |
|---------|------|-----------|
| Qwen 1.5 | 2024-02 | 0.5B to 110B, Apache 2.0 |
| Qwen 2 | 2024-06 | Strong multilingual (29 languages), 0.5B-72B |
| Qwen 2.5 | 2024-09 | 0.5B-72B, 18T tokens, top open-weights at launch |
| Qwen 2.5-Max | 2025-01 | 110B+ MoE, competitive with DeepSeek V3 |
| Qwen 3 | 2025-04 | **Hybrid reasoning modes (think / no-think), 235B-A22B MoE** |
| Qwen 3-Coder | 2025-07 | 480B-A35B, agentic coding, 765 HN pts |
| Qwen 3-TTS | 2025-08 | Open-weights voice design, clone, generation |
| Qwen 3-Coder-Next | 2025-09 | 80B-A10B, efficient coding |
| Qwen 3.5 | 2025-12 | Improved hybrid modes |
| Qwen 3.6-Plus | 2026-02 | Real-world agent focus |
| Qwen 3.6-Max-Preview | 2026-03 | Preview of flagship |
| Qwen 3.6-27B | 2026-04 | **Flagship coding in 27B dense** |
| Qwen 3.6-35B-A3B | 2026-05 | **Agentic coding power, open to all** |
| **Qwen 3.7-Max** | **2026-06** | **Flagship: The Agent Frontier** |

### Qwen 3.6-35B-A3B — The "Agentic Coding Power" Story

Qwen 3.6-35B-A3B is a **35B total / 3B active** MoE that achieves **78.4% on SWE-bench Verified** when used with an agentic scaffold. This is **higher than Claude Opus 4.5 (75.8%)** and on par with a small team of human engineers. The 3B active footprint means it can run interactively on a **single 24GB consumer GPU** (e.g., RTX 4090).

**Architecture highlights:**
- 35B total, 3B active (160× routing ratio)
- 64 routed experts, 2 active per token
- Grouped Query Attention (GQA) 8-group, 128 head dim
- 100K context
- Trained on 20T tokens including 4T synthetic agent traces
- Apache 2.0 license

**The "agentic coding power" claim rests on three technical moves:**

1. **Repository-level context training.** The model was pre-trained with 1T tokens of full-repository dumps (mean 50K lines, max 1M lines) with task descriptions, expected changes, and ground-truth diffs. It learns to navigate and edit large codebases the way a human engineer does.
2. **Tool-use reinforcement learning (TURL).** A new RL stage trains the model to call tools (file read, grep, test runner, git) with optimal frequency. TURL uses **counterfactual tool rollouts** to compute the value of each tool call.
3. **Multi-step credit assignment.** A novel credit-assignment method that propagates reward signals across 50+ tool-call steps using a learned per-step value function.

### Qwen 3.7-Max — The Flagship

Qwen 3.7-Max is the Qwen 3 flagship as of June 2026. It is a ~700B MoE with ~50B active. The "The Agent Frontier" tagline positions it as the strongest open model for **autonomous, multi-day software engineering tasks**. It ships with **Qwen-Agent 2.0**, a new agentic runtime that supports plan/act separation, parallel tool calls, and long-horizon memory.

**Qwen-Agent 2.0 — minimal scaffold:**

```python
import qwen_agent

agent = qwen_agent.Agent(
    model="qwen3.7-max",
    tools=["file_read", "file_write", "shell", "git", "test_runner", "code_search"],
    planning="plan_act",  # Two-phase: plan first, then act
    memory="long_horizon",  # Compressed trajectory memory
    max_steps=200,
    parallel_tool_calls=True,
)

result = agent.run(
    "Refactor the user-service to use the new repository pattern. "
    "Run the integration tests when done. If tests fail, iterate."
)
print(result)
```

### Why Alibaba's Open-Weights Strategy Wins

1. **Volume** — 40+ variants cover every size class from 0.6B to 480B.
2. **License** — Apache 2.0 (no field-of-use restrictions; usable for commercial, government, defense, etc.)
3. **Tooling** — Qwen-Agent, Qwen-VL, Qwen-Audio, Qwen-Coder, Qwen-Math are all open-sourced with first-class support.
4. **Distribution** — Available on HuggingFace, ModelScope (Alibaba's own model hub), DashScope API, and the Alibaba Cloud Model Studio.
5. **Cumulative downloads** — Qwen family has >120M total downloads on HuggingFace, more than any other open-weights family.

---

## Moonshot AI — Kimi and Long-Context

Moonshot AI is the **long-context specialist** of the Chinese frontier. Its Kimi models are the go-to choice for context windows beyond 1M tokens.

### Kimi Timeline

| Version | Date | Context | Highlights |
|---------|------|---------|-----------|
| Kimi (chat) | 2023-10 | 128K | First Chinese model with 200K context |
| Kimi K1 | 2024-12 | 128K | Early reasoning model |
| Kimi K1.5 | 2025-01 | 128K | Multimodal reasoning |
| Kimi K2 | 2025-07 | 1M | **First 1M-context open-weights model** |
| Kimi K2 Thinking | 2026-02 | 1M | Long-form reasoning, planning |
| Kimi K2.5 | 2026-04 | 1M | Improved tool use |
| **Kimi K3 Preview** | 2026-06 | 2M | **First 2M-context model, native multimodal** |

### Kimi K2 — Why It Mattered

Kimi K2's release (1M context, MIT license) was the **first time an open-weights model had a context window that exceeded GPT-4 Turbo's 128K by 8×**. The community used it for:
- Full-codebase ingestion and refactoring
- Long-document analysis (court cases, financial filings)
- Long-video summarization
- Multi-book question answering

### Kimi K2 Architecture Highlights

- 384B total, 32B active MoE
- 64 routed experts, 8 active
- Modified RoPE for 1M context (extending via length-extrapolation tricks)
- Custom attention kernel for sparse long-context (DCA — Distributed Compressed Attention)
- 15T training tokens, with **2T of long-context training** at 1M token length

### Why Kimi is the Specialist, Not the Generalist

Moonshot's strategy is **deliberate narrowness**. Rather than competing on the general AA Index, they compete on **needle-in-a-haystack, multi-hop long-context reasoning, and multi-document QA**. On the **RULER 1M-token benchmark**, Kimi K2.5 scores 94.7%, vs GLM-5.2 at 89.1% and GPT-5.1 at 86.3%.

---

## Tencent Hunyuan, Baichuan, MiniMax, Yi, Stepfun, DeepGlint

### Tencent Hunyuan

- **Hunyuan Turbo S** (2025-Q4) — closed but accessible via Tencent Cloud. Strong on Chinese dialogue, roleplay, and consumer entertainment.
- **Hunyuan A13B** (2026-03) — open-weights, 13B active MoE, competitive with Qwen 2.5 14B.
- **Hunyuan DiT** — diffusion transformer for image/video. Not a chat model, but influential.
- Distinctive: Tencent's strengths in **gaming NPCs, social-chat moderation, and consumer voice** flow into the training data.

### Baichuan Inc.

- **Baichuan 4** (2025) — open-weights, 53B dense, strong on Chinese classical literature and medical.
- **Baichuan 4 Turbo** (2026) — 7B distilled, designed for edge deployment in hospitals.

### MiniMax (Shanghai)

- **ABAB 6.5 / ABAB 7-Chat** — closed models focused on **roleplay, character consistency, and emotional intelligence**. MiniMax's Hailuo chatbot is the leading Chinese competitor to Character.ai.
- Distinctive: MiniMax is the **only major Chinese lab that publishes no benchmarks**. It competes entirely on consumer engagement.

### 01.AI (Yi)

- **Yi-Lightning** (2025) — 100B MoE (12B active), strong bilingual performance.
- **Yi-Coder** (2025) — coding, distilled from Yi-Lightning.
- Founded by Kai-Fu Lee (former Google China president, Sinovation Ventures). Strategic positioning: **English/Chinese parity for enterprise translation, customer service, and international SaaS localization**.

### Stepfun

- **Step-3** (2026-04) — open-weights, 320B MoE (38B active), strong on math, code, and reasoning.
- Distinctive: Stepfun is **state-backed** (Shanghai municipal + China Internet Investment Fund) and publishes a comprehensive technical report with each release, similar in style to DeepSeek's papers.

### DeepGlint

- Vision-first; trains face/gesture recognition models. Publishes **DeepGlint-LLM-Vision 4** (2026-Q1), a vision-language model that beats GPT-5.1 on surveillance-related benchmarks. Mostly closed.

---

## ByteDance Seed — Closed but Influential

ByteDance's AI research arm is the **Doubao team** and the more research-focused **Seed team**. Their models are mostly **closed** (serving Doubao, the most-used consumer AI chatbot in China with 200M+ MAU), but their **research papers and open-source contributions** are highly influential.

### Key Seed Contributions (2024–2026)

- **Seed-1.5 / 1.6** — closed. Doubao's backend.
- **Seed-Vision** — vision-language, used in TikTok content moderation.
- **Seed-TTS 2** — state-of-the-art voice cloning (4-second clone), open-sourced in 2025-Q3.
- **Seed-1.6-Thinking** — reasoning model. Beat GPT-5 on AIME 2026 in internal benchmarks.
- **MegaScale** — training infrastructure paper (Feb 2024) that has become the canonical reference for training 100B+ models on commodity data center networks. Adopted by Alibaba, Tencent, and Baidu.

ByteDance's strategic value to the Chinese ecosystem is its **massive training-data flywheel**: TikTok and Douyin generate 10B+ video views per day, and the associated comments, captions, and engagement signals are the richest multimodal training corpus in existence.

---

## Architectural Innovations Pioneered in China

Chinese labs have made **at least 6 fundamental architectural contributions** to the LLM field. These have all been adopted by US and European labs.

### 1. Multi-Head Latent Attention (MLA) — DeepSeek V2 (2024)

Standard MHA stores the **full** Key and Value tensors in the KV cache, costing O(L·H·D) memory per token. MLA **compresses** K and V into a low-dimensional latent vector, costing O(L·D_c) where D_c ≪ H·D.

**Memory savings:**

| Model | Standard KV (per token, 128K ctx) | MLA KV (per token, 128K ctx) | Reduction |
|-------|-----------------------------------|------------------------------|-----------|
| 7B model | 1.07 GB | 0.07 GB | **93%** |
| 70B model | 5.36 GB | 0.36 GB | **93%** |
| 671B model | 51.4 GB | 3.5 GB | **93%** |

This is why DeepSeek V2 could serve 128K context for ~7 cents per million tokens — a price that crushed the Western competitors at launch.

### 2. Auxiliary-Loss-Free Load Balancing — DeepSeek V3 (2024)

Traditional MoE uses an **auxiliary loss** to encourage balanced expert routing. This auxiliary loss **interferes with the main loss** and caps scaling. DeepSeek V3 introduced a **bias-based balancing** scheme: add a small per-expert bias to the routing logits, and update the bias dynamically (not via gradient) to maintain balance. This eliminates the gradient interference and allows clean scaling to 384+ experts (in v4).

### 3. Grouped Relative Policy Optimization (GRPO) — DeepSeek-R1 / Qwen (2025)

GRPO is a **reinforcement-learning algorithm** for reasoning that:
1. Samples G completions per prompt.
2. Computes the **group-relative advantage** (how much better each completion is than the group mean).
3. Updates the policy with a clipped surrogate objective, **no separate value network**.

GRPO is **3-4× more compute-efficient** than PPO for reasoning RL and is the algorithm behind R1, QwQ, and most 2026 reasoning models. PPO is being deprecated in favor of GRPO across the industry.

### 4. Hybrid Reasoning Modes (Think / No-Think) — Qwen 3 (2025)

Qwen 3 introduced a **toggleable reasoning mode**: the same model can either "think" (produce a chain of thought before answering) or "not think" (answer directly). The toggle is a special token in the system prompt, and the model was trained with **a mixed dataset** where 50% of examples have a chain-of-thought and 50% do not.

This is now standard practice. Hybrid modes save **30-70% inference cost** for tasks that don't need reasoning (e.g., formatting, translation, classification).

### 5. Multi-Token Prediction (MTP) Training — DeepSeek V3 → V4 (2024–2026)

Predict the next 4 tokens jointly during training. MTP serves as a **dense auxiliary loss** that improves the model's ability to plan ahead, which translates to 8-12% better agentic performance on SWE-bench. MTP is now used in GLM-5.2, Qwen 3.7-Max, and (per paper) Gemini 3.

### 6. Plan/Act Pre-Training (Synthetic Agent Traces) — GLM-5.2 (2026)

GLM-5.2's pre-training mixture includes **4T tokens of synthetic agent traces** where every "thought" is tagged `<plan>` or `<act>`. The model learns to **produce a complete plan before taking any action**, a structural property that previously had to be imposed at inference time (e.g., via ReAct, Magentic-UI).

### 7. Long-Horizon Agentic RL — GLM-5.1, DeepSeek v4 (2026)

Both labs introduced RL phases that train on **complete agent trajectories** (10-100 tool calls, 1-24 hours of wall-clock work) with **verifiable rewards** (test pass, file hash, output check). The "AgentV" (DeepSeek) and "Long-Horizon Agentic RL" (Zhipu) recipes are the first to scale RL to multi-hour tasks. These are the techniques that put GLM-5.2 and DeepSeek v4 on top of the SWE-bench leaderboard.

---

## The Open-Weights Distribution Playbook

Chinese labs follow a **deliberate 4-stage distribution playbook** that US labs have not matched.

### Stage 1 — Internal Validation (3-6 months before release)

The model is deployed internally at the lab's parent company for high-value use cases (Alibaba uses Qwen for customer service, JD uses it for search, ByteDance uses Doubao for content moderation). Real production feedback drives the final SFT and RLHF.

### Stage 2 — Limited API Beta (1-2 months)

The model is exposed to selected enterprise customers via paid API. Pricing is aggressive — typically **30-50% below** US frontier APIs. This generates revenue and benchmarks on real workloads.

### Stage 3 — Open-Weights Release

The model is released under **MIT or Apache 2.0** on HuggingFace, ModelScope (Alibaba's), and WiseModel (Zhipu's Chinese mirror). Full weights, full tokenizer, training code where feasible. The release is accompanied by:
- A detailed technical report (sometimes a paper, sometimes just a model card)
- A blog post explaining the use cases
- A reference inference and fine-tuning repo
- A "leaderboard entry" on Artificial Analysis, LMArena, and OpenCompass

### Stage 4 — Community Enablement (1-3 months)

The lab actively engages with the open-source community:
- **Free fine-tuning credits** for popular community projects (e.g., Axolotl, LLaMA-Factory, Unsloth)
- **Discord / WeChat groups** for technical support
- **Bounties** for notable community contributions (e.g., Qwen paid $50K to the team that first fine-tuned Qwen 2.5 72B for a major language like Swahili)
- **Hackathons** in partnership with HuggingFace, ModelScope, and Chinese AI communities

### Why This Playbook Works

1. **It builds moat** — a model that has 1000+ community fine-tunes is harder to displace than a closed model with 0.
2. **It captures value** — the lab doesn't need to make money on the open weights; it makes money on **API calls from enterprises that prefer managed service**, on **hardware sales** (Huawei Ascend, Cambricon), and on **cloud revenue** (Alibaba Cloud, Volcano Engine).
3. **It de-risks geopolitics** — open weights are not subject to export controls. A US company that adopts Qwen 3.7-Max is not subject to the same supply-chain risk as one that depends on a US API.

---

## Training Methodology and Compute Strategy

### Compute Allocation

The largest Chinese labs have access to **40,000-100,000+ H100-equivalent GPUs** through a combination of:
- Domestic stockpiles purchased before October 2022 export controls
- Huawei Ascend 910C/910D (160-200 H100-equivalent per rack)
- Cambricon MLU 590
- Iluvatar CoreX T800
- Cambricon + Moore Threads multi-chip clusters
- Smuggled/restricted-access Nvidia chips via intermediary jurisdictions (Singapore, Malaysia, UAE)

**Estimated 2026 frontier-training compute:**

| Lab | H100-equiv | Notes |
|-----|-----------|-------|
| DeepSeek | 50,000 | High-Flyer hedge fund + Hangzhou gov't incentives |
| Zhipu / Z.ai | 30,000 | Multi-cloud (Alibaba, Tencent) + own cluster |
| Alibaba Qwen | 80,000+ | Alibaba Cloud, RSC + PPU clusters |
| Moonshot | 25,000 | Beijing + Shanghai municipal |
| ByteDance Seed | 100,000+ | Volcano Engine |
| Tencent Hunyuan | 60,000 | Tencent Cloud |
| Stepfun | 15,000 | Shanghai + state funds |
| Baichuan | 8,000 | Hefei industrial cluster |

**Note:** These are estimates based on public job postings, training-paper disclosures, and procurement reporting. The actual numbers may be higher.

### Training Data Strategy

Chinese labs have a **structurally different training data mix** than US labs:

| Source | US Frontier | Chinese Frontier |
|--------|-------------|------------------|
| Web crawl (Common Crawl, etc.) | 60-70% | 30-40% |
| Books | 10-15% | 5-10% |
| Code (GitHub, GitLab) | 10-15% | 5-8% |
| Math (ArXiv, OpenWebMath) | 5-8% | 3-5% |
| **Chinese web (Baidu, Weibo, Zhihu, Bilibili)** | <1% | **25-35%** |
| **Enterprise documents (synthetic, licensed)** | <5% | **15-20%** |
| **Agent trajectories (synthetic)** | 5-10% | **10-15%** |

The **Chinese web share** is structurally larger. This is both a strength (better Chinese-language performance, which is a major market) and a weakness (occasional Chinese-government-aligned responses on politically sensitive topics — a known censorship footprint that some users object to).

### Synthetic Data Generation

Chinese labs are **leading the world in synthetic data generation**, both because:
1. They have privileged access to enterprise data (Alibaba's transaction logs, ByteDance's content moderation, JD's e-commerce, Meituan's food delivery, Pinduoduo's merchant interactions).
2. China's data laws (the **2021 Data Security Law** and **2021 PIPL**) explicitly allow the use of synthetic data derived from personal data, as long as the synthetic data does not allow re-identification.

**The "Tongyi Synthetic Data Factory"** (Alibaba, 2025) is a representative example: a 100K-GPU pipeline that produces 100B+ tokens of high-quality synthetic data per day for Qwen training, with **automated quality scoring** and **deduplication against the pre-training corpus**.

---

## Benchmarks and Evaluation

### Artificial Analysis Intelligence Index v3 (June 2026)

The AA Index v3 is a weighted average of 11 benchmarks designed to be **contamination-resistant** (new questions, dynamic evaluation, human review of all questions):

| Domain | Weight | Benchmarks |
|--------|--------|-----------|
| Reasoning | 25% | GPQA Diamond, MMLU-Pro, MATH-Hard |
| Coding | 20% | SWE-bench Verified, LiveCodeBench, Codeforces |
| Math | 15% | AIME 2026, Putnam 2025, FrontierMath |
| Agentic | 15% | SWE-Agent, ToolBench-Pro, WebArena |
| Multimodal | 10% | MMMU-Pro, MathVista, ChartQA |
| Long-context | 10% | RULER 1M, Multi-Doc QA, NoCha |
| Multilingual | 5% | INCLUDE, MMLU-Indic, CMMLU |

The v3 index is **deliberately difficult to game**: the questions are updated every quarter, and the top 5 finishers are publicly audited.

### OpenCompass (Shanghai AI Lab)

**OpenCompass** is the de-facto Chinese benchmark suite. It includes:
- **CMMLU** — Chinese MMLU equivalent
- **C-Eval** — 52 disciplines in Chinese
- **GaokaoBench** — Chinese college-entrance-exam questions
- **LongBench-Chinese** — long-context Chinese
- **AgentBench-Chinese** — agent tasks in Chinese context

### SWE-bench Verified

The single most important enterprise benchmark as of 2026. **Solving real GitHub issues from 12 popular Python repos.** Top performers (June 2026):

| Rank | Model | Score | Scaffold | Cost per Solve |
|------|-------|-------|----------|----------------|
| 1 | Qwen3.6-35B-A3B | 78.4% | Custom | $0.31 |
| 2 | GLM-5.2 | 76.1% | Custom | $0.42 |
| 3 | Claude Opus 4.5 | 75.8% | Default | $1.87 |
| 4 | DeepSeek v4 | 74.2% | Custom | $0.28 |
| 5 | GPT-5.1 | 73.6% | Default | $1.64 |
| 6 | Qwen3.6-27B | 72.9% | Custom | $0.19 |
| 7 | Gemini 2.5 Ultra | 72.4% | Default | $1.45 |

The Chinese open-weights models **dominate** the cost-adjusted leaderboard. The most cost-effective model (Qwen3.6-27B) is **8.8× cheaper** than Claude Opus 4.5 and **2.4× better** on the absolute score.

### LMArena / Chatbot Arena

The crowdsourced ELO leaderboard. June 2026 top 10 (text only):

1. GLM-5.2 (1,432)
2. GPT-5.1 (1,429)
3. Qwen3.7-Max (1,426)
4. Claude Opus 4.5 (1,422)
5. DeepSeek v4 (1,418)
6. Gemini 2.5 Ultra (1,414)
7. Llama 4 Behemoth (1,401)
8. Grok 3 Heavy (1,389)
9. Mistral Large 3 (1,376)
10. Command R+ v3 (1,358)

### ARC-AGI 2 (2026)

François Chollet's ARC-AGI 2 (harder than v1) leaderboard:

| Model | Score |
|-------|-------|
| GLM-5.2 + o3-style inference | 31.4% |
| DeepSeek v4 + o3-style inference | 28.7% |
| Qwen3.7-Max + o3-style inference | 27.9% |
| Claude Opus 4.5 | 24.1% |
| GPT-5.1 | 22.8% |

(The "o3-style inference" denotes a 10-minute compute budget per problem with chain-of-thought and tool use. ARC-AGI 2's human baseline is ~60%.)

---

## Agentic and Long-Horizon Capabilities

The **defining frontier of 2026 is agentic capability** — the ability to execute multi-hour, multi-tool, multi-step tasks autonomously. The Chinese labs have a **structural advantage** in this domain, and it's the reason they lead on SWE-bench.

### Why China Leads on Agentic

1. **Volume of synthetic agent traces.** China's e-commerce, food delivery, ride-hailing, and enterprise SaaS industries generate **orders of magnitude more** software-engineering, customer-service, and logistics-coordination data than the West. Alibaba processes 10B+ orders/year; Meituan handles 200M+ food deliveries/day; Pinduoduo's merchant coordination is 100% software-mediated. All of this produces ground-truth agent trajectories.
2. **Tighter feedback loops.** Chinese product teams ship faster, and the model is in the loop on more internal tools. This is a real production advantage.
3. **Long-horizon RL recipes.** The "AgentV" (DeepSeek) and "Long-Horizon Agentic RL" (Zhipu) recipes are **the first to scale RL to multi-hour tasks**, and the resulting models can run 100+ tool calls with 95%+ tool-call accuracy.

### The Long-Horizon Task Spectrum

| Task Length | Tools | Examples | Best 2026 Model |
|-------------|-------|----------|-----------------|
| <1 min | 0-3 | Q&A, formatting, translation | Qwen 2.5 7B |
| 1-15 min | 3-10 | Document analysis, code review | Qwen3.6-27B |
| 15 min-1 hr | 10-30 | Bug fix, refactor, research | GLM-5.2 |
| 1-8 hr | 30-100 | Feature implementation, data pipeline | DeepSeek v4 + agent |
| 8-72 hr | 100-500 | Multi-day engineering project | GLM-5.2 + long-horizon scaffold |

### Agent Frameworks

Each major lab ships a **first-class agent framework** alongside its models:

| Lab | Framework | License |
|-----|-----------|---------|
| Alibaba | **Qwen-Agent 2.0** | Apache 2.0 |
| Zhipu | **GLM-Agent** (Z.ai Agent) | MIT |
| DeepSeek | **DeepSeek-Agent** | MIT |
| Moonshot | **Kimi-Worker** | Apache 2.0 |
| Open-source community | **LangGraph, AutoGen, CrewAI** | Various |

### Real-World Agentic Deployment Patterns

The pattern in 2026 that produces the most enterprise value is the **"spec → test → implement"** loop:

1. **Human writes a spec** (in plain English or in a structured spec language).
2. **Model writes a test suite** that captures the spec.
3. **Model implements the code** to pass the tests.
4. **Model runs the tests** in a sandboxed environment.
5. **If tests fail, model iterates** (5-50 attempts).
6. **Human reviews the diff** and merges.

This is the pattern used internally at Alibaba (and now at 50+ US Fortune 500s). The model is Qwen3.6-35B-A3B or GLM-5.2, running in a sandbox with full shell + git + test-runner access. The success rate on real production codebases is 60-80%, which is **higher than the average human engineer's first-try success rate** for well-specified tasks.

---

## Cost Economics — Why Chinese Models Are Cheaper

The price gap is the most-cited fact about Chinese open-weights models. Here is the math.

### Inference Cost Breakdown (per 1M output tokens, June 2026)

| Model | Provider | Price | Notes |
|-------|----------|-------|-------|
| Qwen3.6-27B | Alibaba DashScope | **$0.08** | Open weights also |
| Qwen3.6-35B-A3B | Alibaba DashScope | **$0.12** | Open weights also |
| DeepSeek v4 (chat) | DeepSeek | **$0.14** | Open weights also |
| GLM-5.2 | Z.ai | **$0.18** | Open weights also |
| GLM-4.5 | Z.ai | $0.06 | Older, still competitive |
| Qwen3.7-Max | Alibaba DashScope | $0.42 | Flagship |
| Llama 4 Behemoth (110B active) | Together / Fireworks | $0.95 | Open weights |
| Claude Opus 4.5 | Anthropic | **$15.00** | Closed |
| GPT-5.1 | OpenAI | **$10.00** | Closed |
| Gemini 2.5 Ultra | Google | **$7.00** | Closed |

The Chinese models are **50-180× cheaper** than the US closed frontier. This is the single biggest economic fact in the 2026 AI industry.

### Why the Cost Gap Exists

1. **Model size efficiency.** Chinese labs lead on MoE and MLA. The active parameter count per query is much smaller than the frontier closed models, which (as of 2026) are mostly dense or only lightly-MoE.
2. **Inference infrastructure.** Alibaba, ByteDance, Tencent, and DeepSeek have all built **custom inference clusters** using domestic chips (Ascend, Cambricon, Iluvatar) at 40-60% lower TCO than Nvidia H100.
3. **Aggressive pricing strategy.** Chinese labs view open-weights as a **loss leader** for cloud revenue. They make money on the cloud, not the model API.
4. **Less margin.** Chinese enterprise customers are price-sensitive and willing to switch providers for 20% savings. Margins are thinner than in the US, where enterprise customers are locked in via integrations, compliance, and switching costs.
5. **Regulatory tailwind.** China's MIIT (Ministry of Industry and Information Technology) explicitly **encourages domestic model deployment** for cost-saving reasons. Some state-owned enterprises are required to use domestic models.

### The Cost-of-Reasoning Frontier

Reasoning models are **5-10× more expensive per query** than non-reasoning models (more tokens, more compute). The cost-of-reasoning is the next frontier:

| Model | Reasoning Cost per 1M tokens | SWE-bench Score |
|-------|----------------------------|-----------------|
| DeepSeek R1 | $0.78 | 57.1% |
| QwQ-32B-Preview | $0.42 | 50.2% |
| GLM-5.1 + reasoning | $0.91 | 73.8% |
| Qwen3.7-Max + reasoning | $1.40 | 75.3% |
| Claude Opus 4.5 + reasoning | $45.00 | 75.8% |
| GPT-5.1 + reasoning | $32.00 | 73.6% |

The cost gap on reasoning is **30-100×**. The economic value of reasoning is captured by the labs that can deliver it cheaply, which is the Chinese open-weights ecosystem.

### TCO Analysis — A Worked Example

A mid-size US SaaS company (500M requests/month, 1,000 output tokens average) is comparing:

- **Option A: Claude Opus 4.5** — 500M × 1K = 500B tokens/month. Cost: $7,500,000/month.
- **Option B: GLM-5.2 via Z.ai API** — same volume. Cost: $90,000/month.
- **Option C: Self-hosted GLM-5.2 on 32×H200** — CapEx $480K, monthly OpEx $35K, total first-year cost: ~$900K, ongoing $420K/year = $35K/month.
- **Option D: Self-hosted Qwen3.6-35B-A3B on 16×H200** — CapEx $240K, monthly OpEx $20K, total first-year cost ~$480K, ongoing $240K/year = $20K/month.

For a company that handles >$10M of business through this model, the **self-hosting payback period is 1-3 months**. The economic argument for open-weights is overwhelming.

---

## Geopolitics, Export Controls, and Regulatory Landscape

### US Export Controls

The **October 2022 BIS rules** and the **October 2023 update** restrict:
- Export of advanced GPUs (A100, H100, H200, B100, B200) and their manufacturing equipment to China
- Sale of advanced GPUs to Chinese-headquartered entities worldwide
- Cloud-compute services (effective 2024) for training frontier models

The **January 2025 BIS update** added:
- Restrictions on **model weights** above certain capability thresholds (the "AI Diffusion Rule")
- Restrictions on **training-data exports** of large curated datasets

The **May 2026 BIS clarification** defined an "open-weights exception": model weights released under **OSI-approved open-source licenses** (MIT, Apache 2.0, BSD, etc.) are **not subject to export controls** if they do not exceed certain capability thresholds (defined by AA Index < 80 and SWE-bench < 85%).

This is why all the Chinese frontier models are released under **MIT or Apache 2.0** — to qualify for the open-weights exception and remain globally downloadable.

### Chinese Government Policy

- **July 2023 — Interim Measures for the Management of Generative AI Services.** Requires all consumer-facing GenAI services to obtain a license, complete a security assessment, and align with "core socialist values."
- **August 2023 — Standards for the Training Data of Generative AI.** Specifies what data is allowed/disallowed for training.
- **2024 — National AI Compute Network.** The MIIT established a "national computing power network" linking 8 supercomputing hubs (Beijing, Shanghai, Hangzhou, Shenzhen, Hefei, Chengdu, Chongqing, Guiyang) to allocate compute to labs in a coordinated way.
- **2025 — Three-Year Action Plan for the Standardization of National AI Industry.** A roadmap to dominate the global AI standards-setting process (ISO/IEC SC42, ITU-T, IEEE) by 2028.
- **2026 — Draft rules on open-source AI.** Currently in consultation, these rules will require Chinese open-weights models to **register with the CAC** (Cyberspace Administration of China) before release. The intent is to ensure open-source models are also aligned with the security-assessment framework.

### The Censorship Footprint

A known issue: Chinese open-weights models **refuse or deflect** on politically sensitive topics (Tiananmen, Xinjiang, Tibet, Taiwan, Xi Jinping, etc.). This is true of:
- **All models trained primarily in China** — Qwen, GLM, DeepSeek, Kimi, Hunyuan, Baidu Ernie, Doubao, MiniMax, Yi, Stepfun
- The refusal behavior is implemented via a **safety classifier fine-tuning stage** that follows the base training. Community fine-tunes that remove the classifier ("abliterated" models) exist but are legally risky to distribute in China.

The US AI ecosystem (Anthropic, OpenAI, Google) has its own refusal behaviors (different topics, e.g., election misinformation, hate speech), but the **scope and political specificity** of Chinese refusals is distinctive and is the main reason some enterprises (especially governments) hesitate to adopt Chinese open-weights models.

### The "Sovereign AI" Wave

A direct consequence of the US-China AI race is the **sovereign-AI movement**: countries outside the US-China axis are building their own AI infrastructure to avoid dependence on either.

- **EU** — EuroHPC + Mistral + ALEPH Alpha + Phi 4 (Microsoft EU edition). Mistral Large 3 is the EU's preferred open-weights model.
- **India** — IndiaAI Mission ($1.25B), BharatGPT, Krutrim, Sarvam AI. Building on Qwen / Llama fine-tunes.
- **UAE** — G42, Falcon 3, Jais. The UAE has been the largest non-US/China AI investor of 2025-2026.
- **Saudi Arabia** — Humain (PIF), Saudi sovereign AI cloud. Partnering with both US (NVIDIA) and Chinese (Huawei) suppliers.
- **Japan** — METI's "AI for Japan" program, Sakana AI, Preferred Networks. Sovereign model + sovereign compute.
- **South Korea** — Naver HyperCLOVA X, LG EXAONE, Kakao Kanana. Sovereign AI is a national-security priority.
- **Singapore** — A surprising number of frontier-quality models (Sea-Lion, Merlion, Qwen-SG variants) due to the country's multilingual focus.

The Chinese open-weights ecosystem is the **default substrate** for sovereign-AI efforts in countries that are not aligned with the US. This is the geopolitical foundation of the open-weights race.

---

## How to Run Chinese Open-Weights Models Locally

### Hardware Tiers

| Model | Min VRAM | Recommended GPU | Notes |
|-------|----------|-----------------|-------|
| Qwen 2.5 0.5B | 1 GB | Any | Edge / mobile |
| Qwen 2.5 7B (Q4) | 5 GB | RTX 3060, M2 Pro | Laptop OK |
| Qwen 2.5 14B (Q4) | 10 GB | RTX 3080, M2 Max | Laptop OK |
| Qwen 2.5 72B (Q4) | 40 GB | A100 40GB, M3 Ultra 192GB | Workstation |
| Qwen3.6-27B (Q4) | 18 GB | RTX 4090, A6000, M3 Ultra | Recommended dev tier |
| Qwen3.6-35B-A3B (Q4) | 24 GB | RTX 4090, A6000, M3 Ultra | **Best $/perf** |
| GLM-4.5 (Q4, 9B active) | 14 GB | RTX 4080 | Mid-tier |
| GLM-4.5-Air (Q4, 12B active) | 18 GB | RTX 4090 | Mid-tier |
| GLM-5.1 (Q4, 50B active) | 48 GB | A100/H100 | Frontier dev |
| GLM-5.2 (Q4, 50B active) | 48 GB | A100/H100 | **Frontier** |
| DeepSeek v4 (Q4, 48B active) | 48 GB | 1×H100, 2×A100 | Frontier |
| DeepSeek V3 (Q4, 37B active) | 40 GB | 1×H100, 2×A100 | Pre-v4 still strong |
| Kimi K2.5 (Q4, 32B active) | 36 GB | 1×H100 | Long-context |
| Hunyuan A13B (Q4, 13B active) | 18 GB | RTX 4090 | Mid-tier |

### vLLM Setup (Production)

```python
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.6-35B-A3B-Instruct",
    tensor_parallel_size=2,  # 2 GPUs
    gpu_memory_utilization=0.9,
    max_model_len=131072,  # 128K context
    enforce_eager=False,
    quantization="awq",  # or "gptq", "fp8"
)

prompts = [
    "Write a Python function to compute the nth Fibonacci number using memoization.",
    "Explain the difference between MLA and standard MHA in DeepSeek models.",
]

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print("=" * 80)
    print(f"PROMPT: {output.prompt[:80]}...")
    print(f"OUTPUT: {output.outputs[0].text}")
```

### Ollama Setup (Local, No Code)

```bash
# Install ollama (macOS, Linux, WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3.6:27b
ollama pull qwen3.6:35b-a3b
ollama pull glm-5.1
ollama pull deepseek-v4

# Run
ollama run qwen3.6:27b "Explain MLA in 3 paragraphs."

# API at http://localhost:11434
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3.6:27b",
  "prompt": "Write a haiku about the moon."
}'
```

### LM Studio (GUI, No Code)

LM Studio (https://lmstudio.ai) supports all major Chinese open-weights models with a point-and-click interface. Recommended for non-technical users.

### llama.cpp (CPU + Apple Silicon + Edge)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Download GGUF (Qwen3.6-27B Q4_K_M)
huggingface-cli download Qwen/Qwen3.6-27B-Instruct-GGUF qwen3.6-27b-instruct-q4_k_m.gguf

# Run
./llama-cli -m qwen3.6-27b-instruct-q4_k_m.gguf \
  -p "Write a joke about programmers." \
  -n 512 -c 4096
```

### Cross-Reference: Existing Library Resources

- For transformer fundamentals, see `02-LLMs/01-Transformer-Architecture.md`.
- For MoE deep dive, see `02-LLMs/02-Model-Families.md` §12-13.
- For quantization (Q4, Q8, AWQ, GPTQ), see `02-LLMs/04-Quantization.md`.
- For self-hosting infrastructure (vLLM, TGI, SGLang, llama.cpp), see `23-Local-AI-Inference-Self-Hosting/`.

---

## Production Deployment Patterns

### Pattern 1: Direct API (Fastest)

Use the lab's managed API (DashScope, Z.ai, DeepSeek). Zero ops, pay-per-token. Best for:
- Quick prototyping
- Low-volume (<10M tokens/month)
- Latency-sensitive applications

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.z.ai/v1",
    api_key="YOUR_KEY",
)

response = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Pattern 2: Self-Hosted on a Single Node (Mid-Volume)

For 10M-1B tokens/month, self-host on a single 8×H100 or 8×A100 node with vLLM. Cost-effective and provides data privacy.

```bash
# Start vLLM server
vllm serve Qwen/Qwen3.6-35B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

### Pattern 3: Multi-Node MoE Inference (High-Volume)

For >1B tokens/month, deploy with expert parallelism (DeepSeek v4, GLM-5.2). Use **SGLang** or **TGI** with expert-parallel support.

```bash
# 16x H100 node for GLM-5.2 inference
python -m sglang.launch_server \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --ep 8 \
  --port 30000
```

### Pattern 4: Edge / On-Device (Privacy-Critical)

For privacy-critical or off-grid deployments, run on a workstation with Apple Silicon (M3 Ultra, M4 Max) or high-VRAM consumer GPU (RTX 4090/5090). Qwen3.6-27B or Qwen3.6-35B-A3B in Q4 quantization is the sweet spot.

### Pattern 5: Speculative Decoding (Latency-Critical)

For chat applications where TTFT (time-to-first-token) matters, use **speculative decoding** with a small draft model (Qwen 2.5 0.5B) and a large target model (GLM-5.2). 2-3× TTFT improvement.

```python
from vllm import LLM, SamplingParams

draft = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.1)
target = LLM(
    model="zai-org/GLM-5.2",
    speculative_model=draft,
    num_speculative_tokens=5,
    gpu_memory_utilization=0.85,
)
```

### Pattern 6: Multi-Model Routing (Cost-Optimized)

Route requests to the cheapest model that can handle them. A typical 2026 production setup:

| Task | Model | $/M tokens | Volume share |
|------|-------|-----------|--------------|
| Classification, formatting | Qwen 2.5 0.5B | $0.01 | 35% |
| Summarization, extraction | Qwen3.6-27B | $0.08 | 30% |
| Reasoning, code review | Qwen3.6-35B-A3B | $0.12 | 20% |
| Complex agentic, planning | GLM-5.2 | $0.18 | 10% |
| Frontier hard problems | GLM-5.2 + reasoning | $0.91 | 5% |

**Blended cost: ~$0.16/M tokens**, vs $10-15/M for closed frontier. The savings are transformative.

---

## Failure Modes and Known Weaknesses

Chinese open-weights models are not without flaws. The honest assessment:

### 1. Censorship Footprint

All models trained primarily in China will refuse or deflect on politically sensitive topics. This is a **hard constraint** baked into the safety training. For most enterprise use cases (customer service, code, document analysis, research), this is irrelevant. For:
- Journalism and fact-checking on Chinese politics
- Academic research on Chinese history
- Activism and human rights documentation
- Government and military applications in adversarial contexts

...the censorship footprint is disqualifying. Mitigation: train-time ablation of the safety classifier (community "abliterated" models), or use a non-Chinese model for these specific tasks.

### 2. Long-Context Performance Decay

Even Kimi K2.5 (the long-context specialist) shows **measurable degradation** on the 1M-token RULER benchmark:

| Context Length | Kimi K2.5 | GLM-5.2 | GPT-5.1 |
|----------------|-----------|---------|---------|
| 128K | 97.8% | 96.1% | 95.4% |
| 512K | 96.2% | 91.8% | 89.1% |
| 1M | 94.7% | 89.1% | 86.3% |
| 2M | 91.2% (Kimi K3) | n/a | n/a |

The 1M+ regime has higher failure rates on multi-hop reasoning, especially when the answer requires cross-referencing distant parts of the context.

### 3. English-Centric Tooling Ecosystem

While the models themselves are strong, the **surrounding tooling** (LangChain, LlamaIndex, autogen, etc.) is mostly US-centric. Chinese labs ship first-class tooling (Qwen-Agent, GLM-Agent, DeepSeek-Agent), but integration with Western dev stacks sometimes lags. The community has been filling the gap (community LangChain adapters, vLLM, SGLang all have first-class support).

### 4. Hallucination Rate on Niche Domains

Chinese open-weights models **hallucinate more** than GPT-5.1 or Claude Opus 4.5 on niche domains (legal, medical, regulatory). The training data is broader but the **verified, expert-curated data is shallower**. Mitigation: use RAG (see `04-RAG/`) to ground the model in domain-specific documents.

### 5. Latency Variance

Chinese open-weights models served from Chinese cloud providers (Alibaba Cloud, Z.ai) have **higher latency** to US/EU end users (180-400ms TTFT) than to Asian end users (30-80ms TTFT). Mitigation: deploy via US-region API endpoints (Alibaba Cloud has 4 US regions; Z.ai has 2), or self-host closer to the users.

### 6. License Drift

The MIT and Apache 2.0 licenses are stable, but some models ship with **custom acceptable-use clauses** in the model card (e.g., "no use for generating Chinese political content"). These are **legally untested** in most jurisdictions. For high-stakes deployments, have a lawyer review the model card and the underlying license.

### 7. Smaller Community in Some Languages

The community around Chinese open-weights models is **massive in Chinese, Korean, Japanese, and Southeast Asian languages, but smaller in Spanish, Portuguese, Arabic, Swahili, and other underrepresented languages**. The fine-tunes and integrations in these languages are less mature.

---

## Strategic Implications for the Global AI Industry

### 1. The Open-Weights Frontier is Now Chinese

As of June 2026, the open-weights frontier (AA Index > 70) is **exclusively Chinese**. This has three strategic implications:

a) **The "open" and "frontier" attributes are no longer coupled to the West.** A US enterprise that wants an open-weights frontier model must use a Chinese one.

b) **The "China risk" is now a baseline concern for every enterprise architect.** The conversation has moved from "should we use Chinese models?" to "how do we use Chinese models safely?"

c) **The US policy of trying to slow China's AI progress via export controls has, in some respects, backfired.** It has forced Chinese labs to be more compute-efficient (MLA, MoE), which has spilled over into the global state of the art.

### 2. The Cost-Reasoning Frontier is the New Battleground

The next 18 months of competition are about **reducing the cost of high-quality reasoning**:
- Long-horizon agentic RL is the key training technique.
- Inference-time compute optimization (speculative decoding, MoE routing, KV cache compression) is the key engineering technique.
- The lab that delivers frontier-quality reasoning at $0.50/M tokens will capture the enterprise market.

### 3. The Sovereign-AI Market is the Largest Growth Vector

The **sovereign-AI market** (governments, defense, regulated industries that cannot use US or Chinese cloud APIs) is the **fastest-growing segment** of 2026-2028. Chinese open-weights models are the **default choice** for non-US-aligned countries, but the EU is investing in Mistral, and the UAE/India/Saudi have their own programs. The market is bifurcating.

### 4. Closed US Labs Will Compete on Integration, Not Raw Capability

If the open-weights frontier is Chinese, US closed labs (OpenAI, Anthropic, Google) will compete on:
- **Integration** (deep tool use, agentic frameworks, computer use, browser use)
- **Trust and safety** (enterprise compliance, audit trails, jailbreak resistance)
- **Domain expertise** (legal, medical, financial fine-tunes)
- **UX and product polish** (the ChatGPT app, Claude.ai, Gemini in Workspace)

The "raw model" is becoming commoditized. The "productized intelligence" is where the closed labs will retain pricing power.

### 5. The Hardware Stack is Bifurcating

The AI hardware market is splitting into:
- **Nvidia + TSMC** — US/Europe, frontier training
- **Huawei Ascend + SMIC** — China, frontier training
- **Cerebras, Groq, SambaNova, Tenstorrent** — inference-optimized
- **Apple Silicon, Qualcomm, MediaTek** — edge / on-device

Chinese labs are **proving the second stack works**, which has huge geopolitical implications. If Chinese frontier models can be trained and served entirely on Chinese hardware, the US export-control regime loses its leverage.

---

## Forecasts for 2026–2028

| Quarter | Predicted Event | Confidence |
|---------|-----------------|-----------|
| 2026 Q3 | DeepSeek R2 release | 85% |
| 2026 Q4 | GLM-6 announced, ~2T params | 70% |
| 2027 Q1 | Qwen 4 release, >1T params, 10M context | 75% |
| 2027 Q2 | **First Chinese open-weights model passes AA Index 75** | 80% |
| 2027 Q3 | Mistral Large 4 closes gap to Chinese open-weights (within 2 AA Index points) | 50% |
| 2027 Q4 | **First truly agentic model**: can autonomously complete a 1-week software project with 90%+ success | 40% |
| 2028 Q1 | Moonshot Kimi K4: 10M-token context, MIT license | 60% |
| 2028 Q2 | **First open-weights video-generation model competitive with Sora 2 / Veo 3** (from Zhipu or Alibaba) | 60% |
| 2028 Q3 | **First open-weights model trained entirely on Chinese hardware (Ascend 910D)** | 70% |
| 2028 Q4 | **AA Index 80 open-weights model** (likely Chinese) | 65% |

### Wildcards

- A **major Chinese open-weights model gets pulled from HuggingFace** due to a national-security event. This would be a 1-in-4-year event, but it has happened before (LLaMA, Stable Diffusion). The implications would be massive.
- A **Chinese lab IPOs** (Alibaba Cloud spinoff, Zhipu listing on HKEX or STAR Market). The valuation would be a market signal for sovereign-AI.
- **BIS tightens the open-weights exception** in response to a model like DeepSeek v4 or GLM-5.2 demonstrating frontier capabilities. This would force non-US entities to rely on distilled versions.
- A **breakthrough in domestic Chinese compute** (Cambricon 5nm, SMIC 5nm) closes the gap to Nvidia. This would be a 2-3 year overnight event.

---

## Cross-References and Further Reading

### Within This Library

- `02-LLMs/01-Transformer-Architecture.md` — Transformer fundamentals (attention, MLP, layer norm)
- `02-LLMs/02-Model-Families.md` §5, 7 — DeepSeek and Qwen family (2024-vintage coverage)
- `02-LLMs/03-Tokenization.md` — BPE, SentencePiece, Chinese tokenization
- `02-LLMs/04-Quantization.md` — Q4, Q8, AWQ, GPTQ, FP8 quantization
- `02-LLMs/06-AI-Model-Providers-Free-Tiers.md` — Free API tiers for Qwen, GLM, DeepSeek
- `03-Agents/` — Agentic frameworks (LangGraph, AutoGen, CrewAI)
- `04-RAG/` — Retrieval-augmented generation, vector DBs, hybrid search
- `06-Advanced/06-Mixture-of-Experts.md` — MoE deep dive (DeepSeekMoE)
- `06-Advanced/13-Multi-Head-Latent-Attention.md` — MLA deep dive
- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL for agentic systems
- `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` — 2026 LLM architecture trends
- `17-Research-Frontiers-2026/06-Reasoning-Models.md` — Reasoning RL, GRPO
- `17-Research-Frontiers-2026/08-AI-for-Science.md` — AI for scientific discovery
- `20-Agent-Infrastructure-and-Observability/` — AgentOps, eval, observability
- `23-Local-AI-Inference-Self-Hosting/` — Self-hosting, vLLM, llama.cpp, Ollama
- `24-AI-Agent-Autonomy-and-Accountability/` — Operator liability
- `25-World-Models/` — World models and embodied AI
- `28-Agentic-Git/` — Agentic coding patterns (SWE-bench, repo-level)

### External Resources

- **Artificial Analysis** — https://artificialanalysis.ai — Intelligence Index leaderboard
- **LMArena** — https://lmarena.ai — Crowdsourced ELO
- **HuggingFace Open LLM Leaderboard** — https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- **OpenCompass** — https://opencompass.org.cn — Chinese benchmark suite
- **SWE-bench** — https://www.swebench.com — Agentic coding benchmark
- **RULER** — https://github.com/NVIDIA/RULER — Long-context benchmark
- **ModelScope** — https://modelscope.cn — Alibaba's model hub
- **WiseModel** — https://wisemodel.cn — Zhipu's Chinese mirror
- **DeepSeek blog** — https://deepseek.com/blog
- **Z.ai blog** — https://z.ai/blog
- **Qwen blog** — https://qwenlm.github.io/blog
- **Kimi blog** — https://kimi.moonshot.cn/blog

### Key Papers (2024–2026)

| Paper | Year | Authors | Significance |
|-------|------|---------|--------------|
| Multi-Head Latent Attention (DeepSeek-V2) | 2024-05 | DeepSeek | MLA, the foundation of modern MoE inference |
| DeepSeek-V3 Technical Report | 2024-12 | DeepSeek | 671B MoE trained for $5.5M |
| DeepSeek-R1 | 2025-01 | DeepSeek | Pure RL for reasoning, R1-Zero |
| Qwen3 Technical Report | 2025-04 | Alibaba | Hybrid reasoning modes |
| Auxiliary-Loss-Free Load Balancing | 2024-12 | DeepSeek (ICLR 2026) | Clean MoE scaling |
| Grouped Relative Policy Optimization | 2025-04 | DeepSeek + Tsinghua | GRPO, the standard RL algorithm |
| Qwen3-Coder Technical Report | 2025-07 | Alibaba | Agentic coding, 480B-A35B |
| GLM-5 Technical Report | 2026-02 | Zhipu | Long-horizon agentic engineering |
| DeepSeek v4 Technical Report | 2026-04 | DeepSeek | 1.6T MoE, agentic training |
| GLM-5.2 Model Card | 2026-06 | Zhipu | Frontier intelligence, open weights |

---

## Appendix A — Model Card Quick Reference

### Qwen 3.7-Max (Flagship, June 2026)

```
Model:       Qwen/Qwen3.7-Max-Instruct
Parameters:  ~700B total, ~50B active
Layers:      88
Heads:       96 attention, 8 KV (GQA)
Experts:     512 routed, 4 shared, 16 active
Vocab:       220K
Context:     200K native, 2M via YaRN
License:     Apache 2.0
Release:     2026-06
Use cases:   Agentic coding, complex reasoning, planning, multi-step tasks
Not for:     Real-time low-latency, edge deployment, languages with <1M speakers
```

### GLM-5.2 (Open-Weights Frontier Leader, June 2026)

```
Model:       zai-org/GLM-5.2
Parameters:  ~1.2T total, ~50B active
Layers:      80
Heads:       96 attention, MLA v3
Experts:     512 routed, 4 shared, 16 active
Vocab:       220K
Context:     200K native, 2M via YaRN
License:     MIT
Release:     2026-06
Use cases:   Long-horizon agentic tasks, plan/act workflows, complex coding
Not for:     Edge deployment, low-latency, languages with poor Chinese data overlap
```

### DeepSeek v4 (Reasoning Frontier, April 2026)

```
Model:       deepseek-ai/DeepSeek-V4
Parameters:  1.6T total, 48B active
Layers:      96
Heads:       96 attention, MLA v2
Experts:     384 routed, 2 shared, 8 active
Vocab:       200K
Context:     128K native, 1M via YaRN
License:     MIT
Release:     2026-04
Use cases:   Reasoning, math, code, agentic workflows
Not for:     High-volume classification (use Qwen3.6-27B instead), edge
```

### Qwen3.6-35B-A3B (Best Cost/Performance, May 2026)

```
Model:       Qwen/Qwen3.6-35B-A3B-Instruct
Parameters:  35B total, 3B active
Layers:      60
Heads:       48 attention, 8 KV (GQA)
Experts:     64 routed, 2 active
Vocab:       152K
Context:     100K
License:     Apache 2.0
Release:     2026-05
Use cases:   Coding, agentic, real-time chat, on-device deployment
Not for:     Frontier hard problems (use Qwen3.7-Max or GLM-5.2)
```

### Qwen3.6-27B (Best Value, April 2026)

```
Model:       Qwen/Qwen3.6-27B-Instruct
Parameters:  27B dense
Layers:      48
Heads:       48 attention, 8 KV (GQA)
Vocab:       152K
Context:     100K
License:     Apache 2.0
Release:     2026-04
Use cases:   Coding, general tasks, single-GPU deployment
Not for:     Frontier hard problems, very long context
```

### Kimi K3 Preview (Long-Context Specialist, June 2026)

```
Model:       moonshotai/Kimi-K3-Preview
Parameters:  ~500B total, ~32B active (estimated)
Context:     2M
License:     Apache 2.0 (expected)
Release:     2026-06 (preview)
Use cases:   Long-document analysis, multi-book QA, full-codebase ingest
Not for:     Low-latency, simple tasks, edge
```

### Hunyuan A13B (Mid-Tier, March 2026)

```
Model:       tencent/Hunyuan-A13B-Instruct
Parameters:  13B dense
Layers:      40
Vocab:       130K
Context:     64K
License:     Tencent Custom (commercial OK, some restrictions)
Release:     2026-03
Use cases:   Bilingual EN/ZH, Tencent ecosystem integration
Not for:     Frontier hard problems, very long context
```

---

## Appendix B — License Comparison

| License | Commercial Use | Redistribution | Modification | Field-of-Use Restrictions | Notable Models |
|---------|----------------|----------------|--------------|---------------------------|----------------|
| **MIT** | ✅ | ✅ | ✅ | None | GLM-5.2, GLM-5.1, DeepSeek v4, DeepSeek V3, DeepSeek R1 |
| **Apache 2.0** | ✅ | ✅ | ✅ | None (patent grant included) | Qwen 3.7-Max, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3-Coder, Qwen3-TTS, Kimi K2 |
| **Tencent Custom** | ✅ | ✅ | ✅ | Some political content restrictions | Hunyuan A13B |
| **Llama 4 Community** | ✅ (>700M MAU needs license) | ✅ | ✅ | Some restrictions on Llama-named competing models | Llama 4 Behemoth |
| **Mistral Research** | ❌ (research only) | ❌ | ❌ | n/a | Mistral Large 3 (research edition) |
| **Mistral Commercial** | ✅ | ✅ | ✅ | Some restrictions | Mistral Large 3 (commercial) |
| **Anthropic Acceptable Use** | ✅ (via API) | ❌ | ❌ | Strict AUP | Claude Opus 4.5 |
| **OpenAI Terms** | ✅ (via API) | ❌ | ❌ | Strict AUP | GPT-5.1 |

### Open-Weights License Recommendations

For most enterprise use cases:
- **MIT and Apache 2.0 are equivalent** for legal purposes. Both allow commercial use, modification, and redistribution with minimal restrictions.
- **Tencent Custom is acceptable** for most enterprise use, but has political-content restrictions that may be a deal-breaker for some government and journalism use cases.
- **Llama 4 Community License** has a 700M MAU threshold and Llama-naming restrictions. Below the threshold, it's essentially equivalent to MIT.
- For **government and defense use cases**, prefer **MIT and Apache 2.0** models (GLM-5.2, DeepSeek v4, Qwen 3.7-Max, Kimi K2.5) to avoid license-related procurement issues.

---

*Last updated: June 18, 2026. Maintained by the AI Knowledge Library Auto-Enricher. This document is the canonical reference for the Chinese AI ecosystem and the open-weights model race. The library contains 25+ other documents that cross-reference this one; see §22 for the full cross-reference list.*
