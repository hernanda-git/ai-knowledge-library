# Tools and Frameworks for Fine-Tuning and Post-Training

> A practical map of the 2026 fine-tuning ecosystem: training libraries, high-level wrappers, managed platforms, data tooling, and serving infrastructure — with guidance on when to choose each.

## Table of Contents

- [The Landscape at a Glance](#the-landscape-at-a-glance)
- [Core Training Libraries](#core-training-libraries)
- [High-Level Wrappers: Axolotl and Unsloth](#high-level-wrappers-axolotl-and-unsloth)
- [Managed Fine-Tuning Platforms](#managed-fine-tuning-platforms)
- [Data Curation and Labeling Tools](#data-curation-and-labeling-tools)
- [Experiment Tracking and Evaluation](#experiment-tracking-and-evaluation)
- [Serving Fine-Tuned Models](#serving-fine-tuned-models)
- [Quantization and Export Formats](#quantization-and-export-formats)
- [Choosing Your Stack](#choosing-your-stack)
- [Cross-References](#cross-references)

---

## The Landscape at a Glance

| Layer | Options |
|-------|---------|
| Data | Argilla, Distilabel, Lilac, Cleanlab, custom scripts |
| Training (low-level) | Transformers + PEFT + TRL, DeepSpeed, FSDP |
| Training (wrappers) | Axolotl, Unsloth, LLaMA-Factory, torchtune |
| Managed | OpenAI FT, Together, Fireworks, Predibase, AWS Bedrock, Vertex AI |
| Tracking | Weights & Biases, TensorBoard, MLflow |
| Eval | lm-eval-harness, LightEval, LLM-as-judge harnesses |
| Serving | vLLM, TGI, SGLang, LoRAX, Ollama |
| Quantize/export | bitsandbytes, GPTQ, AWQ, GGUF (llama.cpp) |

---

## Core Training Libraries

**Hugging Face Transformers + PEFT + TRL** is the reference stack.
- *Transformers* provides models/tokenizers.
- *PEFT* provides LoRA, QLoRA, adapters, prefix tuning.
- *TRL* provides `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `RewardTrainer`, `PPOTrainer`.

Best when you need full control and custom logic. Verbose but transparent.

**torchtune** — PyTorch-native fine-tuning library with clean recipes (LoRA, QLoRA, full FT, DPO) configured via YAML. Good middle ground between raw TRL and opinionated wrappers; strong FSDP/distributed support.

**DeepSpeed / FSDP** — not fine-tuning libraries per se but the sharding backends that make full FT of large models feasible. ZeRO-3 (DeepSpeed) and FSDP (PyTorch) shard optimizer/gradient/parameter state across GPUs.

---

## High-Level Wrappers: Axolotl and Unsloth

**Axolotl** — the community-favorite YAML-driven wrapper. You describe the model, dataset, LoRA config, and hyperparameters in one config file and run one command. Supports SFT, DPO, ORPO, GRPO, multi-GPU, packing, and dozens of dataset formats.

```yaml
# axolotl config excerpt
base_model: meta-llama/Llama-3.1-8B
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
datasets:
  - path: ./train.jsonl
    type: chat_template
sequence_len: 4096
sample_packing: true
num_epochs: 2
learning_rate: 0.0002
```
```bash
accelerate launch -m axolotl.cli.train config.yml
```

**Unsloth** — optimized for **speed and memory** on single GPUs via custom Triton kernels and manual autograd. Claims ~2× faster training and ~50–70% less VRAM than vanilla HF, letting you fine-tune larger models on consumer cards. API mirrors Transformers with a `FastLanguageModel` loader.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.1-8b-bnb-4bit", max_seq_length=4096, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)
```

**LLaMA-Factory** — GUI + CLI platform supporting a huge range of models and methods (SFT, DPO, PPO, ORPO, KTO), popular for rapid experimentation.

Guidance: **Unsloth** for single-GPU speed/VRAM; **Axolotl** for reproducible config-driven multi-GPU pipelines; **torchtune** if you want PyTorch-native; **raw TRL** for research-grade custom loops.

---

## Managed Fine-Tuning Platforms

When you'd rather not run GPUs:

| Platform | Notes |
|----------|-------|
| OpenAI Fine-Tuning | SFT + DPO on GPT models; simplest for closed models |
| Together AI | Open models, LoRA/full FT, pay-per-token training |
| Fireworks AI | Fast fine-tuning + serving of open models, multi-LoRA |
| Predibase | LoRA-centric, LoRAX serving, adapter marketplace |
| AWS Bedrock / SageMaker | Enterprise, VPC, custom model import |
| Google Vertex AI | Managed tuning of Gemma/Gemini + open models |
| Azure AI Foundry | Enterprise fine-tuning + governance |

Trade-off: managed platforms remove ops burden but reduce control, may be pricier at scale, and can lock you into their serving. See `25-Multi-Cloud-AI-Strategy/` and `41-AI-Cost-Optimization-and-Enterprise-ROI/`.

---

## Data Curation and Labeling Tools

Because data quality dominates outcomes:

- **Argilla** — open-source data annotation/curation UI for building SFT and preference datasets; human-in-the-loop review.
- **Distilabel** — programmatic synthetic data + AI-feedback pipelines (generate, critique, rank) for building preference pairs (pairs well with `51-Synthetic-Data-Generation/`).
- **Cleanlab** — automatically finds label errors and outliers in datasets.
- **Lilac / dataset viewers** — explore, cluster, and dedupe text datasets at scale.

A common 2026 recipe: Distilabel generates candidate responses from a strong teacher, an AI judge ranks them into chosen/rejected, and Argilla surfaces low-confidence cases for human review — producing a DPO dataset semi-automatically.

---

## Experiment Tracking and Evaluation

- **Weights & Biases** — the standard for logging loss, LR, gradient norms, sample generations, and comparing runs.
- **MLflow / TensorBoard** — open alternatives.
- **lm-evaluation-harness (EleutherAI)** — standardized academic benchmarks (MMLU, GSM8K, etc.) to catch capability regressions.
- **LightEval / custom LLM-as-judge** — task-specific and pairwise preference evals.

Always run a **before/after** eval: base vs fine-tuned on both your task metric *and* a general regression suite. See `52-AI-Hallucination-Detection-and-Mitigation/`.

---

## Serving Fine-Tuned Models

- **vLLM** — high-throughput serving with PagedAttention and native multi-LoRA (`--enable-lora`). The default for open-model serving.
- **TGI (Text Generation Inference)** — Hugging Face's production server, LoRA support, good ecosystem integration.
- **SGLang** — fast serving with strong structured-output and RadixAttention caching.
- **LoRAX (Predibase)** — purpose-built for serving *thousands* of LoRA adapters on one base model.
- **Ollama / llama.cpp** — local/edge serving of merged GGUF models (see `62-Edge-AI-and-On-Device-Inference/`).

---

## Quantization and Export Formats

Post-training, you often quantize for cheaper serving:

| Format/Method | Use case |
|---------------|----------|
| GGUF (llama.cpp) | CPU/consumer/local, Ollama |
| AWQ | Activation-aware 4-bit, good quality on GPU |
| GPTQ | Popular 4-bit GPU quantization |
| bitsandbytes NF4 | Training-time (QLoRA) and quick inference |
| FP8 | Native on H100/B200 for near-lossless speedups |

Merge LoRA → export to the target format → serve. Validate that quantization didn't degrade your task metric.

---

## Choosing Your Stack

| Scenario | Recommended stack |
|----------|-------------------|
| Solo dev, 1 consumer GPU | Unsloth + QLoRA → GGUF → Ollama |
| Reproducible team pipeline | Axolotl + W&B + vLLM |
| Research / custom loss | TRL/torchtune + FSDP |
| No GPUs / enterprise | Together / Fireworks / Bedrock |
| Many per-customer adapters | LoRA + LoRAX or vLLM multi-LoRA |
| Reasoning capability build | TRL GRPO + verifier + vLLM sampling |

---

## Cross-References

- `03-Technical-Deep-Dive.md` — code using these tools
- `51-Synthetic-Data-Generation/` — Distilabel and data creation
- `52-AI-Hallucination-Detection-and-Mitigation/` — evaluation harnesses
- `62-Edge-AI-and-On-Device-Inference/` — GGUF/Ollama serving
- `63-GPU-Kernel-and-Inference-Performance-Engineering/` — vLLM/SGLang performance
- `41-AI-Cost-Optimization-and-Enterprise-ROI/` — build vs managed economics
- `25-Multi-Cloud-AI-Strategy/` — platform selection

Continue to `05-Future-Outlook.md`.
