# Technical Deep-Dive: Fine-Tuning Algorithms and Code

> This document goes hands-on: the math and working code for SFT, LoRA/QLoRA, DPO, and GRPO using the modern Hugging Face + PEFT + TRL stack, plus data pipeline and serving details.

## Table of Contents

- [Environment and Stack](#environment-and-stack)
- [End-to-End SFT with LoRA](#end-to-end-sft-with-lora)
- [QLoRA on a Single GPU](#qlora-on-a-single-gpu)
- [DPO Preference Optimization](#dpo-preference-optimization)
- [GRPO for Reasoning](#grpo-for-reasoning)
- [Merging and Serving Adapters](#merging-and-serving-adapters)
- [Multi-Adapter Serving](#multi-adapter-serving)
- [Distributed and Large-Model Training](#distributed-and-large-model-training)
- [Data Pipeline Code](#data-pipeline-code)
- [Debugging Training Runs](#debugging-training-runs)
- [Cross-References](#cross-references)

---

## Environment and Stack

The de facto 2026 open-source stack:

```bash
pip install "transformers>=4.44" "peft>=0.12" "trl>=0.9" \
            "bitsandbytes>=0.43" "datasets" "accelerate" "flash-attn"
```

- **transformers** — models and tokenizers
- **peft** — LoRA/QLoRA implementations
- **trl** — SFTTrainer, DPOTrainer, GRPOTrainer
- **bitsandbytes** — 4-bit/8-bit quantization for QLoRA
- **accelerate** — multi-GPU/distributed launch
- **flash-attn** — memory-efficient attention

Alternatives that wrap this stack: **Axolotl** and **Unsloth** (see `04-Tools-and-Frameworks.md`).

---

## End-to-End SFT with LoRA

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="bfloat16", attn_implementation="flash_attention_2"
)

# Dataset with a "messages" field in chat format
dataset = load_dataset("json", data_files="train.jsonl", split="train")

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

sft_config = SFTConfig(
    output_dir="./llama3-sft",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=4096,
    packing=True,
    assistant_only_loss=True,        # mask prompt tokens
)

trainer = SFTTrainer(
    model=model, args=sft_config,
    train_dataset=dataset, peft_config=peft_config,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("./llama3-sft/final")
```

Key choices: `assistant_only_loss=True` masks the prompt; `packing=True` improves throughput; effective batch size is controlled via gradient accumulation.

---

## QLoRA on a Single GPU

To fit a large model on one card, load the base in 4-bit:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
    attn_implementation="flash_attention_2",
)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
# ...then attach the same LoraConfig and use SFTTrainer as above.
```

`prepare_model_for_kbit_training` enables gradient checkpointing and casts layer norms to fp32 for stability. Use a **paged optimizer** (`optim="paged_adamw_8bit"` in the config) to survive memory spikes.

---

## DPO Preference Optimization

DPO needs a dataset of `{prompt, chosen, rejected}`. Typically you start from your SFT checkpoint.

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="prefs.jsonl", split="train")
# rows: {"prompt": "...", "chosen": "...", "rejected": "..."}

dpo_config = DPOConfig(
    output_dir="./llama3-dpo",
    beta=0.1,                       # KL strength: lower = more deviation
    learning_rate=5e-6,             # much smaller than SFT
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
)

trainer = DPOTrainer(
    model=model,                    # SFT checkpoint (LoRA ok)
    ref_model=None,                 # None => uses frozen copy / disabled adapter
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

Watch the `rewards/margins` metric — it should grow positive (chosen scoring above rejected). If it stays near zero, your preference pairs are too noisy or β is off.

---

## GRPO for Reasoning

GRPO samples a group of completions per prompt and rewards them with a **verifier function**. Example: train math reasoning where the reward checks the final boxed answer.

```python
from trl import GRPOTrainer, GRPOConfig

def reward_correct(completions, answers, **kwargs):
    rewards = []
    for c, gold in zip(completions, answers):
        pred = extract_boxed_answer(c)      # your parser
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards

def reward_format(completions, **kwargs):
    # small bonus for using <think>...</think> then an answer
    return [0.2 if "<think>" in c and "\\boxed" in c else 0.0
            for c in completions]

grpo_config = GRPOConfig(
    output_dir="./llama3-grpo",
    num_generations=8,              # group size
    max_prompt_length=512,
    max_completion_length=2048,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_correct, reward_format],
    args=grpo_config,
    train_dataset=dataset,          # has "prompt" and "answers"
    processing_class=tokenizer,
)
trainer.train()
```

GRPO advantage per sample is group-normalized: `A_i = (r_i − mean) / std`. Because there is no critic model, memory is lower than PPO. See `29-Reasoning-and-Inference-Scaling/` for reward-shaping strategies and reward-hacking avoidance.

---

## Merging and Serving Adapters

To eliminate inference overhead, merge the LoRA into base weights:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
merged = PeftModel.from_pretrained(base, "./llama3-sft/final").merge_and_unload()
merged.save_pretrained("./llama3-merged")
tokenizer.save_pretrained("./llama3-merged")
```

Note: you cannot merge a LoRA into a 4-bit base cleanly — dequantize to fp16 first, then merge.

Serve the merged model with vLLM:

```bash
vllm serve ./llama3-merged --max-model-len 8192 --dtype bfloat16
```

---

## Multi-Adapter Serving

A major cost win: host **one** base model and swap many small LoRA adapters per request (multi-tenant). vLLM supports this:

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --enable-lora \
  --lora-modules support=./adapters/support sales=./adapters/sales
```

```python
# request selects adapter via the "model" field
client.chat.completions.create(model="support", messages=[...])
```

This lets a SaaS serve per-customer fine-tunes without a GPU per customer. See `63-GPU-Kernel-and-Inference-Performance-Engineering/` and `48-MCP-Cloud-Infrastructure-Agent-as-a-Service/`.

---

## Distributed and Large-Model Training

For full fine-tuning or 70B+ models, use sharding:

- **DeepSpeed ZeRO-3** — shards optimizer states, gradients, and parameters across GPUs.
- **FSDP (Fully Sharded Data Parallel)** — PyTorch-native equivalent, well-integrated with `accelerate`.
- **Gradient checkpointing** — trade compute for memory by recomputing activations.

Launch example:

```bash
accelerate launch --config_file fsdp.yaml train.py
```

Rule of thumb: LoRA/QLoRA avoids most distributed complexity up to ~70B on a single node; full FT of large models needs multi-node ZeRO-3/FSDP.

---

## Data Pipeline Code

Applying the correct chat template and validating stop tokens:

```python
def format_row(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    assert tokenizer.eos_token in text, "Missing EOS — model won't learn to stop!"
    return {"text": text}

dataset = dataset.map(lambda e: format_row(e, tokenizer))
```

Decontaminate against your eval set:

```python
eval_hashes = {hash(x) for x in eval_prompts}
dataset = dataset.filter(lambda e: hash(e["prompt"]) not in eval_hashes)
```

---

## Debugging Training Runs

| Observation | Diagnosis / fix |
|-------------|-----------------|
| Loss not decreasing | LR too low; check data is actually masked correctly |
| Loss → 0 in <100 steps | Overfitting/leakage; reduce epochs, add data |
| OOM at start | Reduce batch/seq len; enable grad checkpointing; use QLoRA |
| OOM mid-run | Memory fragmentation; paged optimizer; lower max_length |
| Gibberish output | Template/EOS mismatch; wrong tokenizer |
| NaN loss | LR too high; disable fp16, use bf16; check for bad rows |
| DPO margins flat | Noisy prefs; raise β; verify chosen≠rejected |

Always log to Weights & Biases or TensorBoard and inspect generations at each checkpoint, not just the loss curve.

---

## Cross-References

- `02-Core-Topics.md` — the theory behind this code
- `04-Tools-and-Frameworks.md` — Axolotl, Unsloth, managed platforms
- `29-Reasoning-and-Inference-Scaling/02-RL-Training-Methodology.md` — GRPO depth
- `63-GPU-Kernel-and-Inference-Performance-Engineering/` — serving performance
- `51-Synthetic-Data-Generation/` — generating the datasets used here

Continue to `04-Tools-and-Frameworks.md`.
