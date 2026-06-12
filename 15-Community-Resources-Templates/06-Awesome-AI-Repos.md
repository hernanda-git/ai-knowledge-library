# 06 — Awesome AI Repositories

> **Purpose:** A curated "awesome list" of GitHub repositories organized by category — the go-to reference for finding high-quality open-source AI projects.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Large Language Models (LLMs)](#large-language-models-llms)
3. [Agent Frameworks](#agent-frameworks)
4. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
5. [Vision & Multimodal](#vision--multimodal)
6. [Audio & Speech](#audio--speech)
7. [Model Deployment & Serving](#model-deployment--serving)
8. [Evaluation & Benchmarks](#evaluation--benchmarks)
9. [Safety & Alignment](#safety--alignment)
10. [Fine-Tuning & Training](#fine-tuning--training)
11. [Data & Datasets](#data--datasets)
12. [Developer Tools](#developer-tools)
13. [Vector Databases](#vector-databases)
14. [Model Orchestration](#model-orchestration)
15. [Research & Papers](#research--papers)
16. [Tutorials & Learning](#tutorials--learning)
17. [How to Contribute](#how-to-contribute)

---

## Introduction

This is a **curated, annotated list** of the most significant open-source AI repositories on GitHub. Each entry includes:
- 📝 **Description** — What the project does
- ⭐ **GitHub Stars** — Approximate count (as of June 2026)
- 🔄 **Activity Status** — Active / Maintenance / Inactive
- 📌 **Notes** — Why it's notable, any caveats

### Legend

| Symbol | Meaning |
|--------|---------|
| 🏆 | Category leader / Essential tool |
| 🆕 | New (less than 6 months old) |
| 🔄 | Recently updated |
| 💤 | Low activity (consider before adopting) |
| ☠️ | Archived or deprecated |
| 📚 | Learning resource |
| 🔧 | Developer tool |

---

## Large Language Models (LLMs)

### Open-Source Base Models

#### 🏆 [meta-llama/llama-models](https://github.com/meta-llama/llama-models)
- **Stars:** ~45K ⭐
- **Activity:** ✅ Active
- **Description:** Meta's Llama model family (3, 3.1, 4). The most widely adopted open-source LLM family. Llama 3.1 includes 8B, 70B, and 405B variants.
- **Notes:** Requires license acceptance. The benchmark for open-weight models.

#### 🏆 [mistralai/mistral-framework](https://github.com/mistralai/mistral-framework)
- **Stars:** ~15K ⭐
- **Activity:** ✅ Active
- **Description:** Mistral's open-weight models (7B, 8x7B, 12B, 123B). Known for efficiency and strong performance.
- **Notes:** Mixtral 8x7B was the first open MoE model. Apache 2.0 licensed.

#### [01-ai/Yi](https://github.com/01-ai/Yi)
- **Stars:** ~8K ⭐
- **Activity:** ✅ Active
- **Description:** Series of bilingual (English/Chinese) models by 01.AI (founded by Kai-Fu Lee). 6B, 9B, 34B variants.
- **Notes:** Strong multilingual performance, especially for Chinese.

#### [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- **Stars:** ~12K ⭐
- **Activity:** ✅ Active
- **Description:** Alibaba's Qwen2.5 model family. Strong in math and coding benchmarks. 0.5B–72B variants.
- **Notes:** Consistently high benchmark scores, Apache 2.0 licensed.

#### [google/gemma](https://github.com/google/gemma)
- **Stars:** ~8K ⭐
- **Activity:** ✅ Active
- **Description:** Google's open-weight models. 2B, 7B, 9B, 27B variants. Built on Gemini research.
- **Notes:** Strong for their size. CodeGemma for code tasks available.

#### [microsoft/Phi-3](https://github.com/microsoft/phi-3)
- **Stars:** ~5K ⭐
- **Activity:** ✅ Active
- **Description:** Microsoft's small language models (3.8B–14B). Remarkably capable for their size.
- **Notes:** Excellent for edge/on-device deployment. MIT licensed.

#### [nvidia/nemotron](https://github.com/nvidia/nemotron)
- **Stars:** ~3K ⭐
- **Activity:** ✅ Active
- **Description:** NVIDIA's open LLM series. Strong in reasoning and instruction following.
- **Notes:** Includes reward model and preference data.

### LLM Utilities & Management

#### 🏆 [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Stars:** ~75K ⭐
- **Activity:** ✅ Active
- **Description:** CPU/GPU inference of LLMs in pure C/C++. Runs models on laptops, Raspberry Pi, and phones.
- **Notes:** Essential for local/edge inference. Supports quantization (GGUF format).

#### 🏆 [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- **Stars:** ~45K ⭐
- **Activity:** ✅ Active
- **Description:** Web UI for running LLMs locally. Supports multiple backends (llama.cpp, ExLlama, Transformers).
- **Notes:** Best all-in-one local LLM UI. Lora loading, chat, and API server.

#### [janhq/jan](https://github.com/janhq/jan)
- **Stars:** ~25K ⭐
- **Activity:** ✅ Active
- **Description:** Offline-first, open-source ChatGPT alternative. Desktop app with built-in model runner.
- **Notes:** Very polished user experience. Cross-platform.

#### [LMFlow](https://github.com/OptimalScale/LMFlow)
- **Stars:** ~9K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Toolkit for finetuning and inference of large models. One-click training.
- **Notes:** Good for quick experimentation with fine-tuning.

---

## Agent Frameworks

#### 🏆 [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- **Stars:** ~98K ⭐
- **Activity:** ✅ Active
- **Description:** Framework for building LLM-powered applications. Chaining, agents, RAG, tool use.
- **Notes:** The most widely adopted LLM framework. Python + JS. Extensive ecosystem. See [04-Agent-Toolkits.md](04-Agent-Toolkits.md).

#### 🏆 [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- **Stars:** ~9K ⭐
- **Activity:** ✅ Active
- **Description:** Build stateful, multi-agent applications as graphs. Extension of LangChain.
- **Notes:** Best for complex agent orchestration with state management.

#### 🏆 [microsoft/autogen](https://github.com/microsoft/autogen)
- **Stars:** ~36K ⭐
- **Activity:** ✅ Active
- **Description:** Multi-agent conversation framework from Microsoft Research.
- **Notes:** Excellent for agent-to-agent and human-in-the-loop conversations.

#### 🏆 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)
- **Stars:** ~25K ⭐
- **Activity:** ✅ Active
- **Description:** Framework for orchestrating role-based AI agents.
- **Notes:** Easiest multi-agent framework to get started with.

#### [geekan/MetaGPT](https://github.com/geekan/MetaGPT)
- **Stars:** ~45K ⭐
- **Activity:** ✅ Active
- **Description:** Multi-agent framework for software development automation. Agents act as PM, architect, engineer, QA.
- **Notes:** Unique software engineering focus. Generates PRDs, design docs, and code.

#### [THUDM/AgentBench](https://github.com/THUDM/AgentBench)
- **Stars:** ~5K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Benchmark for evaluating LLM agents across diverse environments.
- **Notes:** Standard for agent evaluation. 8+ different environments.

#### [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel)
- **Stars:** ~22K ⭐
- **Activity:** ✅ Active
- **Description:** Enterprise-grade AI orchestration SDK. Supports C#, Python, Java.
- **Notes:** Best for .NET/Azure enterprise integration. See [04-Agent-Toolkits.md](04-Agent-Toolkits.md).

#### [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- **Stars:** ~170K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** The original autonomous GPT agent experiment. Self-prompting, task-decomposition.
- **Notes:** Highly influential but less actively developed now. More of a reference architecture.

---

## RAG (Retrieval-Augmented Generation)

#### 🏆 [run-llama/llama_index](https://github.com/run-llama/llama_index)
- **Stars:** ~37K ⭐
- **Activity:** ✅ Active
- **Description:** Data framework for LLM applications. Excellent for indexing, retrieval, and RAG pipelines.
- **Notes:** Best-in-class data connector ecosystem (160+ sources).

#### 🏆 [deepset-ai/haystack](https://github.com/deepset-ai/haystack)
- **Stars:** ~18K ⭐
- **Activity:** ✅ Active
- **Description:** Production-ready RAG framework. Modular pipeline architecture.
- **Notes:** Best for enterprise RAG with document processing. See [04-Agent-Toolkits.md](04-Agent-Toolkits.md).

#### 🏆 [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- **Stars:** ~32K ⭐
- **Activity:** ✅ Active
- **Description:** Knowledge base QA with local LLMs. Full Chinese/English RAG application.
- **Notes:** Production-ready RAG app with local model support.

#### [weaviate/Verba](https://github.com/weaviate/Verba)
- **Stars:** ~6K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source RAG application with Weaviate backend. Built for easy experimentation.
- **Notes:** Good for quick RAG prototyping. Includes UI.

#### [microsoft/graphrag](https://github.com/microsoft/graphrag)
- **Stars:** ~20K ⭐
- **Activity:** ✅ Active
- **Description:** Graph-based RAG. Builds knowledge graphs from documents for structured retrieval.
- **Notes:** Novel approach combining graphs + RAG. Strong for multi-hop reasoning.

#### [Qdrant/fastembed](https://github.com/qdrant/fastembed)
- **Stars:** ~1K ⭐
- **Activity:** ✅ Active
- **Description:** Fast, lightweight embedding generation library. 10x faster than sentence-transformers.
- **Notes:** Good for high-throughput RAG systems.

---

## Vision & Multimodal

### Image Generation

#### 🏆 [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- **Stars:** ~145K ⭐
- **Activity:** ✅ Active
- **Description:** Web UI for Stable Diffusion. The most popular SD interface.
- **Notes:** Extensive plugin ecosystem, model browser, inpainting, ControlNet support.

#### 🏆 [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **Stars:** ~60K ⭐
- **Activity:** ✅ Active
- **Description:** Node-based Stable Diffusion interface. Powerful workflow system.
- **Notes:** More technical than A1111 but more flexible. The standard for advanced workflows.

#### [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stars:** ~25K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Stability AI's generative models (Stable Diffusion 3, Stable Video Diffusion).
- **Notes:** Official implementation repository. Check license for each model.

#### [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- **Stars:** ~42K ⭐
- **Activity:** ✅ Active
- **Description:** SD-focused image generation with minimalist UI. Optimized for quality with minimal prompting.
- **Notes:** Great for users who want quality without complex workflows.

### Multimodal LLMs

#### 🏆 [openai/CLIP](https://github.com/openai/CLIP)
- **Stars:** ~25K ⭐
- **Activity:** 💤 Low Activity
- **Description:** Contrastive Language-Image Pre-training. Foundation for many multimodal systems.
- **Notes:** Foundational. Used by most modern vision-language models.

#### [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **Stars:** ~20K ⭐
- **Activity:** ✅ Active
- **Description:** Large Language and Vision Assistant. Strong multimodal LMM.
- **Notes:** Excellent open-source vision-language model. Easy to fine-tune.

#### [InternLM/InternVL](https://github.com/InternLM/InternVL)
- **Stars:** ~8K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source vision-language foundation model. Competitive with GPT-4V on benchmarks.
- **Notes:** One of the strongest open multimodal models.

### Computer Vision

#### 🏆 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Stars:** ~35K ⭐
- **Activity:** ✅ Active
- **Description:** YOLOv5/v8/v9/v10 object detection and image segmentation.
- **Notes:** Industry standard for real-time object detection. Well-maintained.

#### [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- **Stars:** ~30K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** FAIR's object detection and segmentation platform.
- **Notes:** Excellent research platform. Less actively developed since DINO.

---

## Audio & Speech

#### 🏆 [openai/whisper](https://github.com/openai/whisper)
- **Stars:** ~75K ⭐
- **Activity:** ✅ Active
- **Description:** General-purpose speech recognition model. 97 languages. Multiple model sizes.
- **Notes:** Best open-source ASR. Large-v3 is state-of-the-art for many languages.

#### 🏆 [mozilla/TTS](https://github.com/mozilla/TTS)
- **Stars:** ~10K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Text-to-Speech deep learning toolkit. XTTS model.
- **Notes:** Good TTS. Coqui TTS forked and continues active development.

#### [Coqui-ai/TTS](https://github.com/Coqui-ai/TTS)
- **Stars:** ~35K ⭐
- **Activity:** ✅ Active
- **Description:** Deep learning TTS toolkit. XTTS v2 supports voice cloning.
- **Notes:** Actively developed. Best open-source multi-speaker TTS.

#### [suno-ai/bark](https://github.com/suno-ai/bark)
- **Stars:** ~36K ⭐
- **Activity:** 💤 Low Activity
- **Description:** Text-prompted generative audio model. Music, sound effects, speech.
- **Notes:** Innovative but limited by licensing and activity.

#### [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **Stars:** ~21K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Audio generation library. MusicGen, AudioGen, EnCodec.
- **Notes:** Best for music generation. MusicGen models are high quality.

#### 🏆 [lm-sys/FastChat](https://github.com/lm-sys/FastChat)
- **Stars:** ~37K ⭐
- **Activity:** ✅ Active
- **Description:** Platform for training, serving, and evaluating LLMs. Vicuna model. Supports multi-model serving.
- **Notes:** Excellent for model evaluation (MT-Bench). Serves multiple models with one API.

---

## Model Deployment & Serving

#### 🏆 [ollama/ollama](https://github.com/ollama/ollama)
- **Stars:** ~110K ⭐
- **Activity:** ✅ Active
- **Description:** Run LLMs locally with simple CLI. Pull and run models with one command.
- **Notes:** Best developer experience for local models. Docker-like simplicity.

#### 🏆 [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Stars:** ~45K ⭐
- **Activity:** ✅ Active
- **Description:** High-throughput LLM serving engine. PagedAttention for memory efficiency.
- **Notes:** Production standard for LLM serving. 2-10x higher throughput than HF Transformers.

#### 🏆 [khoj-ai/khoj](https://github.com/khoj-ai/khoj)
- **Stars:** ~15K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source AI assistant for your digital brain. Self-hostable.
- **Notes:** Unique personal knowledge base assistant. Integrates with Obsidian, Emacs.

#### [berkeley-reasoning/star-gate](https://github.com/berkeley-reasoning/star-gate)
- **Stars:** ~2K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source inference engine for reasoning models.
- **Notes:** New but promising for specialized inference.

#### [kubeflow/kubeflow](https://github.com/kubeflow/kubeflow)
- **Stars:** ~14K ⭐
- **Activity:** ✅ Active
- **Description:** ML workflow platform on Kubernetes. Pipelines, training, serving.
- **Notes:** Enterprise ML platform. Complex but comprehensive.

#### [ray-project/ray](https://github.com/ray-project/ray)
- **Stars:** ~35K ⭐
- **Activity:** ✅ Active
- **Description:** Unified compute framework for AI and Python applications. Distributed serving.
- **Notes:** Ray Serve for model deployment. Excellent for distributed inference and training.

#### [bentoml/BentoML](https://github.com/bentoml/BentoML)
- **Stars:** ~7K ⭐
- **Activity:** ✅ Active
- **Description:** Unified model serving framework. Build, ship, and scale AI applications.
- **Notes:** Good DX for model serving. OpenTelemetry integration.

---

## Evaluation & Benchmarks

#### 🏆 [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Stars:** ~8K ⭐
- **Activity:** ✅ Active
- **Description:** Standard framework for evaluating language models on 200+ tasks.
- **Notes:** The defacto standard for LLM evaluation. Supports most common benchmarks.

#### 🏆 [openai/human-eval](https://github.com/openai/human-eval)
- **Stars:** ~4K ⭐
- **Activity:** 💤 Low Activity
- **Description:** Code generation benchmark. Function completion from docstrings.
- **Notes:** Standard for code gen evaluation. Simple but effective.

#### [opencompass/opencompass](https://github.com/opencompass/opencompass)
- **Stars:** ~5K ⭐
- **Activity:** ✅ Active
- **Description:** Comprehensive LLM evaluation platform. 100+ datasets, 20+ tasks.
- **Notes:** Strong alternative to lm-eval-harness. Better parallel evaluation.

#### [truera/trulens](https://github.com/truera/trulens)
- **Stars:** ~3K ⭐
- **Activity:** ✅ Active
- **Description:** LLM application evaluation and monitoring.
- **Notes:** Good for RAG evaluation and LLM app quality monitoring.

#### [confident-ai/deepeval](https://github.com/confident-ai/deepeval)
- **Stars:** ~6K ⭐
- **Activity:** ✅ Active
- **Description:** LLM evaluation framework with unit-test-like interface.
- **Notes:** Best Python developer experience for evaluation.

#### [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **Stars:** ~7K ⭐
- **Activity:** ✅ Active
- **Description:** RAG evaluation framework. Metrics for retrieval and generation quality.
- **Notes:** The standard for RAG evaluation. Faithfulness, answer relevancy, context precision.

#### [stanford-crfm/helm](https://github.com/stanford-crfm/helm)
- **Stars:** ~3K ⭐
- **Activity:** ✅ Active
- **Description:** Holistic Evaluation of Language Models. Stanford CRFM's comprehensive benchmark.
- **Notes:** Broad coverage but resource-intensive to run.

---

## Safety & Alignment

#### 🏆 [openai/moderation](https://github.com/openai/moderation)
- **Stars:** ~1K ⭐
- **Activity:** ✅ Active
- **Description:** OpenAI's moderation API reference implementation. Content safety classification.
- **Notes:** Enterprise content moderation standard.

#### 🏆 [lm-sys/FastChat Safety](https://github.com/lm-sys/FastChat)
- **Stars:** ~37K ⭐
- **Activity:** ✅ Active
- **Description:** Includes safety evaluation (MT-Bench, safety categories).
- **Notes:** Safety evaluation integrated into broader platform.

#### [thu-coai/Safety-Prompts](https://github.com/thu-coai/Safety-Prompts)
- **Stars:** ~2K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Safety evaluation dataset and red-teaming prompts for LLMs.
- **Notes:** Good collection of safety test cases.

#### [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)
- **Stars:** ~4K ⭐
- **Activity:** ✅ Active
- **Description:** Safe RLHF framework. Constrained preference optimization.
- **Notes:** Implementation of Safe RLHF paper with BeaverTails dataset.

#### [Anthropic/hh-rlhf](https://github.com/anthropics/hh-rlhf)
- **Stars:** ~2K ⭐
- **Activity:** 💤 Low Activity
- **Description:** Helpful & Harmless RLHF preference dataset.
- **Notes:** Foundational RLHF dataset. Not actively updated but still relevant.

#### [deepseek-ai/DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)
- **Stars:** ~4K ⭐
- **Activity:** ✅ Active
- **Description:** DeepSeek models with strong safety alignment.
- **Notes:** Competitive open-source models with good safety properties.

---

## Fine-Tuning & Training

#### 🏆 [huggingface/peft](https://github.com/huggingface/peft)
- **Stars:** ~16K ⭐
- **Activity:** ✅ Active
- **Description:** Parameter-Efficient Fine-Tuning. LoRA, QLoRA, DoRA, IA3, AdaLoRA, etc.
- **Notes:** Essential for efficient fine-tuning. Integrates with Transformers library.

#### 🏆 [huggingface/transformers](https://github.com/huggingface/transformers)
- **Stars:** ~140K ⭐
- **Activity:** ✅ Active
- **Description:** State-of-the-art ML for PyTorch, TensorFlow, JAX. 100K+ pretrained models.
- **Notes:** The foundational library for modern NLP/ML. Everything depends on it.

#### 🏆 [OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- **Stars:** ~10K ⭐
- **Activity:** ✅ Active
- **Description:** Streamlined LLM fine-tuning framework. Supports most model architectures.
- **Notes:** Best DX for fine-tuning. YAML config + one command.

#### [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook)
- **Stars:** ~6K ⭐
- **Activity:** ✅ Active
- **Description:** Recipes for aligning language models. SFT, DPO, ORPO, KTO, PPO.
- **Notes:** Official HuggingFace alignment recipes. Production code.

#### [unslothai/unsloth](https://github.com/unslothai/unsloth)
- **Stars:** ~18K ⭐
- **Activity:** ✅ Active
- **Description:** 2x faster fine-tuning with half memory. Drop-in replacement for PEFT.
- **Notes:** Significant speed and memory improvements. Easy to use.

#### [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- **Stars:** ~19K ⭐
- **Activity:** 💤 Low Activity
- **Description:** LoRA fine-tuning of LLaMA on Alpaca dataset. The original "fine-tune on a single GPU" repo.
- **Notes:** Historical importance. Most users should use axolotl or unsloth now.

#### [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)
- **Stars:** ~9K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** Decentralized LLM inference. Run large models collaboratively over the internet.
- **Notes:** Unique distributed approach. Good for community-run models.

---

## Data & Datasets

#### 🏆 [huggingface/datasets](https://github.com/huggingface/datasets)
- **Stars:** ~20K ⭐
- **Activity:** ✅ Active
- **Description:** Access and share datasets for ML. 100K+ datasets with one API.
- **Notes:** The standard library for dataset loading and processing.

#### 🏆 [argilla-io/argilla](https://github.com/argilla-io/argilla)
- **Stars:** ~4K ⭐
- **Activity:** ✅ Active
- **Description:** Data annotation and curation platform for LLMs. Human + AI feedback.
- **Notes:** Best open-source data annotation tool for LLMs.

#### [argilla-io/distilabel](https://github.com/argilla-io/distilabel)
- **Stars:** ~2K ⭐
- **Activity:** ✅ Active
- **Description:** LLM-based synthetic data generation and annotation. Pipeline for AI feedback.
- **Notes:** Excellent for generating and labeling training data at scale.

#### [google-research-datasets/dialog-in-browser](https://github.com/google-research-datasets/dialog-in-browser)
- **Stars:** ~1K ⭐
- **Activity:** 💤 Low Activity
- **Description:** Multi-turn dialog dataset collected in-browser.
- **Notes:** Niche but useful for conversation modeling.

---

## Developer Tools

#### 🏆 [continuedev/continue](https://github.com/continuedev/continue)
- **Stars:** ~22K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source AI code assistant for VS Code and JetBrains. Bring your own model.
- **Notes:** Best open-source copilot alternative. Model agnostic.

#### 🏆 [codediodeio/OpenAI-API-deno](https://github.com/codediodeio/OpenAI-API-deno)
- **Stars:** ~1K ⭐
- **Activity:** ✅ Active
- **Description:** Deno version of OpenAI API wrapper.
- **Notes:** Niche but useful for Deno users.

#### [plandex-ai/plandex](https://github.com/plandex-ai/plandex)
- **Stars:** ~10K ⭐
- **Activity:** ✅ Active
- **Description:** AI coding agent for terminal. Plan and build software autonomously.
- **Notes:** Good terminal-first developer tool.

#### [openai/openai-cookbook](https://github.com/openai/openai-cookbook)
- **Stars:** ~60K ⭐
- **Activity:** ✅ Active
- **Description:** Examples and guides for using the OpenAI API.
- **Notes:** Essential reference for API usage patterns.

#### [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook)
- **Stars:** ~8K ⭐
- **Activity:** ✅ Active
- **Description:** Cookbook for Anthropic API. Prompt engineering, tool use, Claude.
- **Notes:** Best resource for Claude-specific patterns.

#### [gpt-engineer-org/gpt-engineer](https://github.com/gpt-engineer-org/gpt-engineer)
- **Stars:** ~52K ⭐
- **Activity:** 🔄 Maintenance
- **Description:** AI-powered code generation from natural language specifications.
- **Notes:** Ambitious. Quality varies by prompt specificity.

#### [openai/swarm](https://github.com/openai/swarm)
- **Stars:** ~18K ⭐
- **Activity:** ✅ Active
- **Description:** Experimental framework for coordinating multiple AI agents. Educational/experimental.
- **Notes:** Lightweight, focused, educational. Not for production.

---

## Vector Databases

#### 🏆 [weaviate/weaviate](https://github.com/weaviate/weaviate)
- **Stars:** ~12K ⭐
- **Activity:** ✅ Active
- **Description:** Open-source vector database. Hybrid search, generative search, modules.
- **Notes:** Full-featured. Includes built-in LLM integration modules.

#### 🏆 [qdrant/qdrant](https://github.com/qdrant/qdrant)
- **Stars:** ~22K ⭐
- **Activity:** ✅ Active
- **Description:** Vector database written in Rust. Blazing fast, filtering, and rich features.
- **Notes:** Best performance per $ among open-source vector DBs.

#### 🏆 [chroma-core/chroma](https://github.com/chroma-core/chroma)
- **Stars:** ~16K ⭐
- **Activity:** ✅ Active
- **Description:** AI-native open-source embedding database. Simple API, Python-first.
- **Notes:** Best developer experience. Great for prototyping.

#### [milvus-io/milvus](https://github.com/milvus-io/milvus)
- **Stars:** ~32K ⭐
- **Activity:** ✅ Active
- **Description:** Cloud-native vector database. Distributed, GPU-accelerated.
- **Notes:** Best for enterprise-scale deployments. Complex to operate.

#### [pinecone-io/pinecone-cli](https://github.com/pinecone-io/pinecone-cli)
- **Stars:** ~1K ⭐
- **Activity:** ✅ Active
- **Description:** Pinecone CLI and SDK. Managed vector database service.
- **Notes:** Proprietary but generous free tier. CLI for managing indexes.

---

## Model Orchestration

#### 🏆 [lm-sys/FastChat](https://github.com/lm-sys/FastChat)
- **Stars:** ~37K ⭐
- **Activity:** ✅ Active
- **Description:** Platform for training, serving, and evaluating LLM-based chatbots.
- **Notes:** Multi-model serving, MT-Bench evaluation, Vicuna models.

#### [BerriAI/litellm](https://github.com/BerriAI/litellm)
- **Stars:** ~15K ⭐
- **Activity:** ✅ Active
- **Description:** Call all LLM APIs using OpenAI format. 100+ providers. Load balancing.
- **Notes:** Essential for multi-provider setups. Drop-in OpenAI replacement.

#### [portkey-ai/gateway](https://github.com/portkey-ai/gateway)
- **Stars:** ~7K ⭐
- **Activity:** ✅ Active
- **Description:** AI gateway for routing, fallbacks, caching, and observability.
- **Notes:** Production LLM gateway. Rate limiting, retries, spend tracking.

---

## Research & Papers

#### 🏆 [paperswithcode/paperswithcode](https://github.com/paperswithcode/paperswithcode)
- **Stars:** ~1K ⭐ (API)
- **Activity:** ✅ Active
- **Description:** Website + API linking papers to code. ML research tracking.
- **Notes:** Essential for keeping up with ML research.

#### [hendrycks/test](https://github.com/hendrycks/test)
- **Stars:** ~2K ⭐
- **Activity:** 💤 Low Activity
- **Description:** MMLU benchmark. Multitask language understanding.
- **Notes:** The most cited LLM benchmark. Standard evaluation.

#### [mlfoundations/datacomp](https://github.com/mlfoundations/datacomp)
- **Stars:** ~1K ⭐
- **Activity:** ✅ Active
- **Description:** Data filtering benchmark and toolkit for CLIP/LLM training.
- **Notes:** Important for understanding data quality in training.

---

## Tutorials & Learning

#### 🏆 [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)
- **Stars:** ~70K ⭐
- **Activity:** ✅ Active
- **Description:** 18-lesson course on generative AI fundamentals.
- **Notes:** Comprehensive, free, high-quality curriculum.

#### 🏆 [microsoft/ai-for-beginners](https://github.com/microsoft/ai-for-beginners)
- **Stars:** ~35K ⭐
- **Activity:** ✅ Active
- **Description:** 12-week AI curriculum covering ML, NN, CV, NLP, GenAI.
- **Notes:** Broader scope than the GenAI-specific course.

#### [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- **Stars:** ~50K ⭐
- **Activity:** ✅ Active
- **Description:** Comprehensive prompt engineering guide and resources.
- **Notes:** Best community prompt engineering reference.

#### [fastai/fastbook](https://github.com/fastai/fastbook)
- **Stars:** ~22K ⭐
- **Activity:** ✅ Active
- **Description:** "Deep Learning for Coders" book and notebooks. Jupyter-based learning.
- **Notes:** Excellent practical deep learning education.

#### [cohere-ai/notebooks](https://github.com/cohere-ai/notebooks)
- **Stars:** ~2K ⭐
- **Activity:** ✅ Active
- **Description:** Cohere's example notebooks for RAG, embedding, classification.
- **Notes:** Good practical examples with Cohere API.

---

## How to Contribute

To add a repository to this list:

1. **Check for existing entries** — Avoid duplication
2. **Ensure the repo is notable** — Generally >1K stars OR significant impact
3. **Provide complete info** — Name, description, stars, activity status
4. **Use the format:**

```markdown
#### 🏆 [owner/repo](https://github.com/owner/repo)
- **Stars:** ~XK ⭐
- **Activity:** ✅ Active / 🔄 Maintenance / 💤 Low Activity
- **Description:** One-sentence description.
- **Notes:** Why it's notable.
```

5. **Submit a PR** following [08-Contribution-Templates.md](08-Contribution-Templates.md)

### Categories We Welcome

- New model releases (significant open-weight models)
- Developer tools that solve real problems
- Evaluation/safety tools
- Notable learning resources
- RAG/agent frameworks

### Categories We Typically Don't Accept

- Unmaintained repos (< 1 year without update)
- Pure wrappers without new functionality
- Personal projects with < 500 stars
- Commercial products without open-source value

---

## Further Reading

- [04-Agent-Toolkits.md](04-Agent-Toolkits.md) — Detailed agent framework comparisons
- [05-Fine-Tuning-Datasets.md](05-Fine-Tuning-Datasets.md) — Dataset repositories for fine-tuning
- [10-Tools-Ecosystem.md](10-Tools-Ecosystem.md) — Complete ecosystem overview
- [GitHub Explore](https://github.com/explore) — Discover new repositories
- [Best-of ML Python](https://github.com/ml-tooling/best-of-ml-python) — Alternative curated list

---

*Document version 1.0 — Last updated 2026-06-12*
