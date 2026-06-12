# 07 — AI Code Assistant

## Case Study: Fine-Tuned CodeLlama + RAG for Internal Developer Productivity

| Metadata | Value |
|----------|-------|
| **Industry** | Software Engineering / Developer Tools |
| **Domain** | Code generation, code review, documentation |
| **Difficulty** | Advanced |
| **Est. Timeline** | 8-14 weeks |
| **Team Size** | 5-7 engineers (3 ML, 1 IDE plugin, 1 backend, 1 security) |

---

## 🎯 Problem Statement

### Business Context

**Company:** DevPlatform Inc. (SaaS platform, 450 engineers, 200+ microservices)
**Codebase:** 12M+ lines of Python, TypeScript, Java, Go across 450+ repos
**Developer Base:** 450 engineers, 40% juniors, 30% average tenure < 12 months

### Pain Points

1. **Onboarding Time** — New engineers take 6-9 months to be productive; 3 months just understanding codebase
2. **Code Review Bottleneck** — 1,200 PRs/week; senior engineers spend 4+ hours/day reviewing
3. **Context Switching** — Developers spend 35% of time searching for code/docs, not writing code
4. **Inconsistent Patterns** — 8 different ways to implement the same API pattern across teams
5. **Documentation Decay** — Internal wiki has 12K pages, 40% are outdated; engineers don't trust docs
6. **Boilerplate Overhead** — 25% of new code is boilerplate (setup, config, repetitive patterns)

### Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| **Developer Speed** | +30% (PRs/week) | Current velocity |
| **Bug Introduction Rate** | -20% | Baseline |
| **Code Review Time** | -40% | 4 hrs/day |
| **Onboarding Time** | -50% | 6-9 months |
| **Engineer Adoption** | > 80% | 0% (new) |
| **Code Proposal Acceptance** | > 60% | N/A |

### Constraints

- **Code Security**: NEVER send proprietary code to external APIs (OpenAI, Claude). Must run on-premise or self-hosted LLM.
- **Latency**: IDE completions must appear in < 500ms (inline) and < 5s (chat).
- **Compliance**: SOC2, must log all AI interactions for audit.
- **Code Privacy**: No code stored on third-party servers. All inference on-premise.
- **IDE Support**: Must support VS Code and JetBrains IntelliJ.

---

## 🏗️ Solution Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER (IDE)                                │
│                                                                             │
│  ┌──────────────────────┐           ┌──────────────────────┐                │
│  │  VS Code Extension   │           │  JetBrains Plugin     │               │
│  │  - Inline completions│           │  - Inline completions │               │
│  │  - Chat panel        │           │  - Chat panel         │               │
│  │  - Code review       │           │  - Code review        │               │
│  └──────────┬───────────┘           └──────────┬────────────┘               │
│             │                                  │                           │
│             └──────────────┬───────────────────┘                           │
│                            │ WebSocket + REST                              │
└────────────────────────────┼────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────────┐
│                   API GATEWAY & AUTH                                       │
│                            ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Gateway (rates, auth, audit logging, prompt filtering)      │  │
│  └──────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                    │
└─────────────────────────────────────┼──────────────────────────────────────┘
                                       │
┌─────────────────────────────────────┼──────────────────────────────────────┐
│                      CORE INFERENCE LAYER                                 │
│                                       ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     MODEL ORCHESTRATOR                                │  │
│  │                                                                       │  │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐  │  │
│  │  │  Code Completion   │  │  Chat & Explain    │  │  Code Review   │  │  │
│  │  │  Model (fast)      │  │  Model (deep)      │  │  Model (batch) │  │  │
│  │  │                    │  │                    │  │                │  │  │
│  │  │  CodeLlama-7B      │  │  CodeLlama-34B     │  │  GPT-4 (code   │  │  │
│  │  │  Quantized (AWQ)   │  │  Full precision   │  │  review only)  │  │  │
│  │  └─────────┬──────────┘  └─────────┬──────────┘  └───────┬────────┘  │  │
│  │            │                       │                       │          │  │
│  │            └───────────────────────┼───────────────────────┘          │  │
│  │                                    │                                  │  │
│  │  ┌───────────────────────────────┐ │ ┌────────────────────────────┐  │  │
│  │  │  RAG Context Builder          │ │ │  Code Security Scanner    │  │  │
│  │  │  - Retrieves from ChromaDB   │◀┘ │  - PII / secrets          │  │  │
│  │  │  - Private codebase vectors  │   │  - Dependency vulns       │  │  │
│  │  │  - Internal docs + APIs      │   │  - License check          │  │  │
│  │  └──────────────────────────────┘   └────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Hardware: 4× NVIDIA A100 (80GB) — CodeLlama-7B (2× T4), 34B (1× A100)   │
└─────────────────────────────────────┼────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────┼────────────────────────────────────────┐
│                     KNOWLEDGE & STORAGE LAYER                               │
│                                       ▼                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Vector DB   │  │  Git History │  │  Internal     │  │  Code Index      │ │
│  │  (ChromaDB)  │  │  (graph DB)  │  │  Wiki/RFCs    │  │  (tree-sitter)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │  Model        │  │  Audit Log   │  │  Telemetry   │                      │
│  │  Registry    │  │  (PostgreSQL)│  │  (Prometheus)│                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Fine-Tuning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING PIPELINE (CodeLlama-7B)                  │
│                                                                         │
│  Dataset Sources:                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │  Internal  │  │  PR        │  │  Code      │  │  Code      │        │
│  │  Repos     │  │  Reviews   │  │  Comments  │  │  Commits   │        │
│  │  (450)     │  │  (50K)     │  │  (200K)    │  │  (500K)    │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
│        │               │               │               │              │
│        └───────────────┴───────────────┴───────────────┘              │
│                                    │                                   │
│                                    ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │  Data Processing                                              │      │
│  │  - Extract function-level pairs (code before/after PR)       │      │
│  │  - Format as instruction-response pairs                      │      │
│  │  - Filter PII, secrets, API keys                             │      │
│  │  - Deduplicate (85K unique training pairs)                   │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
│                             │                                         │
│                             ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │  Training (QLoRA)                                            │      │
│  │  - Base: codellama-7b-hf                                    │      │
│  │  - Quantization: 4-bit NormalFloat (NF4)                     │      │
│  │  - LoRA rank: 64, alpha: 128                                 │      │
│  │  - Target modules: q_proj, v_proj                            │      │
│  │  - Batch size: 4 (gradient accumulation: 8)                  │      │
│  │  - Learning rate: 2e-4 (cosine schedule)                    │      │
│  │  - Epochs: 3 (with validation early stop)                     │      │
│  │  - Hardware: 1× A100 (80GB) — training time: 6 hours         │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
│                             │                                         │
│                             ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │  Evaluation                                                 │      │
│  │  - HumanEval (Python): pass@1 from 0.28 → 0.42             │      │
│  │  - Internal benchmark (500 tasks): 72% acceptance           │      │
│  │  - Code security scan: 0.1% injection rate                 │      │
│  └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Base Model** | CodeLlama-7B / 34B (HuggingFace) | 7B / 34B | Open-source, permissive license, code-optimized |
| **Fine-Tuning** | QLoRA via HuggingFace PEFT + TRL | 0.11 / 0.9 | 4-bit quantization, parameter-efficient |
| **Inference** | vLLM + TensorRT-LLM | 0.5 / 0.9 | High-throughput, P50 < 200ms |
| **Vector Store** | ChromaDB | 0.5 | Self-hosted, code embeddings |
| **Code Embeddings** | CodeBERT + starencoder | — | Code-aware embeddings |
| **Code Parsing** | tree-sitter (Python, TS, Java, Go) | 0.22 | Parse-before-embed |
| **RAG Framework** | LangChain + LlamaIndex | 0.2 / 0.10 | Retrieval chains |
| **IDE Plugin** | VS Code Extension API + LSP | 1.90 / 3.17 | Standard extension protocol |
| **Security Scanner** | Semgrep + custom rules | 1.74 | PII/secret detection |
| **Monitoring** | LangSmith + MLflow + Prometheus | — | Trace, log, metric |
| **Registry** | HuggingFace Hub (private) | — | Model versioning |

### Installation

```bash
# Fine-tuning & inference
pip install torch==2.1.2 transformers==4.41.2 accelerate==0.31.0
pip install peft==0.11.1 trl==0.9.6 bitsandbytes==0.43.1
pip install vllm==0.5.1 tensorrt-llm==0.9.0

# Code parsing & embeddings
pip install tree-sitter==0.22.3 tree-sitter-languages==1.10.2
pip install sentence-transformers==3.0.1

# RAG & serving
pip install chromadb==0.5.5 langchain==0.2.12 llama-index==0.10.54
pip install fastapi==0.111.1 websockets==12.0
```

---

## ⚙️ Implementation Details

### 1. Code Indexing with tree-sitter

```python
# src/indexing/code_indexer.py
import tree_sitter
from tree_sitter_languages import get_language, get_parser
from typing import List, Dict, Optional

class CodeIndexer:
    """Parse and index code into chunks using AST-aware splitting."""

    LANGUAGE_MAP = {
        ".py": "python", ".ts": "typescript", ".tsx": "tsx",
        ".java": "java", ".go": "go", ".js": "javascript",
    }

    def __init__(self):
        self.parsers = {}
        for ext, lang in self.LANGUAGE_MAP.items():
            try:
                self.parsers[lang] = get_parser(lang)
            except Exception:
                print(f"Warning: parser for {lang} not available")

    def extract_functions(self, filepath: str) -> List[Dict]:
        """Extract all function/method definitions with context."""
        import os
        ext = os.path.splitext(filepath)[1]
        lang = self.LANGUAGE_MAP.get(ext)
        if not lang or lang not in self.parsers:
            return self._fallback_chunk(filepath)

        with open(filepath, "r") as f:
            code = f.read()

        tree = self.parsers[lang].parse(bytes(code, "utf-8"))
        functions = []

        # Walk AST for function definitions
        def visit(node, depth=0):
            if node.type in (
                "function_definition", "method_definition",
                "function_declaration", "method_declaration",
                "class_definition", "interface_declaration",
            ):
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                lines = code.split("\n")[start_line:end_line + 1]
                body = "\n".join(lines)

                # Extract function name
                name_node = node.child_by_field_name("name")
                name = code[
                    name_node.start_byte:name_node.end_byte
                ] if name_node else "unknown"

                # Extract docstring if present
                docstring = self._extract_docstring(node, code)

                functions.append({
                    "type": node.type,
                    "name": name,
                    "filepath": filepath,
                    "start_line": start_line,
                    "end_line": end_line,
                    "body": body,
                    "docstring": docstring,
                })

            for child in node.children:
                visit(child, depth + 1)

        visit(tree.root_node)
        return functions

    def _extract_docstring(self, node, code: str) -> Optional[str]:
        """Extract docstring from function node."""
        for child in node.children:
            if child.type == "comment" or child.type == "expression_statement":
                text = code[child.start_byte:child.end_byte].strip()
                if '"""' in text or "'''" in text:
                    return text
        return None

    def _fallback_chunk(self, filepath: str) -> List[Dict]:
        """Fallback for unsupported languages — split by blank lines."""
        with open(filepath, "r") as f:
            lines = f.readlines()

        chunks = []
        current_chunk = []
        for i, line in enumerate(lines):
            current_chunk.append(line)
            if line.strip() == "" and len(current_chunk) > 5:
                chunks.append({
                    "type": "code_block",
                    "filepath": filepath,
                    "start_line": i - len(current_chunk) + 1,
                    "end_line": i,
                    "body": "".join(current_chunk),
                })
                current_chunk = []
        return chunks
```

### 2. Fine-Tuning with QLoRA

```python
# src/training/finetune.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

class CodeLlamaFineTuner:
    """Fine-tune CodeLlama with QLoRA on internal codebase."""

    MODEL_NAME = "codellama/CodeLlama-7b-hf"
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.1

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 4-bit quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=self.LORA_R,
            lora_alpha=self.LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def prepare_dataset(self, samples: List[dict]) -> Dataset:
        """Format code pairs as instruction-tuning data."""
        formatted = []
        for sample in samples:
            instruction = sample.get("instruction", "")
            input_code = sample.get("input", "")
            output_code = sample.get("output", "")

            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_code:
                prompt += f"### Input:\n```\n{input_code}\n```\n\n"
            prompt += f"### Response:\n```\n{output_code}\n```"
            prompt += self.tokenizer.eos_token

            formatted.append({"text": prompt})

        return Dataset.from_list(formatted)

    def train(self, train_dataset, val_dataset, output_dir: str = "./codellama-finetuned"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=3,
            logging_steps=10,
            eval_steps=100,
            save_strategy="epoch",
            evaluation_strategy="steps",
            fp16=False,
            bf16=True,
            report_to="wandb",
            run_name="codellama-7b-code-assistant",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=4096,
            dataset_text_field="text",
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer
```

### 3. Inference with vLLM

```python
# src/inference/model_server.py
from vllm import LLM, SamplingParams
from typing import List, Optional
import time

class CodeAssistantInference:
    """Low-latency inference server for fine-tuned CodeLlama."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
    ):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_completion(
        self,
        prefix: str,
        suffix: str = "",
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> dict:
        """Generate inline code completion (fill-in-the-middle)."""
        start_time = time.time()

        # FIM format: <PRE> prefix <SUF> suffix <MID>
        fim_prompt = f"<PRE> {prefix} <SUF> {suffix} <MID>"

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["<EOT>", "\n\n\n"],
        )

        outputs = self.llm.generate([fim_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        latency_ms = (time.time() - start_time) * 1000
        return {
            "completion": generated_text,
            "latency_ms": latency_ms,
            "model": "codellama-7b-fim",
        }

    def generate_chat(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict:
        """Generate chat response with context."""
        start_time = time.time()

        # Format messages as CodeLlama chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        latency_ms = (time.time() - start_time) * 1000
        return {
            "response": response,
            "latency_ms": latency_ms,
        }
```

### 4. RAG Context Building for Code

```python
# src/rag/code_context.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class CodeContextRetriever:
    """Retrieve relevant code context for LLM prompt augmentation."""

    def __init__(self, chromadb_path: str = "./data/chromadb"):
        self.client = PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(
            name="codebase_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(
            "Salesforce/codet5p-110m-embedding"
        )

    def index_code(
        self,
        functions: List[Dict],
        repo: str,
        batch_size: int = 100
    ):
        """Batch index function-level code chunks."""
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            texts = [f["body"] for f in batch]
            embeddings = self.embedder.encode(texts, show_progress_bar=False)

            ids = [f"{repo}::{f['filepath']}::{f['start_line']}" for f in batch]
            metadatas = [
                {
                    "repo": repo,
                    "filepath": f["filepath"],
                    "function_name": f["name"],
                    "type": f["type"],
                    "start_line": f["start_line"],
                    "end_line": f["end_line"],
                }
                for f in batch
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

    def retrieve_context(
        self,
        query: str,
        n_results: int = 10,
        repo_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve relevant code snippets for a query."""
        query_embedding = self.embedder.encode([query])[0]

        where_filters = {}
        if repo_filter:
            where_filters["repo"] = repo_filter
        if type_filter:
            where_filters["type"] = type_filter

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filters if where_filters else None,
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs

    def build_prompt_context(
        self, query: str, max_tokens: int = 2000
    ) -> str:
        """Build a compact context string for LLM prompt augmentation."""
        docs = self.retrieve_context(query, n_results=15)
        context_parts = []
        total_tokens = 0

        for doc in docs:
            snippet = doc["content"]
            meta = doc["metadata"]
            # Rough token count
            tokens = len(snippet.split()) + 20

            if total_tokens + tokens > max_tokens:
                break

            header = f"# {meta['repo']}/{meta['filepath']}:{meta['function_name']}"
            context_parts.append(f"{header}\n```\n{snippet}\n```")
            total_tokens += tokens

        return "\n\n".join(context_parts)
```

### 5. Code Review Agent

```python
# src/agents/code_review.py
from typing import List, Dict, Optional

class CodeReviewAgent:
    """Automated code review using fine-tuned LLM + static analysis."""

    REVIEW_CATEGORIES = [
        "correctness", "performance", "security", "maintainability",
        "style", "test_coverage", "documentation"
    ]

    def __init__(self, llm, static_analyzer=None):
        self.llm = llm
        self.static_analyzer = static_analyzer  # Semgrep or similar

    async def review_patch(
        self, diff: str, filepath: str, language: str
    ) -> Dict:
        """Review a code diff (PR change)."""
        findings = []

        # Step 1: Static analysis
        if self.static_analyzer:
            static_findings = self.static_analyzer.scan(diff, language)
            findings.extend(static_findings)

        # Step 2: LLM-based code review
        prompt = f"""Review this code change in {language}.

        File: {filepath}
        Diff:
        ```diff
        {diff}
        ```

        For each issue found, provide:
        1. Category: {', '.join(self.REVIEW_CATEGORIES)}
        2. Severity: critical / major / minor / suggestion
        3. Line number
        4. Description
        5. Suggested fix

        Format as JSON list:"""

        response = await self.llm.ainvoke(prompt)
        llm_findings = self._parse_review_response(response.content)
        findings.extend(llm_findings)

        return {
            "filepath": filepath,
            "language": language,
            "num_findings": len(findings),
            "findings": findings,
            "passes": self._is_acceptable(findings),
        }

    def _parse_review_response(self, text: str) -> List[Dict]:
        """Parse structured review output from LLM."""
        import json
        import re
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: simple parsing
        return [{"raw": text}]

    def _is_acceptable(self, findings: List[Dict]) -> bool:
        """Check if review passes (no critical or major issues)."""
        for finding in findings:
            severity = finding.get("severity", "").lower()
            if severity in ("critical", "major"):
                return False
        return True
```

### 6. IDE Plugin (VS Code) Stub

```typescript
// vscode-extension/src/extension.ts  (stub for reference)
import * as vscode from 'vscode';
import { CodeAssistantClient } from './client';

export function activate(context: vscode.ExtensionContext) {
    const client = new CodeAssistantClient();

    // 1. Inline code completions
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            {
                async provideInlineCompletionItems(document, position) {
                    const prefix = document.getText(
                        new vscode.Range(new vscode.Position(0, 0), position)
                    );
                    const suffix = document.getText(
                        new vscode.Range(position, document.lineAt(document.lineCount - 1).range.end)
                    );

                    const result = await client.complete(prefix, suffix);
                    return new vscode.InlineCompletionItem(result.completion);
                }
            }
        )
    );

    // 2. Chat panel
    const chatProvider = new CodeChatProvider(client);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('codeAssistant.chat', chatProvider)
    );
}
```

---

## 📊 Metrics & Results

### Model Performance

| Metric | Base CodeLlama-7B | Fine-Tuned (QLoRA) | GPT-4 (External) |
|--------|-------------------|-------------------|-------------------|
| **HumanEval pass@1** | 28.8% | 42.3% | 67.0% |
| **Internal Benchmark pass@1** | 35.2% | 72.1% | 83.5% |
| **P50 Inline Completion Latency** | 180ms | 220ms | 1,200ms |
| **P95 Chat Latency** | 1.2s | 1.8s | 5.5s |
| **Code Security Compliance** | 100% on-prem | 100% on-prem | 0% (external) |
| **API Cost per 1M tokens** | $0.50 (GPU amortized) | $0.50 | $10.00-$30.00 |

### Developer Productivity Impact (6-month study)

| Metric | Before | With AI Assistant | Delta |
|--------|--------|-------------------|-------|
| **PRs per Developer per Week** | 2.8 | 3.7 | +32% |
| **Code Review Time** | 4.2 hrs/day | 2.5 hrs/day | -40% |
| **Onboarding Time (new dev)** | 7.5 months | 3.2 months | -57% |
| **Bug Introduction Rate** | 8.2% of PRs | 6.1% of PRs | -26% |
| **Time Spent Searching Code** | 35% of day | 15% of day | -57% |
| **Boilerplate Code (auto-gen)** | 25% of new | 8% of new | -68% |
| **Engineer Satisfaction (eNPS)** | +12 | +68 | +56 pts |

### Cost-Benefit Analysis

```
┌──────────────────────────────────────────────────────────────┐
│  Investment (Year 1)                                         │
│  GPU Hardware (4× A100 80GB)                    $320,000     │
│  Engineering Time (5 engineers × 3 months)      $450,000     │
│  Infrastructure (servers, storage)               $80,000      │
│  Total Investment                                $850,000     │
│                                                              │
│  Annual Savings                                                │
│  Developer Productivity Gain (32%)                $2,400,000  │
│  Faster Onboarding (57% reduction)                $1,200,000  │
│  Fewer Bugs (26% reduction)                       $900,000    │
│  Code Review Savings (40%)                        $600,000    │
│  Total Annual Savings                             $5,100,000  │
│                                                              │
│  Year-1 ROI                                          ~500%   │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **On-premise model was non-negotiable** — Engineers refused to use any tool that sent code externally. Self-hosting CodeLlama was the only way to get adoption.

2. **QLoRA was sufficient** — Full fine-tuning of 7B model with QLoRA (4-bit) achieved 72% on internal benchmark vs 83% for GPT-4. The 11% gap was acceptable given 20× cost savings and privacy.

3. **RAG context boosts acceptance drastically** — Providing 3-5 relevant code snippets as context improved completion acceptance from 52% to 78%.

4. **Incremental adoption** — Started with chat-only (no inline completions). After 2 weeks, users requested inline completions. Letting users pull, not pushing features.

### ❌ What Went Wrong

1. **Initial model was too big** — Started with CodeLlama-34B. Inference latency was 2-8 seconds for inline completions — users hated it. Switched to 7B-quantized for inline, 34B only for chat.

2. **Repository indexing was slow** — Full codebase index of all 450 repos took 18 hours. Optimized to index only `src/` and `lib/` directories, and only files modified in last 90 days.

3. **Code generation without tests** — Early users accepted AI-generated code without review, introducing subtle bugs. Added mandatory test generation to all completions.

4. **Hallucinated API calls** — Model occasionally invented internal API endpoints. Added retrieval guard: every API usage must cite its source documentation.

### ⚠️ Critical Warnings

```
! WARNING: NEVER send proprietary code to external LLM APIs (OpenAI, etc.).
! WARNING: Generated code must pass security scan before being committed.
! WARNING: Monitor for over-reliance — engineers should review ALL AI output.
! WARNING: Fine-tuned models can memorize PII — filter training data carefully.
```

### Security Guidelines

```
Training Data Filtering:
- ✅ Stripe API keys (regex: sk_live_*, pk_live_*)
- ✅ Internal IP addresses
- ✅ Employee names/emails
- ❌ (Kept) Error messages, stack traces
- ❌ (Kept) Code patterns, architecture decisions

Prompt Filtering (before sending to model):
- ✅ Stripe/API keys → [REDACTED]
- ✅ Internal hostnames → [INTERNAL_HOST]
- ✅ Database connection strings → [DB_CONNECTION]
```

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-CODE-ASSISTANT/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── configs/
│   ├── config.yaml
│   ├── model_config.yaml
│   ├── indexing_config.yaml
│   ├── review_rules.yaml
│   └── security_filters.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── code_indexer.py
│   │   ├── repo_scanner.py
│   │   ├── git_history.py
│   │   └── incremental_indexer.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── finetune.py
│   │   ├── data_prep.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_server.py
│   │   ├── vllm_wrapper.py
│   │   └── fim_generator.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── code_context.py
│   │   ├── code_embedder.py
│   │   └── context_cache.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── code_review.py
│   │   ├── doc_generator.py
│   │   ├── test_generator.py
│   │   └── refactor_agent.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── websocket_handler.py
│   │   ├── schemas.py
│   │   └── auth.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── acceptance_tracker.py
│   │   └── security_log.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── secrets_filter.py
│
├── extensions/
│   ├── vscode/
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── extension.ts
│   │   │   ├── client.ts
│   │   │   ├── inlineCompletion.ts
│   │   │   └── chatPanel.ts
│   │   └── README.md
│   └── jetbrains/
│       ├── src/
│       │   └── main/kotlin/.../
│       └── resources/
│
├── tests/
│   ├── unit/
│   │   ├── test_indexer.py
│   │   ├── test_inference.py
│   │   └── test_review.py
│   ├── integration/
│   │   ├── test_rag.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_code/
│       └── mock_queries.json
│
├── notebooks/
│   ├── 01-finetune-analysis.ipynb
│   ├── 02-rag-quality.ipynb
│   └── 03-adoption-metrics.ipynb
│
├── scripts/
│   ├── index_all_repos.py
│   ├── train_model.sh
│   ├── start_server.sh
│   ├── evaluate_model.py
│   └── sync_embeddings.py
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment-vllm.yaml
│   ├── deployment-api.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── gpu-scheduler.yaml
│
└── docs/
    ├── architecture.md
    ├── model_card.md
    ├── security_review.md
    ├── developer_guide.md
    └── training_data_guide.md
```

### Getting Started

```bash
# 1. Copy template
cp -r TEMPLATE-CODE-ASSISTANT ~/my-code-assistant
cd ~/my-code-assistant

# 2. Install dependencies
make install

# 3. Index a sample repository
python scripts/index_all_repos.py \
  --repo-dir ./tests/fixtures/sample_code/ \
  --output ./data/chromadb

# 4. Start vLLM server (requires GPU)
./scripts/start_server.sh --model codellama/CodeLlama-7b-hf

# 5. Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def fibonacci(n):", "max_tokens": 100}'

# 6. Build VS Code extension
cd extensions/vscode && npm install && npm run compile
```

---

## 📚 References & Further Reading

### Academic Papers
- Rozière et al. (2023) — "Code Llama: Open Foundation Models for Code" — [arXiv:2308.12950](https://arxiv.org/abs/2308.12950)
- Dettmers et al. (2023) — "QLoRA: Efficient Finetuning of Quantized Language Models" — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Fried et al. (2023) — "InCoder: A Generative Model for Code Infilling and Synthesis" — [arXiv:2204.05999](https://arxiv.org/abs/2204.05999)

### Tools & Documentation
- vLLM: https://docs.vllm.ai/
- HuggingFace PEFT: https://huggingface.co/docs/peft
- TRL (Transformer Reinforcement Learning): https://huggingface.co/docs/trl
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- VS Code Extension API: https://code.visualstudio.com/api

### Security
- Semgrep: https://semgrep.dev/docs/
- OWASP Code Review Guide: https://owasp.org/www-project-code-review-guide/

---

> **Next**: [08-Autonomous-Navigation.md](08-Autonomous-Navigation.md) — Warehouse robot navigation with SLAM, path planning, and simulation-to-real deployment.
