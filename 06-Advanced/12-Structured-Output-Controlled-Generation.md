# Structured Output & Controlled Generation

## Table of Contents

1. [Overview](#overview)
2. [Why Structured Output Matters](#why-structured-output-matters)
3. [Techniques for Controlled Generation](#techniques-for-controlled-generation)
4. [JSON Mode & Schema Enforcement](#json-mode--schema-enforcement)
5. [Grammar-Based Generation](#grammar-based-generation)
6. [Tool/Function Calling](#toolfunction-calling)
7. [Outlines Library](#outlines-library)
8. [Instructor Library](#instructor-library)
9. [JSONformer & lm-format-enforcer](#jsonformer--lm-format-enforcer)
10. [Provider-Specific Solutions](#provider-specific-solutions)
11. [Constrained Decoding with llama.cpp](#constrained-decoding-with-llamacpp)
12. [Validation & Retry Strategies](#validation--retry-strategies)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Production Patterns](#production-patterns)
15. [Cross-References](#cross-references)

---

## Overview

Structured output — ensuring LLMs produce responses that conform to a specific schema, grammar, or format — is one of the most critical capabilities for production LLM deployments. Without it, applications face unreliable parsing, silent failures, and cascading errors.

### The Problem

LLMs generate free-form text. When you need a JSON object, a specific enum value, or a regex-constrained string, the default behavior is unreliable:

```
User: Extract the date, amount, and vendor from this invoice.
LLM (unstructured): The invoice date is January 15, 2026. The total amount 
                    is $1,234.56. The vendor is Acme Corp.
LLM (structured): {"date": "2026-01-15", "amount": 1234.56, "vendor": "Acme Corp."}
```

Without controlled generation, you're relying on prompt engineering and hoping the model follows format instructions — which it won't reliably do.

---

## Why Structured Output Matters

### Production Requirements

| Requirement | Without Structured Output | With Structured Output |
|-------------|--------------------------|----------------------|
| **Data extraction** | 60-80% parse rate | 99.9%+ valid JSON |
| **API integration** | Manual error handling | Automatic schema validation |
| **Multi-step agents** | Cascading format failures | Reliable state transitions |
| **Type safety** | Stringly-typed | Full type enforcement |
| **Cost** | Retries on parse errors | First-attempt success |

### Business Impact

- **Reliability:** Schema-constrained output eliminates the most common failure mode in LLM applications — unparseable responses
- **Latency:** No retry loops for malformed output (can save 30-50% on end-to-end latency)
- **Cost:** Fewer tokens wasted on formatting instructions and retries
- **Developer Experience:** Type-safe interfaces between LLMs and application code

---

## Techniques for Controlled Generation

### Taxonomy of Approaches

```
Controlled Generation
├── Prompt-Based
│   ├── JSON mode instructions
│   ├── Few-shot examples
│   └── Output format constraints in system prompt
├── Decoding-Time Constraints
│   ├── Grammar-based (GBNF, EBNF)
│   ├── Logit masking/bias
│   └── Beam search with constraints
├── Schema Enforcement
│   ├── JSON Schema validation + retry
│   ├── Pydantic/Structured output libraries
│   └── Type-safe generation
└── Post-Processing
    ├── Regex extraction
    ├── Template filling
    └── Validation + correction loop
```

### Comparison

| Method | Reliability | Latency Overhead | Complexity | Flexibility |
|--------|------------|-----------------|------------|-------------|
| Prompt-based | Low (70-90%) | None | Low | High |
| Grammar decoding | Very High (99.9%) | +5-15% | Medium | High |
| Logit masking | Very High (99.9%) | +2-10% | High | Medium |
| Schema validation + retry | High (95-99%) | +retry cost | Low | High |
| Instructor library | Very High (99.5%) | +1-5% | Low | High |
| Outlines library | Very High (99.9%) | +5-20% | Medium | High |

---

## JSON Mode & Schema Enforcement

### OpenAI JSON Mode

OpenAI's JSON mode forces the model to produce valid JSON:

```python
from openai import OpenAI
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract invoice data as JSON."},
        {"role": "user", "content": invoice_text}
    ],
    response_format={"type": "json_object"}  # Enforces JSON output
)

data = json.loads(response.choices[0].message.content)
```

**Limitations:**
- Only guarantees valid JSON, **not** a specific schema
- Model may produce extra fields, wrong types, or different structure
- Requires careful prompt engineering for schema alignment

### OpenAI Structured Outputs (Schema Mode)

OpenAI's structured outputs mode enforces a specific JSON Schema:

```python
from pydantic import BaseModel
from openai import OpenAI

class Invoice(BaseModel):
    invoice_number: str
    date: str
    vendor: str
    amount: float
    tax_rate: float | None = None
    line_items: list[str]

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": invoice_text}],
    response_format=Invoice,
)

invoice: Invoice = completion.choices[0].message.parsed
# → Invoice(invoice_number="INV-2026-001", date="2026-01-15", ...)
```

**How it works:** OpenAI fine-tunes the model's output distribution to constrain tokens to schema-valid paths. This is a server-side implementation of constrained decoding.

**Supported types:** Pydantic models, JSON Schema, typed dicts, enums, unions, optional fields, nested models, lists, and more.

---

## Grammar-Based Generation

### GBNF (GGML BNF) Grammar

llama.cpp and related projects support GBNF grammars for fine-grained output control:

```
# Grammar for a simple JSON object
root ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= string ws ":" ws value
string ::= "\"" [a-zA-Z0-9_-]+ "\""
value ::= string | number | "true" | "false" | "null" | object | array
number ::= "-"? [0-9]+ ("." [0-9]+)?
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
array ::= "[" ws (value ("," ws value)*)? ws "]"
ws ::= [ \t\n]*
```

### Using Grammar in llama.cpp

```bash
# Generate a response constrained to valid JSON
./main -m model.gguf \
  --grammar-file json.gbnf \
  -p "Extract: The invoice total is $1234.56 from Acme Corp on 2026-01-15"
```

### Python Integration

```python
import llama_cpp

llm = llama_cpp.Llama(model_path="model.gguf", grammar="json.gbnf")

output = llm(
    "Extract: The invoice total is $1234.56 from Acme Corp on 2026-01-15",
    max_tokens=100
)
# Output is guaranteed valid JSON
```

---

## Tool/Function Calling

Tool calling is the most widely used structured output mechanism in production:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_invoice",
            "description": "Extract invoice data from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "amount": {"type": "number"},
                    "vendor": {"type": "string"},
                    "date": {"type": "string", "format": "date"}
                },
                "required": ["invoice_number", "amount", "vendor"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": invoice_text}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_invoice"}}
)
```

**Provider support:**

| Provider | Tool Calling | Parallel Tools | Strict Schema |
|----------|-------------|----------------|---------------|
| OpenAI | ✅ | ✅ | ✅ (Structured Outputs) |
| Anthropic | ✅ | ✅ | ⚠️ (via tool descriptions) |
| Google Gemini | ✅ | ✅ | ✅ |
| AWS Bedrock | ✅ | ✅ | ⚠️ |
| Ollama (local) | ✅ | ⚠️ | ⚠️ |
| vLLM | ✅ | ✅ | ⚠️ |

---

## Outlines Library

Outlines is a Python library for structured text generation using regular expressions, JSON Schema, and context-free grammars.

### JSON Schema Mode

```python
from outlines import generate, models

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "occupation": {"type": "string", "enum": ["engineer", "doctor", "teacher"]}
    },
    "required": ["name", "age", "occupation"]
}

generator = generate.json(model, schema)
result = generator("Extract person info: John is a 30-year-old engineer.")
# → {"name": "John", "age": 30, "occupation": "engineer"}
```

### Regex Mode

```python
# Generate text that must match a regex
generator = generate.regex(model, r"(Mr|Ms|Dr)\. [A-Z][a-z]+ [A-Z][a-z]+")
result = generator("Generate a formal name:")
# → "Dr. Jane Smith"
```

### How Outlines Works

Outlines uses **FSM-based decoding** — it converts the schema/grammar into a finite state machine that constrains which tokens the model can generate at each step:

1. Convert JSON Schema → context-free grammar
2. Convert grammar → deterministic finite automaton
3. At each decoding step, mask logits for tokens not in the valid next-token set
4. Sample only from allowed tokens → guaranteed valid output

---

## Instructor Library

Instructor provides Pydantic-validated structured outputs across multiple LLM providers.

### Basic Usage

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class UserDetail(BaseModel):
    name: str
    age: int
    role: str

# Patch the client
client = instructor.from_openai(OpenAI())

# Structured extraction — guaranteed valid Pydantic model
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserDetail,
    messages=[{"role": "user", "content": "John is 25 and a developer"}]
)
# → UserDetail(name='John', age=25, role='developer')
```

### Advanced Features

```python
from typing import Literal
from pydantic import Field, BaseModel

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str]
    urgency: Literal["low", "medium", "high"]

# Streaming with structured output
for chunk in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=SentimentAnalysis,
    messages=[{"role": "user", "content": "I loved this product!"}],
    stream=True,
):
    if chunk.key_phrases:
        print(f"Partial phrases: {chunk.key_phrases}")

# With validation
class ValidatedInvoice(BaseModel):
    amount: float = Field(gt=0)
    date: str
    
    @field_validator("date")
    @classmethod
    def valid_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v
```

**Provider support:** OpenAI, Anthropic, Google, Cohere, Mistral, Together, Groq, Ollama, vLLM, and any OpenAI-compatible endpoint.

---

## JSONformer & lm-format-enforcer

### JSONformer

JSONformer generates structured JSON by masking logits at each token position:

```python
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "score": {"type": "number", "minimum": 0, "maximum": 10}
    }
}

builder = Jsonformer(model, tokenizer, schema, prompt="Analyze this review:")
output = builder()
```

### lm-format-enforcer

```python
from lm_format_enforcer import CharacterLevelParser, JsonSchemaParser

# JSON schema enforcement
parser = JsonSchemaParser({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name"]
})

# Use with HuggingFace pipeline
from transformers import pipeline
pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

# RegEx enforcement
regex_parser = CharacterLevelParser(r"([A-Z][a-z]+ )+\((\d{4})\)")  # "Name (2026)"
```

---

## Provider-Specific Solutions

### Anthropic — Tool Use

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-4-opus",
    max_tokens=500,
    tools=[{
        "name": "extract_data",
        "description": "Extract structured data",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["tech", "finance", "healthcare"]
                }
            },
            "required": ["title"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_data"},
    messages=[{"role": "user", "content": "..."}]
)
```

### Google Gemini — Response Schema

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content(
    "Extract: John is 30 years old",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "age": {"type": "INTEGER"}
            }
        }
    )
)
```

### Ollama — Structured Output via Grammar

```bash
curl -X POST http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "..."}],
  "format": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "confidence": {"type": "number"}
    },
    "required": ["summary"]
  },
  "options": {
    "temperature": 0
  }
}'
```

---

## Constrained Decoding with llama.cpp

llama.cpp supports grammar-based constrained decoding at the C++ level, making it the fastest option for local structured output:

```python
import llama_cpp

llm = llama_cpp.Llama(
    model_path="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
    n_gpu_layers=-1,  # Use all GPU layers
    n_ctx=4096,
    grammar="""
root ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= string ws ":" ws value
string ::= "\"" [a-zA-Z0-9_-]+ "\""
value ::= string | number | "true" | "false"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n]*
"""
)

output = llm.create_completion("Extract: price is $99.99", max_tokens=100)
```

### Performance: Grammar vs Unconstrained

| Model | Grammar | Unconstrained | Overhead |
|-------|---------|---------------|----------|
| Llama 3.2 7B (Q4) | 45 tok/s | 52 tok/s | +15% |
| Mistral 7B (Q4) | 48 tok/s | 55 tok/s | +13% |
| Phi-3 Medium (Q4) | 50 tok/s | 58 tok/s | +14% |

---

## Validation & Retry Strategies

Even with structured output techniques, validation is essential:

### Validation Pipeline

```python
from pydantic import BaseModel, ValidationError
from typing import Optional
import instructor

class ExtractedEntity(BaseModel):
    name: str
    type: str
    confidence: float

class ExtractionResponse(BaseModel):
    entities: list[ExtractedEntity]
    error: Optional[str] = None

def extract_with_fallback(text: str, max_retries: int = 3):
    """Extract entities with validation and retry."""
    
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model="gpt-4o",
                response_model=ExtractionResponse,
                messages=[{
                    "role": "user",
                    "content": f"Extract entities from: {text}"
                }]
            )
            
            # Validate
            validated = ExtractionResponse.model_validate(result)
            
            # Business rule validation
            for entity in validated.entities:
                if entity.confidence < 0.0 or entity.confidence > 1.0:
                    raise ValueError(f"Invalid confidence: {entity.confidence}")
            
            return validated
            
        except (ValidationError, ValueError) as e:
            if attempt == max_retries - 1:
                return ExtractionResponse(
                    entities=[],
                    error=f"Failed after {max_retries} attempts: {str(e)}"
                )
            continue
```

### Retry Strategy Comparison

| Strategy | Success Rate | Avg Latency | Best For |
|----------|-------------|-------------|----------|
| Single attempt | 85-95% | 1x | Non-critical | 
| Validate + retry (3x) | 99.5% | 1.5x | Production | 
| Validate + regenerate prompt | 99.8% | 2x | Complex schemas |
| Grammar-constrained (single) | 99.9% | 1.1x | High-reliability systems |

---

## Performance Benchmarks

### Structured Output Libraries Comparison

| Library | Provider | Token Overhead | Latency Impact | Schema Support | Ease of Use |
|---------|----------|---------------|----------------|---------------|-------------|
| OpenAI Structured Outputs | OpenAI | Minimal | +2-5% | JSON Schema / Pydantic | ⭐⭐⭐⭐⭐ |
| Instructor | Multi-provider | Low | +1-5% | Pydantic | ⭐⭐⭐⭐⭐ |
| Outlines | Open-source | Low | +5-20% | JSON Schema, Regex, CFG | ⭐⭐⭐⭐ |
| JSONformer | Open-source | Medium | +10-30% | JSON Schema | ⭐⭐⭐ |
| lm-format-enforcer | Open-source | Low | +5-15% | JSON Schema, Regex | ⭐⭐⭐⭐ |
| llama.cpp grammar | C++ local | Minimal | +10-15% | GBNF | ⭐⭐⭐ |
| Guidance | Multi-provider | Medium | +5-20% | Custom templates | ⭐⭐⭐⭐ |

### Accuracy by Schema Complexity

```
Schema Type          Prompt-Only    Instructor    Grammar-Decoding
Simple JSON             82%           99.5%          99.9%
Nested JSON             65%           98.2%          99.7%
Enum fields             78%           99.8%          99.9%
Regex patterns          55%            N/A           99.8%
Multi-type arrays       60%           97.5%          99.5%
Deeply nested (5+ lvls) 40%           95.0%          99.0%
```

---

## Production Patterns

### Pattern 1: Agent with Structured State

```python
from pydantic import BaseModel
from enum import Enum

class AgentState(Enum):
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    RESPONDING = "responding"
    ESCALATING = "escalating"

class AgentContext(BaseModel):
    state: AgentState
    query: str
    search_results: list[str] = []
    analysis: str | None = None
    response: str | None = None
    confidence: float = 0.0

# Agent loop with structured state transitions
context = AgentContext(state=AgentState.SEARCHING, query=user_input)

while context.state != AgentState.RESPONDING:
    next_state = llm_with_schema(
        prompt=f"Current state: {context.state}. Determine next state.",
        schema=AgentState
    )
    context.state = next_state
    # Process based on new state...
```

### Pattern 2: Multi-Output Extraction

```python
class MultiOutputExtractor(BaseModel):
    """Extract multiple structured outputs from a single prompt."""
    entities: list[dict] = Field(
        description="List of extracted entities",
        max_length=20
    )
    relations: list[dict] = Field(
        description="Relations between entities",
        default_factory=list
    )
    summary: str = Field(description="Brief summary")

# Single call, multiple structured outputs
result = client.chat.completions.create(
    model="gpt-4o",
    response_model=MultiOutputExtractor,
    messages=[{"role": "user", "content": long_document}]
)
```

### Pattern 3: Streaming Structured Output

```python
# Real-time structured output with partial validation
for chunk in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=StockAnalysis,
    messages=[{"role": "user", "content": "Analyze AAPL stock"}],
    stream=True
):
    if chunk.sentiment:
        update_ui(sentiment=chunk.sentiment)
    if chunk.risk_factors:
        for risk in chunk.risk_factors:
            add_risk_indicator(risk)
```

---

## Cross-References

- **LLM Architectures** → [17-Research-Frontiers-2026/03-LLM-Architectures-2026.md](../17-Research-Frontiers-2026/03-LLM-Architectures-2026.md) — Decoding algorithm research
- **Agent Development** → [13-Top-Demand/02-AI-Agent-Development.md](../13-Top-Demand/02-AI-Agent-Development.md) — Tool calling patterns
- **Fine-Tuning** → [13-Top-Demand/07-Fine-Tuning-Custom-Models.md](../13-Top-Demand/07-Fine-Tuning-Custom-Models.md) — Output format fine-tuning
- **Local Inference** → [23-Local-AI-Inference-Self-Hosting/04-Local-LLM-Indexing-and-Search.md](../23-Local-AI-Inference-Self-Hosting/04-Local-LLM-Indexing-and-Search.md) — Grammar-based generation
- **Real-Time Systems** → [13-Top-Demand/11-Real-Time-AI-Systems.md](../13-Top-Demand/11-Real-Time-AI-Systems.md) — Low-latency structured output
- **Evaluation** → [06-Advanced/03-Evaluation-Benchmarks.md](../06-Advanced/03-Evaluation-Benchmarks.md) — Output quality measurement

---

*Last updated: June 2026 | 450+ lines covering all structured output techniques, libraries, benchmarks, and production patterns*
