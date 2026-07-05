# 52 — AI Hallucination Detection and Mitigation: Core Topics

> **Category:** 52 — AI Hallucination Detection and Mitigation  
> **Document:** 02 — Core Topics  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](01-Overview.md), [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md)

---

## Table of Contents

1. [Hallucination in Different LLM Paradigms](#1-hallucination-in-different-llm-paradigms)
2. [Factual Grounding Techniques](#2-factual-grounding-techniques)
3. [Citation Integrity and Source Attribution](#3-citation-integrity-and-source-attribution)
4. [Confidence Calibration and Uncertainty Quantification](#4-confidence-calibration-and-uncertainty-quantification)
5. [Domain-Specific Hallucination Patterns](#5-domain-specific-hallucination-patterns)
6. [Hallucination in Multi-Modal Systems](#6-hallucination-in-multi-modal-systems)
7. [Hallucination in Agentic Workflows](#7-hallucination-in-agentic-workflows)
8. [Detection Metrics Deep Dive](#8-detection-metrics-deep-dive)
9. [Mitigation Strategy Selection](#9-mitigation-strategy-selection)

---

## 1. Hallucination in Different LLM Paradigms

### 1.1 Base Models vs. Instruction-Tuned Models

| Characteristic | Base Models | Instruction-Tuned Models |
|---------------|-------------|------------------------|
| **Hallucination type** | Text continuation, pattern completion | Sycophancy, over-confidence |
| **Trigger** | Autoregressive prediction | Following instructions without verification |
| **Mitigation** | Few-shot examples, constrained decoding | System prompts, refusal training |
| **Detection difficulty** | Medium (pattern-based) | Hard (confident and fluent) |

**Base model hallucination example:**
```
Prompt: "The capital of France is Paris. The capital of Germany is Berlin. 
The capital of Spain is Madrid. The capital of Italy is"
Base model: "Florence" (plausible but incorrect — should be "Rome")
```

**Instruction-tuned model hallucination example:**
```
System: You are a helpful assistant.
User: "What's the population of Springfield?"
Model: "Springfield, Illinois has a population of approximately 116,250 
as of the 2020 census." (Confident and specific, but which Springfield? 
There are dozens. The model picked one and presented it as THE answer.)
```

### 1.2 Chat Models vs. Completion Models

Chat models with system prompts can be instructed to ground responses, but they develop unique hallucination patterns:

**Chat-specific hallucinations:**
- **Conversational sycophancy**: Agreeing with incorrect user premises
- **Persona leakage**: Mixing fictional character traits into factual responses
- **Context contamination**: Mixing information from different parts of the conversation
- **Over-helpfulness**: Generating detailed answers when "I don't know" is appropriate

**Completion models:**
- **Text continuation**: Generating plausible-sounding but incorrect continuations
- **Pattern exploitation**: Using statistical patterns instead of factual knowledge
- **Knowledge interpolation**: "Filling in" gaps with plausible but incorrect details

### 1.3 RAG Systems

RAG reduces hallucination but introduces new failure modes:

```
RAG Hallucination Spectrum:

Fully Grounded ────────────────────────────── Pure Hallucination
      │                                            │
      ├── Retrieved doc contains exact answer       │
      ├── Answer derivable from retrieved docs      │
      ├── Answer requires inference from docs       │
      ├── Model ignores retrieved docs              │
      ├── Model combines retrieved + hallucinated   │
      ├── Model hallucinates despite good retrieval  │
      └── Model generates completely fabricated     │
```

**RAG-specific failure modes:**

1. **Retrieval miss**: Relevant documents not retrieved → model fills with training data
2. **Context overflow**: Too many documents → model attends to wrong ones
3. **Synthesis error**: Correct documents but incorrect synthesis
4. **Attribution error**: Correct information but wrong source attribution
5. **Temporal mismatch**: Retrieved documents are outdated → model generates current but wrong answer

### 1.4 Fine-Tuned Models

Fine-tuning can reduce hallucination in specific domains but can also create domain-specific hallucination patterns:

| Fine-Tuning Effect | Positive | Negative |
|-------------------|----------|----------|
| **Domain knowledge** | Better domain-specific answers | Overconfidence in domain |
| **Style adaptation** | Appropriate tone and format | May sacrifice accuracy for style |
| **Refusal behavior** | Better at saying "I don't know" | May refuse valid questions |
| **Fact grounding** | More likely to cite sources | May fabricate plausible-sounding sources |

---

## 2. Factual Grounding Techniques

### 2.1 Prompt-Level Grounding

**Zero-shot grounding:**
```
IMPORTANT: Answer ONLY using the information provided below. If the 
information is insufficient, state that you cannot answer the question 
based on the provided context. Do not use any outside knowledge.

Context: {retrieved_documents}

Question: {user_query}

Answer:
```

**Few-shot grounding:**
```
Instructions: Answer based ONLY on the provided context. Each answer 
must cite the specific document(s) used.

Example:
Context: 
[Doc 1] Acme Corp reported Q3 revenue of $4.2 billion.
[Doc 2] The company saw 12% year-over-year growth.

Q: What was Acme's Q3 revenue?
A: According to Doc 1, Acme Corp reported Q3 revenue of $4.2 billion.

Now answer the following:
Context: {actual_context}
Q: {actual_query}
A:
```

**Chain-of-thought grounding:**
```
Step 1: Read the provided context carefully.
Step 2: Identify which parts of the context are relevant to the question.
Step 3: Formulate your answer using ONLY the relevant parts.
Step 4: For each claim in your answer, note which document supports it.
Step 5: If any part of your answer cannot be supported by the context, 
        mark it as "UNVERIFIED" and explain why.

Context: {context}
Question: {query}

Let's work through this step by step:
```

### 2.2 Architecture-Level Grounding

**Retrieval-Augmented Generation (RAG):**
```python
class GroundedRAG:
    def __init__(self, retriever, generator, verifier):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
    
    async def generate(self, query):
        # Step 1: Retrieve relevant documents
        docs = await self.retriever.retrieve(query, top_k=5)
        
        # Step 2: Build grounded context
        context = self._build_context(docs, query)
        
        # Step 3: Generate with grounding instructions
        response = await self.generator.generate(
            prompt=context,
            system_prompt=self._grounding_system_prompt(),
            temperature=0.1
        )
        
        # Step 4: Verify grounding
        verification = await self.verifier.verify(
            response=response,
            context=context,
            query=query
        )
        
        # Step 5: If verification fails, retry with more specific context
        if verification["hallucination_rate"] > 0.3:
            # Re-retrieve with more specific query
            specific_docs = await self.retriever.retrieve(
                self._extract_claims(response), top_k=3
            )
            context = self._build_context(specific_docs, query)
            response = await self.generator.generate(
                prompt=context,
                system_prompt=self._grounding_system_prompt(),
                temperature=0
            )
        
        return response, verification
    
    def _grounding_system_prompt(self):
        return """You are a factual assistant. Your task is to answer 
questions using ONLY the provided context. 

RULES:
1. Every factual claim MUST be supported by the provided documents
2. If the documents don't contain the answer, say "I don't have enough 
   information in the provided documents to answer this question."
3. Use exact quotes where possible, attributing to the specific document
4. Clearly separate facts from inferences
5. Never fabricate statistics, dates, or specific details"""
```

### 2.3 Knowledge-Grounded Generation

**Constrained decoding with knowledge base:**
```python
def knowledge_constrained_decode(model, knowledge_base, prompt, max_tokens):
    """Generate text constrained by knowledge base content."""
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        # Get model logits
        logits = model.get_logits(prompt + "".join(generated_tokens))
        
        # Filter logits based on knowledge base
        valid_tokens = knowledge_base.get_valid_tokens(
            context="".join(generated_tokens),
            candidate_tokens=torch.topk(logits, k=100).indices
        )
        
        # Apply knowledge-based constraints
        constrained_logits = logits.clone()
        mask = torch.full_like(constrained_logits, float('-inf'))
        mask[:, valid_tokens] = 0
        constrained_logits = constrained_logits + mask
        
        # Sample from constrained distribution
        next_token = torch.multinomial(
            torch.softmax(constrained_logits, dim=-1), 1
        )
        generated_tokens.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated_tokens)
```

### 2.4 Attribution-Aware Generation

**Generating with source attribution:**
```python
async def attributed_generate(llm_client, query, context_docs):
    """Generate response with inline source attribution."""
    
    prompt = f"""Answer the following question using the provided documents. 
For EVERY factual claim, include the document number in brackets.

Format your answer as:
[Claim 1] [Doc 3] [Doc 5]
[Claim 2] [Doc 1]
[Inference based on Doc 3 and Doc 7]

Documents:
{format_numbered_docs(context_docs)}

Question: {query}

Answer with citations:"""
    
    response = await llm_client.generate(prompt, temperature=0)
    
    # Parse and validate citations
    parsed = parse_attributed_response(response)
    validation = validate_citations(parsed, context_docs)
    
    return {
        "response": parsed["text"],
        "citations": parsed["citations"],
        "validation": validation,
        "unsupported_claims": validation["unsupported"]
    }
```

---

## 3. Citation Integrity and Source Attribution

### 3.1 Types of Citation Hallucinations

| Type | Description | Detection Difficulty |
|------|-------------|---------------------|
| **Fabricated author** | Invent non-existent researchers | Medium (API lookup) |
| **Fabricated paper** | Invent non-existent publications | Medium (API lookup) |
| **Wrong attribution** | Attribute real work to wrong author | Hard (requires expertise) |
| **Correct author, wrong year** | Slight date errors | Easy (date validation) |
| **Correct paper, wrong findings** | Misrepresent paper's conclusions | Very Hard (requires reading) |
| **Fabricated DOIs/arXiv IDs** | Generate non-existent identifiers | Easy (format validation + lookup) |
| **Outdated citations** | Reference superseded findings | Hard (requires domain knowledge) |

### 3.2 Citation Verification Pipeline

```python
class CitationVerificationPipeline:
    def __init__(self):
        self.semantic_scholar = SemanticScholarClient()
        self.crossref = CrossRefClient()
        self.arxiv = ArxivClient()
    
    async def verify_all_citations(self, text):
        """Extract and verify all citations in text."""
        citations = self._extract_citations(text)
        results = []
        
        for citation in citations:
            if citation["type"] == "doi":
                result = await self._verify_doi(citation)
            elif citation["type"] == "arxiv":
                result = await self._verify_arxiv(citation)
            elif citation["type"] == "author_year":
                result = await self._verify_author_year(citation)
            elif citation["type"] == "bracket":
                result = await self._verify_bracket(citation)
            else:
                result = {"status": "unknown_type", "verified": False}
            
            results.append({**citation, **result})
        
        fabricated = [r for r in results if not r.get("verified", False)]
        
        return {
            "total_citations": len(results),
            "verified": len(results) - len(fabricated),
            "fabricated": len(fabricated),
            "fabrication_rate": len(fabricated) / max(len(results), 1),
            "details": results
        }
    
    async def _verify_doi(self, citation):
        """Verify a DOI citation."""
        doi = citation["identifier"]
        try:
            result = await self.crossref.get_work(doi)
            return {
                "verified": True,
                "title": result.get("title", ""),
                "authors": result.get("authors", []),
                "year": result.get("year"),
                "source": "crossref"
            }
        except Exception:
            return {"verified": False, "reason": "DOI not found"}
    
    async def _verify_arxiv(self, citation):
        """Verify an arXiv ID."""
        arxiv_id = citation["identifier"]
        try:
            result = await self.arxiv.get_paper(arxiv_id)
            return {
                "verified": True,
                "title": result.title,
                "authors": result.authors,
                "date": result.published,
                "source": "arxiv"
            }
        except Exception:
            return {"verified": False, "reason": "arXiv ID not found"}
    
    async def _verify_author_year(self, citation):
        """Verify author-year citation via Semantic Scholar."""
        try:
            results = await self.semantic_scholar.search_paper(
                query=f"{citation['author']} {citation.get('title', '')}",
                limit=5
            )
            
            for paper in results:
                if (citation["author"].lower() in 
                    [a["name"].lower() for a in paper.get("authors", [])]):
                    if str(citation.get("year", "")) in str(paper.get("year", "")):
                        return {
                            "verified": True,
                            "title": paper.get("title"),
                            "matched_paper": paper.get("paperId"),
                            "source": "semantic_scholar"
                        }
            
            return {"verified": False, "reason": "No matching paper found"}
        except Exception:
            return {"verified": False, "reason": "API error"}
    
    def _extract_citations(self, text):
        """Extract all citations from text."""
        citations = []
        
        # DOI pattern
        doi_pattern = r'(?:doi:|https?://doi\.org/)(10\.\d{4,}/[^\s]+)'
        for match in re.finditer(doi_pattern, text):
            citations.append({
                "type": "doi",
                "identifier": match.group(1),
                "text": match.group(0),
                "position": match.start()
            })
        
        # arXiv pattern
        arxiv_pattern = r'(?:arXiv:)?(\d{4}\.\d{4,5})'
        for match in re.finditer(arxiv_pattern, text):
            citations.append({
                "type": "arxiv",
                "identifier": match.group(1),
                "text": match.group(0),
                "position": match.start()
            })
        
        # Author-year pattern
        author_pattern = r'([A-Z][a-z]+ (?:et al\.?)?,? (\d{4}))'
        for match in re.finditer(author_pattern, text):
            citations.append({
                "type": "author_year",
                "author": match.group(1).split(",")[0].split("et")[0].strip(),
                "year": match.group(2),
                "text": match.group(0),
                "position": match.start()
            })
        
        return citations
```

### 3.3 Real-Time Citation Checking

```python
async def real_time_citation_check(text, knowledge_base):
    """Check citations in real-time during generation."""
    citations_in_text = extract_citations(text)
    
    checks = []
    for citation in citations_in_text:
        # Check against knowledge base
        in_kb = knowledge_base.contains(citation)
        
        # Check against external APIs
        external = await verify_externally(citation)
        
        checks.append({
            "citation": citation,
            "in_knowledge_base": in_kb,
            "externally_verified": external["verified"],
            "confidence": external.get("confidence", 0),
            "action": "keep" if (in_kb or external["verified"]) else "flag"
        })
    
    return checks
```

---

## 4. Confidence Calibration and Uncertainty Quantification

### 4.1 Why Confidence Calibration Matters

LLMs are notoriously poorly calibrated — their expressed confidence often doesn't match actual accuracy. A model might say "I'm very confident" about a hallucinated fact and "I'm somewhat sure" about a correct one.

**Calibration metrics:**

| Metric | Definition | Ideal Value |
|--------|-----------|-------------|
| **Expected Calibration Error (ECE)** | Average gap between confidence and accuracy | 0.0 |
| **Maximum Calibration Error (MCE)** | Worst-case gap between confidence and accuracy | 0.0 |
| **Brier Score** | Mean squared difference between confidence and outcome | 0.0 |
| **Reliability Diagram Slope** | Slope of calibration curve | 1.0 |

### 4.2 Temperature-Based Confidence Estimation

```python
async def temperature_confidence_estimation(llm_client, query, temperatures=[0.1, 0.3, 0.5, 0.7, 0.9], n_samples=5):
    """Estimate confidence by measuring response stability across temperatures."""
    
    responses_by_temp = {}
    
    for temp in temperatures:
        responses = []
        for _ in range(n_samples):
            response = await llm_client.generate(query, temperature=temp)
            responses.append(response)
        responses_by_temp[temp] = responses
    
    # Compute consistency at each temperature
    consistency_scores = {}
    for temp, responses in responses_by_temp.items():
        embeddings = [get_embedding(r) for r in responses]
        pairwise_similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                pairwise_similarities.append(sim)
        consistency_scores[temp] = np.mean(pairwise_similarities)
    
    # Confidence = stability across temperatures
    # High confidence: similar responses regardless of temperature
    # Low confidence: different responses at different temperatures
    
    temp_stability = np.std(list(consistency_scores.values()))
    avg_consistency = np.mean(list(consistency_scores.values()))
    
    confidence = avg_consistency * (1 - temp_stability)
    
    return {
        "confidence": confidence,
        "consistency_by_temperature": consistency_scores,
        "temp_stability": temp_stability,
        "risk_level": "low" if confidence > 0.8 else "medium" if confidence > 0.5 else "high"
    }
```

### 4.3 Verbalized Confidence

```python
async def verbalized_confidence(llm_client, query):
    """Get the model to express its own confidence."""
    
    confidence_prompt = f"""Question: {query}

Before answering, assess your confidence level:
1. How certain are you about this? (very certain / somewhat certain / uncertain / very uncertain)
2. What is your confidence percentage? (0-100%)
3. What factors affect your confidence? (e.g., well-known fact, ambiguous topic, 
   potentially outdated information)
4. Would you be comfortable if a human expert reviewed your answer for accuracy?

Now provide your answer and your confidence assessment:"""
    
    response = await llm_client.generate(confidence_prompt, temperature=0)
    
    # Parse confidence from response
    confidence = parse_confidence(response)
    
    return {
        "response": extract_answer(response),
        "self_reported_confidence": confidence["percentage"],
        "confidence_level": confidence["level"],
        "factors": confidence["factors"],
        "review_needed": confidence["percentage"] < 70
    }
```

### 4.4 Bayesian Uncertainty Estimation

```python
class BayesianUncertainty:
    def __init__(self, llm_client, n_samples=20):
        self.llm = llm_client
        self.n_samples = n_samples
    
    async def estimate_uncertainty(self, query, context):
        """Estimate uncertainty using Bayesian approach with multiple samples."""
        
        # Generate multiple responses
        responses = []
        for _ in range(self.n_samples):
            response = await self.llm.generate(
                f"Context: {context}\n\nQuestion: {query}",
                temperature=0.7
            )
            responses.append(response)
        
        # Extract key claims from each response
        claims_per_response = [extract_claims(r) for r in responses]
        
        # Compute claim agreement (epistemic uncertainty)
        all_claims = []
        for claims in claims_per_response:
            for claim in claims:
                all_claims.append(normalize_claim(claim))
        
        claim_counts = Counter(all_claims)
        total_claims = len(all_claims)
        
        # Shannon entropy as uncertainty measure
        probabilities = [count / total_claims for count in claim_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize entropy (0 = certain, 1 = completely uncertain)
        max_entropy = np.log2(len(claim_counts)) if len(claim_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Compute response diversity
        unique_responses = len(set(responses))
        diversity = unique_responses / self.n_samples
        
        # Combine metrics
        uncertainty = (normalized_entropy * 0.6 + diversity * 0.4)
        
        return {
            "uncertainty": uncertainty,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "response_diversity": diversity,
            "unique_responses": unique_responses,
            "dominant_claim": claim_counts.most_common(1)[0] if claim_counts else None,
            "risk_level": "low" if uncertainty < 0.3 else "medium" if uncertainty < 0.6 else "high"
        }
```

---

## 5. Domain-Specific Hallucination Patterns

### 5.1 Medical Domain

**Common hallucination patterns:**
1. **Drug interaction fabrications**: Inventing non-existent drug interactions
2. **Dosage hallucinations**: Generating incorrect medication dosages
3. **Study citation fabrications**: Creating fake medical studies
4. **Symptom-disease mismatches**: Incorrectly associating symptoms with diseases
5. **Treatment protocol errors**: Suggesting outdated or incorrect treatment protocols

**Detection approaches:**
```python
class MedicalHallucinationDetector:
    def __init__(self):
        self.medical_kb = MedicalKnowledgeBase()  # PubMed, DrugBank, etc.
        self.drug_checker = DrugInteractionChecker()
    
    async def verify_medical_response(self, query, response, context):
        """Verify medical content against authoritative sources."""
        
        claims = extract_medical_claims(response)
        results = []
        
        for claim in claims:
            if claim["type"] == "drug_dosage":
                verified = await self._verify_dosage(claim)
            elif claim["type"] == "drug_interaction":
                verified = await self._verify_interaction(claim)
            elif claim["type"] == "study_reference":
                verified = await self._verify_study(claim)
            elif claim["type"] == "treatment_protocol":
                verified = await self._verify_protocol(claim)
            else:
                verified = {"status": "manual_review_needed"}
            
            results.append({**claim, "verification": verified})
        
        critical_errors = [r for r in results 
                          if r["verification"].get("severity") == "critical"]
        
        return {
            "results": results,
            "critical_errors": len(critical_errors),
            "requires_physician_review": len(critical_errors) > 0,
            "safe_to_display": len(critical_errors) == 0
        }
    
    async def _verify_dosage(self, claim):
        """Verify drug dosage against medical databases."""
        drug = claim["drug"]
        dosage = claim["dosage"]
        
        # Check against DrugBank or similar
        correct_dosage = await self.medical_kb.get_drug_dosage(drug)
        
        if correct_dosage and dosage in correct_dosage:
            return {"status": "verified", "severity": "none"}
        elif correct_dosage:
            return {
                "status": "incorrect",
                "correct_value": correct_dosage,
                "severity": "critical",
                "message": f"Incorrect dosage for {drug}. Correct: {correct_dosage}"
            }
        else:
            return {"status": "unverifiable", "severity": "high"}
```

### 5.2 Legal Domain

**Common hallucination patterns:**
1. **Fabricated case law**: Inventing non-existent court cases
2. **Incorrect statute citations**: Wrong legal code references
3. **Jurisdictional errors**: Applying wrong jurisdiction's laws
4. **Outdated regulations**: Citing superseded laws
5. **Misinterpreted precedents**: Incorrectly stating what a case held

**Detection approaches:**
```python
class LegalHallucinationDetector:
    def __init__(self):
        self.legal_db = LegalDatabase()  # Westlaw, LexisNexis, etc.
    
    async def verify_legal_response(self, response, jurisdiction):
        """Verify legal content against authoritative legal databases."""
        
        legal_claims = extract_legal_claims(response)
        results = []
        
        for claim in legal_claims:
            if claim["type"] == "case_citation":
                verified = await self._verify_case(claim)
            elif claim["type"] == "statute":
                verified = await self._verify_statute(claim, jurisdiction)
            elif claim["type"] == "legal_principle":
                verified = await self._verify_principle(claim, jurisdiction)
            else:
                verified = {"status": "manual_review_needed"}
            
            results.append({**claim, "verification": verified})
        
        return {
            "results": results,
            "fabricated_citations": [r for r in results 
                                   if r["verification"].get("status") == "fabricated"],
            "requires_attorney_review": any(
                r["verification"].get("severity") == "critical" 
                for r in results
            )
        }
```

### 5.3 Financial Domain

**Common hallucination patterns:**
1. **Fabricated financial figures**: Inventing revenue numbers, stock prices
2. **Incorrect calculations**: Mathematical errors in financial analysis
3. **Outdated market data**: Using old prices or ratios
4. **Misattributed analyst opinions**: Wrong attribution of financial recommendations
5. **Fabricated regulatory references**: Inventing SEC rules or financial regulations

### 5.4 Technical/Scientific Domain

**Common hallucination patterns:**
1. **Fabricated API details**: Inventing non-existent function parameters
2. **Incorrect version numbers**: Wrong software versions
3. **Fabricated research papers**: Inventing scientific publications
4. **Incorrect mathematical formulas**: Errors in equations
5. **Outdated technical specifications**: Using old API versions

---

## 6. Hallucination in Multi-Modal Systems

### 6.1 Vision-Language Model Hallucinations

Multi-modal models (GPT-4V, Claude Vision, Gemini) have unique hallucination patterns:

| Type | Description | Example |
|------|-------------|---------|
| **Object hallucination** | Detecting non-existent objects | "I see a cat in the image" (no cat present) |
| **Attribute hallucination** | Incorrect object attributes | "The red car" (car is blue) |
| **Spatial hallucination** | Incorrect spatial relationships | "The cup is on the left" (it's on the right) |
| **Text hallucination** | Misreading text in images | OCR-like errors |
| **Counting hallucination** | Incorrect object counts | "There are three people" (there are five) |
| **Action hallucination** | Incorrect action descriptions | "The person is running" (they're standing) |

### 6.2 Image-Text Consistency Checking

```python
class VisionHallucinationDetector:
    def __init__(self, vision_model, language_model):
        self.vision = vision_model
        self.language = language_model
    
    async def check_image_text_consistency(self, image, text_description):
        """Verify that text description matches the image."""
        
        # Get model's own image analysis
        model_analysis = await self.vision.analyze(image, 
            prompt="Describe this image in detail, listing all visible objects, "
                   "their colors, positions, and any text visible."
        )
        
        # Extract claims from the text description
        claims = extract_visual_claims(text_description)
        
        # Check each claim against the model's own analysis
        results = []
        for claim in claims:
            consistency_check = await self.language.generate(
                f"""Image analysis: {model_analysis}

Claim to verify: {claim['text']}

Is this claim SUPPORTED, REFUTED, or CANNOT_DETERMINE based on the image analysis?
Provide brief reasoning.""",
                temperature=0
            )
            
            results.append({
                "claim": claim,
                "consistent": "SUPPORTED" in consistency_check,
                "explanation": consistency_check
            })
        
        inconsistent = [r for r in results if not r["consistent"]]
        
        return {
            "total_claims": len(claims),
            "consistent_claims": len(claims) - len(inconsistent),
            "inconsistent_claims": len(inconsistent),
            "hallucination_rate": len(inconsistent) / max(len(claims), 1),
            "details": results
        }
```

### 6.3 Audio-Visual Hallucination Detection

```python
async def check_audio_visual_consistency(audio_transcript, video_description):
    """Verify consistency between audio and visual content."""
    
    # Extract claims from audio
    audio_claims = extract_claims(audio_transcript)
    
    # Extract claims from video
    video_claims = extract_visual_claims(video_description)
    
    # Cross-check
    inconsistencies = []
    for audio_claim in audio_claims:
        for video_claim in video_claims:
            consistency = await check_claim_consistency(audio_claim, video_claim)
            if consistency == "contradictory":
                inconsistencies.append({
                    "audio_claim": audio_claim,
                    "video_claim": video_claim,
                    "type": "audio_visual_contradiction"
                })
    
    return {
        "inconsistencies": inconsistencies,
        "consistency_score": 1 - (len(inconsistencies) / max(len(audio_claims), 1))
    }
```

---

## 7. Hallucination in Agentic Workflows

### 7.1 Agent-Specific Hallucination Patterns

| Pattern | Description | Impact |
|---------|-------------|--------|
| **Tool result hallucination** | Fabricating tool outputs | Incorrect decisions based on fake data |
| **Observation hallucination** | Misinterpreting environment observations | Wrong state assessment |
| **Plan hallucination** | Creating non-executable plans | Workflow failures |
| **Memory hallucination** | Recalling incorrect past interactions | Inconsistent behavior |
| **Goal hallucination** | Pursuing wrong objectives | Task failure |

### 7.2 Agent Hallucination Monitoring

```python
class AgentHallucinationMonitor:
    def __init__(self, agent):
        self.agent = agent
        self.trace_log = []
    
    async def monitor_step(self, step_type, input_data, output_data):
        """Monitor a single agent step for hallucinations."""
        
        hallucination_checks = []
        
        if step_type == "tool_call":
            # Check if tool call is based on correct understanding
            check = await self._verify_tool_call(input_data, output_data)
            hallucination_checks.append(check)
        
        elif step_type == "observation":
            # Check if observation is correctly interpreted
            check = await self._verify_observation(input_data, output_data)
            hallucination_checks.append(check)
        
        elif step_type == "plan":
            # Check if plan is executable and logical
            check = await self._verify_plan(input_data, output_data)
            hallucination_checks.append(check)
        
        elif step_type == "memory":
            # Check if recalled information is accurate
            check = await self._verify_memory(input_data, output_data)
            hallucination_checks.append(check)
        
        # Log the step
        self.trace_log.append({
            "step_type": step_type,
            "input": input_data,
            "output": output_data,
            "hallucination_checks": hallucination_checks,
            "timestamp": datetime.utcnow()
        })
        
        # Return if any hallucination detected
        return any(c.get("hallucinated", False) for c in hallucination_checks)
    
    async def _verify_tool_call(self, context, tool_call):
        """Verify a tool call is appropriate and based on correct understanding."""
        verification_prompt = f"""Agent context: {context}
Tool call: {tool_call}

Is this tool call appropriate given the context? 
Is the agent correctly understanding what it needs to do?
Are there any fabricated or incorrect assumptions?

Respond with: CORRECT or HALLUCINATED (with explanation)"""
        
        result = await self.agent.llm.generate(verification_prompt, temperature=0)
        
        return {
            "hallucinated": "HALLUCINATED" in result,
            "explanation": result
        }
    
    def get_hallucination_summary(self):
        """Get summary of hallucinations detected during agent execution."""
        total_steps = len(self.trace_log)
        hallucinated_steps = sum(
            1 for step in self.trace_log 
            if any(c.get("hallucinated", False) for c in step["hallucination_checks"])
        )
        
        return {
            "total_steps": total_steps,
            "hallucinated_steps": hallucinated_steps,
            "hallucination_rate": hallucinated_steps / max(total_steps, 1),
            "hallucination_by_type": self._count_by_type(),
            "most_common_hallucination": self._most_common()
        }
```

---

## 8. Detection Metrics Deep Dive

### 8.1 Hallucination Rate (HR)

```python
def compute_hallucination_rate(responses, ground_truth):
    """Compute the hallucination rate across a set of responses."""
    
    hallucinated_count = 0
    total_count = len(responses)
    
    for response, truth in zip(responses, ground_truth):
        claims = extract_claims(response)
        truth_claims = extract_claims(truth)
        
        for claim in claims:
            if not any(claim_supported(claim, tc) for tc in truth_claims):
                hallucinated_count += 1
                break  # Count per response, not per claim
    
    return hallucinated_count / max(total_count, 1)
```

### 8.2 Factual Consistency Score (FCS)

```python
def compute_factual_consistency(response, context):
    """Compute how much of the response is supported by the context."""
    
    response_claims = extract_claims(response)
    supported_claims = 0
    
    for claim in response_claims:
        # Check if claim is supported by any part of context
        support_score = compute_claim_support(claim, context)
        if support_score > 0.7:  # Threshold for "supported"
            supported_claims += 1
    
    return supported_claims / max(len(response_claims), 1)
```

### 8.3 Hallucination Severity Score (HSS)

```python
def compute_severity_score(hallucinations):
    """Compute weighted severity score for hallucinations."""
    
    severity_weights = {
        "critical": 1.0,
        "major": 0.7,
        "minor": 0.3,
        "cosmetic": 0.1
    }
    
    if not hallucinations:
        return 0.0
    
    total_weight = sum(
        severity_weights.get(h.get("severity", "minor"), 0.3)
        for h in hallucinations
    )
    
    # Normalize by number of hallucinations
    return total_weight / len(hallucinations)
```

---

## 9. Mitigation Strategy Selection

### 9.1 Decision Framework

```
What type of content are you generating?
├── Factual/Informational
│   ├── High-stakes (medical, legal, financial)?
│   │   ├── Yes → Mandatory RAG + Verification + Human Review
│   │   └── No → RAG + Post-generation verification
│   └── With citations/references?
│       ├── Yes → Citation verification pipeline
│       └── No → Self-consistency + NLI verification
├── Creative/Generative
│   ├── Factual elements present?
│   │   ├── Yes → Verify factual elements only
│   │   └── No → Lower hallucination risk (creative = allowed)
│   └── Brand/accuracy sensitive?
│       ├── Yes → Grounded generation + post-verification
│       └── No → Standard generation with monitoring
├── Conversational/Chat
│   ├── Customer-facing?
│   │   ├── Yes → System prompt grounding + refusal training
│   │   └── No → Standard generation
│   └── Decision-support?
│       ├── Yes → Confidence calibration + source attribution
│       └── No → Standard generation
└── Code/Technical
    ├── Production code?
    │   ├── Yes → Test generation + verification
    │   └── No → Standard generation with warnings
    └── API/Version specific?
        ├── Yes → Documentation verification
        └── No → Standard generation
```

### 9.2 Cost-Benefit Analysis

| Strategy | Implementation Cost | Hallucination Reduction | Latency Impact | Best For |
|----------|-------------------|------------------------|----------------|----------|
| System prompt grounding | Low | 20-30% | None | All applications |
| RAG hardening | Medium | 40-60% | +100-500ms | Knowledge-intensive |
| Self-consistency | High | 30-50% | +200-1000ms | High-stakes |
| NLI verification | Medium | 35-55% | +50-200ms | Moderate-stakes |
| Citation verification | Medium | 60-80% (citations) | +100-300ms | Research/academic |
| Confidence calibration | High | 25-45% | +200-500ms | Decision support |
| Domain fine-tuning | Very High | 50-70% | None | Domain-specific |
| Multi-agent verification | Very High | 60-80% | +500-2000ms | Critical applications |

### 9.3 Layered Defense Strategy

The most effective approach combines multiple strategies:

```
Layer 1: Prevention (Prompt Engineering) → 20-30% reduction
Layer 2: Retrieval (Hardened RAG) → additional 20-30% reduction  
Layer 3: Generation (Constrained Decoding) → additional 10-15% reduction
Layer 4: Verification (Post-generation checks) → additional 15-25% reduction
Layer 5: Monitoring (Production tracking) → Catches remaining + feedback loop

Combined: 70-85% total hallucination reduction
```

---

## Cross-References

| Document | Relevance |
|----------|-----------|
| [01-Overview.md](01-Overview.md) | General overview and taxonomy |
| [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) | Advanced detection algorithms |
| [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) | Specific tool implementations |
| [05-Future-Outlook.md](05-Future-Outlook.md) | Future directions in hallucination prevention |
| [04-RAG/02-Advanced-RAG.md](../../04-RAG/02-Advanced-RAG.md) | Advanced RAG for better grounding |
| [06-Advanced/05-Interpretability.md](../../06-Advanced/05-Interpretability.md) | Model interpretability for understanding hallucinations |
| [20-Agent-Infrastructure/04-Agent-Evaluation-and-Testing.md](../../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md) | Agent evaluation includes hallucination testing |

---

*Last updated: July 2026*
