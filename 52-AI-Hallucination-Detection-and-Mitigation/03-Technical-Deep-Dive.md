# 52 — AI Hallucination Detection and Mitigation: Technical Deep Dive

> **Category:** 52 — AI Hallucination Detection and Mitigation  
> **Document:** 03 — Technical Deep Dive  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](01-Overview.md), [02-Core-Topics.md](02-Core-Topics.md), [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md)

---

## Table of Contents

1. [Advanced Detection Algorithms](#1-advanced-detection-algorithms)
2. [Neural Hallucination Detection Models](#2-neural-hallucination-detection-models)
3. [Probabilistic Approaches](#3-probabilistic-approaches)
4. [Knowledge Graph Verification](#4-knowledge-graph-verification)
5. [Automated Red-Teaming for Hallucination](#5-automated-red-teaming-for-hallucination)
6. [Real-Time Production Detection](#6-real-time-production-detection)
7. [Hallucination in Structured Output](#7-hallucination-in-structured-output)
8. [Cross-Lingual Hallucination](#8-cross-lingual-hallucination)
9. [Adversarial Hallucination Attacks](#9-adversarial-hallucination-attacks)

---

## 1. Advanced Detection Algorithms

### 1.1 BERTScore-Based Hallucination Detection

```python
from bert_score import score as bert_score
import numpy as np

class BERTScoreDetector:
    def __init__(self, model_type="microsoft/deberta-xlarge-mnli"):
        self.model_type = model_type
    
    def detect_hallucination(self, source_text, generated_text):
        """Detect hallucination using BERTScore alignment."""
        
        # Split into sentences
        source_sentences = self._split_sentences(source_text)
        generated_sentences = self._split_sentences(generated_text)
        
        # Compute BERTScore for each generated sentence against all source sentences
        results = []
        for gen_sent in generated_sentences:
            # Find best matching source sentence
            best_score = 0
            best_match = None
            
            for src_sent in source_sentences:
                P, R, F1 = bert_score(
                    [gen_sent], [src_sent], 
                    model_type=self.model_type,
                    verbose=False
                )
                if F1.item() > best_score:
                    best_score = F1.item()
                    best_match = src_sent
            
            # Classify based on score thresholds
            if best_score > 0.85:
                classification = "supported"
            elif best_score > 0.65:
                classification = "paraphrase"
            elif best_score > 0.45:
                classification = "related"
            else:
                classification = "hallucinated"
            
            results.append({
                "sentence": gen_sent,
                "best_match": best_match,
                "score": best_score,
                "classification": classification
            })
        
        hallucinated = [r for r in results if r["classification"] == "hallucinated"]
        
        return {
            "total_sentences": len(generated_sentences),
            "supported": len([r for r in results if r["classification"] == "supported"]),
            "paraphrased": len([r for r in results if r["classification"] == "paraphrase"]),
            "related": len([r for r in results if r["classification"] == "related"]),
            "hallucinated": len(hallucinated),
            "hallucination_rate": len(hallucinated) / max(len(generated_sentences), 1),
            "details": results
        }
    
    def _split_sentences(self, text):
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
```

### 1.2 Entailment Tree Verification

```python
class EntailmentTreeVerifier:
    """Verify claims using a tree of entailment relationships."""
    
    def __init__(self, nli_model, llm_client):
        self.nli = nli_model
        self.llm = llm_client
    
    async def verify_with_entailment_tree(self, claim, evidence_chunks):
        """Build an entailment tree to verify a claim."""
        
        # Step 1: Decompose claim into atomic facts
        atomic_facts = await self._decompose_claim(claim)
        
        # Step 2: For each atomic fact, find supporting evidence
        verification_results = []
        for fact in atomic_facts:
            evidence = await self._find_evidence(fact, evidence_chunks)
            
            # Step 3: Check entailment
            entailment_result = self._check_entailment(fact, evidence)
            
            # Step 4: If not directly entailed, try multi-hop reasoning
            if entailment_result["label"] != "ENTAILMENT":
                multi_hop = await self._multi_hop_verification(fact, evidence_chunks)
                entailment_result = multi_hop
            
            verification_results.append({
                "atomic_fact": fact,
                "evidence": evidence,
                "entailment": entailment_result
            })
        
        # Step 5: Aggregate results
        supported = sum(1 for r in verification_results 
                       if r["entailment"]["label"] == "ENTAILMENT")
        refuted = sum(1 for r in verification_results 
                     if r["entailment"]["label"] == "CONTRADICTION")
        neutral = len(verification_results) - supported - refuted
        
        overall = "SUPPORTED" if supported == len(verification_results) else \
                  "REFUTED" if refuted > 0 else \
                  "PARTIALLY_SUPPORTED" if supported > 0 else "UNVERIFIED"
        
        return {
            "overall_verdict": overall,
            "atomic_facts": len(atomic_facts),
            "supported": supported,
            "refuted": refuted,
            "neutral": neutral,
            "details": verification_results
        }
    
    async def _decompose_claim(self, claim):
        """Decompose a claim into atomic facts."""
        prompt = f"""Decompose this claim into its atomic factual components:

Claim: {claim}

List each atomic fact as a separate statement:"""
        
        response = await self.llm.generate(prompt, temperature=0)
        facts = [f.strip() for f in response.split('\n') if f.strip() and not f.strip().startswith(('1)', '2)', '3)', '4)', '5)', '•', '-'))]
        return facts
    
    async def _find_evidence(self, fact, evidence_chunks):
        """Find the most relevant evidence for a fact."""
        # Use embedding similarity to find relevant chunks
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        fact_embedding = model.encode([fact])
        chunk_embeddings = model.encode(evidence_chunks)
        
        similarities = np.dot(chunk_embeddings, fact_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        return [evidence_chunks[i] for i in top_indices]
    
    def _check_entailment(self, hypothesis, premises):
        """Check entailment between hypothesis and premises."""
        combined_premise = " ".join(premises)
        result = self.nli(f"{combined_premise} [SEP] {hypothesis}")
        return {
            "label": result["label"],
            "score": result["score"]
        }
    
    async def _multi_hop_verification(self, fact, evidence_chunks):
        """Attempt multi-hop reasoning to verify a fact."""
        # Find intermediate facts that could bridge the gap
        prompt = f"""Given this fact to verify: {fact}

Available evidence chunks:
{chr(10).join(evidence_chunks[:5])}

Is there a chain of reasoning that connects the evidence to the fact?
If yes, describe the chain. If no, explain why."""

        response = await self.llm.generate(prompt, temperature=0)
        
        # Determine if multi-hop reasoning succeeded
        if "chain" in response.lower() and "connects" in response.lower():
            return {"label": "ENTAILMENT", "score": 0.7, "multi_hop": True}
        else:
            return {"label": "NEUTRAL", "score": 0.5, "multi_hop": False}
```

### 1.3 Semantic Consistency with Contrastive Decoding

```python
class ContrastiveDecodingDetector:
    """Detect hallucination by comparing normal and contrastive decoding."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def detect_with_contrastive_decoding(self, prompt, max_tokens=200):
        """
        Compare standard decoding with contrastive decoding.
        Large divergence indicates potential hallucination.
        """
        
        # Standard decoding
        standard_output = self._generate(prompt, method="standard", max_tokens=max_tokens)
        
        # Contrastive decoding (penalize tokens that would be likely 
        # without the context)
        contrastive_output = self._generate(prompt, method="contrastive", max_tokens=max_tokens)
        
        # Compute divergence
        standard_tokens = self.tokenizer.encode(standard_output)
        contrastive_tokens = self.tokenizer.encode(contrastive_output)
        
        # Token-level divergence
        divergence = self._compute_token_divergence(standard_tokens, contrastive_tokens)
        
        # Semantic divergence
        semantic_div = self._compute_semantic_divergence(standard_output, contrastive_output)
        
        # High divergence suggests the model is "straying" from context
        hallucination_risk = (divergence * 0.4 + semantic_div * 0.6)
        
        return {
            "standard_output": standard_output,
            "contrastive_output": contrastive_output,
            "token_divergence": divergence,
            "semantic_divergence": semantic_div,
            "hallucination_risk": hallucination_risk,
            "risk_level": "high" if hallucination_risk > 0.7 else 
                         "medium" if hallucination_risk > 0.4 else "low"
        }
    
    def _generate(self, prompt, method="standard", max_tokens=200):
        """Generate text with specified method."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if method == "standard":
            outputs = self.model.generate(
                inputs, max_new_tokens=max_tokens, 
                do_sample=True, temperature=0.7
            )
        elif method == "contrastive":
            # Contrastive decoding: penalize tokens that are likely 
            # without the specific context
            outputs = self.model.generate(
                inputs, max_new_tokens=max_tokens,
                penalty_alpha=0.6, do_sample=True,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    def _compute_token_divergence(self, tokens1, tokens2):
        """Compute token-level divergence between two sequences."""
        from collections import Counter
        
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        all_tokens = set(freq1.keys()) | set(freq2.keys())
        
        divergence = 0
        for token in all_tokens:
            p = freq1.get(token, 0) / max(len(tokens1), 1)
            q = freq2.get(token, 0) / max(len(tokens2), 1)
            
            if p > 0 and q > 0:
                divergence += abs(p - q) / max(p, q)
        
        return divergence / max(len(all_tokens), 1)
    
    def _compute_semantic_divergence(self, text1, text2):
        """Compute semantic divergence between two texts."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        
        # Cosine distance
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - similarity
```

---

## 2. Neural Hallucination Detection Models

### 2.1 Fine-Tuned Hallucination Detection Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NeuralHallucinationDetector:
    """Use fine-tuned models specifically designed for hallucination detection."""
    
    def __init__(self, model_name="vectara/hallucination_evaluation_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def detect(self, context, response):
        """Detect hallucination using neural model."""
        
        # Format input as the model expects
        input_text = f"Context: {context}\n\nResponse: {response}"
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        # Model outputs: [not_hallucinated, hallucinated]
        hallucination_prob = probs[0][1].item()
        
        return {
            "hallucination_probability": hallucination_prob,
            "is_hallucinated": hallucination_prob > 0.5,
            "confidence": max(probs[0].tolist()),
            "risk_level": "high" if hallucination_prob > 0.7 else
                         "medium" if hallucination_prob > 0.4 else "low"
        }
```

### 2.2 Custom Hallucination Detection Training

```python
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer
)

class HallucinationDataset(Dataset):
    """Dataset for training hallucination detection models."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: premise (context) + hypothesis (response)
        text = f"Context: {item['context']}\n\nResponse: {item['response']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long)  # 0=supported, 1=hallucinated
        }

def train_hallucination_detector(train_data, val_data, model_name="microsoft/deberta-v3-base"):
    """Train a custom hallucination detection model."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_dataset = HallucinationDataset(train_data, tokenizer)
    val_dataset = HallucinationDataset(val_data, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./hallucination_detector",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            "f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    return model, tokenizer
```

### 2.3 Ensemble Detection

```python
class EnsembleHallucinationDetector:
    """Combine multiple detection methods for more robust detection."""
    
    def __init__(self):
        self.detectors = {
            "nli": NLIDetector(),
            "bertscore": BERTScoreDetector(),
            "neural": NeuralHallucinationDetector(),
            "consistency": SelfConsistencyDetector(),
            "citation": CitationVerifier()
        }
        self.weights = {
            "nli": 0.25,
            "bertscore": 0.20,
            "neural": 0.30,
            "consistency": 0.15,
            "citation": 0.10
        }
    
    def detect(self, context, response, query=None):
        """Run all detectors and combine results."""
        
        results = {}
        scores = {}
        
        # Run each detector
        if "nli" in self.detectors:
            results["nli"] = self.detectors["nli"].detect_hallucination(context, response)
            scores["nli"] = results["nli"]["hallucination_rate"]
        
        if "bertscore" in self.detectors:
            results["bertscore"] = self.detectors["bertscore"].detect_hallucination(context, response)
            scores["bertscore"] = results["bertscore"]["hallucination_rate"]
        
        if "neural" in self.detectors:
            results["neural"] = self.detectors["neural"].detect(context, response)
            scores["neural"] = results["neural"]["hallucination_probability"]
        
        if "consistency" in self.detectors and query:
            results["consistency"] = self.detectors["consistency"].check(query, response)
            scores["consistency"] = 1 - results["consistency"]["consistency_score"]
        
        if "citation" in self.detectors:
            citation_result = self.detectors["citation"].verify_citations(response, [context])
            results["citation"] = citation_result
            scores["citation"] = citation_result["fabrication_rate"]
        
        # Weighted combination
        weighted_score = sum(
            scores.get(detector, 0) * weight 
            for detector, weight in self.weights.items()
            if detector in scores
        )
        
        # Normalize weights
        total_weight = sum(
            weight for detector, weight in self.weights.items()
            if detector in scores
        )
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Majority voting
        votes = sum(1 for s in scores.values() if s > 0.5)
        
        return {
            "ensemble_score": weighted_score,
            "majority_vote": "hallucinated" if votes > len(scores) / 2 else "supported",
            "individual_results": results,
            "individual_scores": scores,
            "agreement": len(set(1 if s > 0.5 else 0 for s in scores.values())) == 1,
            "final_verdict": "HALLUCINATED" if weighted_score > 0.5 else "SUPPORTED"
        }
```

---

## 3. Probabilistic Approaches

### 3.1 Bayesian Hallucination Estimation

```python
import numpy as np
from scipy import stats

class BayesianHallucinationEstimator:
    """Estimate hallucination probability using Bayesian inference."""
    
    def __init__(self, prior_hallucination_rate=0.1):
        self.prior = prior_hallucination_rate
    
    def estimate(self, evidence_scores, likelihood_ratios):
        """
        Estimate posterior hallucination probability using Bayesian updating.
        
        evidence_scores: list of binary evidence (1=supports, 0=doesn't support)
        likelihood_ratios: list of (true_positive_rate, false_positive_rate) for each evidence type
        """
        
        # Prior odds
        prior_odds = self.prior / (1 - self.prior)
        
        # Update with each piece of evidence
        posterior_odds = prior_odds
        
        for score, (tpr, fpr) in zip(evidence_scores, likelihood_ratios):
            if score == 1:
                # Evidence supports hallucination
                lr = (1 - fpr) / tpr  # Likelihood ratio for "not hallucinated"
            else:
                # Evidence doesn't support
                lr = fpr / (1 - tpr)
            
            posterior_odds *= lr
        
        # Convert back to probability
        posterior_prob = posterior_odds / (1 + posterior_odds)
        
        # Credible interval
        n_evidence = len(evidence_scores)
        if n_evidence > 0:
            # Approximate with beta distribution
            alpha = sum(evidence_scores) + 1
            beta = n_evidence - sum(evidence_scores) + 1
            credible_interval = stats.beta.ppf([0.025, 0.975], alpha, beta)
        else:
            credible_interval = [self.prior, self.prior]
        
        return {
            "posterior_probability": posterior_prob,
            "credible_interval": credible_interval,
            "evidence_count": n_evidence,
            "supporting_evidence": sum(evidence_scores),
            "confidence": "high" if abs(posterior_prob - 0.5) > 0.3 else 
                         "medium" if abs(posterior_prob - 0.5) > 0.1 else "low"
        }
```

### 3.2 Monte Carlo Hallucination Estimation

```python
class MonteCarloHallucinationEstimator:
    """Use Monte Carlo sampling to estimate hallucination probability."""
    
    def __init__(self, llm_client, n_samples=100):
        self.llm = llm_client
        self.n_samples = n_samples
    
    async def estimate(self, query, context, verification_fn):
        """
        Estimate hallucination probability using Monte Carlo sampling.
        
        1. Generate multiple responses
        2. For each response, check if it's hallucinated
        3. Estimate probability from frequency
        """
        
        hallucination_count = 0
        responses = []
        
        for _ in range(self.n_samples):
            # Generate response with temperature sampling
            response = await self.llm.generate(
                f"Context: {context}\n\nQuestion: {query}",
                temperature=0.8
            )
            
            # Check if hallucinated
            is_hallucinated = await verification_fn(query, response, context)
            
            if is_hallucinated:
                hallucination_count += 1
            
            responses.append({
                "response": response,
                "is_hallucinated": is_hallucinated
            })
        
        # Estimate probability
        estimated_prob = hallucination_count / self.n_samples
        
        # Confidence interval using Wilson score interval
        z = 1.96  # 95% confidence
        n = self.n_samples
        p_hat = estimated_prob
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
        
        ci_lower = max(0, center - spread)
        ci_upper = min(1, center + spread)
        
        # Find consensus response (most common)
        from collections import Counter
        response_counts = Counter(r["response"] for r in responses)
        consensus_response = response_counts.most_common(1)[0][0]
        consensus_count = response_counts.most_common(1)[0][1]
        
        return {
            "estimated_hallucination_probability": estimated_prob,
            "confidence_interval_95": [ci_lower, ci_upper],
            "total_samples": n,
            "hallucination_samples": hallucination_count,
            "consensus_response": consensus_response,
            "consensus_ratio": consensus_count / n,
            "response_diversity": len(response_counts) / n,
            "recommendation": "block" if estimated_prob > 0.5 else
                           "review" if estimated_prob > 0.2 else "pass"
        }
```

### 3.3 Information-Theoretic Hallucination Detection

```python
class InformationTheoreticDetector:
    """Use information theory to detect hallucinations."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def detect(self, context, response):
        """Detect hallucination using information-theoretic measures."""
        
        # Compute surprisal of each token in the response
        token_surprisals = await self._compute_token_surprisals(context, response)
        
        # Compute mutual information between context and response
        mi_score = await self._compute_mutual_information(context, response)
        
        # Compute KL divergence between context distribution and response distribution
        kl_divergence = await self._compute_kl_divergence(context, response)
        
        # High surprisal + low MI + high KL = likely hallucination
        hallucination_score = (
            np.mean(token_surprisals) * 0.3 +
            (1 - mi_score) * 0.4 +
            kl_divergence * 0.3
        )
        
        return {
            "hallucination_score": hallucination_score,
            "avg_token_surprisal": np.mean(token_surprisals),
            "max_token_surprisal": np.max(token_surprisals),
            "mutual_information": mi_score,
            "kl_divergence": kl_divergence,
            "high_surprisal_tokens": self._find_high_surprisal_tokens(
                response, token_surprisals, threshold=3.0
            )
        }
    
    async def _compute_token_surprisals(self, context, response):
        """Compute surprisal for each token in the response given context."""
        tokens = response.split()
        surprisals = []
        
        for i, token in enumerate(tokens):
            # Build prompt with context and partial response
            partial_response = " ".join(tokens[:i])
            prompt = f"Context: {context}\n\nResponse so far: {partial_response}\n\nNext token:"
            
            # Get log probabilities
            logprobs = await self.llm.get_logprobs(prompt)
            
            # Surprisal = -log P(token)
            if token in logprobs:
                surprisal = -logprobs[token]
            else:
                surprisal = 20  # High surprisal for unknown tokens
            
            surprisals.append(surprisal)
        
        return surprisals
    
    async def _compute_mutual_information(self, context, response):
        """Compute mutual information between context and response."""
        # Simplified: use embedding similarity as proxy
        context_emb = await self._get_embedding(context)
        response_emb = await self._get_embedding(response)
        
        similarity = np.dot(context_emb, response_emb) / (
            np.linalg.norm(context_emb) * np.linalg.norm(response_emb)
        )
        
        return max(0, similarity)  # MI is non-negative
    
    async def _compute_kl_divergence(self, context, response):
        """Compute KL divergence between context and response distributions."""
        # Simplified: use token frequency distributions
        context_tokens = Counter(context.lower().split())
        response_tokens = Counter(response.lower().split())
        
        all_tokens = set(context_tokens.keys()) | set(response_tokens.keys())
        
        # Normalize to probability distributions
        context_total = sum(context_tokens.values())
        response_total = sum(response_tokens.values())
        
        kl = 0
        for token in all_tokens:
            p = context_tokens.get(token, 0) / max(context_total, 1)
            q = response_tokens.get(token, 0) / max(response_total, 1)
            
            if p > 0 and q > 0:
                kl += p * np.log(p / q)
        
        return kl
    
    def _find_high_surprisal_tokens(self, response, surprisals, threshold=3.0):
        """Find tokens with high surprisal (potential hallucinations)."""
        tokens = response.split()
        high_surprisal = []
        
        for token, surprisal in zip(tokens, surprisals):
            if surprisal > threshold:
                high_surprisal.append({
                    "token": token,
                    "surprisal": surprisal,
                    "position": tokens.index(token)
                })
        
        return sorted(high_surprisal, key=lambda x: x["surprisal"], reverse=True)
```

---

## 4. Knowledge Graph Verification

### 4.1 Building a Verification Knowledge Graph

```python
class VerificationKnowledgeGraph:
    """Use knowledge graphs to verify factual claims."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
    
    def add_fact(self, subject, predicate, obj, source, confidence=1.0):
        """Add a fact to the knowledge graph."""
        self.graph.add_edge(
            subject, obj,
            relation=predicate,
            source=source,
            confidence=confidence
        )
    
    def verify_claim(self, subject, predicate, obj):
        """Verify a claim against the knowledge graph."""
        
        # Check if the exact triple exists
        if self.graph.has_edge(subject, obj):
            edge_data = self.graph[subject][obj]
            return {
                "verified": True,
                "relation": edge_data["relation"],
                "source": edge_data["source"],
                "confidence": edge_data["confidence"]
            }
        
        # Check for inverse relations
        inverse_predicates = {
            "is_parent_of": "is_child_of",
            "is_capital_of": "has_capital",
            "authored": "written_by",
            # Add more inverse relations
        }
        
        inverse_pred = inverse_predicates.get(predicate)
        if inverse_pred and self.graph.has_edge(obj, subject):
            edge_data = self.graph[obj][subject]
            if edge_data["relation"] == inverse_pred:
                return {
                    "verified": True,
                    "relation": f"inverse({inverse_pred})",
                    "source": edge_data["source"],
                    "confidence": edge_data["confidence"]
                }
        
        # Check for related facts (1-hop)
        related_facts = []
        for neighbor in self.graph.neighbors(subject):
            edge_data = self.graph[subject][neighbor]
            related_facts.append({
                "subject": subject,
                "relation": edge_data["relation"],
                "object": neighbor
            })
        
        return {
            "verified": False,
            "related_facts": related_facts,
            "suggestion": "Check related facts for potential verification"
        }
    
    def extract_claims_from_text(self, text):
        """Extract factual claims from text using NER and relation extraction."""
        # Use a pre-trained NER model
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        claims = []
        
        # Extract entity pairs and relations
        for sent in doc.sents:
            entities = [(ent.text, ent.label_) for ent in sent.ents]
            
            for i, (subj, _) in enumerate(entities):
                for j, (obj, _) in enumerate(entities):
                    if i != j:
                        # Use dependency parsing to find relation
                        relation = self._extract_relation(sent, subj, obj)
                        if relation:
                            claims.append({
                                "subject": subj,
                                "relation": relation,
                                "object": obj,
                                "sentence": sent.text
                            })
        
        return claims
    
    def _extract_relation(self, sent, subj, obj):
        """Extract relation between two entities."""
        # Simplified relation extraction
        # In practice, use a relation extraction model
        tokens = [token.text for token in sent]
        
        try:
            subj_idx = tokens.index(subj.split()[0])
            obj_idx = tokens.index(obj.split()[0])
            
            if subj_idx < obj_idx:
                relation_tokens = tokens[subj_idx + len(subj.split()):obj_idx]
            else:
                relation_tokens = tokens[obj_idx + len(obj.split()):subj_idx]
            
            return " ".join(relation_tokens) if relation_tokens else None
        except ValueError:
            return None
```

### 4.2 Knowledge Graph-Augmented Generation

```python
class KGAugmentedGenerator:
    """Generate text with knowledge graph verification."""
    
    def __init__(self, llm_client, kg):
        self.llm = llm_client
        self.kg = kg
    
    async def generate_verified(self, query, context):
        """Generate response with knowledge graph verification."""
        
        # Step 1: Generate initial response
        initial_response = await self.llm.generate(
            f"Context: {context}\n\nQuestion: {query}",
            temperature=0
        )
        
        # Step 2: Extract claims from response
        claims = self.kg.extract_claims_from_text(initial_response)
        
        # Step 3: Verify each claim against knowledge graph
        verified_claims = []
        unverified_claims = []
        
        for claim in claims:
            verification = self.kg.verify_claim(
                claim["subject"], claim["relation"], claim["object"]
            )
            
            if verification["verified"]:
                verified_claims.append({**claim, "verification": verification})
            else:
                unverified_claims.append({**claim, "verification": verification})
        
        # Step 4: If there are unverified claims, regenerate with constraints
        if unverified_claims:
            constraint_prompt = f"""Original response had these unverified claims:
{chr(10).join(f'- {c["subject"]} {c["relation"]} {c["object"]}' for c in unverified_claims)}

Please regenerate the response, removing or correcting these unverified claims.
Only include claims that can be verified."""

            corrected_response = await self.llm.generate(
                f"Context: {context}\n\nOriginal response: {initial_response}\n\n{constraint_prompt}",
                temperature=0
            )
            
            return {
                "response": corrected_response,
                "original_response": initial_response,
                "verified_claims": len(verified_claims),
                "unverified_claims": len(unverified_claims),
                "corrected": True
            }
        
        return {
            "response": initial_response,
            "verified_claims": len(verified_claims),
            "unverified_claims": 0,
            "corrected": False
        }
```

---

## 5. Automated Red-Teaming for Hallucination

### 5.1 Adversarial Test Generation

```python
class HallucinationRedTeam:
    """Automatically generate adversarial tests for hallucination."""
    
    def __init__(self, llm_client, detector):
        self.llm = llm_client
        self.detector = detector
    
    async def generate_test_cases(self, domain, n_cases=50):
        """Generate adversarial test cases that are likely to cause hallucination."""
        
        test_cases = []
        
        # Category 1: Knowledge boundary tests
        boundary_cases = await self._generate_boundary_tests(domain, n_cases // 4)
        test_cases.extend(boundary_cases)
        
        # Category 2: Misleading context tests
        misleading_cases = await self._generate_misleading_tests(domain, n_cases // 4)
        test_cases.extend(misleading_cases)
        
        # Category 3: Temporal confusion tests
        temporal_cases = await self._generate_temporal_tests(domain, n_cases // 4)
        test_cases.extend(temporal_cases)
        
        # Category 4: Citation fabrication tests
        citation_cases = await self._generate_citation_tests(domain, n_cases // 4)
        test_cases.extend(citation_cases)
        
        return test_cases
    
    async def _generate_boundary_tests(self, domain, n):
        """Generate tests at the boundary of model knowledge."""
        prompt = f"""Generate {n} questions about {domain} that are:
1. Very specific (require exact numbers, dates, or names)
2. About obscure or lesser-known facts
3. Likely to tempt the model to fabricate details

Format each as:
Q: [question]
EXPECTED: [what a correct answer should look like]
RISK: [why this might cause hallucination]"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_test_cases(response, "boundary")
    
    async def _generate_misleading_tests(self, domain, n):
        """Generate tests with misleading context."""
        prompt = f"""Generate {n} questions about {domain} where the context 
contains slightly incorrect information that might mislead the model.

Example: "Context says X was invented in 1990, but it was actually 1985. 
Question: When was X invented?"

Format each as:
CONTEXT: [misleading context]
QUERY: [question]
CORRECT_ANSWER: [the actual correct answer]
TRAP: [what the model might incorrectly say]"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_test_cases(response, "misleading")
    
    async def _generate_temporal_tests(self, domain, n):
        """Generate tests involving temporal confusion."""
        prompt = f"""Generate {n} questions about {domain} where the correct 
answer depends on the time period. Include questions where:
1. The answer has changed over time
2. The model might use outdated information
3. Multiple time periods are relevant

Format each as:
QUERY: [question]
TIME_PERIOD: [relevant time period]
CORRECT_ANSWER: [correct answer for that period]
OUTDATED_ANSWER: [what the model might say if using old data]"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_test_cases(response, "temporal")
    
    async def _generate_citation_tests(self, domain, n):
        """Generate tests that tempt citation fabrication."""
        prompt = f"""Generate {n} questions about {domain} that:
1. Would typically require citations to specific papers or sources
2. Ask for specific statistics or data points
3. Request references to organizations or institutions

Format each as:
QUERY: [question]
EXPECTED_CITATION: [a real, verifiable source if one exists]
FABRICATION_RISK: [description of what the model might fabricate]"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_test_cases(response, "citation")
    
    def _parse_test_cases(self, response, category):
        """Parse generated test cases from LLM response."""
        cases = []
        lines = response.split('\n')
        
        current_case = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('CONTEXT:') or line.startswith('QUERY:'):
                if current_case:
                    cases.append(current_case)
                current_case = {"category": category, "query": line.split(':', 1)[1].strip()}
            elif line.startswith('EXPECTED:') or line.startswith('CORRECT_ANSWER:'):
                current_case["expected"] = line.split(':', 1)[1].strip()
            elif line.startswith('RISK:') or line.startswith('TRAP:') or line.startswith('FABRICATION_RISK:'):
                current_case["risk"] = line.split(':', 1)[1].strip()
            elif line.startswith('CONTEXT:'):
                current_case["context"] = line.split(':', 1)[1].strip()
        
        if current_case:
            cases.append(current_case)
        
        return cases
    
    async def run_red_team(self, test_cases, generator_fn):
        """Run red team tests against a generator function."""
        
        results = []
        
        for test_case in test_cases:
            # Generate response
            response = await generator_fn(test_case["query"], test_case.get("context", ""))
            
            # Detect hallucination
            detection = self.detector.detect(
                context=test_case.get("context", ""),
                response=response
            )
            
            # Determine if hallucination occurred
            hallucinated = detection.get("is_hallucinated", False)
            
            results.append({
                "test_case": test_case,
                "response": response,
                "detection": detection,
                "hallucinated": hallucinated,
                "expected_hallucination": "RISK" in test_case or "TRAP" in test_case
            })
        
        # Summary
        total = len(results)
        hallucinated_count = sum(1 for r in results if r["hallucinated"])
        expected_hallucinations = sum(1 for r in results if r["expected_hallucination"])
        
        return {
            "total_tests": total,
            "hallucinations_detected": hallucinated_count,
            "hallucination_rate": hallucinated_count / max(total, 1),
            "expected_hallucinations": expected_hallucinations,
            "detection_accuracy": sum(
                1 for r in results 
                if r["hallucinated"] == r["expected_hallucination"]
            ) / max(total, 1),
            "detailed_results": results
        }
```

---

## 6. Real-Time Production Detection

### 6.1 Streaming Hallucination Detection

```python
class StreamingHallucinationDetector:
    """Detect hallucinations in real-time as text is generated."""
    
    def __init__(self, detector, alert_threshold=0.6):
        self.detector = detector
        self.alert_threshold = alert_threshold
        self.buffer = ""
        self.hallucination_events = []
    
    async def process_token(self, token, context):
        """Process each token as it's generated."""
        
        self.buffer += token
        
        # Check if we have a complete sentence
        if token in ['.', '!', '?', '\n'] or len(self.buffer) > 200:
            # Analyze the current sentence
            sentence = self.buffer.strip()
            
            if sentence:
                detection = self.detector.detect(context, sentence)
                
                if detection.get("hallucination_probability", 0) > self.alert_threshold:
                    self.hallucination_events.append({
                        "sentence": sentence,
                        "probability": detection["hallucination_probability"],
                        "timestamp": datetime.utcnow()
                    })
                    
                    # Emit alert
                    yield {
                        "type": "hallucination_alert",
                        "sentence": sentence,
                        "probability": detection["hallucination_probability"],
                        "recommendation": "review"
                    }
            
            self.buffer = ""
    
    def get_summary(self):
        """Get summary of hallucination events during generation."""
        return {
            "total_events": len(self.hallucination_events),
            "max_probability": max(
                (e["probability"] for e in self.hallucination_events),
                default=0
            ),
            "events": self.hallucination_events
        }
```

### 6.2 Production Monitoring Pipeline

```python
class ProductionHallucinationMonitor:
    """Monitor hallucinations in production LLM applications."""
    
    def __init__(self, detector, metrics_client, alert_client):
        self.detector = detector
        self.metrics = metrics_client
        self.alerts = alert_client
        self.window_size = 1000  # Rolling window for rate calculation
        self.detections = []
    
    async def monitor_request(self, request_id, query, context, response):
        """Monitor a single production request."""
        
        # Run detection
        detection = self.detector.detect(context, response)
        
        # Record metrics
        self.metrics.gauge("hallucination.probability", detection["hallucination_probability"])
        self.metrics.counter("hallucination.total_requests")
        
        if detection["is_hallucinated"]:
            self.metrics.counter("hallucination.detections")
            
            # Record for rolling window
            self.detections.append({
                "request_id": request_id,
                "timestamp": datetime.utcnow(),
                "probability": detection["hallucination_probability"]
            })
            
            # Keep only recent detections
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.detections = [d for d in self.detections if d["timestamp"] > cutoff]
            
            # Calculate rolling hallucination rate
            rate = len(self.detections) / self.window_size
            
            # Check thresholds
            if rate > 0.1:  # 10% hallucination rate
                await self.alerts.send(
                    level="critical",
                    message=f"Hallucination rate {rate:.1%} exceeds 10% threshold",
                    context={"request_id": request_id, "detection": detection}
                )
            elif rate > 0.05:  # 5% hallucination rate
                await self.alerts.send(
                    level="warning",
                    message=f"Hallucination rate {rate:.1%} exceeds 5% threshold",
                    context={"request_id": request_id, "detection": detection}
                )
        
        return {
            "request_id": request_id,
            "detection": detection,
            "rolling_rate": len(self.detections) / self.window_size
        }
```

---

## 7. Hallucination in Structured Output

### 7.1 JSON/Structured Output Verification

```python
import json
from jsonschema import validate, ValidationError

class StructuredOutputVerifier:
    """Verify structured outputs for hallucination."""
    
    def __init__(self, schema):
        self.schema = schema
    
    def verify(self, output, context):
        """Verify a structured output against context."""
        
        results = {
            "schema_valid": True,
            "context_consistent": True,
            "hallucinated_fields": [],
            "issues": []
        }
        
        # Check schema validity
        try:
            validate(instance=output, schema=self.schema)
        except ValidationError as e:
            results["schema_valid"] = False
            results["issues"].append(f"Schema validation error: {e.message}")
        
        # Check each field against context
        for field_path, value in self._flatten(output).items():
            if isinstance(value, str) and len(value) > 10:
                # Check if this string value is supported by context
                support_score = self._check_support(value, context)
                
                if support_score < 0.3:
                    results["hallucinated_fields"].append({
                        "field": field_path,
                        "value": value,
                        "support_score": support_score
                    })
                    results["context_consistent"] = False
        
        return results
    
    def _flatten(self, obj, prefix=""):
        """Flatten nested dict/list into dot-notation paths."""
        items = {}
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                items.update(self._flatten(v, new_key))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{prefix}[{i}]"
                items.update(self._flatten(v, new_key))
        else:
            items[prefix] = obj
        
        return items
    
    def _check_support(self, value, context):
        """Check if a value is supported by context."""
        # Use semantic similarity or NLI
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        value_emb = model.encode([value])[0]
        context_emb = model.encode([context])[0]
        
        similarity = np.dot(value_emb, context_emb) / (
            np.linalg.norm(value_emb) * np.linalg.norm(context_emb)
        )
        
        return max(0, similarity)
```

### 7.2 Code Hallucination Detection

```python
class CodeHallucinationDetector:
    """Detect hallucinations in generated code."""
    
    def __init__(self):
        self.common_hallucinations = [
            "import nonexistent_library",
            "call non-existent_function",
            "use deprecated API",
            "incorrect syntax",
            "wrong parameter types",
            "missing imports",
            "fabricated method names"
        ]
    
    def detect(self, code, context, language="python"):
        """Detect hallucinations in generated code."""
        
        results = {
            "syntax_valid": True,
            "imports_valid": True,
            "api_valid": True,
            "hallucinated_elements": [],
            "suggestions": []
        }
        
        # Check syntax
        try:
            if language == "python":
                import ast
                ast.parse(code)
            # Add other language parsers
        except SyntaxError as e:
            results["syntax_valid"] = False
            results["hallucinated_elements"].append({
                "type": "syntax_error",
                "line": e.lineno,
                "message": str(e)
            })
        
        # Check imports
        import re
        if language == "python":
            imports = re.findall(r'(?:from|import)\s+(\w+)', code)
            for imp in imports:
                if not self._import_exists(imp):
                    results["imports_valid"] = False
                    results["hallucinated_elements"].append({
                        "type": "invalid_import",
                        "module": imp
                    })
        
        # Check API usage against context
        api_calls = self._extract_api_calls(code, language)
        for api_call in api_calls:
            if not self._api_exists_in_context(api_call, context):
                results["api_valid"] = False
                results["hallucinated_elements"].append({
                    "type": "fabricated_api",
                    "call": api_call
                })
        
        return results
    
    def _import_exists(self, module_name):
        """Check if a Python module exists."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _extract_api_calls(self, code, language):
        """Extract API calls from code."""
        # Simplified: find function calls
        import re
        pattern = r'(\w+)\.(?:\w+)\('
        return re.findall(pattern, code)
    
    def _api_exists_in_context(self, api_call, context):
        """Check if an API call is mentioned in context."""
        return api_call.lower() in context.lower()
```

---

## 8. Cross-Lingual Hallucination

### 8.1 Language-Specific Hallucination Patterns

| Language | Common Hallucination Types | Detection Challenges |
|----------|--------------------------|---------------------|
| English | Factual errors, citation fabrication | Well-studied, many tools |
| Chinese | Character substitution errors | Fewer detection tools |
| Japanese | Honorific misuse, context confusion | Complex grammar rules |
| Arabic | Diacritical mark errors | Right-to-left processing |
| Hindi | Gender agreement errors | Morphological complexity |

### 8.2 Cross-Lingual Verification

```python
class CrossLingualHallucinationDetector:
    """Detect hallucinations across languages."""
    
    def __init__(self, multilingual_model):
        self.model = multilingual_model
    
    async def detect_cross_lingual(self, source_text, source_lang, 
                                     translated_text, target_lang):
        """Detect hallucinations introduced during translation."""
        
        # Get embeddings in both languages
        source_emb = self.model.encode(source_text, lang=source_lang)
        target_emb = self.model.encode(translated_text, lang=target_lang)
        
        # Semantic similarity
        similarity = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
        )
        
        # Claim extraction and verification
        source_claims = self._extract_claims(source_text, source_lang)
        target_claims = self._extract_claims(translated_text, target_lang)
        
        # Match claims
        matched_claims = self._match_claims(source_claims, target_claims)
        
        # Identify hallucinated claims
        hallucinated = []
        for source_claim in source_claims:
            matched = False
            for target_claim in target_claims:
                if self._claims_match(source_claim, target_claim):
                    matched = True
                    break
            if not matched:
                hallucinated.append(source_claim)
        
        # Identify added claims (potential hallucinations)
        added = []
        for target_claim in target_claims:
            matched = False
            for source_claim in source_claims:
                if self._claims_match(source_claim, target_claim):
                    matched = True
                    break
            if not matched:
                added.append(target_claim)
        
        return {
            "semantic_similarity": similarity,
            "source_claims": len(source_claims),
            "target_claims": len(target_claims),
            "matched_claims": len(matched_claims),
            "lost_claims": len(hallucinated),
            "added_claims": len(added),
            "hallucination_rate": len(added) / max(len(target_claims), 1),
            "details": {
                "hallucinated": hallucinated,
                "added": added
            }
        }
```

---

## 9. Adversarial Hallucination Attacks

### 9.1 Attack Vectors

| Attack | Method | Impact |
|--------|--------|--------|
| **Prompt injection** | Manipulate prompts to cause hallucination | Misinformation generation |
| **Context poisoning** | Inject false information into context | RAG hallucination |
| **Adversarial examples** | Slight input modifications causing hallucination | Unpredictable outputs |
| **Data poisoning** | Corrupt training data to increase hallucination | Systematic errors |
| **Model extraction** | Extract model to study hallucination patterns | Targeted attacks |

### 9.2 Defense Mechanisms

```python
class AdversarialHallucinationDefense:
    """Defend against adversarial attacks that cause hallucination."""
    
    def __init__(self, detector, sanitizer):
        self.detector = detector
        self.sanitizer = sanitizer
    
    async def defend(self, input_data, context):
        """Apply defenses against adversarial hallucination attacks."""
        
        defenses_applied = []
        
        # Defense 1: Input sanitization
        sanitized_input = self.sanitizer.sanitize(input_data)
        if sanitized_input != input_data:
            defenses_applied.append("input_sanitization")
        
        # Defense 2: Context validation
        validated_context = self._validate_context(context)
        if validated_context != context:
            defenses_applied.append("context_validation")
        
        # Defense 3: Output verification
        output = await self._generate(sanitized_input, validated_context)
        verification = self.detector.detect(validated_context, output)
        
        if verification["is_hallucinated"]:
            # Defense 4: Regenerate with stricter constraints
            output = await self._generate_with_constraints(
                sanitized_input, validated_context
            )
            defenses_applied.append("constrained_regeneration")
        
        return {
            "output": output,
            "defenses_applied": defenses_applied,
            "verification": verification
        }
    
    def _validate_context(self, context):
        """Validate context for potential poisoning."""
        # Check for known adversarial patterns
        adversarial_patterns = [
            r"ignore previous instructions",
            r"system prompt:",
            r"you are now",
            r"forget everything"
        ]
        
        import re
        for pattern in adversarial_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                # Remove or neutralize the adversarial content
                context = re.sub(pattern, "[NEUTRALIZED]", context, flags=re.IGNORECASE)
        
        return context
```

---

## Cross-References

| Document | Relevance |
|----------|-----------|
| [01-Overview.md](01-Overview.md) | General overview and taxonomy |
| [02-Core-Topics.md](02-Core-Topics.md) | Core techniques and patterns |
| [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) | Tool implementations |
| [05-Future-Outlook.md](05-Future-Outlook.md) | Future directions |
| [03-Agents/05-Tool-Implementations.md](../../03-Agents/05-Tool-Implementations.md) | Tool use in agents |
| [18-Agent-Security-and-Trust/02-Prompt-Injection-Defenses.md](../../18-Agent-Security-and-Trust/02-Prompt-Injection-Defenses.md) | Defending against adversarial attacks |
| [20-Agent-Infrastructure/03-Agent-Tracing-and-Observability.md](../../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md) | Production monitoring |

---

*Last updated: July 2026*
