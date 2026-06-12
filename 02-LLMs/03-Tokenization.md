# Tokenization — Comprehensive Reference

> This document covers everything about tokenization in Large Language Models: algorithms, training, vocabulary design, impact on performance, multilingual challenges, and practical implementation. Written for AI engineers and NLP researchers.

---

## Table of Contents

1. [What is Tokenization?](#what-is-tokenization)
2. [Subword Tokenization Algorithms](#subword-tokenization-algorithms)
3. [Byte-Level Tokenization](#byte-level-tokenization)
4. [Tokenizer Training](#tokenizer-training)
5. [Vocabulary Size Tradeoffs](#vocabulary-size-tradeoffs)
6. [Special Tokens](#special-tokens)
7. [Tokenization Comparison Across Models](#tokenization-comparison-across-models)
8. [Tokenizer Impact on Performance](#tokenizer-impact-on-performance)
9. [Multilingual Tokenization Challenges](#multilingual-tokenization-challenges)
10. [Tokenization and Context Length](#tokenization-and-context-length)
11. [tiktoken Library Internals](#tiktoken-library-internals)
12. [HuggingFace Tokenizers Library](#huggingface-tokenizers-library)
13. [Adding New Tokens to Existing Tokenizers](#adding-new-tokens-to-existing-tokenizers)
14. [Tokenizer Merging](#tokenizer-merging)
15. [Custom Tokenizer Creation](#custom-tokenizer-creation)
16. [Advanced Topics](#advanced-topics)

---

## What is Tokenization?

Tokenization is the process of converting raw text into a sequence of integers (token IDs) that the model can process. It is the **first and last step** in any LLM pipeline:

```
Raw text → [Tokenizer] → Token IDs → [LLM] → Token IDs → [Detokenizer] → Output text
```

**Why tokenization matters:**
- Determines the **vocabulary** the model can produce
- Affects **compression ratio** (chars per token) — directly impacts effective context length
- Drives **multilingual capability** — some tokenizers handle non-English poorly
- Influences **computational cost** — more tokens = more compute
- Impacts **model performance** — bad tokenization harms reasoning, code gen, translation
- Creates **tokenization artifacts** — suboptimal splits can introduce bias or information loss

**Key concepts:**
- **Token:** A unit of text (word, subword, or character/byte)
- **Vocabulary:** The set of all tokens the model knows
- **Token ID:** The integer index of a token in the vocabulary
- **Encoder:** Text → Token IDs
- **Decoder:** Token IDs → Text
- **Special tokens:** Reserved tokens with specific functions (BOS, EOS, PAD, etc.)

---

## Subword Tokenization Algorithms

### Byte-Pair Encoding (BPE)

BPE is the most widely used tokenization algorithm in modern LLMs. It was originally proposed as a data compression technique by Philip Gage in 1994 and adapted for NLP by Rico Sennrich et al. in 2016 (Neural Machine Translation of Rare Words with Subword Units).

**Core algorithm:**

1. **Initialization:** Start with a vocabulary of all individual characters (or bytes) in the training corpus
2. **Count pairs:** Find the most frequent adjacent pair of tokens in the corpus
3. **Merge:** Replace all occurrences of that pair with a new token
4. **Add to vocabulary:** Add the new merged token to the vocabulary
5. **Repeat:** Steps 2-4 until the vocabulary reaches the desired size

**Pseudocode:**
```
function train_bpe(corpus, vocab_size):
    # Step 1: Initialize with character-level vocabulary
    vocab = set of all unique characters in corpus
    tokens = [char for char in corpus]  # initial segmentation
    
    # Step 2-5: Iteratively merge most frequent pairs
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_counts = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += 1
        
        # Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # Merge this pair everywhere
        new_token = best_pair[0] + best_pair[1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        
        # Add merged token to vocabulary
        vocab.add(new_token)
    
    return vocab, merge_rules
```

**Merge rules:**
- The final vocabulary is stored as a set of tokens
- Additionally, the **merge rules** (ordered list of pair merges) must be saved
- At inference time, new text is tokenized by greedily applying these merge rules in order
- **Greedy matching:** For each position, find the longest vocabulary token that matches the text
- Some implementations use the merge rules directly instead of vocabulary lookup

**Frequency-based nature:**
- BPE is purely frequency-based — it doesn't use any linguistic knowledge
- Common pairs merge first: ("th",  "he", "the"), ("ing"), ("tion")
- Rare characters or pairs remain as subwords or individual characters
- This creates a bias: common words stay as single tokens, rare words get split

**BPE tokenization example:**
```
Corpus: "low lower lowest low"

Step 1 — Characters: l, o, w, space, e, r, s, t
  Initial vocabulary: {l, o, w, space, e, r, s, t}
  Most common pair: ('l', 'o') appears 3 times → merge to "lo"

Step 2:
  Vocabulary: {l, o, w, space, e, r, s, t, lo}
  Most common pair: ('lo', 'w') appears 3 times → merge to "low"

Step 3:
  Vocabulary: {l, o, w, ..., low, ...}
  Most common pair: ('low', 'e') appears 2 times → merge to "lowe"

Step 4:
  Vocabulary: ... continues until target vocabulary size
```

**Problems with raw BPE:**
- Character-by-character initialization means the vocabulary initially contains all characters
- For large vocabularies, the algorithm is O(vocab_size × corpus_size) — expensive
- Merging decisions are greedy and cannot be reversed
- BPE has no notion of "word boundaries" — it can merge across word boundaries
- Solution: **pre-tokenization** (split on whitespace/punctuation) before BPE

### WordPiece

WordPiece is the tokenization algorithm used by BERT and its variants (originated by Google for Japanese/Korean segmentation, then adapted for BERT).

**Key difference from BPE:**
- BPE merges the most **frequent** pair
- WordPiece merges the pair that **maximizes the likelihood** of the training data

**Algorithm:**
1. Initialize vocabulary with all characters (like BPE)
2. Start with a "base" vocabulary of single characters
3. Try merging every possible pair of tokens
4. For each candidate merge, compute the **likelihood improvement**:
   - L = log P(merge) — log P(no merge)
   - where P(merge) = pair_count / total_pairs
   - and P(no merge) = first_count / total × second_count / total
5. Keep the merge that gives the **largest likelihood increase**
6. Repeat until target vocabulary size

**Likelihood-based merge scoring:**
```
score(A, B) = count(A,B) / (count(A) × count(B))
```

This score prefers merging tokens that frequently appear together **relative to their individual frequencies**. A high score means the pair is more "cohesive" than expected by chance.

**WordPiece vs BPE example:**
- BPE might merge "th" → "t" + "h" if "th" is very common
- WordPiece might NOT merge "th" if "t" and "h" also appear commonly apart
- WordPiece tends to create more "meaningful" subwords because it captures statistical dependence

**BERT's WordPiece:**
- Pre-tokenization: Basic whitespace + punctuation splitting
- Vocabulary: 30,000 tokens
- Special tokens: [CLS], [SEP], [MASK], [PAD], [UNK]
- WordPiece uses "##" prefix to indicate continuation of a word:
  - "playing" → "play" + "##ing"
  - "unhappiness" → "un" + "##happiness"
- Unknown words are split into characters (if "##" prefix on single character) or mapped to [UNK]

**Limitations:**
- WordPiece requires more computation than BPE (must evaluate all candidate pairs)
- The likelihood-based scoring can be less stable than frequency-based
- BERT's implementation has a size limit on intermediate merges, which limits token creation
- The "##" notation is a design choice — other implementations use different markers

### Unigram Language Model

Unigram tokenization (Kudo, 2018 — Subword Regularization) takes a **probabilistic** approach. Instead of greedily building a vocabulary, it starts with a large vocabulary and iteratively prunes it.

**Core idea:**
- Each token is assigned a probability based on its frequency
- The segmentation of a word is not deterministic — there are multiple possible segmentations
- The best segmentation is the one with the highest probability under the unigram model

**Algorithm:**

1. **Initialization:** Create a large "seed" vocabulary — typically all characters plus the most common substrings (from BPE or other methods)
2. **Expectation-Maximization (EM) training:**
   - **E-step:** Given the current vocabulary, compute the most likely segmentation of each word using the Viterbi algorithm (or the forward-backward algorithm for all possible segmentations)
   - **M-step:** Re-estimate the probability of each token based on its expected count under the current segmentation
3. **Pruning:** Remove tokens with the lowest probability (typically remove 10-20% of tokens each iteration)
4. **Repeat:** EM training + pruning until the target vocabulary size

**EM training details:**

**E-step (computing token probabilities):**
For a given word w, the probability of a segmentation S = (t₁, t₂, ..., tₙ) is:
```
P(S) = Πᵢ P(tᵢ)  (unigram assumption — tokens are independent)
```

The probability of the word is:
```
P(w) = Σ_{S ∈ segmentations(w)} P(S)
```

The most likely segmentation is found via Viterbi (dynamic programming):
```
dp[0] = 1.0  (empty prefix has probability 1)
dp[i] = max_{j < i, w[j:i] ∈ vocab} dp[j] × P(w[j:i])
```

**M-step (re-estimating token probabilities):**
For each token t, its expected count is:
```
count(t) = Σ_{word w} Σ_{segmentation S of w} count_S(t) × P(S|w)
```
where P(S|w) = P(S) / P(w) is the posterior probability of segmentation S given word w.

Normalized probabilities:
```
P(t) = count(t) / Σ_{t'} count(t')
```

**Pruning criterion:**
Tokens with the lowest P(t) are removed. The loss incurred by removing token t is:
```
loss(t) = Σ_{w} log P(w)  (with t removed) — Σ_{w} log P(w)  (with t)
```

**Key advantages of Unigram:**
- **Principled probabilistic model:** Provides a well-founded way to trade off vocabulary size vs coverage
- **Subword regularization:** During training, you can sample from multiple segmentations instead of using the single best one — this acts as a regularizer
- **Better multilingual support:** The EM algorithm naturally handles different character distributions across languages
- **More efficient than BPE:** Training is typically faster because EM converges quickly

**Used in:**
- SentencePiece (Unigram mode is the default)
- T5, mT5, XLNet, ALBERT

### SentencePiece

SentencePiece (Google, 2018) is a **language-independent tokenization library** that treats the input as a raw byte sequence (or Unicode string) without requiring pre-tokenization.

**Key innovation — No pre-tokenization needed:**
- Traditional tokenizers (BPE, WordPiece) require pre-tokenization: splitting text by whitespace/punctuation
- This is language-dependent — doesn't work well for CJK (Chinese, Japanese, Korean) which have no spaces
- SentencePiece directly processes raw text (or bytes) without language-specific pre-tokenization

**Two training modes:**

1. **BPE mode (--model_type=bpe):**
   - Standard BPE merges, applied to the character-level representation
   - Same algorithm as BPE, but SentencePiece handles normalization separately

2. **Unigram mode (--model_type=unigram, default):**
   - EM-based unigram language model
   - SentencePiece's default and most common mode

**Normalization:**

SentencePiece applies **normalization** to the input text before tokenization:

- **NFKC normalization** (default): Unicode normalization form KC
  - Converts full-width to half-width characters (ｅ → e)
  - Converts ligatures to their components (ﬁ → fi)
  - Standardizes Unicode representation of characters
- **Custom normalization:** Users can define normalization rules via a text file
- **No normalization:** Also supported (for code or specialized domains)

**Normalization rules file format:**
```
# Comment
<tab>	 <tab> <half-width tab>  (replace tab with half-width version)
Ａ<tab>A  (full-width A → half-width A)
```
Rules are applied as longest-match-first string replacement.

**Byte Fallback (--byte_fallback=true):**

When a character is not in the vocabulary and cannot be decomposed into known tokens, SentencePiece can fall back to **UTF-8 byte-level encoding**:

- Unknown character → sequence of UTF-8 byte tokens (e.g., "😀" → <0xF0>, <0x9F>, <0x98>, <0x80>)
- Byte tokens are special tokens in the format `<0xNN>` where NN is the byte value (00-FF)
- This guarantees **no unknown tokens** — every possible Unicode string can be represented
- Trade-off: byte sequences are long (~4 bytes per character for non-BMP chars)

**Tokenization algorithms in SentencePiece:**

**BPE mode tokenization:**
```
Raw text → Normalize → BPE merges (greedy, left-to-right) → Token IDs
```

**Unigram mode tokenization:**
```
Raw text → Normalize → Find best segmentation via Viterbi → Token IDs
```

**SentencePiece model file format:**
The `.model` file contains:
- Header (magic number, model type, vocab size)
- Normalization rules (as a serialized proto)
- Token vocabulary with scores (log-probabilities for Unigram, merge ranks for BPE)
- Byte fallback tokens (if enabled)
- Reserved IDs for special tokens

**Configuration parameters:**
```
--input: Training data file
--model_prefix: Output model name prefix  
--vocab_size: Target vocabulary size (default: 8000)
--model_type: unigram (default), bpe, char, word
--character_coverage: Fraction of characters covered (default: 0.9995)
--max_sentence_length: Max sentence length in bytes (default: 4192)
--normalization_rule_name: nfkc (default), identity, or custom
--add_dummy_prefix: Add space at beginning of text (default: true)
--remove_extra_whitespaces: Collapse multiple whitespaces (default: true)
--byte_fallback: Enable byte-level fallback (default: false)
--split_by_whitespace: Split by whitespace (default: true for BPE)
--allow_unknown: Allow unknown tokens (default: false for unigram)
```

---

## Byte-Level Tokenization

Byte-level tokenization operates on **raw UTF-8 bytes** instead of Unicode characters or characters after normalization. This provides 100% coverage of all possible Unicode strings.

### GPT-2's Byte-Level BPE

GPT-2 introduced byte-level BPE to solve the problem of unknown tokens.

**Problem with character-level BPE:**
- Character-level BPE starts with all Unicode characters
- Unicode has 150,000+ characters — the base vocabulary is huge
- New characters (emoji, mathematical symbols, historical scripts) are not covered
- Any character not in the initial vocabulary becomes [UNK]

**Solution: Base vocabulary = 256 byte values**
- Instead of starting with Unicode characters, start with all 256 possible byte values
- UTF-8 encodes all Unicode characters as 1-4 bytes
- Every possible Unicode character is already represented
- No UNK tokens possible — complete Unicode coverage

**GPT-2's approach:**

1. **Byte-level vocabulary:** Start with all 256 byte values (0x00-0xFF)
2. **Add word-level tokens:** Add special token Ġ (representing space) and common words
3. **BPE merges:** Apply standard BPE merging on the byte sequences
4. **Final vocabulary:** 50,257 tokens (256 byte tokens + ~50,000 merged tokens + 1 end-of-text token)

**Byte-level BPE tokenization example:**
```
Input: "Hello 👋 world"
UTF-8 bytes: H(48) e(65) l(6C) l(6C) o(6F) 20 F0 9F 91 8B 20 w(77) o(6F) r(72) l(6C) d(64)

The tokenizer processes these bytes and applies BPE merges:
"Hello" → single token (common word)
" " + "👋" → might stay as individual bytes (rare emoji)
" " + "world" → "world" as single token (common word)
```

**Strengths:**
- Complete Unicode coverage (no UNK tokens)
- Can handle any script, emoji, mathematical symbols
- Simple — just bytes in, tokens out

**Weaknesses:**
- Non-English text uses significantly more bytes per character
  - English ASCII: 1 byte per character
  - CJK characters: 3 bytes per character
  - Emoji: 4 bytes per character
- This means non-English text is tokenized into **more tokens** — reducing effective context
- For example, "你好" = 6 bytes = up to 6 initial tokens (before BPE merges)

### GPT-4's tiktoken (cl100k_base)

GPT-4 uses the **tiktoken** library and the **cl100k_base** tokenizer. This is the most widely-used byte-level BPE tokenizer in production.

**cl100k_base specifications:**
- Algorithm: Byte-level BPE
- Vocabulary size: **100,256** tokens
- Base vocabulary: 256 byte tokens
- Regular tokens: ~99,000 learned BPE merges
- Special tokens: 0 (special tokens handled separately, not merged)
- Regex-based pre-tokenization: Yes (different rules for different character classes)

**Regex pre-tokenization rules (GPT-4 split pattern):**
```
'(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This regex captures:
1. English contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive)
2. Letter sequences: optionally preceded by space, one or more Unicode letters
3. Number sequences: optionally preceded by space, one or more Unicode digits
4. Other characters: non-letter, non-digit, non-space sequences
5. Whitespace captures

**Effect of pre-tokenization:**
- Words are separated from their preceding space
- Numbers are separated from letters
- Punctuation stands alone
- This ensures that BPE merges only happen within these "word-like" units
- Prevents BPE from merging across word boundaries (a problem with pure byte-level BPE)

**Encoding process (tiktoken):**
```
1. Apply regex pre-tokenization → list of "chunks"
2. For each chunk:
   a. Encode to UTF-8 bytes
   b. Apply BPE merges (greedy, longest match)
   c. Map each merged token to its ID
3. Concatenate all token IDs
```

**Performance characteristics:**
- English text: ~3.8 characters per token (excellent compression)
- Code: ~4.5 chars per token (including whitespace)
- Non-English (Latin script): ~3-4 chars per token
- CJK: ~1.5 characters (Chinese chars) per token (poor — 3 bytes per char → wasted)
- Processing speed: ~50 MB/s on CPU (Rust core)

### Llama 3's tiktoken-based Tokenizer

Llama 3 uses a **custom tiktoken-based tokenizer** with vocabulary size **128,256**.

**Key differences from GPT-4's cl100k_base:**

1. **Larger vocabulary:** 128,256 vs 100,256 tokens
   - More room for common subwords and multi-byte sequences
   - Better multilingual coverage
   - Better code tokenization

2. **Different regex pre-tokenization:**
   - Llama 3 uses a simplified regex pattern compared to GPT-4
   - Less aggressive separation of punctuation
   - Better handling of whitespace in code

3. **No special tokens in vocabulary:**
   - Like GPT-4, special tokens (BOS, EOS, etc.) are handled outside the tokenizer
   - Added during chat template formatting, not during tokenization

4. **Improved multilingual tokenization:**
   - Additional tokens for common multi-byte character sequences
   - Better compression for CJK languages (~1.2 chars/token vs ~1.5 for GPT-4)

**Compression benchmark:**
```
GPT-4 (cl100k):  "Hello world!如何用Python写一个快速排序？" → 24 tokens
Llama 3 (128k):   "Hello world!如何用Python写一个快速排序？" → 20 tokens
```

The improvement comes from dedicated tokens for common Chinese character pairs and code patterns.

**Training of Llama 3 tokenizer:**
- Trained on a multilingual corpus (English, code, and 30+ other languages)
- Target vocabulary size: 128,256
- BPE merging with byte-level base vocabulary
- Regex pre-tokenization tuned for code (preserving indentation)
- Extensive evaluation on compression ratio across languages before finalizing

### Claude Tokenizer (Anthropic)

Anthropic's Claude uses a proprietary tokenizer (not publicly documented in detail).

**Known characteristics:**
- Byte-level BPE (like GPT-4 and Llama 3)
- Estimated vocabulary size: ~100K-150K
- Optimized for helpful and harmless chat
- Good multilingual performance (Claude 3+)

**Unique aspects (inferred):**
- Claude appears to have better tokenization for safety-critical tokens
- The tokenizer is integrated with Claude's "constitutional" framework
- Vision tokens are integrated into the same tokenization scheme

---

## Tokenizer Training

### Training from Scratch

Training a tokenizer from scratch requires:

1. **A representative corpus:** The training data should match the distribution of text the model will see
2. **A target vocabulary size:** Typically 16K-256K tokens
3. **An algorithm choice:** BPE, WordPiece, or Unigram
4. **Pre-tokenization decisions:** Regex rules, normalization, byte fallback

**General workflow:**
```
1. Collect training corpus (5-50GB of text)
2. Apply normalization (NFKC, lowercase, etc.)
3. Define pre-tokenization rules
4. Initialize vocabulary (characters or bytes)
5. Train the tokenizer (iterative merging or EM)
6. Evaluate compression ratio, coverage, and token distribution
7. Iterate on parameters until satisfied
```

### Vocabulary Selection

The vocabulary is determined by the training algorithm, but there are key considerations:

**Inclusion criteria for tokens:**
- **Frequency:** Tokens should appear enough times in the corpus
- **Length:** Very long tokens (>15 characters) are rarely useful
- **Compositionality:** Subwords should be meaningful units
- **Coverage:** The vocabulary should cover >99.99% of tokens in the corpus

**Manual vocabulary additions:**
- Domain-specific terms (medical, legal, scientific)
- Common multi-token expressions that should be single tokens
- Code keywords and operators
- Important entity names

**Vocabulary pruning:**
- Remove tokens that never appear in the corpus (can happen with pre-defined vocabularies)
- Remove tokens that are too long (token length distribution is typically capped)
- Remove tokens with very low probability (in unigram models)

### Merge Frequency in BPE

Understanding how merges are counted is important for analyzing BPE behavior:

**Raw frequency counting:**
- Each adjacent pair of tokens is counted
- The pair with the highest **global** frequency is merged
- Problem: many merges happen in very specific contexts

**Weighted frequency:**
- Each occurrence of a pair can be weighted
- Common weighting: inverse document frequency (IDF)
- This reduces the impact of repeated pairs in the same document

**Entropy-based merging:**
- Instead of pure frequency, use entropy reduction as the merge criterion
- Merge the pair that provides the most "information gain"
- More principled but computationally expensive

### Learning BPE Merges Algorithm

The standard BPE training algorithm has O(V × D) complexity where V is vocab size and D is corpus size.

**Optimizations:**

1. **Indexed pair counting:**
   - Maintain a hash map from (token, token) → count
   - After each merge, update only affected positions rather than recounting everything
   - Affected positions: where the merged token appears

2. **Boundary-aware counting:**
   - Only count pairs within the same word (if using pre-tokenization)
   - This is how SentencePiece BPE works

3. **Approximate counting:**
   - Sample the corpus instead of full counting
   - Use reservoir sampling for large corpora

**Algorithm with optimization:**
```
function train_bpe_fast(corpus, vocab_size):
    # Pre-tokenize into words
    words = pre_tokenize(corpus)
    
    # Convert each word to list of character tokens
    token_lists = [to_chars(w) for w in words]
    
    # Build initial pair count map
    pair_counts = defaultdict(int)
    for tlist in token_lists:
        for i in range(len(tlist)-1):
            pair_counts[(tlist[i], tlist[i+1])] += 1
    
    # Main training loop
    vocab = set(all_chars)
    while len(vocab) < vocab_size:
        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # Create merged token
        new_token = ''.join(best_pair)
        
        # Update all token lists and pair counts
        new_pair_counts = defaultdict(int)
        for tlist in token_lists:
            # Perform this merge
            new_list = []
            i = 0
            while i < len(tlist):
                if i < len(tlist)-1 and tlist[i] == best_pair[0] and tlist[i+1] == best_pair[1]:
                    new_list.append(new_token)
                    # Add new adjacent pairs involving the merged token
                    if i > 0:
                        new_pair_counts[(tlist[i-1], new_token)] += 1
                    i += 2
                else:
                    new_list.append(new_list[-1] if new_list else None)  # handle boundaries
                    ...  # complex boundary tracking
                    i += 1
            token_lists = [new_list for each word]
        
        vocab.add(new_token)
    
    return vocab
```

The boundary tracking is the most complex part — after merging, new adjacent pairs involving the merged token must be added to the counts.

### Unigram EM Algorithm (Detailed)

The Unigram EM algorithm is more complex than BPE but produces better tokenizers for multilingual use.

**Forward-Backward Algorithm for computing expected counts:**

For a word w = [c₁, c₂, ..., cₙ] (characters), we want the expected count of each token under all possible segmentations.

**Forward pass:**
```
α[i] = P(token[1:i]) = Σ_{j < i, w[j:i] ∈ vocab} α[j] × P(w[j:i])
```
where α[0] = 1.0

**Backward pass:**
```
β[i] = P(w[i+1:n]) = Σ_{j > i, w[i:j] ∈ vocab} P(w[i:j]) × β[j]
```
where β[n] = 1.0

**Token posterior probability:**
For a token t matching span w[i:j]:
```
P(t at position i | w) = α[i] × P(t) × β[j] / α[n]
```

**Expected count of token t:**
```
expected_count(t) = Σ_{word w} Σ_{positions i,j where w[i:j] == t} P(t at i | w)
```

**M-step:**
```
P_new(t) = expected_count(t) / Σ_{t'} expected_count(t')
```

**Convergence:**
The algorithm converges in 5-20 iterations for most corpora.

---

## Vocabulary Size Tradeoffs

### Large Vocabulary vs Small Vocabulary

**Large vocabulary (100K-256K):**

| Pros | Cons |
|------|------|
| Better compression (fewer tokens for same text) | Larger embedding table (memory) |
| Fewer tokens per sequence → more effective context | More parameters to train |
| Better multilingual coverage | Slower training (softmax over more classes) |
| More "whole words" preserved | More training data needed to learn rare tokens |
| Lower inference latency (fewer tokens to process) | Larger model checkpoint files |

**Small vocabulary (8K-32K):**

| Pros | Cons |
|------|------|
| Smaller embedding table | More tokens per sequence |
| Faster softmax (output projection) | Less multilingual coverage |
| Better for limited hardware | Frequent subword splitting |
| Faster tokenizer training | Characters or bytes as fallback |
| More parameter budget for transformer layers | Longer sequences reduce effective context |

### Efficiency vs Coverage

**The coverage-efficiency tradeoff:**
- Coverage: percentage of text that can be represented without splitting into small units
- Efficiency: average characters per token (higher is more efficient)

```
Small vocab (e.g., 32K):
  Coverage: 99.5% of English tokens are single tokens
  Efficiency: ~4.2 chars/token for English
  Efficiency: ~1.2 chars/token for Chinese
  
Large vocab (e.g., 128K):
  Coverage: 99.9%+ 
  Efficiency: ~4.8 chars/token for English
  Efficiency: ~2.0 chars/token for Chinese
```

**Diminishing returns:**
- Increasing vocab from 8K to 32K: big improvement
- Increasing from 32K to 128K: moderate improvement
- Increasing from 128K to 256K: small improvement
- The curve is logarithmic — most gains come from the first 50K tokens

**Optimal vocab size (empirical):**
- English-only models: 32K-50K
- Multilingual models: 100K-250K
- Code-focused models: 32K-50K (but with code-specific tokens)
- General-purpose: 128K (sweet spot for Llama 3, GPT-4)

### Embedding Table Size

The embedding table is the **lookup table** that maps token IDs to vector representations.

**Memory cost:**
```
embedding_params = vocab_size × embedding_dim × dtype_bytes

Example:
  vocab_size = 128,256
  embedding_dim = 4096
  dtype = FP16 (2 bytes)
  → 128,256 × 4096 × 2 = 1.05 billion parameters = 2.1 GB
  
Compare:
  vocab_size = 32,000 → 32,000 × 4096 × 2 = 0.26 billion params = 0.5 GB
```

The embedding table can be 5-20% of total model parameters, depending on architecture.

**Techniques to reduce embedding cost:**

1. **Factorized embeddings (ALBERT):**
   - Decompose embedding into V × E + E × H where E << H
   - Example: 128K × 128 + 128 × 4096 vs 128K × 4096
   - Savings: 16M + 0.5M = 16.5M vs 524M (97% reduction)

2. **Tied embeddings (Weight Tying):**
   - Share embedding weights with the output projection (pre-softmax layer)
   - Common in T5 and some LMs
   - Requires embedding dim to equal hidden dim
   - Saves 50% of the embedding + output parameter count

3. **Low-rank approximations:**
   - Use SVD or learnable low-rank projections
   - Effective but adds computational overhead

4. **Cross-layer sharing:**
   - Share embeddings across layers (used in ALBERT)

---

## Special Tokens

### BOS (Beginning of Sequence)

`<s>`, `<|begin_of_text|>`, `<bos>`

- Marks the start of a sequence
- Used in: Llama 2 (as `<s>`), Llama 3 (as `<|begin_of_text|>`)
- Optional but helps the model identify sequence boundaries
- Some models omit BOS (GPT-4) and rely on the chat template structure

### EOS (End of Sequence)

`</s>`, `<|end_of_text|>`, `<eos>`, `<|im_end|>`

- Marks the end of a sequence
- Halts generation when the model outputs this token
- Critical for: stopping generation at the right point
- Used in: all autoregressive models
- Llama 2: `</s>`, Llama 3: `<|end_of_text|>`

### PAD (Padding)

`<pad>`, `<|pad|>`

- Used to pad sequences to equal length in a batch
- Attention masks exclude PAD tokens
- Index usually 0 (but configurable)
- Not needed for generative inference (samples processed individually)

### UNK (Unknown)

`<unk>`, `<|unknown|>`

- Represents tokens not in the vocabulary
- Rare in modern models (byte-level BPE eliminates UNK)
- Still used in WordPiece (BERT) and some SentencePiece models without byte fallback

### MASK

`[MASK]`, `<mask>`

- Used in Masked Language Modeling (BERT-style pre-training)
- Represents a token to be predicted
- NOT used during fine-tuning or inference (only during pre-training)
- Special position encoding: the model learns to "fill in the blank"

### Separator Tokens

`<sep>`, `[SEP]` (BERT)
- Separates different segments in multi-segment inputs
- BERT: [CLS] sentence A [SEP] sentence B [SEP]
- The model learns segment-level relationships through attention

### Chat Templates

Chat templates encode the conversation structure into token sequences.

**Llama 2 format:**
```
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] {assistant_response}</s>
```

**Llama 3 format:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{assistant_response}<|eot_id|>
```

**ChatML (OpenAI):**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

**DeepSeek format:**
```
<User: {user_message}
Assistant: {assistant_response}
<User: {next_user_message}
Assistant:
```

**The importance of chat template consistency:**
- The chat template must match exactly what was used during training
- Mismatched templates cause quality degradation
- The template is part of the tokenization process (applied before encoding)
- Libraries like HuggingFace's `apply_chat_template()` handle this

### Tool Use Tokens

Models trained for tool use (function calling) add special tokens:

**OpenAI format:**
```
<|tool_call|>{"name": "get_weather", "arguments": {"city": "London"}}<|tool_result|>...
```

**Claude format:**
```
<function_calls>
<invoke function="get_weather">
<parameter name="city">London</parameter>
</invoke>
</function_calls>
```

**GQA (General Tool Use):**
```
<|tool|>get_weather(city="London")<|result|>...
```

These tokens help the model structure tool calls and parse tool results.

### Vision Tokens

Multimodal models integrate vision tokens into the text token stream:

**GPT-4V/GPT-4o approach:**
- Images are processed by a vision encoder (ViT-like)
- The encoder outputs a grid of visual features
- These features are projected to match the language model's embedding dimension
- Visual token IDs are injected into the text token sequence

**Vision token format:**
```
Token 1: <|image_start|>
Token 2-257: <|image_patch_0|>, <|image_patch_1|>, ..., <|image_patch_255|>
Token 258: <|image_end|>
```

**Number of vision tokens per image:**
- GPT-4o: 256-1024 tokens per image (depending on resolution)
- Llama 4: ~700 tokens per image (with dynamic resolution)
- Gemini: variable (depends on image size and encoding scheme)

**Position encoding for vision tokens:**
- Vision tokens have their own position encoding scheme
- Typically: 2D position encoding (row, column in the image grid)
- The language model attends to vision tokens through cross-attention or concatenated token sequences

---

## Tokenization Comparison Across Models

### Vocabulary Size and Tokenizer Type

| Model | Vocab Size | Algorithm | Base Units | Pre-tokenization | Byte Fallback |
|-------|-----------|-----------|------------|-----------------|---------------|
| GPT-1 | 40,000 | BPE | Unicode chars | Whitespace/punctuation | No |
| GPT-2 | 50,257 | Byte-level BPE | Bytes | None | N/A (byte-level) |
| GPT-3 | 50,257 | Byte-level BPE | Bytes | None | N/A |
| GPT-4 (cl100k) | 100,256 | Byte-level BPE | Bytes | Regex-based | N/A |
| GPT-4o (o200k) | 200,000* | Byte-level BPE | Bytes | Regex-based | N/A |
| Llama 1 | 32,000 | SentencePiece BPE | Unicode chars | Whitespace | Yes |
| Llama 2 | 32,000 | SentencePiece BPE | Unicode chars | Whitespace | Yes |
| Llama 3 | 128,256 | tiktoken BPE | Bytes | Regex-based | N/A |
| Claude 1/2 | ~32K* | SentencePiece BPE | Unicode chars | Unknown | Yes |
| Claude 3/4 | ~100K* | Byte-level BPE* | Bytes | Regex-based | N/A |
| Gemini 1.0 | 256,000 | SentencePiece Unigram | Unicode chars | None | Yes |
| Gemini 1.5 | 256,000 | SentencePiece Unigram | Unicode chars | None | Yes |
| DeepSeek-V2 | 129,280 | Byte-level BPE | Bytes | Regex-based | N/A |
| DeepSeek-V3 | 129,280 | Byte-level BPE | Bytes | Regex-based | N/A |
| Mistral 7B | 32,000 | SentencePiece BPE | Unicode chars | Whitespace | Yes |
| Mixtral | 32,000 | SentencePiece BPE | Unicode chars | Whitespace | Yes |
| Qwen | 152,000 | BPE | Unicode chars | Whitespace | Yes |
| Qwen2 | 151,936 | BPE | Unicode chars | Whitespace | Yes |
| BERT | 30,000 | WordPiece | Unicode chars | Whitespace + punct | No |
| RoBERTa | 50,000 | BPE | Unicode chars | Whitespace + punct | No |
| T5 | 32,000 | SentencePiece Unigram | Unicode chars | None | Yes |
| mT5 | 250,000 | SentencePiece Unigram | Unicode chars | None | Yes |

*Estimated values.

### Compression Ratio Comparison

Compression ratio = characters per token (higher is better).

| Language | GPT-4 (cl100k) | Llama 3 (128k) | Llama 2 (32k) | Gemini (256k) | BERT (30k) |
|----------|---------------|----------------|---------------|---------------|------------|
| English | 3.8 | 4.0 | 4.2 | 4.5 | 3.5 |
| German | 3.2 | 3.5 | 3.6 | 4.0 | 2.8 |
| French | 3.3 | 3.6 | 3.7 | 4.1 | 2.9 |
| Spanish | 3.4 | 3.7 | 3.8 | 4.2 | 3.0 |
| Chinese | 1.5 | 2.0 | 1.2 | 2.5 | 0.8 |
| Japanese | 1.8 | 2.2 | 1.3 | 2.8 | 1.0 |
| Korean | 1.7 | 2.1 | 1.2 | 2.6 | 0.9 |
| Arabic | 2.5 | 2.8 | 2.0 | 3.2 | 1.8 |
| Russian | 2.8 | 3.1 | 2.2 | 3.5 | 2.0 |
| Hindi | 2.0 | 2.4 | 1.5 | 2.8 | 1.2 |
| Code (Python) | 4.5 | 5.0 | 3.5 | 4.8 | 3.0 |
| Code (HTML) | 5.0 | 5.5 | 4.0 | 5.2 | 3.5 |

**Key observations:**
- Llama 3's 128K vocabulary provides 30-50% better compression than Llama 2's 32K for non-English
- Gemini's 256K vocabulary achieves the best compression overall, especially for CJK
- BERT's WordPiece tokenizer is the worst for CJK (approaching character-level)
- Code compression is good across most tokenizers due to repetitive patterns (indentation, keywords)

### Multilingual Performance

| Model | English | CJK | Indic | Arabic | Code Mixing |
|-------|---------|-----|-------|---------|-------------|
| GPT-4 | Excellent | Good | Fair | Good | Excellent |
| Llama 3 | Excellent | Good | Fair | Good | Excellent |
| Llama 2 | Excellent | Poor | Poor | Fair | Good |
| Gemini | Excellent | Excellent | Good | Excellent | Excellent |
| DeepSeek | Excellent | Excellent | Good | Good | Excellent |
| Qwen2 | Excellent | Excellent | Fair | Good | Good |
| BERT | Excellent | Fair | Poor | Fair | Fair |
| mT5 | Good | Good | Good | Good | Good |

---

## Tokenizer Impact on Performance

### Tokenization Effects on Reasoning

**Tokenization can directly affect reasoning quality:**

1. **Number tokenization:**
   - Multi-digit numbers should ideally be single tokens
   - GPT-4 cl100k: "12345" = 5 tokens (each digit separate!) — this is BAD for arithmetic
   - Llama 3: "12345" = 1-3 tokens (better)
   - Impact: models with per-digit number tokenization struggle with arithmetic
   - "5678 + 9876 = ?" → model must align digit-level tokens across the addition

2. **Subword splitting artifacts:**
   - "understand" = "under" + "stand" (meaningful)
   - "understand" = "und" + "erstand" (confusing)
   - Poor splits make it harder for the model to learn word semantics
   - The model must learn to compose subword representations into word meanings

3. **Sufficient statistics:**
   - Each token carries ~2.8 bits of information (empirical)
   - Subword splitting means the model has fewer "information-dense" units
   - This can reduce reasoning efficiency (more tokens to process for same information)

### Multilingual Capability

**Tokenization is the primary bottleneck for multilingual performance:**

- **English bias:** Most tokenizers are trained on English-dominated corpora, so English gets the best compression
- **Token count disparity:** The same paragraph translated to different languages uses very different token counts:
  - English: 100 tokens
  - Chinese: 250 tokens (GPT-4)
  - Korean: 200 tokens
  - This means Chinese text fills up context 2.5× faster

**Quantified bias (GPT-4 cl100k):**
| Language | Tokens per 100 chars | Relative to English |
|----------|---------------------|-------------------|
| English | 26.3 | 1.0× |
| German | 31.2 | 1.19× |
| French | 30.3 | 1.15× |
| Spanish | 29.4 | 1.12× |
| Chinese | 66.7 | 2.53× |
| Japanese | 55.6 | 2.11× |
| Korean | 58.8 | 2.24× |
| Arabic | 40.0 | 1.52× |
| Hindi | 50.0 | 1.90× |

This 2-2.5× penalty for CJK languages means:
- A 128K context window effectively becomes 50K for Chinese
- Chinese text costs more (more tokens → more compute)
- Chinese conversations hit the context limit faster

### Code Understanding

**Tokenization for code:**

1. **Whitespace handling (Python):**
   - Indentation is semantically meaningful in Python
   - Tokenizers that collapse whitespace (SentencePiece default) break Python
   - Solution: custom pre-tokenization rules for code
   - Llama 3's tokenizer handles this well

2. **Operator tokenization:**
   - "==" should ideally be a single token (meaning "equals")
   - "==" = "=" + "=" (two tokens) loses the semantic unit
   - "->" should be single token in Python function annotations
   - "!=" as single token helps with code understanding

3. **Keyword tokenization:**
   - "def", "class", "if", "for", "while" should ideally be single tokens
   - Most modern tokenizers handle this
   - Rare keywords in less common languages (e.g., "delegate" in C#) may be split

4. **String literal tokenization:**
   - Multi-line strings in Python / template literals in JS
   - Tokenizers must handle these without losing structure
   - Byte-level tokenizers handle this better than character-level

**Code tokenization comparison (100 chars of Python):**
```
GPT-4: 22 tokens
Llama 3: 20 tokens
Llama 2: 29 tokens
Gemini: 21 tokens
```

### Bias in Tokenization

**Tokenization can introduce or amplify biases:**

1. **Name tokenization bias:**
   - Common English names: single token (James, Mary, John)
   - Non-English names: multiple tokens (Muhammad → Muh + ammad, Wei → W + ei)
   - This can create subtle bias — the model perceives familiar names as "simpler"

2. **Sociolect bias:**
   - Different dialects tokenize differently
   - AAVE (African American Vernacular English) may tokenize more poorly than Standard English
   - This can disadvantage speakers of non-standard dialects

3. **Gender representation:**
   - "he" is typically a single token, "she" also single
   - But "doctor" + "he" and "nurse" + "she" are statistical associations
   - Tokenization doesn't directly create gender bias, but can amplify it through subword associations

4. **Domain bias:**
   - Legal/medical/scientific jargon that's common in training data gets single tokens
   - Less common terms get split, potentially making them harder to learn

### Token Bleeding

**Token bleeding** occurs when the tokenizer splits a word in a way that "bleeds" information across boundaries.

**Examples:**
- "unethical" → "un" + "ethical" (good — preserves meaning)
- "unethical" → "uneth" + "ical" (bad — loses meaning, creates artifacts)
- "restaurant" → "rest" + "aurant" (bad — "rest" has different meaning)
- "tokenization" → "token" + "ization" (good — preserves root)
- "tokenization" → "tok" + "enization" (bad — creates confusing prefix)

**Impact:**
- The model must learn to "de-bleed" — compose the actual meaning from subwords
- More training data is needed to overcome bad tokenization
- Token bleeding reduces the model's ability to handle novel words

### Subword Splitting Artifacts

Specific artifacts caused by subword splitting:

1. **Word boundary confusion:**
   - "New York" → "New" + "York" (two tokens, potentially losing the compound meaning)
   - "New York" → "New York" (single token — ideal, preserves compound meaning)

2. **Punctuation splitting:**
   - "don't" → "don" + "'t" (good for grammar)
   - "don't" → "do" + "n't" (alternative, also common)
   - "we'll" → "we" + "'ll" vs "well" (same tokens, different meaning!)

3. **Case sensitivity:**
   - "USA" → "US" + "A" (abbreviation split, meaning preserved)
   - "Usa" → "Us" + "a" (different meaning entirely!)

4. **Number splitting:**
   - "2024" → "20" + "24" (loses the semantic unit)
   - "3.14" → "3" + "." + "14" (mathematical constant split)

---

## Multilingual Tokenization Challenges

### Character-Based vs Subword

**Character-based tokenization:**
- Each character (Unicode code point) is a token
- Pros: complete coverage, no UNK tokens, fair across languages
- Cons: very long sequences (e.g., "Hello" → 5 tokens, "你好" → 2 tokens)
- Used in: character-level models (very rare, mostly for specialized tasks)

**Subword tokenization:**
- Common substrings become tokens
- Pros: efficient for common patterns, balanced across languages
- Cons: language bias, subword artifacts
- Used in: all modern LLMs

**Character vs subword comparison:**
```
English: "Hello" → 5 characters vs 1-2 subwords
Chinese: "你好" → 2 characters vs 1-2 subwords
Thai: "สวัสดี" → 6 characters vs 2-3 subwords
Arabic: "السلام" → 4 characters vs 2 subwords
```

### Byte Fallback Mechanisms

Byte fallback ensures that any Unicode string can be represented:

**How byte fallback works:**
1. Normalize text (NFKC)
2. Attempt to encode using learned subword vocabulary
3. For any remaining characters, split into UTF-8 bytes
4. Map each byte to a special `<0xNN>` token
5. The model learns to interpret byte sequences

**Byte fallback token:**
```
"😀" (U+1F600 — GRINNING FACE)
→ UTF-8: F0 9F 98 80 (4 bytes)
→ Tokens: <0xF0>, <0x9F>, <0x98>, <0x80>
→ 4 tokens for a single emoji!
```

**Trade-offs:**
- Byte fallback guarantees zero UNK tokens
- But rare characters use many tokens (4 per emoji, 3 per CJK character)
- The model must learn to interpret byte sequences, which is inefficient
- Byte fallback is a "safety net" — ideally, most characters are covered by the vocabulary

### Language Mixing

Mixed-language text (code-switching) is challenging for tokenizers:

1. **Segmentation ambiguity:**
   - "Je vais au supermarket" (French + English)
   - Should "supermarket" be a French or English token?
   - The tokenizer decides based on which merge pattern wins

2. **Vocabulary overlap:**
   - Many words overlap across languages (false friends, cognates, loanwords)
   - "table" exists in English and French with different meanings
   - The tokenizer assigns a single token, losing the language distinction

3. **Script mixing:**
   - "Hello世界" (Latin + CJK)
   - The tokenizer must handle both scripts
   - Byte-level tokenizers handle this naturally (both are just bytes)
   - Character-level tokenizers may struggle (different base vocabularies)

4. **Transliterated text:**
   - "nihao" (pinyin for 你好)
   - Tokenizer may treat this as English or Chinese depending on frequency

### Tokenizer Bias Towards English

**Why English dominates:**

1. **Training data distribution:**
   - Most web text is English (~50% of CommonCrawl)
   - Even "multilingual" corpora are English-heavy
   - BPE merges favor English patterns first

2. **Token position in vocabulary:**
   - English words are low-numbered token IDs
   - Rare language tokens have high IDs
   - This could affect the model's preference for English tokens

3. **Compression ratio disparity:**
   - English: 4 chars/token
   - Chinese: 1.5 chars/token
   - This means Chinese text uses 2.7× more tokens for the same content
   - The model has fewer opportunities to "see" Chinese patterns

**Mitigation strategies:**
- **Language-balanced tokenizer training:** Sample data proportionally by language
- **Language-specific vocabularies:** Add dedicated tokens for each language's common subwords
- **Byte-fallback:** Less biased, but less efficient per-token
- **Separate tokenizers per language:** Rare but possible for specialized multilingual systems

---

## Tokenization and Context Length

### How Tokenizer Affects Effective Context

The relationship between text length and token count is determined by the tokenizer's compression ratio.

**Effective context formula:**
```
effective_context = nominal_context / (chars_per_token / baseline_chars_per_token)

Where baseline is usually English at ~4 chars/token
```

**Example:**
- Model with 128K token context limit
- GPT-4 tokenizer: English (4 chars/tok) → 128K effective context for English
- GPT-4 tokenizer: Chinese (1.5 chars/tok) → 128K × (1.5/4) = 48K effective context for Chinese

**Implications:**
- Multilingual applications need to account for this disparity
- The "128K context" claim only applies to English
- Non-English languages get 30-50% of the advertised context

### Token-to-Character Ratio Across Languages

Language-specific ratios for common tokenizers:

**GPT-4 (cl100k_base):**
```
English:      1 token ≈ 3.8 characters
Chinese:      1 token ≈ 1.5 characters (2.5× worse)
Japanese:     1 token ≈ 1.8 characters (2.1× worse)
Korean:       1 token ≈ 1.7 characters (2.2× worse)
Arabic:       1 token ≈ 2.5 characters (1.5× worse)
Hindi:        1 token ≈ 2.0 characters (1.9× worse)
Thai:         1 token ≈ 1.8 characters (2.1× worse)
Russian:      1 token ≈ 2.8 characters (1.4× worse)
```

**Llama 3 (128K vocabulary, improved):**
```
English:      1 token ≈ 4.0 characters
Chinese:      1 token ≈ 2.0 characters (2.0× worse)
Japanese:     1 token ≈ 2.2 characters (1.8× worse)
Korean:       1 token ≈ 2.1 characters (1.9× worse)
Arabic:       1 token ≈ 2.8 characters (1.4× worse)
```

**Real-world implications:**
- Translating English content to Chinese requires 2× the tokens
- The model's context window fills up twice as fast
- For multilingual customer support, budget for 2× more tokens

---

## tiktoken Library Internals

Tiktoken is OpenAI's **fast open-source tokenization library**, written in Rust with Python bindings.

**Architecture:**
```
Python API (tiktoken module)
    ↓
Rust core (tokenization engine)
    ↓
C FFI (bridging Rust and Python)
```

**Key design decisions:**

1. **Rust core for speed:**
   - BPE merges are applied via a **trie** (prefix tree) for O(length) lookups
   - Regex pre-tokenization uses the `regex` crate (Rust's regex library)
   - Memory: vocabulary is stored in a compact hash map

2. **Byte-level BPE:**
   - UTF-8 bytes are the fundamental unit
   - No normalization step (text is used as-is)
   - Regex splits the text before BPE

3. **Rank-based BPE:**
   - Each token has a "rank" (its merge order in training)
   - Lower rank = earlier merge = more preferred
   - Tokenization finds the segmentation with the lowest total rank

**Core data structures:**

1. **Encoder:**
   - `special_tokens`: Map from token string to ID (for special tokens)
   - `token_byte_values`: Map from token ID to UTF-8 bytes
   - `mergeable_ranks`: Map from bytes to rank (for BPE)
   - `pattern`: Regex string for pre-tokenization

2. **Trie for byte encoding:**
   - Characters are encoded via a byte-level trie
   - Each node represents a (byte, rank) pair
   - The trie supports longest-match-first search

**Encoding algorithm (simplified):**
```
def encode(text, encoder):
    tokens = []
    for chunk in regex_findall(text, encoder.pattern):
        # Convert chunk to bytes
        chunk_bytes = chunk.encode('utf-8')
        
        # Apply BPE merges using the trie
        piece_tokens = []
        i = 0
        while i < len(chunk_bytes):
            # Find longest matching token
            longest_match = None
            for j in range(i, len(chunk_bytes)):
                prefix = chunk_bytes[i:j+1]
                if prefix in encoder.mergeable_ranks:
                    longest_match = (prefix, j-i+1)
            if longest_match:
                piece_tokens.append(longest_match)
                i += longest_match[1]
            else:
                # Should never happen (byte-level coverage)
                piece_tokens.append((chunk_bytes[i:i+1], 1))
                i += 1
        
        # Convert piece tokens to IDs
        for piece, _ in piece_tokens:
            tokens.append(encoder.mergeable_ranks[piece])
    
    return tokens
```

**Decoding algorithm:**
```
def decode(token_ids, encoder):
    byte_parts = []
    for tid in token_ids:
        if tid == encoder.eos_token_id:
            break
        if tid in encoder.token_byte_values:
            byte_parts.append(encoder.token_byte_values[tid])
        elif tid in encoder.special_tokens:
            # Handle special tokens
            pass
    
    return bytes(byte_parts).decode('utf-8', errors='replace')
```

**Available encodings:**
- `cl100k_base`: GPT-4, GPT-3.5-Turbo, text-embedding-ada-002
- `p50k_base`: Codex models, text-davinci-002, text-davinci-003
- `p50k_edit`: For edit models (code-davinci-edit)
- `r50k_base`: GPT-3 (Davinci)
- `o200k_base`: GPT-4o (newer, larger vocabulary)

**Usage (Python):**
```python
import tiktoken

# Load encoding
enc = tiktoken.get_encoding("cl100k_base")

# Encode
tokens = enc.encode("Hello, world!")
print(tokens)  # [15496, 11, 995, 0]

# Decode
text = enc.decode(tokens)
print(text)  # "Hello, world!"

# Count tokens (for context management)
n_tokens = len(enc.encode("Long text here..."))

# With special tokens
tokens = enc.encode_with_special("<|im_start|>user\nHello<|im_end|>")
```

**Performance:**
- Encoding speed: ~50 MB/s (single core)
- Decoding speed: ~100 MB/s
- Memory: ~100 MB for cl100k_base
- Concurrency: thread-safe (read-only encoder)

---

## HuggingFace Tokenizers Library

The HuggingFace `tokenizers` library provides a **fast, production-ready** tokenizer implementation in Rust with Python bindings.

**Key features:**
- BPE, WordPiece, Unigram, and Byte-level BPE implementations
- Multi-threaded training
- Pre-tokenization with regex
- Post-processing (adding special tokens)
- Full serialization/deserialization
- Integrates with the HuggingFace Transformers library

**Training a tokenizer (Python):**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

# Create tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Normalization
tokenizer.normalizer = NFKC()

# Pre-tokenization
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

# Trainer configuration
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Train on files
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# Post-processing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
)

# Save and load
tokenizer.save("tokenizer.json")
tokenizer = Tokenizer.from_file("tokenizer.json")
```

**Training with multiple workers:**
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
trainer = tokenizer.train(
    files=["data.txt"],
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    show_progress=True
)
# Multi-threading: uses all available cores by default
```

**Unigram tokenizer training:**
```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

tokenizer = Tokenizer(Unigram())
trainer = UnigramTrainer(
    vocab_size=32000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"],
    unk_token="[UNK]"
)

files = ["corpus.txt"]
tokenizer.train(files, trainer)
```

**WordPiece tokenizer training:**
```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
trainer = WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2,
    limit_alphabet=1000  # controls character vocabulary growth
)
```

**Decoding with HuggingFace tokenizers:**
```python
# Decode token IDs back to text
decoded = tokenizer.decode(ids, skip_special_tokens=True)

# Batch decoding
decoded_batch = tokenizer.decode_batch([ids_1, ids_2])

# With token offsets (for span alignment)
encoding = tokenizer.encode("Hello world!")
print(encoding.offsets)
# [(0, 5), (5, 6), (6, 11)]  # character offsets for each token
```

---

## Adding New Tokens to Existing Tokenizers

### Why Add New Tokens?

1. **Domain-specific vocabulary:** Medical, legal, scientific, or technical terminology
2. **New languages:** Adding support for languages not well-covered
3. **Custom special tokens:** For tool use, image processing, or application-specific features
4. **Fixing poor tokenization:** Ensuring important terms stay as single tokens

### Embedding Initialization for New Tokens

When adding tokens to an existing model, the new token embeddings must be properly initialized:

**Mean initialization:**
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModel.from_pretrained("model_name")

# Get existing embedding matrix
old_embeddings = model.get_input_embeddings().weight.data

# Calculate mean of similar existing tokens
def get_similar_tokens_embedding(new_token):
    # Find existing tokens that are substrings of the new token
    similar_ids = []
    for i in range(tokenizer.vocab_size):
        existing = tokenizer.decode([i])
        if existing in new_token or new_token in existing:
            similar_ids.append(i)
    
    if similar_ids:
        # Initialize as mean of similar existing tokens
        return old_embeddings[similar_ids].mean(dim=0)
    else:
        # Fall back to mean of all embeddings
        return old_embeddings.mean(dim=0)

# Resize token embeddings
new_token = "[NEW_TOKEN]"
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))

# Initialize the new token's embedding
new_id = tokenizer.convert_tokens_to_ids(new_token)
with torch.no_grad():
    model.get_input_embeddings().weight[new_id] = get_similar_tokens_embedding(new_token)
```

**Alternative initialization strategies:**

1. **Random initialization:**
   - Sample from N(0, 0.02) (matching typical embedding initialization)
   - Simple but may require fine-tuning to stabilize

2. **Token decomposition:**
   - If the new token would normally be split (e.g., "machinelearning" → "machine" + "##learning")
   - Initialize as the average of the sub-token embeddings
   - This preserves information from the existing tokenizer

3. **Trainable initialization:**
   - Add a small MLP that predicts embeddings for new tokens
   - Train the MLP on a small corpus while keeping the base model frozen
   - More expensive but potentially better quality

### Expansion for Fine-tuning

When fine-tuning a model with new tokens:

```python
# Step 1: Add tokens to tokenizer
num_added = tokenizer.add_tokens(new_tokens_list)
print(f"Added {num_added} tokens")

# Step 2: Resize model embeddings
model.resize_token_embeddings(len(tokenizer))
# This adds a new embedding matrix row (or rows)
# The new rows are randomly initialized by default

# Step 3: (Optional) Initialize embeddings thoughtfully
# See above for initialization strategies

# Step 4: Ensure the output projection is also resized
# (model.resize_token_embeddings handles both input and output)
```

**Important considerations:**
- Resizing embeddings adds parameters to the model — this slightly increases memory and compute
- The output projection (lm_head) is also resized (typically tied weights or separate)
- Fine-tuning is required for the model to learn the new tokens' meanings
- A few hundred new tokens usually don't degrade existing performance
- Thousands of new tokens may require careful learning rate scheduling

---

## Tokenizer Merging

### Why Merge Tokenizers?

1. **Adding new languages:** Merge a monolingual model's tokenizer with a language-specific tokenizer
2. **Domain-specific extensions:** Add medical/legal vocabulary to an existing tokenizer
3. **Combining strengths:** Use one tokenizer's compression with another's coverage

### Tokenizer Merging Techniques

**Union merging:**
Take the union of both tokenizers' vocabularies.

```python
def merge_tokenizers_vocab(tokenizer_a, tokenizer_b):
    """Merge two tokenizers by taking the union of their vocabularies."""
    vocab_a = set(tokenizer_a.get_vocab().keys())
    vocab_b = set(tokenizer_b.get_vocab().keys())
    
    merged_vocab = vocab_a.union(vocab_b)
    new_tokenizer = tokenizer_a.clone()
    
    # Add new tokens from B that aren't in A
    new_tokens = list(vocab_b - vocab_a)
    new_tokenizer.add_tokens(new_tokens)
    
    return new_tokenizer
```

**Intersection merging:**
Keep only tokens present in both tokenizers (rarely useful).

**Priority merging:**
When conflicts arise, prefer one tokenizer over the other (based on merge order).

**Rank-based merging:**
- Each tokenizer has merge ranks
- When merging, reconcile ranks by assigning new ranks based on priority
- Lower rank = higher priority in the merged tokenizer

### Challenges in Tokenizer Merging

1. **Incompatible tokenization algorithms:**
   - BPE vs Unigram can't be directly merged
   - Byte-level BPE vs character-level BPE have different base units
   - Solution: align representations or use byte-level as common ground

2. **ID collision:**
   - Token "the" may be ID 5 in tokenizer A and ID 123 in tokenizer B
   - The merged tokenizer must assign a single ID per token
   - Requires re-indexing all vocabulary items

3. **Merge rule compatibility:**
   - BPE tokenizers have ordered merge rules
   - Merging two sets of merge rules may create contradictions
   - Solution: retrain a tokenizer on the union vocabulary

4. **Model compatibility:**
   - The model's embedding table must be expanded for all new tokens
   - Existing token IDs shift if merging changes the vocabulary ordering
   - This breaks the pre-trained model (embeddings are tied to specific IDs)

**Best practice:** Instead of merging tokenizers after training, design a single tokenizer that covers all desired use cases before model training. Merging post-training is fragile.

---

## Custom Tokenizer Creation

### Domain-Specific Tokenizer Design

**When to create a custom tokenizer:**

1. **Highly specialized domain:** Medical, legal, scientific, programming language
   - Standard tokenizers waste tokens on general-purpose subwords
   - Domain vocabulary is not well-represented
2. **New language or script:** Low-resource language not covered by existing tokenizers
3. **Specialized format:** Code, mathematics, DNA sequences, log data

**Design process:**

1. **Collect domain corpus:** 1-10GB of representative text
2. **Analyze tokenization of existing tokenizers:**
   - How many tokens does your domain text use?
   - What terms are being split that shouldn't be?
   - What's the OOV (out-of-vocabulary) rate?
3. **Define target vocabulary size:**
   - Larger for broader coverage (128K+)
   - Smaller for efficiency on limited hardware (32K)
4. **Choose algorithm:**
   - BPE: Good for most use cases, fast to train
   - Unigram: Better for multilingual, probabilistic
   - Byte-level BPE: Best for universal coverage
5. **Train and evaluate:**
   - Train on domain corpus
   - Measure compression ratio, coverage, token count distribution
   - Test on held-out data

**Code example: Training a custom BPE tokenizer**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.processors import TemplateProcessing
import json

# Build domain-specific tokenizer for medical text
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Custom normalization for medical text
tokenizer.normalizer = Lowercase()

# Byte-level pre-tokenization (handles Unicode)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

# Trainer with domain-specific parameters
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=5,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    initial_alphabet=ByteLevel.alphabet(),
    limit_alphabet=1000,
)

# Train on domain corpus
files = [
    "medical_papers.txt",
    "clinical_notes.txt",
    "drug_descriptions.txt",
]
tokenizer.train(files, trainer)

# Add post-processing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Evaluate
test_text = "The patient presented with hypertension and hyperlipidemia."
encoded = tokenizer.encode(test_text)
print(f"Original: {len(test_text)} chars, {len(encoded.tokens)} tokens")
print(f"Tokens: {encoded.tokens}")

# Save
tokenizer.save("medical_tokenizer.json")
```

### Evaluating Custom Tokenizers

**Metrics to evaluate:**

1. **Compression ratio:**
   - chars_per_token (CPT) = total_chars / total_tokens
   - Higher is better (fewer tokens for same content)

2. **Coverage:**
   - Percentage of text that can be tokenized without UNK or byte fallback
   - >99.9% is expected for a good tokenizer

3. **Token length distribution:**
   - Ideally: few very short tokens (1-2 chars), most medium tokens (3-6 chars), some long tokens (7-15 chars)
   - Very long tokens (>15 chars) are usually not useful

4. **Vocabulary utilization:**
   - Percentage of tokens that actually appear in the corpus
   - <50% utilization means the vocabulary is too large

5. **Mean token frequency:**
   - Average number of times each token appears in the corpus
   - Very rare tokens (appearing <5 times) are candidates for removal

6. **Subword regularity:**
   - Domain-specific terms should be single tokens
   - Measure: percentage of domain terms that are single tokens

**Evaluation script:**
```python
def evaluate_tokenizer(tokenizer, corpus_path, output_path=None):
    total_chars = 0
    total_tokens = 0
    token_freqs = defaultdict(int)
    unknown_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            encoded = tokenizer.encode(line)
            total_chars += len(line)
            total_tokens += len(encoded.ids)
            
            for token, id in zip(encoded.tokens, encoded.ids):
                token_freqs[token] += 1
                if token == tokenizer.unk_token:
                    unknown_count += 1
    
    # Metrics
    cpt = total_chars / total_tokens
    unknown_rate = unknown_count / total_tokens
    vocab_used = len(token_freqs)
    vocab_total = tokenizer.get_vocab_size()
    utilization = vocab_used / vocab_total * 100
    
    # Token length distribution
    token_lengths = [len(t) for t in token_freqs.keys()]
    
    results = {
        "chars_per_token": cpt,
        "unknown_rate": unknown_rate,
        "vocab_utilization": utilization,
        "total_tokens": total_tokens,
        "unique_tokens_used": vocab_used,
        "avg_token_length": sum(token_lengths) / len(token_lengths),
        "max_token_length": max(token_lengths),
        "min_token_length": min(token_lengths),
    }
    
    return results
```

---

## Advanced Topics

### Tokenization for Different Model Architectures

**Encoder-only (BERT):**
- Uses WordPiece with [CLS], [SEP], [MASK] special tokens
- Vocabulary typically 30K-50K
- Fixed context length (usually 512)
- Pre-tokenization: basic whitespace + punctuation splitting
- BERT tokenizer adds special tokens during tokenization, not in post-processing

**Decoder-only (GPT, Llama, Mistral):**
- Byte-level BPE or SentencePiece
- Larger vocabulary (32K-128K)
- Variable context length
- Chat template integration
- Special tokens usually handled outside the tokenizer (in the chat template formatting)

**Encoder-Decoder (T5):**
- SentencePiece Unigram
- Vocabulary: 32K-250K tokens
- Additional sentinel tokens for span corruption pre-training
- T5 uses `<extra_id_0>` through `<extra_id_99>` for corruption spans
- The sentinel tokens are single token IDs, not generated by the tokenizer

### Tokenization in Multimodal Models

**Vision tokens:**
- Images are split into patches (e.g., 16×16 pixel patches)
- Each patch is encoded into a visual feature vector
- Visual features are projected to the language model's token embedding space
- These "vision tokens" are treated like text token IDs

**Audio tokens:**
- Raw audio is encoded into discrete tokens using a neural codec (e.g., EnCodec, SoundStream, HuBERT)
- Typically 50-100 audio tokens per second of audio
- Each audio token is ∈ [0, 32000) (depending on codebook size)
- Audio tokens are interleaved or concatenated with text tokens

**Video tokens:**
- Video is processed as frame sequences
- Each frame is encoded like an image (patch tokens)
- Temporal tokens may be added to encode motion/frame order
- A 1-minute video at 1 FPS generates ~60×N patches of tokens

### Ethical Considerations in Tokenization

1. **Linguistic equity:**
   - Tokenizers that favor English perpetuate digital inequality
   - Non-English speakers pay more (in tokens) for the same content
   - Solutions: multilingual tokenizers, language-specific models

2. **Name-based bias:**
   - Common English names are single tokens; non-English names split
   - This can affect model behavior for named entity recognition and bias
   - Mitigation: ensure training data includes diverse names

3. **Dialect and sociolect inclusion:**
   - Non-standard dialects are often poorly tokenized
   - This can lead to worse model performance for minority speakers
   - Mitigation: include diverse text sources in tokenizer training

4. **Domain exclusion:**
   - Specialized domains (indigenous knowledge, traditional medicine) may be poorly tokenized
   - This reinforces the digital divide between knowledge systems

### Future Directions in Tokenization

1. **Adaptive tokenization:**
   - Tokenizers that adjust to the input domain dynamically
   - Multiple vocabularies for different modalities/languages
   - Early work: Adaptive BPE, dynamic vocabulary expansion

2. **End-to-end learned tokenization:**
   - Neural tokenizers trained jointly with the language model
   - No fixed vocabulary — tokens are latent representations
   - Examples: CANINE (character-level encoder without explicit tokenization), ByT5 (byte-level T5)

3. **Semantic tokenization:**
   - Tokens based on meaning rather than frequency
   - Morpheme-level or concept-level units
   - Could improve multilingual performance and reduce token count

4. **Hardware-aware tokenization:**
   - Designing vocabularies that align with GPU compute patterns
   - Token count optimization for specific hardware topologies
   - Variable-bit token representations

5. **Compression-optimized tokenization:**
   - Using compression algorithms (zstd, gzip) as inspiration
   - Arithmetic coding-like tokenization for optimal compression
   - Trade-off: improved compression vs computational overhead

---

*This document provides comprehensive coverage of tokenization concepts, algorithms, and practical considerations. Understanding tokenization is fundamental to working effectively with Large Language Models — it affects every aspect of model behavior, from reasoning to multilingual capability to computational efficiency.*
