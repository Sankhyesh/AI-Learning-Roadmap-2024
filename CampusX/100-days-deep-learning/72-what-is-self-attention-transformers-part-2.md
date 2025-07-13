# What is Self Attention - Transformers Part 2

## Introduction

**Paper References:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper by Vaswani et al.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar's visual guide

**Video Context:** This video is the first part of a three-part series on self-attention mechanism, which is the foundational building block of Transformers. Understanding self-attention deeply is crucial for mastering Transformers and modern LLMs.

**Learning Journey:** By the end of this video, you'll understand what self-attention is and why it's needed. The subsequent parts will cover how it works and the detailed geometry behind it.

**Connection to Broader Concepts:** Self-attention is the key innovation that powers Transformers, which in turn are the foundation of all modern language models including GPT, BERT, and other LLMs.

---

## The Fundamental NLP Challenge: Converting Words to Numbers

Before diving into self-attention, let's understand the core requirement of any NLP application. Whether you're building sentiment analysis, named entity recognition, or machine translation systems, the first and most critical step is **converting words to numbers** (vectorization).

```mermaid
graph LR
    A[Text Input] --> B[Vectorization]
    B --> C[Numbers/Vectors]
    C --> D[ML Model]
    D --> E[NLP Output]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#9ff,stroke:#333,stroke-width:2px
```

Computers only understand numbers, not words. So the evolution of NLP has been largely about finding better ways to represent text as numbers.

## Evolution of Text Representation Techniques

### 1. One-Hot Encoding (The Beginning)

Let's say we have two sentences:
- "mat cat mat"
- "cat rat rat"

First, we identify unique words: mat, cat, rat

```mermaid
graph TD
    subgraph "Vocabulary"
        V1[mat - Position 0]
        V2[cat - Position 1]
        V3[rat - Position 2]
    end
    
    subgraph "One-Hot Vectors"
        M[mat = 1,0,0]
        C[cat = 0,1,0]
        R[rat = 0,0,1]
    end
    
    V1 --> M
    V2 --> C
    V3 --> R
```

So "mat cat mat" becomes: [1,0,0] [0,1,0] [1,0,0]

**Problem:** This representation is inefficient and doesn't capture any semantic meaning.

### 2. Bag of Words (Slight Improvement)

Bag of Words counts the frequency of each word:

```python
# For "mat cat mat"
vector = [2, 1, 0]  # mat appears 2x, cat 1x, rat 0x

# For "cat rat rat"
vector = [0, 1, 2]  # mat 0x, cat 1x, rat 2x
```

### 3. TF-IDF (Further Refinement)

TF-IDF adds importance weighting to word frequencies, but still lacks semantic understanding.

### 4. Word Embeddings (The Game Changer)

Word embeddings revolutionized NLP by capturing **semantic meaning**. They convert words into dense vectors where similar words have similar vectors.

```mermaid
graph TD
    subgraph "Training Process"
        A[Large Text Corpus<br/>Wikipedia, Books, etc.] --> B[Neural Network]
        B --> C[Learn Context & Meaning]
        C --> D[Generate Embeddings]
    end
    
    subgraph "Result: Word Vectors"
        D --> E[king = 0.6, 0.2, 0.1, 0.9, 0.3]
        D --> F[queen = 0.6, 0.2, 0.1, 0.8, 0.4]
        D --> G[cricket = 0.1, 0.9, 0.8, 0.2, 0.1]
    end
    
    style A fill:#ffd,stroke:#333,stroke-width:2px
    style D fill:#dff,stroke:#333,stroke-width:2px
```

## The Power and Limitation of Word Embeddings

### Power: Semantic Similarity

Word embeddings capture semantic relationships. In a 5-dimensional embedding space (simplified for illustration):

```mermaid
graph LR
    subgraph "Semantic Space"
        K[King<br/>High Royalty<br/>Low Athletics]
        Q[Queen<br/>High Royalty<br/>Low Athletics]
        C[Cricketer<br/>Low Royalty<br/>High Athletics]
        
        K -.similar.- Q
        K -.different.- C
        Q -.different.- C
    end
```

Each dimension captures some aspect:
- Dimension 1: Royalty (high for king/queen, low for cricketer)
- Dimension 2: Athleticism (low for king/queen, high for cricketer)
- Dimension 3: Human-ness (high for all three)

### Limitation: Static Nature

Word embeddings capture **average meaning** across the training data, not context-specific meaning.

```mermaid
graph TD
    subgraph "Training Data Distribution"
        A[10,000 sentences total]
        B[9,000 sentences:<br/>Apple as fruit]
        C[1,000 sentences:<br/>Apple as company]
        A --> B
        A --> C
    end
    
    subgraph "Resulting Embedding"
        D[Apple = 0.9, 0.1<br/>High taste, Low tech]
    end
    
    B --> D
    C --> D
    
    style B fill:#faa,stroke:#333,stroke-width:3px
    style C fill:#aaf,stroke:#333,stroke-width:1px
```

## The Problem: Context Matters!

Consider this translation task:
**English:** "Apple launched a new phone while I was eating an orange"
**Task:** Translate to Hindi

```mermaid
graph TB
    subgraph "Static Embeddings Problem"
        A[Apple = 0.9, 0.3<br/>Static: mostly fruit]
        B[In this context:<br/>Apple is a tech company!]
        C[Wrong translation:<br/>फल ने फोन लॉन्च किया]
        
        A --> B
        B --> C
    end
    
    style A fill:#faa,stroke:#333,stroke-width:2px
    style C fill:#f66,stroke:#333,stroke-width:2px
```

The static embedding assumes "Apple" is mostly used as a fruit, but in this sentence, it's clearly the technology company!

## The Solution: Self-Attention

Self-attention solves this by creating **contextual embeddings** - embeddings that change based on the surrounding words.

```mermaid
graph LR
    subgraph "Self-Attention Mechanism"
        A[Static Embeddings<br/>Apple=0.9,0.3<br/>launched=...<br/>phone=...] --> B[Self-Attention<br/>Function]
        B --> C[Contextual Embeddings<br/>Apple=0.3,0.9<br/>Tech context!]
    end
    
    style A fill:#fdd,stroke:#333,stroke-width:2px
    style B fill:#dfd,stroke:#333,stroke-width:3px
    style C fill:#ddf,stroke:#333,stroke-width:2px
```

### How Self-Attention Creates Contextual Understanding

```mermaid
sequenceDiagram
    participant S as Sentence
    participant SA as Self-Attention
    participant CE as Contextual Embeddings
    
    S->>SA: Apple [0.9, 0.3]
    S->>SA: launched [...]
    S->>SA: phone [...]
    S->>SA: orange [...]
    
    Note over SA: Analyzes relationships<br/>"launched" + "phone"<br/>suggests tech context
    
    SA->>CE: Apple [0.3, 0.9] ← Tech!
    SA->>CE: launched [...] ← Action
    SA->>CE: phone [...] ← Product
    SA->>CE: orange [...] ← Fruit
```

The self-attention mechanism looks at all words in the sentence and adjusts each word's embedding based on its context. It's smart enough to:
- Recognize "launched" and "phone" indicate a technology context
- Adjust Apple's embedding to reflect company meaning
- Keep "orange" as a fruit despite "Apple" being in the same sentence

## Visual Summary: Static vs Contextual Embeddings

```mermaid
graph TD
    subgraph "Traditional Approach"
        A1[Word] --> B1[Static Embedding]
        B1 --> C1[Same vector<br/>everywhere]
    end
    
    subgraph "Self-Attention Approach"
        A2[Word + Context] --> B2[Self-Attention]
        B2 --> C2[Dynamic vector<br/>based on context]
    end
    
    style C1 fill:#faa,stroke:#333,stroke-width:2px
    style C2 fill:#afa,stroke:#333,stroke-width:2px
```

## Key Takeaways

- **Word embeddings** are powerful but static - they represent average meaning
- **Context matters** - the same word can have different meanings in different sentences
- **Self-attention** is a mechanism that converts static embeddings to contextual embeddings
- **Contextual embeddings** understand word meaning based on surrounding words
- This is the foundation that makes Transformers so powerful for NLP tasks

## What's Next?

In the next video, we'll dive into **how** self-attention actually works:
- Query, Key, and Value vectors
- The attention calculation process
- How context influences the final embeddings

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need**. In Advances in neural information processing systems (pp. 5998-6008).

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). **BERT: Pre-training of deep bidirectional transformers for language understanding**. arXiv preprint arXiv:1810.04805.

[End of Notes]