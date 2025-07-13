# Day 76: Why is Self-Attention Called "Self"? - Self-Attention vs Luong Attention

## Paper References
- **Attention Is All You Need** - Vaswani et al., 2017
  - [Original Paper](https://arxiv.org/abs/1706.03762)
  - Introduces the self-attention mechanism in Transformers
- **Effective Approaches to Attention-based Neural Machine Translation** - Luong et al., 2015
  - [Original Paper](https://arxiv.org/abs/1508.04025)
  - Describes Luong attention mechanism
- **Neural Machine Translation by Jointly Learning to Align and Translate** - Bahdanau et al., 2014
  - [Original Paper](https://arxiv.org/abs/1409.0473)
  - Introduces attention mechanism in NMT

## Introduction: The Essential Question

Today's video addresses a fundamental yet often overlooked question: Why is self-attention called "self"? This isn't just academic curiosity - it's a crucial interview question and a gateway to deeply understanding the attention mechanism family. The instructor emphasizes that before diving into transformer architecture, we must sharpen our understanding of self-attention, just as one would sharpen an axe for four hours before spending two hours cutting a tree.

**Learning Journey**: We'll first understand why self-attention qualifies as an attention mechanism despite looking quite different from Bahdanau or Luong attention, then explore what makes it "self" attention specifically.

```mermaid
graph TD
    subgraph "Today's Learning Path"
        A[Question: Why 'Self' Attention?] --> B[Step 1: Prove it's Attention]
        B --> C[Compare with Luong/Bahdanau]
        C --> D[Identify Common Patterns]
        D --> E[Step 2: Understand 'Self']
        E --> F[Inter vs Intra Sequence]
        F --> G[Complete Understanding]
    end
    
    style A fill:#ffeb3b
    style G fill:#4caf50
```

## The Attention Mechanism Recap: Setting the Foundation

Before comparing self-attention with traditional attention, let's refresh our understanding of how attention mechanisms emerged and why they were needed.

**Video Context**: "Thoda sa hum log peeche jaate hain aur ek recap lete hain attention mechanism ka"

### The Original Problem: Sequence-to-Sequence Translation

```mermaid
graph LR
    subgraph "Language Translation Task"
        A["Turn off the lights<br/>(English)"] --> B["लाइट बंद करो<br/>(Hindi)"]
    end
    
    subgraph "Solution: Encoder-Decoder"
        C[Encoder LSTM] --> D[Context Vector]
        D --> E[Decoder LSTM]
    end
    
    subgraph "The Bottleneck"
        F["All information in<br/>ONE context vector"]
        G["Performance degrades<br/>for >30 words"]
    end
    
    D --> F
    F --> G
    
    style G fill:#ffcdd2
```

### Traditional Encoder-Decoder Architecture

The original encoder-decoder architecture processes input sequences word by word, maintaining hidden states at each timestep:

```mermaid
sequenceDiagram
    participant Input as Input Words
    participant Encoder as Encoder LSTM
    participant Hidden as Hidden States
    participant Context as Context Vector
    participant Decoder as Decoder LSTM
    participant Output as Output Words
    
    Input->>Encoder: Turn (x₁)
    Encoder->>Hidden: h₁
    Input->>Encoder: off (x₂)
    Encoder->>Hidden: h₂
    Input->>Encoder: the (x₃)
    Encoder->>Hidden: h₃
    Input->>Encoder: lights (x₄)
    Encoder->>Hidden: h₄
    Hidden->>Context: h₄ (final state)
    Context->>Decoder: Initialize
    Decoder->>Output: लाइट
    Decoder->>Output: बंद
    Decoder->>Output: करो
```

**The Critical Limitation**: The entire input sentence must be compressed into a single context vector (h₄), creating an information bottleneck.

## The Attention Solution: Multiple Context Vectors

Attention mechanism revolutionized this by introducing a key insight: instead of one context vector for all decoder timesteps, create a unique context vector for each output word.

```mermaid
graph TD
    subgraph "Attention Innovation"
        A["One Context Vector<br/>for ALL outputs"] --> B["PROBLEM:<br/>Information Loss"]
        
        C["Unique Context Vector<br/>for EACH output"] --> D["SOLUTION:<br/>Focused Attention"]
    end
    
    subgraph "Context Vector Calculation"
        E["c₁ for 'लाइट'"]
        F["c₂ for 'बंद'"]
        G["c₃ for 'करो'"]
        
        H["Each cᵢ = weighted sum<br/>of encoder states"]
    end
    
    B -.->|Attention Solves This| C
    C --> E
    C --> F
    C --> G
    E --> H
    F --> H
    G --> H
```

### The Attention Formula: Three Key Equations

**Video Explanation**: "Teen equations se milkar ke hamara Luong bana hai"

```mermaid
graph TD
    subgraph "Equation 1: Context Vector"
        A["cᵢ = Σⱼ αᵢⱼ hⱼ<br/>Weighted sum of encoder states"]
    end
    
    subgraph "Equation 2: Attention Weights"
        B["αᵢⱼ = softmax(eᵢⱼ)<br/>Normalized alignment scores"]
    end
    
    subgraph "Equation 3: Alignment Scores"
        C["eᵢⱼ = sᵢ · hⱼ<br/>Dot product similarity"]
    end
    
    C --> B
    B --> A
    
    style A fill:#e8f5e9
    style B fill:#e3f2fd
    style C fill:#fff3e0
```

Let's break down each equation with concrete examples:

**1. Alignment Score Calculation (eᵢⱼ)**:
```
e₁₁ = s₁ · h₁  (How relevant is 'Turn' for generating 'लाइट'?)
e₁₂ = s₁ · h₂  (How relevant is 'off' for generating 'लाइट'?)
e₁₃ = s₁ · h₃  (How relevant is 'the' for generating 'लाइट'?)
e₁₄ = s₁ · h₄  (How relevant is 'lights' for generating 'लाइट'?)
```

**2. Attention Weights (αᵢⱼ)**:
```
[α₁₁, α₁₂, α₁₃, α₁₄] = softmax([e₁₁, e₁₂, e₁₃, e₁₄])
Example: [0.1, 0.1, 0.1, 0.7] - 'lights' gets 70% attention
```

**3. Context Vector (c₁)**:
```
c₁ = 0.1×h₁ + 0.1×h₂ + 0.1×h₃ + 0.7×h₄
```

## Bridging to Self-Attention: The Conceptual Parallel

Now comes the crucial insight - let's see how self-attention follows the exact same mathematical pattern, just in a different setting.

**Video Quote**: "Main is poore process ko side by side compare karke dikhaunga... aap notice karoge ki dono bahut similar hai"

### Self-Attention Setup: No Encoder-Decoder Split

```mermaid
graph TD
    subgraph "Traditional Attention"
        A[Encoder Sequence] --> B[Decoder Sequence]
        B --> C[Inter-sequence Attention]
    end
    
    subgraph "Self-Attention"
        D[Single Sequence:<br/>Turn off the lights]
        D --> E[Embeddings]
        E --> F[Contextual Embeddings]
        F --> G[Intra-sequence Attention]
    end
    
    style C fill:#ffeb3b
    style G fill:#4caf50
```

### The Three Vectors: Q, K, V

In self-attention, we transform each word embedding into three vectors:

```mermaid
graph LR
    subgraph "For Each Word"
        A[Word Embedding] --> B[Query Q]
        A --> C[Key K]
        A --> D[Value V]
    end
    
    subgraph "Purpose"
        B --> E[What am I looking for?]
        C --> F[What information do I have?]
        D --> G[What value do I provide?]
    end
```

## The Mathematical Parallel: Side-by-Side Comparison

Let's align the mathematical operations of Luong attention and self-attention to see their fundamental similarity.

**Video Setup**: "Main ek tarike se is poori cheez ko likhta hoon jisse everything will make sense"

```mermaid
graph TD
    subgraph "Setup Comparison"
        A["Luong: Two Sequences<br/>English → Hindi"] 
        B["Self: One Sequence<br/>Turn off the lights"]
    end
    
    subgraph "Query Source"
        C["Luong: Decoder state sᵢ"]
        D["Self: Query vector Qᵢ"]
    end
    
    subgraph "Key Source"
        E["Luong: Encoder states hⱼ"]
        F["Self: Key vectors Kⱼ"]
    end
    
    subgraph "Value Source"
        G["Luong: Same as Keys (hⱼ)"]
        H["Self: Value vectors Vⱼ"]
    end
    
    A --> C
    B --> D
    A --> E
    B --> F
    A --> G
    B --> H
```

### Generating Contextual Embedding for "Turn"

Let's trace through the self-attention process for the first word "Turn":

```mermaid
graph TD
    subgraph "Step 1: Similarity Scores"
        A["Q_turn · K_turn = s₁₁<br/>(Turn attending to Turn)"]
        B["Q_turn · K_off = s₁₂<br/>(Turn attending to off)"]
        C["Q_turn · K_the = s₁₃<br/>(Turn attending to the)"]
        D["Q_turn · K_lights = s₁₄<br/>(Turn attending to lights)"]
    end
    
    subgraph "Step 2: Attention Weights"
        E["[w₁₁, w₁₂, w₁₃, w₁₄] <br/>= softmax([s₁₁, s₁₂, s₁₃, s₁₄])"]
    end
    
    subgraph "Step 3: Weighted Sum"
        F["Y_turn = w₁₁×V_turn <br/>+ w₁₂×V_off + w₁₃×V_the<br/> + w₁₄×V_lights"]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
```

**The Parallel Formula**:
```
Luong:         cᵢ = Σⱼ αᵢⱼ × hⱼ
Self-Attention: Yᵢ = Σⱼ wᵢⱼ × Vⱼ
```

### Visual Alignment of Operations

```mermaid
graph LR
    subgraph "Luong Attention"
        A1["sᵢ (Query)"] --> B1["Dot Product"]
        A2["hⱼ (Key)"] --> B1
        B1 --> C1["eᵢⱼ (Score)"]
        C1 --> D1["Softmax"]
        D1 --> E1["αᵢⱼ (Weight)"]
        E1 --> F1["Weighted Sum"]
        A3["hⱼ (Value)"] --> F1
        F1 --> G1["cᵢ (Context)"]
    end
    
    subgraph "Self-Attention"
        A4["Qᵢ (Query)"] --> B2["Dot Product"]
        A5["Kⱼ (Key)"] --> B2
        B2 --> C2["sᵢⱼ (Score)"]
        C2 --> D2["Softmax"]
        D2 --> E2["wᵢⱼ (Weight)"]
        E2 --> F2["Weighted Sum"]
        A6["Vⱼ (Value)"] --> F2
        F2 --> G2["Yᵢ (Output)"]
    end
    
    style G1 fill:#4caf50
    style G2 fill:#4caf50
```

**Key Insight**: The mathematical operations are identical - only the source and purpose of the vectors differ!

## Why "Self"? The Core Distinction

Now we arrive at the heart of the matter - what makes self-attention "self"?

**Video Revelation**: "Aapka idhar jo sentence hai aur idhar jo sentence hai, yeh dono same hi toh hai!"

```mermaid
graph TD
    subgraph "Inter-Sequence<br/> Attention (Luong/Bahdanau)"
        A["Sequence 1: English"] --> C["Attention between<br/>DIFFERENT sequences"]
        B["Sequence 2: Hindi"] --> C
        C --> D["Cross-attention/<br/>Inter-attention"]
    end
    
    subgraph "Intra-Sequence<br/> Attention (Self)"
        E["Sequence: Turn off the lights"]
        E --> F["Attention within<br/>SAME sequence"]
        E --> F
        F --> G["Self-attention/<br/>Intra-attention"]
    end
    
    subgraph "The 'Self' Meaning"
        H["Words attend to themselves<br/>and their companions<br/>in the SAME sequence"]
    end
    
    G --> H
    
    style D fill:#ffeb3b
    style G fill:#4caf50
    style H fill:#e91e63
```

### Concrete Example: Word Relationships

```mermaid
graph TD
    subgraph "Traditional Attention Example"
        A["'lights' (English)"] --> B["attends to"]
        B --> C["'लाइट' (Hindi)"]
        D["Different Sequences"]
    end
    
    subgraph "Self-Attention Example"
        E["'lights' (position 4)"] --> F["attends to"]
        F --> G["'Turn' (position 1)<br/>'off' (position 2)<br/>'the' (position 3)<br/>'lights' (position 4)"]
        H["Same Sequence"]
    end
    
    style D fill:#ffcdd2
    style H fill:#c8e6c9
```

## Implementation: Demonstrating the Mathematical Equivalence

Let's implement both mechanisms to show their mathematical similarity:

```python
import numpy as np

def luong_attention(decoder_state, encoder_states):
    """
    Luong attention mechanism
    decoder_state: Current decoder hidden state (query)
    encoder_states: All encoder hidden states (keys & values)
    """
    # Step 1: Calculate alignment scores (dot product)
    scores = np.dot(encoder_states, decoder_state)
    
    # Step 2: Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Step 3: Calculate context vector (weighted sum)
    context = np.sum(weights[:, np.newaxis] * encoder_states, axis=0)
    
    return context, weights

def self_attention(query, keys, values):
    """
    Self-attention mechanism
    query: Query vector for current position
    keys: Key vectors for all positions
    values: Value vectors for all positions
    """
    # Step 1: Calculate similarity scores (dot product)
    scores = np.dot(keys, query)
    
    # Step 2: Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Step 3: Calculate output (weighted sum)
    output = np.sum(weights[:, np.newaxis] * values, axis=0)
    
    return output, weights

# Example usage showing mathematical equivalence
hidden_dim = 4
seq_len = 4

# Luong attention setup
decoder_state = np.random.randn(hidden_dim)  # s₁
encoder_states = np.random.randn(seq_len, hidden_dim)  # [h₁, h₂, h₃, h₄]

# Self-attention setup (for one position)
Q = np.random.randn(hidden_dim)  # Q_turn
K = np.random.randn(seq_len, hidden_dim)  # [K_turn, K_off, K_the, K_lights]
V = np.random.randn(seq_len, hidden_dim)  # [V_turn, V_off, V_the, V_lights]

# Compare outputs
luong_output, luong_weights = luong_attention(decoder_state, encoder_states)
self_output, self_weights = self_attention(Q, K, V)

print("Luong attention weights shape:", luong_weights.shape)
print("Self-attention weights shape:", self_weights.shape)
print("\nBoth use the same mathematical operations!")
```

## The Complete Picture: Attention Family Tree

```mermaid
graph TD
    subgraph "Attention Mechanisms Family"
        A[Attention Mechanism]
        A --> B[Inter-Sequence <br/> Attention]
        A --> C[Intra-Sequence <br/>Attention]
        
        B --> D[Bahdanau Attention<br/>Concatenate + MLP]
        B --> E[Luong Attention<br/>Dot Product]
        
        C --> F[Self-Attention<br/>Q, K, V Transformations]
        C --> G[Multi-Head Attention<br/>Parallel Self-Attention]
    end
    
    subgraph "Key Differentiator"
        H["Inter: Between <br/>different sequences<br/>Intra: Within <br/>same sequence"]
    end
    
    style F fill:#4caf50
    style H fill:#e91e63
```

## Visual Summary: The Three Operations

```mermaid
graph LR
    subgraph "Common to All Attention"
        A[1. Query-Key Similarity] --> B[2. Softmax Normalization]
        B --> C[3. Value Weighting]
    end
    
    subgraph "What Changes"
        D[Source of Q, K, V]
        E[Inter vs Intra Sequence]
    end
    
    C --> F[Attention Output]
    D --> A
    E --> A
```

## Key Takeaways

- **Self-Attention IS Attention**: It uses the exact same mathematical operations (dot product, softmax, weighted sum) as traditional attention mechanisms

- **The "Self" Distinction**: While Luong/Bahdanau attention operates between two different sequences (encoder-decoder), self-attention operates within a single sequence

- **Mathematical Equivalence**:
  - Luong: `cᵢ = Σⱼ softmax(sᵢ·hⱼ) × hⱼ`
  - Self: `Yᵢ = Σⱼ softmax(Qᵢ·Kⱼ) × Vⱼ`

- **Conceptual Mapping**:
  - Decoder states (s) → Query vectors (Q)
  - Encoder states (h) → Key vectors (K) and Value vectors (V)
  - Context vector (c) → Contextual embedding (Y)

- **Interview Insight**: When asked "Why is it called self-attention?", the answer is: "Because unlike traditional attention that aligns between two different sequences, self-attention computes alignment scores within the same sequence - words attend to themselves and other words in their own sentence."

## Research Impact

The introduction of self-attention revolutionized NLP by:

1. **Removing Sequential Dependencies**: Unlike RNNs, all positions can be processed in parallel
2. **Capturing Long-Range Dependencies**: Direct connections between any two positions
3. **Computational Efficiency**: Parallel processing enables faster training
4. **Foundation for Transformers**: Self-attention became the building block for the transformer revolution

## References

1. **Vaswani, A., et al.** (2017). Attention Is All You Need. *NIPS 2017*.
2. **Luong, M., et al.** (2015). Effective Approaches to Attention-based Neural Machine Translation. *EMNLP 2015*.
3. **Bahdanau, D., et al.** (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.

[End of Notes]