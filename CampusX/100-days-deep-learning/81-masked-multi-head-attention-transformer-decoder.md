# Day 81: Masked Multi-head Attention in Transformer Decoder

## Paper References
- **Attention Is All You Need** (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **The Annotated Transformer** (Harvard NLP): [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Introduction: The Decoder's Dilemma

The Transformer decoder presents a fascinating architectural challenge that sits at the heart of modern language models. While the encoder processes all input tokens simultaneously using parallel self-attention, the decoder must maintain sequential generation during inference while achieving efficient parallel training. This fundamental tension between training efficiency and inference causality is resolved through **masked multi-head attention** - a sophisticated mechanism that prevents future information leakage while enabling batch processing.

This video explores the critical sentence: *"The Transformer decoder is autoregressive at inference time and non-autoregressive at training time"* - a statement that reveals the ingenious solution to one of sequence modeling's most significant challenges.

## Understanding Autoregressive Models

```mermaid
graph TD
    A[Autoregressive Model Definition] --> B[Sequential Generation]
    B --> C[Current Token Depends on Previous Tokens]
    C --> D[Example: Stock Prediction]
    
    subgraph "Stock Price Prediction"
        D1[Wednesday: $29] --> D2[Thursday: $30]
        D2 --> D3[Friday: Depends on Wed + Thu]
    end
    
    D --> D1
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
```

In deep learning context, autoregressive models generate data points in sequence by conditioning each new point on previously generated ones. This is the natural behavior for language generation tasks where:

- Each word depends on the context of previous words
- Future tokens are unknown during generation
- Sequential processing is inherent to the task

## The Training vs Inference Paradox

### Inference Time: Forced Autoregression

```mermaid
sequenceDiagram
    participant U as User Input
    participant E as Encoder
    participant D as Decoder
    participant O as Output
    
    U->>E: "I am fine"
    E->>D: Context vectors
    
    Note over D: Step 1: <START> token
    D->>O: "मैं"
    
    Note over D: Step 2: Use "मैं" as input
    D->>O: "बढ़िया" (incorrect)
    
    Note over D: Step 3: Use "बढ़िया" as input
    D->>O: "हूं"
    
    Note over D: Step 4: Use "हूं" as input
    D->>O: <END>
    
    Note right of O: Final: "मैं बढ़िया हूं"
```

During inference, the decoder **must** be autoregressive because:
1. The next token's input depends on the previous token's output
2. Future tokens are genuinely unknown
3. No alternative exists - you cannot predict what hasn't been generated yet

### Training Time: The Efficiency Challenge

```mermaid
graph LR
    subgraph "Sequential Training (Slow)"
        A1[आप] --> A2[आप → कैसे]
        A2 --> A3[आप → कैसे → हैं]
        A3 --> A4[Complete]
    end
    
    subgraph "Parallel Training (Fast but Problematic)"
        B1[आप]
        B2[कैसे]
        B3[हैं]
        B1 & B2 & B3 --> B4[Process All Together]
    end
    
    style A1 fill:#ffcdd2
    style A2 fill:#ffcdd2
    style A3 fill:#ffcdd2
    style B1 fill:#c8e6c9
    style B2 fill:#c8e6c9
    style B3 fill:#c8e6c9
```

The training dilemma arises because:
- **Sequential training** is accurate but extremely slow
- **Parallel training** is fast but creates data leakage
- For long sequences (300+ tokens), sequential training becomes prohibitively expensive

## The Data Leakage Problem

```mermaid
graph TD
    subgraph "Data Leakage in Parallel Training"
        A["आप (Position 1)"]
        B["कैसे (Position 2)"]
        C["हैं (Position 3)"]
        
        A --> |"Can see future"| B
        A --> |"Can see future"| C
        B --> |"Can see future"| C
        
        A -.-> A1["Contextual embedding<br/>uses future information"]
        B -.-> B1["Contextual embedding<br/>uses future information"]
        C -.-> C1["Valid: can use all<br/>previous context"]
    end
    
    style A fill:#ffcdd2
    style B fill:#ffcdd2
    style C fill:#c8e6c9
    style A1 fill:#ffcdd2
    style B1 fill:#ffcdd2
    style C1 fill:#c8e6c9
```

**The Core Problem**: In parallel training, when calculating contextual embeddings:
- Token "आप" can see future tokens "कैसे" and "हैं"
- Token "कैसे" can see future token "हैं" 
- This creates unfair advantage during training that won't exist during inference

**Mathematical Representation**:
```
Contextual(आप) = 0.8 × Emb(आप) + 0.1 × Emb(कैसे) + 0.1 × Emb(हैं)
```
During inference, tokens "कैसे" and "हैं" don't exist when processing "आप"!

## Self-Attention Mechanics: The Foundation

```mermaid
graph LR
    subgraph "Input Embeddings"
        E1[आप_emb]
        E2[कैसे_emb]
        E3[हैं_emb]
    end
    
    subgraph "Weight Matrices"
        W1[W_Q]
        W2[W_K]
        W3[W_V]
    end
    
    subgraph "Generated Vectors"
        Q[Query Vectors]
        K[Key Vectors]
        V[Value Vectors]
    end
    
    E1 --> W1
    E2 --> W1
    E3 --> W1
    W1 --> Q
    
    E1 --> W2
    E2 --> W2
    E3 --> W2
    W2 --> K
    
    E1 --> W3
    E2 --> W3
    E3 --> W3
    W3 --> V
    
    style Q fill:#e3f2fd
    style K fill:#f3e5f5
    style V fill:#e8f5e8
```

**Self-Attention Process**:
1. **Linear Transformations**: Each embedding is multiplied by WQ, WK, WV matrices
2. **Attention Score Calculation**: Q · K^T produces attention scores
3. **Scaling**: Divide by √(d_k) for numerical stability
4. **Softmax**: Convert scores to probabilities
5. **Weighted Sum**: Multiply attention weights with value vectors

## The Masking Solution

```mermaid
graph TD
    subgraph "Attention Score Matrix"
        A1[आप→आप: 0.8]
        A2[आप→कैसे: 0.6]
        A3[आप→हैं: 0.4]
        B1[कैसे→आप: 0.3]
        B2[कैसे→कैसे: 0.7]
        B3[कैसे→हैं: 0.5]
        C1[हैं→आप: 0.2]
        C2[हैं→कैसे: 0.4]
        C3[हैं→हैं: 0.9]
    end
    
    subgraph "Mask Matrix"
        M1[0]
        M2[-∞]
        M3[-∞]
        M4[0]
        M5[0]
        M6[-∞]
        M7[0]
        M8[0]
        M9[0]
    end
    
    subgraph "Masked Scores"
        R1[0.8]
        R2[-∞]
        R3[-∞]
        R4[0.3]
        R5[0.7]
        R6[-∞]
        R7[0.2]
        R8[0.4]
        R9[0.9]
    end
    
    A1 --> R1
    A2 --> R2
    A3 --> R3
    B1 --> R4
    B2 --> R5
    B3 --> R6
    C1 --> R7
    C2 --> R8
    C3 --> R9
    
    style M2 fill:#ffcdd2
    style M3 fill:#ffcdd2
    style M6 fill:#ffcdd2
    style R2 fill:#ffcdd2
    style R3 fill:#ffcdd2
    style R6 fill:#ffcdd2
```

**Masking Process**:
1. **Create Mask Matrix**: Same dimensions as attention scores
2. **Set Future Positions**: -∞ for future tokens, 0 for allowed positions
3. **Add Mask**: Attention_scores + Mask_matrix
4. **Apply Softmax**: softmax(-∞) = 0, eliminating future information

## Step-by-Step Masking Implementation

```mermaid
sequenceDiagram
    participant QK as Q·K^T
    participant S as Scaling
    participant M as Mask Addition
    participant SM as Softmax
    participant V as Value Multiplication
    
    QK->>S: Raw attention scores
    S->>M: Scaled scores ÷ √d_k
    
    Note over M: Add mask matrix<br/>Future positions = -∞
    
    M->>SM: Masked scores
    
    Note over SM: softmax(-∞) = 0<br/>Future attention = 0
    
    SM->>V: Attention weights
    V->>V: Final contextual embeddings
```

**Mathematical Implementation**:
```python
# Attention calculation with masking
def masked_attention(Q, K, V, mask):
    # Calculate attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (set future positions to -∞)
    scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Calculate contextual embeddings
    contextual = torch.matmul(attention_weights, V)
    
    return contextual
```

## The Elegant Solution: Best of Both Worlds

```mermaid
graph TD
    subgraph "Training Benefits"
        T1[Parallel Processing]
        T2[Fast Training]
        T3[Batch Efficiency]
        T4[GPU Utilization]
    end
    
    subgraph "Inference Consistency"
        I1[No Data Leakage]
        I2[Causal Dependencies]
        I3[Sequential Logic]
        I4[Identical Behavior]
    end
    
    subgraph "Masked Self-Attention"
        M[Masking Mechanism]
    end
    
    T1 --> M
    T2 --> M
    T3 --> M
    T4 --> M
    
    M --> I1
    M --> I2
    M --> I3
    M --> I4
    
    style M fill:#fff3e0
    style T1 fill:#e8f5e8
    style T2 fill:#e8f5e8
    style T3 fill:#e8f5e8
    style T4 fill:#e8f5e8
    style I1 fill:#e3f2fd
    style I2 fill:#e3f2fd
    style I3 fill:#e3f2fd
    style I4 fill:#e3f2fd
```

**Achievement**: Masked self-attention enables:
- **Parallel training** for efficiency
- **Causal consistency** for correctness
- **Identical behavior** between training and inference logic
- **Scalable processing** for long sequences

## Detailed Mathematical Walkthrough

### Without Masking (Problematic)
```
Contextual(आप) = w₁₁ × V(आप) + w₁₂ × V(कैसे) + w₁₃ × V(हैं)
Contextual(कैसे) = w₂₁ × V(आप) + w₂₂ × V(कैसे) + w₂₃ × V(हैं)
Contextual(हैं) = w₃₁ × V(आप) + w₃₂ × V(कैसे) + w₃₃ × V(हैं)
```

**Problem**: w₁₂, w₁₃, w₂₃ represent future information access

### With Masking (Correct)
```
Contextual(आप) = w₁₁ × V(आप) + 0 × V(कैसे) + 0 × V(हैं)
Contextual(कैसे) = w₂₁ × V(आप) + w₂₂ × V(कैसे) + 0 × V(हैं)
Contextual(हैं) = w₃₁ × V(आप) + w₃₂ × V(कैसे) + w₃₃ × V(हैं)
```

**Solution**: Future weights (w₁₂, w₁₃, w₂₃) are zeroed through masking

## Practical Implementation Considerations

```mermaid
graph LR
    subgraph "Mask Creation"
        A[Sequence Length] --> B[Lower Triangular Matrix]
        B --> C[Boolean Mask]
        C --> D[Infinity Substitution]
    end
    
    subgraph "Attention Computation"
        E[Q·K^T] --> F[Scale]
        F --> G[Add Mask]
        G --> H[Softmax]
        H --> I[Multiply V]
    end
    
    D --> G
    
    style B fill:#e8f5e8
    style G fill:#fff3e0
    style H fill:#e3f2fd
```

**Key Implementation Details**:
1. **Mask Shape**: [seq_len, seq_len] lower triangular matrix
2. **Mask Values**: 0 for allowed positions, -∞ for blocked positions
3. **Numerical Stability**: Use -1e9 instead of true -∞
4. **Broadcasting**: Ensure mask dimensions match attention scores

## Training vs Inference Behavior Comparison

```mermaid
graph TD
    subgraph "Training Phase"
        TR1[Full Target Sequence Available]
        TR2[Teacher Forcing]
        TR3[Parallel Processing]
        TR4[Masked Self-Attention]
        TR5[Fast Batch Training]
    end
    
    subgraph "Inference Phase"
        IN1[Generate Token by Token]
        IN2[Use Previous Outputs]
        IN3[Sequential Processing]
        IN4[Standard Self-Attention]
        IN5[Slower Generation]
    end
    
    TR1 --> TR2
    TR2 --> TR3
    TR3 --> TR4
    TR4 --> TR5
    
    IN1 --> IN2
    IN2 --> IN3
    IN3 --> IN4
    IN4 --> IN5
    
    style TR4 fill:#e8f5e8
    style IN4 fill:#e3f2fd
```

**Critical Insight**: The same model behaves differently in training vs inference:
- **Training**: Non-autoregressive (parallel) with masking
- **Inference**: Autoregressive (sequential) by necessity

## Performance Impact Analysis

```mermaid
pie title Training Speed Comparison
    "Sequential Training" : 25
    "Parallel Training (Masked)" : 75
```

**Performance Metrics**:
- **Sequential Training**: O(n) time complexity for n tokens
- **Parallel Training**: O(1) time complexity with masking
- **Memory Usage**: Higher for parallel due to full attention matrix
- **GPU Utilization**: Dramatically improved with parallel processing

## Key Takeaways

1. **Autoregressive Necessity**: Sequence generation inherently requires autoregressive behavior during inference due to causal dependencies.

2. **Training Efficiency**: Masked self-attention enables parallel training while maintaining causal consistency.

3. **Data Leakage Prevention**: Masking prevents future information access during training, ensuring fair evaluation.

4. **Architectural Elegance**: The solution achieves optimal training efficiency without compromising inference correctness.

5. **Scalability**: The approach scales effectively to long sequences where sequential training becomes prohibitive.

6. **Implementation Simplicity**: Despite conceptual complexity, the implementation requires only a mask matrix addition.

The masked multi-head attention mechanism represents a masterful solution to the fundamental tension between training efficiency and inference causality in sequence modeling. By understanding this mechanism deeply, we gain insight into why Transformers have become the foundation of modern language models and how they achieve their remarkable balance of performance and accuracy.

[End of Notes]