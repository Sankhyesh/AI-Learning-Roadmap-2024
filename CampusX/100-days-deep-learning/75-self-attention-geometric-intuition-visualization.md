# Day 75: Self-Attention Geometric Intuition - How to Visualize Self-Attention

## Paper References
- **Attention Is All You Need** - Vaswani et al., 2017
  - [Original Paper](https://arxiv.org/abs/1706.03762)
  - Foundational paper introducing the Transformer architecture and self-attention mechanism
- **Geometric Understanding of Deep Learning** - Lei et al., 2020
  - Geometric perspectives on neural network operations and transformations

## Introduction: The Visual Journey into Self-Attention

In this video, we embark on a unique geometric journey to understand what happens mathematically behind the scenes when we apply self-attention to words. While many tutorials skip this conceptual depth, this exploration provides the geometric intuition that transforms abstract mathematical operations into visual understanding.
![alt text](images/75/image.png)
**Why This Matters**: Self-attention operates as a sophisticated geometric transformation that repositions word embeddings in high-dimensional space based on contextual relationships. Understanding this geometry reveals why self-attention is so powerful for capturing contextual meaning.

**Learning Journey**: We'll start with word embeddings as vectors in 2D space, apply each step of self-attention geometrically, and visualize how words "pull" each other through contextual gravity, transforming static embeddings into dynamic, context-aware representations.

```mermaid
graph TD 
    subgraph "Learning Flow"
        A[Word2Vec Visualization] --> B[2D Embedding Setup]
        B --> C[Linear Transformations Q,K,V]
        C --> D[Similarity Calculations]
        D --> E[Attention Weights Geometry]
        E --> F[Contextual Vector Composition]
        F --> G[Geometric Intuition: Contextual Gravity]
    end
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

## Quick Recap: Self-Attention Mathematical Framework
![alt text](images/75/image-1.png)
Before diving into geometric intuition, let's review the mathematical foundation we've built in previous videos. This visual recap sets the stage for our geometric exploration:

```mermaid
graph TD
    subgraph "Input Processing"
        A["Money Bank<br/>(Input Sentence)"] --> B["Word Embeddings<br/>E_money, E_bank"]
    end
    
    subgraph "Weight Matrices"
        C["W_Q<br/>(Query Weights)"]
        D["W_K<br/>(Key Weights)"] 
        E["W_V<br/>(Value Weights)"]
    end
    
    subgraph "Vector Generation"
        F["Q_money, Q_bank<br/>(Query Vectors)"]
        G["K_money, K_bank<br/>(Key Vectors)"]
        H["V_money, V_bank<br/>(Value Vectors)"]
    end
    
    subgraph "Attention Computation"
        I["Similarity Scores<br/>Q·K^T"]
        J["Scaled Scores<br/>÷√d_k"]
        K["Attention Weights<br/>Softmax"]
        L["Contextual Embeddings<br/>Y_money, Y_bank"]
    end
    
    B --> F
    B --> G  
    B --> H
    C --> F
    D --> G
    E --> H
    F --> I
    G --> I
    I --> J
    J --> K
    K --> L
    H --> L
```

**Core Formula**: 
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This mathematical framework now becomes our geometric playground where each operation has a clear visual interpretation.

## Word Embeddings as Geometric Vectors

The journey begins with understanding how words exist in geometric space. The instructor shows us a fascinating Word2Vec visualization where words cluster based on semantic similarity.

**Video Context**: "Yeh ek visualization hai jahan pe Word2Vec jo hamaari word embeddings hoti hai unka visualization aap dekh sakte ho"

### Interactive Visualization Resources

For deeper exploration of geometric intuitions in transformers, these trusted educational resources provide excellent interactive visualizations:

**🎯 Jay Alammar's Illustrated Transformer**
- **URL**: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- **What it shows**: Step-by-step visual breakdown of transformer architecture with geometric transformations
- **Key features**: Self-attention visualization, encoder-decoder flow, multi-head attention geometry
- **Educational value**: "The best explanation of transformer neural network architecture" - used in Stanford, Harvard, MIT courses

**🔬 BertViz - Interactive Attention Visualization** 
- **GitHub**: [github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
- **What it shows**: Real-time attention patterns in BERT, GPT-2, T5 models
- **Key features**: Head view, model view, and neuron view of attention mechanisms
- **Educational value**: Observe actual geometric attention patterns in trained models

**🌟 LLM Visualization (3D Interactive)**
- **URL**: [bbycroft.net/llm](https://bbycroft.net/llm)
- **What it shows**: 3D animated walkthrough of LLM processing
- **Key features**: Interactive 3D geometry of transformer operations
- **Educational value**: Spatial understanding of information flow through layers

**📊 Distill.pub Attention Articles**
- **URL**: [distill.pub/2016/augmented-rnns](https://distill.pub/2016/augmented-rnns/)
- **What it shows**: Interactive attention mechanisms with reactive diagrams
- **Key features**: Visual attention flow, geometric positioning, interactive exploration
- **Educational value**: Dynamic visualizations showing attention as geometric interface

```mermaid
graph TD
    subgraph "Word2Vec Semantic Clustering"
        A["Chemical Elements:<br/>Chloride, Magnesium,<br/>Sodium, Oxide"]
        B["Formal Language:<br/>Indeed, Regard,<br/>Consequently, Practice"]
        C["Objects/Properties:<br/>Object, Increases,<br/>Properties"]
    end
    
    subgraph "Geometric Properties"
        D["200D Original Space"] --> E["2D Projection (PCA)"]
        E --> F["Semantic Similarity<br/>= Geometric Proximity"]
    end
    
    A --> D
    B --> D
    C --> D
```

**Key Insight**: Words with similar meanings cluster together in geometric space. This clustering provides the foundation for self-attention's contextual transformations.

### Setting Up Our 2D Geometric Playground

For visualization purposes, we simplify to 2D space while maintaining the geometric principles:

**Video Quote**: "Hum thodi der ke liye maan lete hain ki yeh jo dono ke dono embedding values hai yeh 2D hai so that hum visualize kar paaye"

```mermaid
graph LR
    subgraph "Our Simplified Setup"
        A["E_money = [2, 5]<br/>(Money Embedding)"]
        B["E_bank = [7, 3]<br/>(Bank Embedding)"]
    end
    
    subgraph "Weight Matrices (2×2)"
        C["W_Q = [[1,0], [0,1]]"]
        D["W_K = [[0,1], [1,0]]"] 
        E["W_V = [[1,1], [0,1]]"]
    end
    
    subgraph "Note"
        F["All values are<br/>hypothetical for<br/>visualization"]
    end
    
    style F fill:#fff3e0
```

**Important Note**: "All values are hypothetical" - these aren't trained values but carefully chosen examples to illustrate geometric principles.

## Step 1: Linear Transformations - Creating Q, K, V Vectors

The first geometric transformation occurs when we multiply embeddings with weight matrices. This is where linear algebra becomes visual geometry.

**Video Explanation**: "Jab aap ek vector ka dot product karte ho ek matrix ke saath toh essentially hota kya hai ki linear transformation hota hai"

```mermaid
graph TD
    subgraph "Original Embedding Space"
        A["E_money [2,5]<br/>📍 Original Position"]
        B["E_bank [7,3]<br/>📍 Original Position"]
    end
    
    subgraph "Linear Transformation Process"
        C["Matrix Multiplication<br/>= Geometric Transformation"]
        D["One Vector → Three New Vectors<br/>(Q, K, V transformations)"]
    end
    
    subgraph "Transformed Vector Space"
        E["Q_money, K_money, V_money<br/>🎯 New Positions"]
        F["Q_bank, K_bank, V_bank<br/>🎯 New Positions"]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    D --> F
    
    style C fill:#e8f5e8
```

### Visualizing the Linear Transformations

Let's see how each weight matrix transforms our original embedding:

```mermaid
graph TD
    subgraph "Money Word Transformations"
        A["E_money [2,5]"] --> B["× W_Q"] 
        A --> C["× W_K"]
        A --> D["× W_V"]
        
        B --> E["Q_money [2,5]<br/>📍 Query Position"]
        C --> F["K_money [5,2]<br/>📍 Key Position"] 
        D --> G["V_money [7,5]<br/>📍 Value Position"]
    end
    
    subgraph "Bank Word Transformations"
        H["E_bank [7,3]"] --> I["× W_Q"]
        H --> J["× W_K"] 
        H --> K["× W_V"]
        
        I --> L["Q_bank [7,3]<br/>📍 Query Position"]
        J --> M["K_bank [3,7]<br/>📍 Key Position"]
        K --> N["V_bank [10,3]<br/>📍 Value Position"]
    end
    
    style E fill:#ffebee
    style F fill:#e3f2fd
    style G fill:#e8f5e8
    style L fill:#ffebee
    style M fill:#e3f2fd
    style N fill:#e8f5e8
```

**Geometric Result**: From 2 original vectors, we now have 6 vectors positioned differently in our geometric space, each serving a specific role in the attention mechanism.

## Step 2: Similarity Calculation - Geometric Distance and Angles

Now comes the crucial step where we calculate how "similar" or "related" words are using their query and key vectors. This is where geometry becomes semantics.

**Video Setup**: "Ab humein kya karna hai ki humein yeh poora process do baar karna hai... main bank word ka contextual embedding banaaunga"

```mermaid
graph TD
    subgraph "Similarity Calculation Setup"
        A["Q_bank<br/>📍 Query Vector"] --> C["Dot Product<br/>Q_bank · K_money"]
        B["K_money<br/>📍 Key Vector"] --> C
        
        A --> D["Dot Product<br/>Q_bank · K_bank"]
        E["K_bank<br/>📍 Key Vector"] --> D
        
        C --> F["Similarity Score 1<br/>Raw value: 10"]
        D --> G["Similarity Score 2<br/>Raw value: 32"]
    end
    
    subgraph "Geometric Interpretation"
        H["Angular Distance Large<br/>→ Lower Dot Product"]
        I["Angular Distance Small<br/>→ Higher Dot Product"]
    end
    
    F --> H
    G --> I
    
    style F fill:#ffcdd2
    style G fill:#c8e6c9
```

### The Geometry of Similarity

**Video Insight**: "Since dekh ke samajh mein aa raha hai ki angular distance zyada hai toh dot product aapka kam hoga"

```mermaid
graph LR
    subgraph "Vector Angles and Similarity"
        A["Large Angle<br/>Between Vectors"] --> B["Low Dot Product<br/>= Low Similarity"]
        C["Small Angle<br/>Between Vectors"] --> D["High Dot Product<br/>= High Similarity"]
    end
    
    subgraph "Our Example"
        E["Q_bank ∠ K_money<br/>Large angle → Score: 10"]
        F["Q_bank ∠ K_bank<br/>Small angle → Score: 32"]
    end
    
    style B fill:#ffcdd2
    style D fill:#c8e6c9
    style E fill:#ffcdd2
    style F fill:#c8e6c9
```

## Step 3: Scaling and Normalization - Geometric Stabilization

The raw similarity scores need geometric stabilization through scaling and normalization.

```mermaid
graph TD
    subgraph "Scaling Process"
        A["Raw Scores:<br/>10, 32"] --> B["÷ √d_k = √2<br/>÷ 1.414"]
        B --> C["Scaled Scores:<br/>7.09, 22.6"]
    end
    
    subgraph "Softmax Normalization"
        C --> D["Softmax Function"] 
        D --> E["Normalized Weights:<br/>w₂₁ = 0.2<br/>w₂₂ = 0.8"]
    end
    
    subgraph "Geometric Meaning"
        F["Scaling prevents<br/>extreme values"]
        G["Softmax creates<br/>probability distribution"]
        H["Weights sum to 1<br/>for meaningful averaging"]
    end
    
    B --> F
    D --> G
    E --> H
    
    style F fill:#fff3e0
    style G fill:#e8f5e8
    style H fill:#e3f2fd
```

**Key Geometric Insight**: Scaling and normalization transform raw geometric distances into meaningful probability weights for contextual blending.

## Step 4: Contextual Vector Composition - The Geometric Magic

This is where the geometric magic happens. We use attention weights to create a weighted combination of value vectors, geometrically repositioning our word in context-aware space.

**Video Process**: "Ab humein kya karna hai ki yeh jo do orange vectors hai humein in dono ke beech vector addition karwaana hai"

```mermaid
graph TD
    subgraph "Weighted Value Vectors"
        A["w₂₁ × V_money<br/>0.2 × [7,5]<br/>= [1.4, 1.0]"] --> C["Vector Addition"]
        B["w₂₂ × V_bank<br/>0.8 × [10,3]<br/>= [8.0, 2.4]"] --> C
    end
    
    subgraph "Geometric Addition Process"
        C --> D["Parallelogram Law<br/>Vector Addition"]
        D --> E["Resultant Vector:<br/>Y_bank = [9.4, 3.4]"]
    end
    
    subgraph "Geometric Interpretation"
        F["Scaling vectors<br/>by attention weights"]
        G["Geometric composition<br/>creates new position"]
        H["Context-aware<br/>embedding location"]
    end
    
    A --> F
    D --> G
    E --> H
    
    style E fill:#4caf50
```

original 
![alt text](images/75/4.png)
after contextual embedding 
![alt text](images/75/4-1.png)

### Visualizing Vector Addition Geometry

**Video Explanation**: "Vector addition ke liye aapke paas do option hote hain... parallelogram law of vector addition"

```mermaid
graph TD
    subgraph "Vector Addition Visualization"
        A["Original V_money<br/>📍 [7,5]"] --> D["Scaled: 0.2 × V_money<br/>📍 [1.4, 1.0]"]
        B["Original V_bank<br/>📍 [10,3]"] --> E["Scaled: 0.8 × V_bank<br/>📍 [8.0, 2.4]"]
        
        D --> F["Parallelogram<br/>Construction"]
        E --> F
        F --> G["Resultant Vector<br/>Y_bank [9.4, 3.4]<br/>🎯 New Context Position"]
    end
    
    subgraph "Geometric Method"
        H["Place vectors tail-to-head<br/>OR"]
        I["Use parallelogram law<br/>for visual clarity"]
    end
    
    style G fill:#2e7d32
```

## The Geometric Revelation: Contextual Gravity

Now comes the most profound geometric insight - comparing the original embedding with the context-aware result.

**Video's Key Insight**: "Yahan pe dekho maine kya kiya... yeh hai mera E_bank jo mera embedding vector of bank hai aur yeh jo aapko dikhaai de raha hai yeh hai Y_bank"

### Advanced Visualization Tools for Geometric Understanding

**🎮 Interactive Attention Exploration**
- **Attention Viz**: [attentionviz.com](http://attentionviz.com/) - Interactive attention pattern exploration
- **Transformer Debugger**: Real-time geometric transformations in transformer layers

**🧭 Geometric Intuition Animations**
- **Medium Articles**: Detailed geometric breakdowns with animated explanations
  - [Visualizing Self-Attention: A Geometric Intuition](https://medium.com/@navdeepsharmatu/visualizing-self-attention-a-geometric-intuition-bb8b1b82239c)
  - [Transformers (Self Attention Geometric Intuition)](https://medium.com/@virajkaralay/transformers-self-attention-geometric-intuition-d5daa9becbb6)

**📐 Mathematical Visualization Tools**
- **GeoGebra**: Create custom geometric representations of vector transformations
- **Manim**: Python library for creating mathematical animations (used by 3Blue1Brown)
- **Observable**: Interactive notebooks for transformer visualizations

```mermaid
graph TD
    subgraph "Before Self-Attention"
        A["E_money [2,5]<br/>📍 Original Position"]
        B["E_bank [7,3]<br/>📍 Original Position"]
        C["Large Distance<br/>Different semantic regions"]
    end
    
    subgraph "After Self-Attention"
        D["Y_money [context-aware]<br/>📍 New Position"]
        E["Y_bank [9.4, 3.4]<br/>📍 New Position"]
        F["Reduced Distance<br/>Contextually aligned"]
    end
    
    subgraph "Geometric Transformation"
        G["Self-Attention acts<br/>like Gravity"]
        H["Words 'pull' each other<br/>based on context"]
        I["Contextual relationships<br/>→ Geometric proximity"]
    end
    
    A --> D
    B --> E
    C --> F
    G --> H
    H --> I
    
    style G fill:#ff9800
    style H fill:#ff9800
    style I fill:#4caf50
```

### The Gravity Analogy

**Video Quote**: "Self-attention acts like gravity in a way - money word ne bank wale word ko apni taraf pull kiya"

```mermaid
graph LR
    subgraph "Contextual Gravity Effect"
        A["Money<br/>🌟 Gravitational Mass"] --> C["Pulls Bank<br/>→ Towards Money's<br/>semantic region"]
        B["Bank<br/>🌍 Affected Object"] --> C
    end
    
    subgraph "Context Sensitivity"
        D["Money Bank<br/>(Financial Context)"] --> E["Bank moves toward<br/>Money's position"]
        F["River Bank<br/>(Geographic Context)"] --> G["Bank would move toward<br/>River's position"]
    end
    
    subgraph "Key Insight"
        H["Same word 'Bank'<br/>Different contextual<br/>destinations"]
    end
    
    E --> H
    G --> H
    
    style A fill:#ffc107
    style C fill:#4caf50
    style H fill:#e91e63
```

## The Complete Geometric Journey

Let's visualize the complete transformation journey from static embeddings to dynamic, context-aware representations:

```mermaid
graph TD
    subgraph "Stage 1: Static Embeddings"
        A["E_money [2,5]<br/>E_bank [7,3]<br/>🔴 Fixed positions"]
    end
    
    subgraph "Stage 2: Linear Transformations"
        B["6 new vectors<br/>Q, K, V for each word<br/>🔵 Transformed positions"]
    end
    
    subgraph "Stage 3: Attention Computation"
        C["Similarity calculations<br/>Geometric distances<br/>🟡 Relationship mapping"]
    end
    
    subgraph "Stage 4: Weight Generation"
        D["Attention weights<br/>Contextual importance<br/>🟠 Influence factors"]
    end
    
    subgraph "Stage 5: Contextual Composition"
        E["Y_money, Y_bank<br/>Context-aware positions<br/>🟢 Final destinations"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    
    style A fill:#ffebee
    style B fill:#e3f2fd
    style C fill:#fffde7
    style D fill:#fff3e0
    style E fill:#e8f5e8
```

### Context-Dependent Positioning

**Video's Profound Insight**: "Kal ko agar main Money Bank ke badle River Bank rakhu... toh ho sakta tha ki self-attention apply karne ke baad Y_bank yahan par aata"

```mermaid
graph TD
    subgraph "Financial Context: Money Bank"
        A["Money [2,5]<br/>🏦 Financial Anchor"] --> C["Bank moves toward<br/>financial semantic space<br/>📊 Y_bank_financial"]
        B["Bank [7,3]<br/>🏛️ Target Word"] --> C
    end
    
    subgraph "Geographic Context: River Bank"
        D["River [1,8]<br/>🌊 Geographic Anchor"] --> F["Bank moves toward<br/>geographic semantic space<br/>🏞️ Y_bank_geographic"]
        E["Bank [7,3]<br/>🏛️ Same Target Word"] --> F
    end
    
    subgraph "Key Realization"
        G["Same word 'Bank'<br/>Different contextual<br/>geometric destinations<br/>🎯 Context-sensitive positioning"]
    end
    
    C --> G
    F --> G
    
    style G fill:#e91e63
```

## Interactive Geometric Exploration

### Creating Your Own Visualizations

**🎯 Python Libraries for Geometric Visualization**
```python
# Essential libraries for geometric self-attention visualization
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# For interactive 3D visualizations
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

# For transformer model exploration
from transformers import AutoModel, AutoTokenizer
import torch
```

**🔬 Online Interactive Playgrounds**
- **Google Colab**: Run BertViz and custom geometric visualizations
- **Observable Notebooks**: Web-based interactive transformer explorations
- **Jupyter Labs**: Local development with interactive widgets

**📊 Recommended Visualization Combinations**
1. **Start with**: Jay Alammar's Illustrated Transformer for conceptual understanding
2. **Explore with**: BertViz for real model attention patterns  
3. **Deep dive with**: Custom matplotlib/plotly geometric visualizations
4. **Experiment with**: Interactive 3D LLM visualization

## Implementation: Bringing Geometry to Code

Let's implement this geometric understanding in code:

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_self_attention_geometry():
    # Step 1: Original embeddings (2D for visualization)
    E_money = np.array([2, 5])
    E_bank = np.array([7, 3])
    
    # Step 2: Weight matrices (hypothetical values)
    W_Q = np.array([[1, 0], [0, 1]])  # Identity for simplicity
    W_K = np.array([[0, 1], [1, 0]])  # Rotation matrix
    W_V = np.array([[1, 1], [0, 1]])  # Shear transformation
    
    # Step 3: Generate Q, K, V vectors
    Q_money = E_money @ W_Q  # [2, 5]
    K_money = E_money @ W_K  # [5, 2]
    V_money = E_money @ W_V  # [7, 5]
    
    Q_bank = E_bank @ W_Q   # [7, 3]
    K_bank = E_bank @ W_K   # [3, 7]
    V_bank = E_bank @ W_V   # [10, 3]
    
    # Step 4: Calculate attention for bank (as in video)
    similarity_1 = np.dot(Q_bank, K_money)  # 10
    similarity_2 = np.dot(Q_bank, K_bank)   # 32
    
    # Step 5: Scale by sqrt(d_k)
    d_k = 2
    scaled_1 = similarity_1 / np.sqrt(d_k)  # 7.09
    scaled_2 = similarity_2 / np.sqrt(d_k)  # 22.6
    
    # Step 6: Apply softmax
    scores = np.array([scaled_1, scaled_2])
    weights = np.exp(scores) / np.sum(np.exp(scores))
    w21, w22 = weights  # [0.2, 0.8] approximately
    
    # Step 7: Weighted combination of value vectors
    Y_bank = w21 * V_money + w22 * V_bank
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original embeddings
    plt.arrow(0, 0, E_money[0], E_money[1], 
              head_width=0.2, head_length=0.3, fc='red', ec='red')
    plt.arrow(0, 0, E_bank[0], E_bank[1], 
              head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    
    # Plot final contextual embedding
    plt.arrow(0, 0, Y_bank[0], Y_bank[1], 
              head_width=0.2, head_length=0.3, fc='green', ec='green', linewidth=3)
    
    # Annotations
    plt.annotate('E_money', xy=E_money, xytext=(E_money[0]+0.5, E_money[1]+0.5))
    plt.annotate('E_bank', xy=E_bank, xytext=(E_bank[0]+0.5, E_bank[1]+0.5))
    plt.annotate('Y_bank (contextual)', xy=Y_bank, xytext=(Y_bank[0]+0.5, Y_bank[1]+0.5))
    
    plt.grid(True, alpha=0.3)
    plt.title('Geometric Visualization of Self-Attention')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(['Original Money', 'Original Bank', 'Contextual Bank'])
    
    return Y_bank, weights

# Execute visualization
final_embedding, attention_weights = visualize_self_attention_geometry()
print(f"Final contextual embedding: {final_embedding}")
print(f"Attention weights: {attention_weights}")
```

## Key Takeaways: The Geometric Essence

- **Self-Attention as Geometric Transformation**: Every step in self-attention has a clear geometric interpretation - from linear transformations to vector compositions

- **Contextual Gravity**: Self-attention acts like a gravitational force where semantically related words pull each other in embedding space

- **Context-Dependent Positioning**: The same word can end up in different regions of semantic space depending on its contextual companions

- **Dynamic vs Static Embeddings**: Traditional embeddings are static points, while self-attention creates dynamic, context-aware positions

- **Geometric Intuition for Complex Operations**: Complex mathematical operations become intuitive when viewed as geometric transformations in high-dimensional space

## Research Impact and Modern Applications

This geometric understanding has profound implications:

**Transformer Architecture**: The geometric intuition explains why Transformers are so effective - they perform sophisticated geometric transformations that capture complex contextual relationships

**Large Language Models**: Models like GPT and BERT leverage these geometric principles across multiple layers and attention heads, creating incredibly nuanced contextual representations

**Multimodal Understanding**: The same geometric principles extend to vision-language models, where images and text inhabit shared geometric spaces

## References

1. **Vaswani, A., et al.** (2017). Attention Is All You Need. *NIPS 2017*.
2. **Rogers, A., et al.** (2020). A Primer on Neural Network Architectures for Natural Language Processing. *Journal of Artificial Intelligence Research*.
3. **Geometric Deep Learning** - Bronstein et al., 2017

[End of Notes]