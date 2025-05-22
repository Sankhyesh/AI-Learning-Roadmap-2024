# Deep Learning Roadmap Overview

---

## Resource Links

### Maths Fundamentals for Deep Learning

- [Maths for ML playlist by CampusX](https://www.youtube.com/playlist?list=PLKnIA16_RmvbYFaaeLY28cWeqV-3vADST)
- [Linear Algebra by 3blue1brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Linear Algebra by Khan Academy](https://www.khanacademy.org/math/linear-algebra)

### Python and Libraries

- [Python by CampusX](https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z)

### Machine Learning Fundamentals

- [100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHHC)

### Courses

- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [CS224N: NLP with Deep Learning (Stanford)](https://web.stanford.edu/class/cs224n/)
- [Practical Deep Learning by Jeremy Howard](https://course.fast.ai/)
- [CS231n: Deep Learning for Computer Vision (Stanford)](https://cs231n.stanford.edu/)
- [Deep Learning - IIT Madras B.S. Degree](https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM)

### YouTube Channels

- [100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)
- [Neural Networks by 3B1B](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Neural Networks by Luis Serrano](https://www.youtube.com/playlist?list=PLs8w1Cdi-zvavXlPXEAsWIh4Cgh83pZPO)
- [Yannic Kilcher](https://www.youtube.com/@YannicKilcher/playlists)
- [Code Emporium](https://www.youtube.com/@CodeEmporium)
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
- [Umar Jamil](https://www.youtube.com/@umarjamilai/videos)
- [Aladdin Persson (PyTorch)](https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson)
- [Deep Learning by Statquest](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)

### Websites

- [Towards Data Science](https://towardsdatascience.com/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Kaggle](https://www.kaggle.com/)
- [Arxiv](https://arxiv.org/)
- [Papers with code](https://paperswithcode.com/)

---

![Neural Network Structure](images/neural_network_example.png)
_Example: Structure of a simple neural network._

---

## Deep Learning Roadmap Overview

The core message is to provide a structured, detailed, and resource-rich pathway to mastering Deep Learning. A systematic approach can lead to significant proficiency. The roadmap is broken down into five key stages:

- **Prerequisites:** Essential foundational knowledge.
- **Curriculum:** A detailed breakdown of Deep Learning topics.
- **Projects:** Practical application of learned concepts.
- **Tools:** Key software and platforms.
- **Resources:** A curated list of learning materials.

---

## 1. Prerequisites to Learn Deep Learning

Certain foundational skills are crucial before diving into Deep Learning.

### Mathematical Foundations

- **Linear Algebra:** Essential for understanding data representation (vectors, matrices, tensors) and manipulations.
- **Calculus:** Primarily for optimization concepts; key topics include differentiation, partial differentiation, and the chain rule.
- **Probability:** Needed for processing results and understanding model behaviors.
- **Statistics:** Basic statistical concepts are used throughout Deep Learning.

### Programming Skills

- **Python Basics:** The dominant language in AI/ML.
- **Python Libraries:** Proficiency in NumPy (numerical operations), Pandas (data manipulation), and Matplotlib (plotting) is expected.

### Machine Learning (ML) Basics

- Understanding the ML workflow, including Data Processing, Model Building, Hyperparameter Tuning, and Model Evaluation. Deep Learning is a subfield of ML.

---

## 2. Detailed Deep Learning Curriculum

The curriculum is structured into several main parts, progressing from basic neural networks to advanced architectures.

### Part 1: Artificial Neural Networks (ANNs) & How to Improve Them

#### Foundations

- **Biological Inspiration:** Understanding biological neuron structure, synapses, and how these concepts translate to artificial neurons.
- **History of Neural Networks:** Covering early models like the Perceptron, the advent of Backpropagation and Multilayer Perceptrons (MLPs), the "AI Winter," the resurgence with neural networks, and the emergence of deep learning.
- **Perceptron and MLPs:** Discussing single-layer perceptron limitations, the XOR problem, and the need for hidden layers in MLP architecture.
- **Layers and Their Functions:** Input Layer (accepting data), Hidden Layers (feature extraction), and Output Layer (producing predictions).
- **Activation Functions:** Including Sigmoid Function, Hyperbolic Tangent (tanh), ReLU (Rectified Linear Unit) and its advantages in mitigating vanishing gradients, Leaky ReLU, Parametric ReLU (addressing dying ReLU problem), and Softmax Function for multi-class classification.

#### Training Fundamentals

- **Forward Propagation:** Mathematical computations at each neuron, passing inputs through the network.
- **Loss Functions:** Such as Mean Squared Error (MSE) for regression, Cross-Entropy Loss for classification, and Hinge Loss (used with SVMs); selecting appropriate loss functions based on tasks.
- **Backpropagation:** Derivation using the chain rule, computing gradients, updating weights and biases, and understanding computational graphs.
- **Gradient Descent Variants:** Batch Gradient Descent, Stochastic Gradient Descent (SGD) and its advantages in large datasets, and Mini-Batch Gradient Descent as a balance.

#### Improving Neural Networks

- **Optimization Algorithms:** Momentum (accelerating SGD), Nesterov Accelerated Gradient, AdaGrad (adaptive learning rates), RMSProp (fixing AdaGrad's diminishing learning rates), and Adam (combining momentum and RMSProp).
- **Regularization Techniques:** L1 and L2 Regularization (adding penalty terms), Dropout (preventing overfitting by randomly dropping neurons), and Early Stopping (halting training when validation loss increases).
- **Hyperparameter Tuning:** Key hyperparameters like Learning Rate, Batch Size, Number of Epochs, and Network Architecture (depth and width). Techniques include Grid search, Random Search, and Bayesian optimization.
- **Vanishing and Exploding Gradients:** Problems in deep networks and solutions like proper weight initialization and the use of ReLU activation functions.
- **Weight Initialization Strategies:** Xavier/Glorot Initialization and He Initialization.
- **Batch Normalization:** Normalizing inputs of each layer to accelerate training and reduce dependence on initialization.

---

### Part 2: Convolutional Neural Networks (CNNs)

#### Context

- Addressing challenges with MLPs for image data like high dimensionality and lack of spatial invariance; advantages of CNNs include parameter sharing and local connectivity.

#### Core Components

- **Convolution Operation:** Understanding Kernels/Filters (e.g., edge detection, feature extraction), mathematical representation (2D and 3D convolution), and hyperparameters like kernel size, depth, stride, and padding (controlling output dimensions, types: same vs. valid).
- **Activation Functions in CNNs:** Primarily ReLU and its variants like Leaky ReLU and ELU (Exponential Linear Unit).
- **Pooling Layers:** For dimensionality reduction and translation invariance; types include Max pooling and Average pooling, considering pooling size and stride.
- **Fully Connected Layers:** Transitioning from convolutional layers by flattening 2D features to 1D.
- **Loss Functions for CNNs:** Typically Cross-Entropy Loss for classification and Mean Squared Error for regression.

#### Architecture and Training

- **CNN Architecture:** Involves layer stacking (Convolutional -> Activation -> Pooling), understanding Feature Maps (depth and channels), and visualization of learned features.
- **Data Preprocessing:** Includes Data Normalization (scaling pixel values, standardization) and Data Augmentation (rotation, flipping, cropping, color jitter, noise addition) to reduce overfitting and increase dataset diversity.

#### Key CNN Architectures and Innovations

- **LeNet-5:** Details on layers, activations, and its contribution to handwritten digit recognition.
- **AlexNet:** Breakthroughs including a deeper network and use of ReLU, impacting the ImageNet Challenge.
- **VGG Networks (VGG-16, VGG-19):** Design philosophy of using small filters (3x3) in deep, uniform architectures.
- **Inception Networks (GoogLeNet):** Featuring Inception Modules with parallel convolutional layers for efficient computation.
- **ResNet (Residual Networks):** Utilizing Residual Blocks with identity mappings (shortcut connections) to solve the vanishing gradient problem; variants include ResNet-50, ResNet-101.
- **MobileNets:** Employing depthwise separable convolutions for optimization on mobile devices.

#### Advanced Applications

- **Pre-trained Models & Transfer Learning:** Using models trained on ImageNet, differentiating between Fine-Tuning vs. Feature Extraction.
- **Object Detection and Localization (Optional):** Traditional methods (Sliding Window) vs. Modern Architectures like Region-Based CNNs (R-CNN, Fast R-CNN, Faster R-CNN), You Only Look Once (YOLO), Single Shot MultiBox Detector (SSD), and Mask R-CNN for instance segmentation.
- **Semantic Segmentation:** Using Fully Convolutional Networks (FCN) by replacing fully connected layers, and U-Net with its encoder-decoder architecture and skip connections.
- **Generative Models with CNNs:** Autoencoders (Convolutional Autoencoders for image reconstruction), Variational Autoencoders (VAE), and Generative Adversarial Networks (GANs) like DCGAN (using CNNs in GANs) for applications like image generation and super-resolution.

---

### Part 3: Recurrent Neural Networks (RNNs)

#### Context

- Addressing sequential data challenges.

#### Core Components

- Basic RNN structure, mathematical formulation, and activation functions.

#### Training

- **Forward Propagation Through Time:** Processing sequence inputs, handling variable-length sequences, and generating outputs.
- **Backpropagation Through Time (BPTT):** Unfolding the RNN, calculating gradients using the chain rule through time steps, and considering computational complexity.

#### Challenges and Solutions

- **Training Challenges:** Vanishing Gradients (gradients diminish over long sequences) and Exploding Gradients (gradients grow exponentially).
- **Solutions:** Gradient clipping and advanced architectures like LSTMs and GRUs.

#### Advanced RNN Architectures

- **LSTM (Long Short-Term Memory):** Core components, gates, intuition, and BPTT.
- **GRU (Gated Recurrent Unit):** Core components, gates, intuition, BPTT, and comparison with LSTM.
- **Deep RNNs:** Stacking RNN layers, challenges with vanishing/exploding gradients in deep RNNs, using LSTMs/GRUs, residual connections, and regularization.
- **Bidirectional RNNs:** Motivation, architecture (forward and backward pass, combining outputs), and Bidirectional LSTMs.

#### Applications

- Language modeling (next word prediction), Sentiment Analysis, POS Tagging, and Time series forecasting.

---

### Part 4: Seq2Seq Networks & Transformers

This section is crucial for understanding Gen AI and LLMs.

#### Encoder-Decoder Networks

- **Introduction:** Purpose is handling variable-length input and output sequences, essential for tasks like machine translation, text summarization, and speech recognition.
- **Components:** Encoder (processes input to a fixed-length context vector, typically using RNNs, LSTMs, or GRUs) and Decoder (generates output from the context vector).
- **Implementation:** Handling variable-length sequences (padding, masking), loss functions (Cross-Entropy), and training techniques like Teacher Forcing.
- **Limitations:** Fixed-length context vector bottleneck; solved by attention mechanisms.

#### Attention Mechanisms

- **Motivation:** Overcoming the bottleneck by allowing the model to access all encoder hidden states, improving performance on long sequences.
- **Types:** Additive Attention (Bahdanau Attention) (uses a feedforward network, more computationally intensive) and Multiplicative Attention (Luong Attention) (uses dot products, more efficient).
- **Steps:** Calculate alignment scores, compute attention weights, compute context vector, update decoder state.
- **Implementation:** Integrating with the decoder, training adjustments, and visualization of attention weights.

#### Transformer Architectures

- **Context:** Addressing limitations of RNN-based Seq2Seq models like sequential processing hindering parallelization and difficulty with long-term dependencies.
- **Key Innovations:** Self-Attention Mechanism (relating different positions of a single sequence) and Positional Encoding (injecting position information). Advantages include improved parallelization and capturing global dependencies.

##### Components

- **Multi-Head Self-Attention:** Multiple attention mechanisms (heads) operating in parallel; computes Query (Q), Key (K), and Value (V) matrices.
- **Positional Encoding:** Provides position information using techniques like sinusoidal functions or learned embeddings.
- **Feedforward Networks:** Position-wise fully connected layers applied independently.
- **Layer Normalization:** Normalizes inputs across features to stabilize training.
- **Residual Connections:** Adding layer input to its output to help train deeper networks.
- **Structure:** Encoder Stack (multiple layers with multi-head self-attention and feedforward networks) and Decoder Stack (similar to encoder but includes masked multi-head self-attention and encoder-decoder attention).
- **Implementation:** Embedding layer, adding positional encoding, building encoder/decoder layers, and an output layer (often with softmax).

##### Types of Transformers

- **BERT (Bidirectional Encoder Representations from Transformers):** Pre-training deep bidirectional representations using only the encoder part; pre-training objectives include Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- **GPT (Generative Pre-trained Transformer):** Focused on language generation, using only the decoder part with masked self-attention; training objective is Causal Language Modeling (CLM).
- **Other notables:** RoBERTa, ALBERT, T5 (Text-to-Text Transfer Transformer).

##### Fine-Tuning Transformers

- **Concept:** Adapting a pre-trained model to a downstream task using Transfer Learning.
- **Steps:** Loading pre-trained model, modifying output layers, adjusting hyperparameters, and training on task-specific data.
- **Best Practices:** Layer-wise learning rates, avoiding catastrophic forgetting, regularization.
- **Common Tasks:** Text Classification, Named Entity Recognition, Question Answering, Text Summarization.

##### Pre-Training Transformers

- **Objectives:** MLM, CLM, Sequence-to-Sequence pre-training.
- **Data Preparation:** Large, diverse corpus selection (e.g., Wikipedia), tokenization strategies (WordPiece, Byte-Pair Encoding (BPE)).
- **Training Strategies:** Distributed Training (multiple GPUs/TPUs), Mixed Precision Training, Adam optimizer with weight decay (AdamW).
- **Challenges:** Compute resources, data quality.
- **Evaluation:** Benchmarking (GLUE, SQuAD), ablation studies.

##### Optimizing Transformers

- **Challenges:** High memory consumption, long training times.
- **Techniques:**
  - Efficient Attention Mechanisms: Sparse Attention, Linearized Attention (Linformer), Reformer.
  - Model Compression: Quantization (reducing weight precision), Pruning (removing weights/neurons), Knowledge Distillation (training a smaller student model from a larger teacher model).
  - Hardware Considerations: GPUs vs. TPUs, parallelism strategies (Data Parallelism, Model Parallelism).
  - Software Tools: Optimized libraries like Hugging Face Transformers, DeepSpeed, NVIDIA Apex.
- **NLP Applications Using Transformers:** Text Classification (Sentiment Analysis, Topic Classification), Question Answering (SQuAD, TriviaQA), Machine Translation (WMT datasets), Text Summarization (CNN/Daily Mail), Language Generation (Chatbots, Story Generation), Named Entity Recognition.

---

### Part 5: Unsupervised Deep Learning

Focus on models that learn from unlabeled data. The roadmap image highlights AutoEncoders and GANs. These are also covered under CNN generative models.

---

## 3. Projects

Hands-on projects are advocated at various stages:

- **ANNs:** MNIST Dataset classification, Sentiment Analysis, Customer Churn prediction, Recommender Systems.
- **CNNs:** Cat Dog Classifier, Emotion Detection from faces, Self Driving Car (likely object detection aspects), X-ray Analysis.
- **RNNs:** Next word prediction, Time Series Forecasting, Sentiment Analysis, POS Tagging.
- **Transformers:** Machine Translation, Text Summarization, Named Entity Recognition (NER), Question Answering.

---

## 4. Tools

A comprehensive list of tools, categorized by their use:

### Deep Learning Libraries

- TensorFlow (with Keras API)
- PyTorch
- HuggingFace TF (Transformers)

### Experimentation & Tracking

- TensorBoard
- MLFlow

### Hyperparameter Tuning (HP Tuning)

- Keras Tuner
- Optuna

### Deployment

- Docker
- FastAPI
- Kubernetes
- TF Serving (TensorFlow Serving)
- TorchServe

### Cloud Platforms

- AWS (Amazon Web Services) (recommended in original transcript)
- GCP (Google Cloud Platform)
- Azure (Microsoft Azure)
- Paperspace

### Distributed Training

- Deepspeed
- Horovod
- PyTorch Lightning

### Pretrained Model Hubs

- HuggingFace
- TF Hub (TensorFlow Hub)
- PyTorch Hub

### Data Versioning

- DVC (Data Version Control)
- Dagster

---

## 5. Resources

A wide array of resources is shared:

### Maths Fundamentals for Deep Learning

- Maths for ML playlist by CampusX
- Linear Algebra by 3Blue1Brown
- Linear Algebra by Khan Academy

### Python and libraries

- Python by CampusX

### Machine Learning Fundamentals

- 100 Days of Machine Learning by CampusX

### Online Courses

- Deep Learning Specialization by Andrew Ng (Coursera)
- CS224N: NLP with Deep Learning (Stanford)
- Practical Deep Learning by Jeremy Howard (fast.ai)
- CS231n: Deep Learning for Computer Vision (Stanford)
- Deep Learning - IIT Madras B.S. Degree

### YouTube Channels

- 100 Days of Deep Learning (CampusX)
- Neural Networks by 3B1B (3Blue1Brown)
- Neural Networks by Luis Serrano
- Yannic Kilcher
- Code Emporium
- Andrej Karpathy
- Umar Jamil
- Aladdin Persson (PyTorch focused)
- Deep Learning by Statquest

### Websites

- Towards Data Science
- Machine Learning Mastery
- Kaggle
- Arxiv
- Papers with code

### Books

- Referenced to a previous video by the original presenter.

---

## Career Paths & Conclusion

After mastering Deep Learning, potential career paths include:

- GenAI / LLM Engineer
- NLP Engineer
- Computer Vision (CV) Engineer

The material concludes by reiterating the transformative power of Deep Learning and encourages learners to approach it with genuine interest and dedication.

---

## Stimulating Learning Prompts

- The curriculum details various Transformer architectures like BERT and GPT, each with unique pre-training objectives (MLM, NSP, CLM). How do these different pre-training strategies influence the models' capabilities and suitability for downstream NLP tasks?
- Considering the Optimization Techniques for Transformers such as efficient attention, model compression (quantization, pruning, knowledge distillation), and hardware considerations, what are the key trade-offs a practitioner must balance when deploying these large models in resource-constrained environments?
