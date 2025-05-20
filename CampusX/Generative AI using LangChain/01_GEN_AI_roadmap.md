# Generative AI using LangChain - Roadmap
* **Why Learn Generative AI?**
    * **Definition of Generative AI**: A type of AI that creates new content (text, images, music, code) by learning patterns from existing data, mimicking **human creativity**.
    * Brief History of AI:
        * Work on AI has been ongoing for 60-70 years.
        * Various techniques evolved:
            * 1980s: **Symbolic AI** (used for **Expert Systems**).
            * **Fuzzy Logic**, **Evolutionary Algorithms**.
            * Separate development in **NLP (Natural Language Processing)** and **Computer Vision**.
        * **Machine Learning (ML)** emerged as a highly impactful technique.
            * ML uses mathematical or statistical models, trained on data, to find patterns and make predictions.
            * Common ML problems: **Regression** (predicting numbers, e.g., stock prices), **Classification** (e.g., cat vs. dog image), **Ranking** (e.g., recommendation systems).
            * Traditionally, ML was not used for tasks requiring **human creativity**.
    * **Generative AI's** core strength: its ability to create *new* content, effectively **mimicking human creativity**, a feat previously thought to be far off.
    * Position in the AI Landscape (Mental Model):
        * Outermost: **AI (Artificial Intelligence)**.
        * Inside AI: **Machine Learning (ML)**.
        * Inside ML: **Deep Learning** (uses neural network-based models; **Transformer Architecture** was a key advancement).
        * Inside Deep Learning: **Generative AI**.

* **Impact Areas of Generative AI**
    * **Customer Support**:
        * Significant impact through AI chatbots handling first-level customer queries.
        * Reduces the need for a large human workforce, making it cost-effective.
    * **Content Creation**:
        * AI tools are now prevalent for creating text (blogs, websites), videos, and other media.
        * It's becoming difficult to distinguish AI-generated content from human-created content.
    * **Education**:
        * Tools like **ChatGPT** act as personal tutors, making learning and exploring new topics easier.
        * Aids in curriculum planning and generating practice questions.
    * **Software Development**:
        * Generative models are proficient at writing **production-ready code**.
        * Potential to change team structures, possibly reducing the number of programmers needed for certain tasks.

* **Is Generative AI a Successful Technology?**
    * Comparison with **Internet** (highly successful) and **Blockchain/Crypto** (full potential perhaps not yet reached).
    * Criteria for success:
        * **Solves real-world problems?**: **Yes** (e.g., customer handling, personalized education).
        * **Useful on a daily basis?**: **Yes**.
        * **Impacting world economics?**: **Absolutely Yes** (e.g., a new Chinese AI model, **DeepSeek Coder V2 (Mistake in transcript, likely referring to a new model, DeepSeek-AI or similar, impact was on US tech stocks)**, reportedly caused a $1 trillion wipeout in US tech company stock shares).
        * **Creating new jobs?**: **Yes** (e.g., the rise of the **AI Engineer** role, demand is increasing).
        * **Accessible?**: **Yes** (tools can be used via natural language, no coding needed for basic use).
    * Conclusion: **Generative AI** is on the path to becoming a highly successful technology, similar to the internet.

* **Speaker's Reasons for Delay in Creating Generative AI Content**
    * **Doubt about the technology**: Initially unsure if it was a truly powerful technology or just a **hype bubble** (now resolved).
    * **Time commitment issues**: Personal life commitments (now resolved).
    * **Fear of the rapid pace of development**: Constant new models, tools, papers, and terminology made it daunting to teach (this led to the curriculum design).

* **Challenges in Learning/Teaching Generative AI**
    * **Rapid Growth**: Daily advancements make it hard to track and extract meaningful information.
    * **Noise and FOMO**: Social media and online discourse create a "Fear Of Missing Out," leading to demotivation.
    * **Lack of a Single Source of Truth**: No established, comprehensive resource exists yet due to the novelty of the field.
    * These factors made **curriculum design** difficult.

* **Mental Model for Understanding Generative AI**
    * The central concept is **Foundation Models**.
        * **Foundation Models** are very large-scale AI models.
        * Trained on **huge amounts of data** (often internet-scale).
        * Require significant **hardware (GPUs)** and incur high training costs (crores of rupees).
        * Key characteristic: They are **generalized models**, not task-specific like traditional ML models. They can perform multiple tasks.
        * Example: **LLMs (Large Language Models)** are the backbone of current **Generative AI**. They can do text generation, sentiment analysis, summarization, Q&A, etc.
        * Also includes **LMMs (Large Multimodal Models)** that work with text, images, videos, and sound.
    * Two Core Activities in **Generative AI**:
        * **Using Foundation Models**: User perspective, application building.
        * **Building Foundation Models**: Builder perspective, typically by large companies.
    * Categorizing terms based on this model:
        * **Prompt Engineering**: **User side** (refining input to LLMs for better output).
        * **RLHF (Reinforcement Learning from Human Feedback)**: **Builder side** (modifying LLM behavior for safety/alignment).
        * **RAG (Retrieval Augmented Generation)**: **User side** (enabling LLMs to answer questions on private documents).
        * **Pre-training**: **Builder side** (the initial, large-scale training of foundation models).
        * **Quantization**: **Builder side** (optimizing model size and performance).
        * **AI Agents**: **User side** (LLMs empowered to perform tasks, not just Q&A).
        * **Vector Databases**: **User side** (used in implementing RAG).
        * **Fine-tuning**: Applicable to **both sides** (adapting a pre-trained model for specific tasks or domains).

* **Proposed Curriculum: Builder Perspective**
    * Focus: How **Foundation Models** are built and deployed. More technical.
    * **Prerequisites**:
        * **Machine Learning fundamentals**.
        * **Deep Learning fundamentals**.
        * Familiarity with a **Deep Learning framework (TensorFlow or PyTorch**, PyTorch preferred).
    * Modules:
        1.  **Transformer Architecture**: In-depth understanding (encoder, decoder, embeddings, self-attention, layer normalization, language modeling).
        2.  **Types of Transformers**: Encoder-only (e.g., BERT), Decoder-only (e.g., GPT), Encoder-Decoder models.
        3.  **Pre-training**: Training objectives, tokenization strategies, training strategies (distributed training), challenges, evaluation.
        4.  **Optimization**: Techniques to make models runnable on normal hardware (e.g., **Quantization**, **Knowledge Distillation**), inference optimization.
        5.  **Fine-tuning**: Task-specific tuning, instruction tuning, continual pre-training, **RLHF**, **PEFT (Parameter-Efficient Fine-Tuning)**.
        6.  **Evaluation**: Metrics, benchmarks, understanding LLM leaderboards.
        7.  **Deployment**: Making trained models accessible.

* **Proposed Curriculum: User Perspective**
    * Focus: How to use existing **Foundation Models** to build applications. Less technical, potentially easier and more fun.
    * Modules:
        1.  **Building Basic LLM Apps**:
            * Understanding types of LLMs (closed-source via **APIs**, open-source).
            * Using tools/libraries like **Hugging Face**, **Ollama** (for local execution), **LangChain** (for building LLM applications).
        2.  **Improving LLM Responses**:
            * **Prompt Engineering**: The art and science of crafting effective prompts.
            * **RAG (Retrieval Augmented Generation)**: Connecting LLMs to private data.
            * **Fine-tuning** (at a user/application level).
        3.  **AI Agents**: Creating applications where LLMs can take actions or use tools (e.g., booking a hotel).
        4.  **LLM Ops**: The process of deploying, monitoring, evaluating, and improving LLM-based applications in production.
        5.  **Miscellaneous**: Working with **Multimodal Foundation Models** (handling audio, video), **Diffusion-based Models** (e.g., Stable Diffusion for image generation).

* **Synergy Between Builder and User Perspectives**
    * **Builder Side**: Typically a **Research Scientist** or **Data Scientist** role, requires ML engineering and MLOps skills.
    * **User Side**: Accessible to anyone with basic **software development** skills for about 80-85% of tasks.
    * The **AI Engineer** role is emerging as someone proficient in both, capable of building applications on top of LLMs with a deeper understanding of how they work. This knowledge leads to better applications and potentially higher salaries.
    * Recommendation: Learn both, but focus can vary based on career goals.
 
 