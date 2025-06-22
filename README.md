# 🤖 AI Learning and Development Hub

## 📚 Overview

Welcome to the AI Learning and Development Hub! This repository is a comprehensive personal learning notebook and development environment for Artificial Intelligence, Machine Learning, Deep Learning, and Coding Interview preparation. It features an intelligent memory system based on the Ebbinghaus Forgetting Curve to optimize long-term retention of complex AI/ML concepts.

## 🧠 Memory Management System

This repository implements a spaced repetition system based on the forgetting curve theory to help combat the natural decline in memory retention:
- **Automated Topic Tracking**: JSON-based storage of learning progress
- **Spaced Repetition Algorithm**: Schedules reviews based on memory strength
- **Interactive UI**: Single-file interface for topic management and revision
- **Progress Analytics**: Track learning efficiency and retention rates

## Repository Structure

This repository is organized into several top-level directories, each serving a specific purpose:

### 🗂️ Core Learning Directories

*   **📊 `CampusX/`**: Comprehensive learning materials from CampusX courses
    *   **🎯 `100 days-ml/`**: 100-day ML program (24 topics completed)
    *   **🔗 `Generative AI using LangChain/`**: 17 modules on GenAI with practical code
    *   **🧠 `100-days-deep-learning/`**: Deep learning fundamentals to CNNs (45 topics)

*   **🎓 `Courseera-Courses/`**: Coursera specialization materials
    *   **🐍 `Machine Learning with Python/`**: Core ML concepts and implementations

*   **💻 `coding/`**: Coding interview preparation (DSA)
    *   **🔍 `Binary_search/`**: 16 binary search problems with solutions
    *   **🪟 `Sliding_window/`**: 8+ sliding window technique problems
    *   **⚡ `dp/`**: Dynamic programming (Knapsack patterns)
    *   **🌐 `graph/`**: Graph algorithms (BFS, DFS, Dijkstra)
    *   **🔄 `recursion/`**: 15 recursion problems with patterns
    *   **📚 `stack/`**: Stack-based algorithmic problems
    *   **❓ `_questions/`**: Curated problem lists (LeetCode, etc.)

### 🛠️ Tools & Applications

*   **🃏 `anki_cards_app/`**: 3D flashcard application with spaced repetition
*   **🏆 `kaggle_comp/`**: Kaggle competition solutions and approaches
*   **⚡ `marathons/`**: Intensive learning sprints (CNN, C++, Python)
*   **🔧 `4-Tools for AI Development/`**: Essential development tools (SQLite, etc.)
*   **📊 `data/`**: Datasets and data utilities
*   **💾 `static/`**: Shared web assets (CSS, JS, templates)
*   **Other Files**:
    *   `LICENSE`: The license for this repository.
    *   `README.md`: This file!
    *   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
    *   `Coding.md`: General notes related to coding.
    *   `quiz_template.html`, `sample_quiz.html`: Templates for creating quizzes.
    *   `start_spark_notebook.bat`: A batch script for starting Spark notebooks.

## 🎯 Learning Pathways

### 🤖 Machine Learning Track
- **Foundation**: [What is ML](CampusX/100%20days-ml/1-What%20is%20ML/) → [AI vs ML vs DL](CampusX/100%20days-ml/2-AI%20vs%20ML%20vs%20DL/) → [Types of ML](CampusX/100%20days-ml/3-Types%20Of%20ML/)
- **Data Handling**: [Understanding Data](CampusX/100%20days-ml/19%20-%20understanding%20your%20data/) → [EDA](CampusX/100%20days-ml/20%20-%20eda%20using%20univariate%20Analysis/) → [Feature Engineering](CampusX/100%20days-ml/23%20What%20is%20feature%20engineering/)
- **Advanced**: [Online ML](CampusX/100%20days-ml/5%20-%20online%20machine%20learning/) → [ML Challenges](CampusX/100%20days-ml/7%20-%20Challenges%20in%20%20Machine%20learning/)

### 🧠 Deep Learning Track  
- **Basics**: [What is Deep Learning](CampusX/100-days-deep-learning/02-what-isdeep-learning.md) → [Neural Networks](CampusX/100-days-deep-learning/03-types-of-neural-network.md)
- **Core Concepts**: [Perceptron](CampusX/100-days-deep-learning/04-what-is-perceptron.md) → [MLP](CampusX/100-days-deep-learning/09-mlp-multi-layer-perceptron.md) → [Backpropagation](CampusX/100-days-deep-learning/15-backpropagation-in-deep-learning-part1.md)
- **Advanced**: [CNNs](CampusX/100-days-deep-learning/40-cnn.md) → [CNN Architecture](CampusX/100-days-deep-learning/45-cnn-architechture.md)

### 🔗 Generative AI Track
- **LangChain Fundamentals**: [Introduction](CampusX/Generative%20AI%20using%20LangChain/03_Introduction_to_LangChain.md) → [Components](CampusX/Generative%20AI%20using%20LangChain/04_Langchain_components.md)
- **Implementation**: [Models](CampusX/Generative%20AI%20using%20LangChain/05_Langchain_model.md) → [Chains](CampusX/Generative%20AI%20using%20LangChain/09_chains_in_langchain.md) → [RAG](CampusX/Generative%20AI%20using%20LangChain/16_what_is_rag.md)

## 💡 Coding Interview Preparation

### 🔍 Algorithmic Patterns
- **🎯 Binary Search**: [16 problems](coding/Binary_search/) covering search variations
- **🪟 Sliding Window**: [8+ techniques](coding/Sliding_window/) for subarray problems  
- **⚡ Dynamic Programming**: [Knapsack patterns](coding/dp/) and optimization
- **🌐 Graph Algorithms**: [BFS/DFS](coding/graph/) implementations and applications
- **🔄 Recursion**: [15 problems](coding/recursion/) with systematic approaches
- **📚 Stack**: [Advanced problems](coding/stack/) including histogram and matrix

### 📋 Problem Collections
- **🌟 Top Lists**: [LeetCode 75](coding/_questions/_100_al/), [Blind 75](coding/_questions/_150/)
- **🏢 Company Specific**: [Google](coding/_questions/_g/), [Microsoft](coding/_questions/_s/)
- **📈 Frequency Based**: Recent and trending problems

## 🚀 Applications & Tools

### 🎯 Learning Applications
- **[3D Anki Cards](anki_cards_app/)**: Interactive flashcard app with spaced repetition
- **[Memory Manager](memory_system/)**: Forgetting curve-based learning optimizer
- **[Quiz Generator](static/)**: Template system for creating interactive quizzes

### 🏆 Project Showcase  
- **[Kaggle Competitions](kaggle_comp/)**: Competition solutions and analysis
- **[Learning Marathons](marathons/)**: Intensive topic deep-dives
- **[Data Tools](data/)**: Utilities for dataset management and processing

## 🧠 How to Use the Memory System

1. **Track Learning**: Topics are automatically indexed from your file structure
2. **Schedule Reviews**: Based on Ebbinghaus forgetting curve principles
3. **Monitor Progress**: JSON-based storage tracks your learning efficiency
4. **Optimize Retention**: Smart intervals prevent forgetting before it happens

### Memory System Files
```
memory_system/
├── memory_tracker.json     # Learning progress database
├── topic_scheduler.py      # Spaced repetition algorithm
└── learning_dashboard.html # Single-file UI interface
```

## 📊 Repository Statistics

- **📝 Study Notes**: 45+ Deep Learning topics, 24+ ML concepts, 17 LangChain modules
- **💻 Code Solutions**: 60+ coding problems with detailed explanations
- **🎯 Interactive Tools**: 3D flashcards, quiz templates, memory tracker
- **📚 Learning Paths**: Structured roadmaps for ML, DL, and GenAI

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:
- Report issues with existing content
- Suggest new learning resources
- Improve the memory system algorithm
- Enhance the learning dashboard UI

---

**Happy Learning!** 🎓✨
