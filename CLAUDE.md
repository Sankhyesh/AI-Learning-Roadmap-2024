# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive AI/ML learning and development repository containing educational materials, practical implementations, and tools for mastering artificial intelligence concepts. It includes structured courses from CampusX, coding interview preparation, and various AI applications.

## Key Commands

### Python Development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies for Docker environment
pip install -r docker/requirements.txt

# Install minimal dependencies
pip install -r docker/requirements-minimal.txt

# Run Python scripts
python <script_name>.py

# Start Jupyter notebook
jupyter notebook
# or
jupyter lab
```

### Docker Environment

```bash
# Build Docker image
cd docker
docker build -t ai-learning-env .

# Run Docker container with GPU support
docker-compose up -d

# Access Jupyter Lab (runs on port 8888)
# No password required by default
```

### Node.js Applications (hello-genai)

```bash
# Navigate to Node.js chatbot
cd hello-genai/node-genai

# Install dependencies
npm install

# Run in development mode
npm run dev

# Run in production mode
npm start
```

### Memory System

```bash
# The memory system is a single HTML file - no build needed
# Simply open in browser:
open memory_system/revision_planner.html
```

### Code Quality

```bash
# Python linting (if installed)
black .
pylint <file_name>.py

# Run tests (when available)
pytest
```

## Architecture & Code Organization

### Educational Content Structure
- **CampusX/**: Contains structured learning tracks
  - `100 days-ml/`: Machine learning fundamentals (24 topics)
  - `100-days-deep-learning/`: Deep learning from basics to transformers (79 topics)
  - `Generative AI using LangChain/`: LangChain and GenAI modules (17 topics)
  - `Natural Language Processing(NLP)/`: NLP course materials
  - `Practical Deep Learning using PyTorch/`: PyTorch implementations

### Coding Interview Preparation
- **coding/**: Algorithm and data structure problems organized by pattern
  - Each subdirectory contains problems with solutions
  - Focus on common interview patterns (sliding window, binary search, DP, etc.)
  - Problems often include multiple solution approaches

### Applications
- **memory_system/**: Spaced repetition system based on Ebbinghaus forgetting curve
  - Single HTML file with CSV data storage
  - No backend required - runs entirely in browser
  
- **hello-genai/**: Multi-language chatbot implementations
  - Same functionality implemented in Python, Node.js, Go, and Rust
  - Demonstrates connecting to local LLM services

- **anki_cards_app/**: 3D flashcard application for learning

### Key Technologies
- **Primary Language**: Python (PyTorch, TensorFlow, scikit-learn)
- **Web Applications**: HTML/CSS/JavaScript, Node.js/Express
- **Data Science Stack**: NumPy, Pandas, Matplotlib, Seaborn
- **ML/DL Frameworks**: PyTorch, TensorFlow, Keras, FastAI
- **NLP Libraries**: Transformers, NLTK, spaCy
- **Development Tools**: Jupyter, Docker, Git

## Development Patterns

### File Naming Conventions
- Learning materials: `XX-topic-name.md` or `XX_topic_name_notes.ext`
- Python modules: snake_case naming
- Jupyter notebooks: descriptive names with topic focus

### Code Style
- Python: Follow PEP 8, use type hints where beneficial
- Use existing utilities and patterns found in neighboring files
- Check imports and dependencies before adding new libraries

### Working with Learning Materials
- Markdown files contain theoretical content and explanations
- Jupyter notebooks contain practical implementations
- Python scripts demonstrate specific concepts or algorithms

### Documentation and Visualization Standards
- **Mermaid Diagrams**: Extensively use Mermaid diagrams for visual explanations
  - Preferred node types: Circular nodes `((text))` for better visual appeal
  - Flowcharts, sequence diagrams, state diagrams, class diagrams supported
  - Mind maps for concept relationships
  - Pie charts for statistical data
  - Color coding with `style` commands for better clarity
- **Diagram Types**:
  - `graph TD` or `graph LR` for flowcharts
  - `sequenceDiagram` for process flows
  - `stateDiagram-v2` for state machines
  - `classDiagram` for architecture
  - `mindmap` for concept mapping
  - `pie` for data visualization
- **Visual Enhancement**:
  - Use subgraphs for logical grouping
  - Apply consistent color schemes
  - Include directional arrows for process flow
  - Add notes and annotations where helpful

## Important Notes

1. **Virtual Environment**: Always activate the virtual environment before installing packages
2. **GPU Support**: Docker setup includes CUDA support for deep learning
3. **Learning Path**: Materials are structured progressively - earlier topics build foundation for later ones
4. **Memory System**: The revision planner helps optimize learning retention through spaced repetition
5. **No Authentication**: Jupyter Lab in Docker runs without password for local development