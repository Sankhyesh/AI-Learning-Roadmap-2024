# GPU-Enabled Deep Learning Docker Environment

This Docker setup provides a complete deep learning environment with GPU support for PyTorch, TensorFlow, and other popular ML/DL frameworks.

## Prerequisites

1. **NVIDIA GPU** with CUDA capability
2. **NVIDIA Docker** (nvidia-docker2) installed
3. **Docker** and **Docker Compose** installed
4. **NVIDIA drivers** installed on host system

## Quick Start

1. Build the Docker image:
```bash
cd docker
docker-compose build
```

2. Start the container:
```bash
docker-compose up -d
```

3. Access Jupyter Lab:
- Open browser: http://localhost:8888
- No password required (configured for development)

4. Test GPU availability:
```bash
docker-compose exec deep-learning python test_environment.py
```

## Included Frameworks & Libraries

### Deep Learning
- PyTorch 2.1.0 with CUDA 11.8
- TensorFlow 2.14.0 with GPU support
- Keras, FastAI, PyTorch Lightning
- Transformers (Hugging Face)

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM

### Data Science
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn, Plotly
- OpenCV
- Jupyter Lab

## Port Mappings

- **8888**: Jupyter Lab
- **6006**: TensorBoard
- **8501**: Streamlit apps
- **5000**: Flask/FastAPI apps

## Volume Mounts

The following directories are mounted:
- `./notebooks` → `/workspace/notebooks`
- `./data` → `/workspace/data`
- `./models` → `/workspace/models`
- `./scripts` → `/workspace/scripts`

## Useful Commands

### Enter container shell:
```bash
docker-compose exec deep-learning bash
```

### Run Python script:
```bash
docker-compose exec deep-learning python scripts/your_script.py
```

### Monitor GPU usage:
```bash
docker-compose exec deep-learning nvidia-smi
```

### View logs:
```bash
docker-compose logs -f deep-learning
```

### Stop container:
```bash
docker-compose down
```

## Troubleshooting

### GPU not detected:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check Docker runtime: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Ensure nvidia-docker2 is installed

### Out of memory errors:
- Reduce batch size in your models
- Use gradient accumulation
- Enable mixed precision training

### Package conflicts:
- Rebuild image after modifying requirements.txt
- Use `--no-cache` flag: `docker-compose build --no-cache`