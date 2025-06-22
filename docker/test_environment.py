#!/usr/bin/env python3
"""
Test script to verify GPU-enabled deep learning environment
"""

import sys
import subprocess

def test_cuda():
    print("=" * 50)
    print("Testing CUDA availability...")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"PyTorch CUDA test failed: {e}")
    
    print("\n")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        print(f"CUDA built: {tf.test.is_built_with_cuda()}")
        print(f"GPU available: {tf.test.is_gpu_available()}")
    except Exception as e:
        print(f"TensorFlow GPU test failed: {e}")

def test_packages():
    print("\n" + "=" * 50)
    print("Testing installed packages...")
    print("=" * 50)
    
    packages = [
        "numpy", "pandas", "scikit-learn", "matplotlib",
        "transformers", "opencv-cv2", "tensorboard"
    ]
    
    for package in packages:
        try:
            if package == "opencv-cv2":
                import cv2
                print(f"✓ OpenCV version: {cv2.__version__}")
            else:
                module = __import__(package.replace("-", "_"))
                version = getattr(module, "__version__", "Unknown")
                print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: Not installed")

def test_nvidia_smi():
    print("\n" + "=" * 50)
    print("NVIDIA-SMI Output...")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi command failed")
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")

if __name__ == "__main__":
    print("Deep Learning Environment Test")
    print("=" * 50)
    
    test_cuda()
    test_packages()
    test_nvidia_smi()
    
    print("\nTest completed!")