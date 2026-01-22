#!/bin/bash

# GPT-OSS-20B ONNX Model Download Script
# This script automates the download and setup of the GPT-OSS-20B ONNX model

set -e  # Exit on error

echo "========================================="
echo "GPT-OSS-20B ONNX Model Download Script"
echo "========================================="
echo ""

# Step 1: Install Hugging Face CLI
echo "[1/5] Installing Hugging Face CLI..."
pip install -q huggingface_hub[cli]
echo "✓ Hugging Face CLI installed"
echo ""

# Step 2: Download the model
echo "[2/5] Downloading GPT-OSS-20B ONNX model..."
echo "This may take a while depending on your internet connection..."
huggingface-cli download onnxruntime/gpt-oss-20b-onnx \
  --include cuda/cuda-int4-kquant-block-32-mixed/* \
  --local-dir .
echo "✓ Model downloaded"
echo ""

# Step 3: Install ONNX Runtime GenAI CUDA
echo "[3/5] Installing ONNX Runtime GenAI CUDA package..."
pip install -q onnxruntime-genai-cuda
echo "✓ ONNX Runtime GenAI CUDA installed"
echo ""

# Step 4: Download example script (if not exists)
if [ ! -f "model-chat.py" ]; then
    echo "[4/5] Downloading example script..."
    curl -s https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-chat.py -o model-chat.py
    echo "✓ Example script downloaded"
else
    echo "[4/5] Example script already exists, skipping..."
fi
echo ""

# Step 5: Verify installation
echo "[5/5] Verifying installation..."
if [ -d "cuda/cuda-int4-kquant-block-32-mixed" ]; then
    echo "✓ Model directory found: cuda/cuda-int4-kquant-block-32-mixed"
    echo ""
    echo "========================================="
    echo "Download completed successfully!"
    echo "========================================="
    echo ""
    echo "To test the model, run:"
    echo "  python model-chat.py -m cuda/cuda-int4-kquant-block-32-mixed -e follow_config"
    echo ""
    echo "Or start the API server:"
    echo "  python api.py"
    echo "  # or"
    echo "  ./run_api.sh"
else
    echo "✗ Error: Model directory not found!"
    echo "Please check the download and try again."
    exit 1
fi
