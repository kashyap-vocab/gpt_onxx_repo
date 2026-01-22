#!/bin/bash
# Script to run the FastAPI server for GPT-OSS-20B ONNX model

cd /media/server/be3a703c-2cfd-4e6e-adda-e6bd3c1aeee1/kashyap/gpt-oss-20b-onnx

echo "Starting FastAPI server..."
echo "Model will be loaded on startup (this may take a moment)"
echo ""
echo "API will be available at: http://0.0.0.0:8001"
echo "API docs will be available at: http://0.0.0.0:8001/docs"
echo ""

python api.py
