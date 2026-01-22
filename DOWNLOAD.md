# Model Download Guide

This guide provides step-by-step instructions to download and set up the GPT-OSS-20B ONNX model.

## Prerequisites

1. **Python 3.8+** installed
2. **Hugging Face CLI** installed:
   ```bash
   pip install huggingface_hub[cli]
   ```
3. **CUDA-capable GPU** (recommended for best performance)
4. **Git** installed (for downloading the example script)

## Step-by-Step Download Instructions

### Step 1: Install Hugging Face CLI (if not already installed)

```bash
pip install huggingface_hub[cli]
```

### Step 2: Download the Model

Download the GPT-OSS-20B ONNX model with CUDA INT4 quantization:

```bash
huggingface-cli download onnxruntime/gpt-oss-20b-onnx \
  --include cuda/cuda-int4-kquant-block-32-mixed/* \
  --local-dir .
```

This will download the model files to the `cuda/cuda-int4-kquant-block-32-mixed/` directory in your current working directory.

**Note:** The download may take some time depending on your internet connection, as the model files are large (~10-20 GB).

### Step 3: Install ONNX Runtime GenAI CUDA Package

Install the CUDA-enabled version of ONNX Runtime GenAI:

```bash
pip install onnxruntime-genai-cuda
```

**Alternative:** If you don't have CUDA or want to use CPU instead:
```bash
pip install onnxruntime-genai
```

### Step 4: Download the Example Script (Optional)

Download the official example script from Microsoft's repository:

```bash
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-chat.py -o model-chat.py
```

**Note:** This script may already exist in your directory. If it does, you can skip this step or overwrite it.

### Step 5: Test the Model

Run the example script to verify the model is working correctly:

```bash
python model-chat.py -m cuda/cuda-int4-kquant-block-32-mixed -e follow_config
```

**Parameters:**
- `-m`: Path to the model directory (adjust if you downloaded to a different location)
- `-e`: Execution provider (`follow_config` uses settings from genai_config.json, or use `cuda`, `cpu`, `dml`)

## Quick Setup Script

You can also run all steps at once using this script:

```bash
#!/bin/bash

# Install Hugging Face CLI
pip install huggingface_hub[cli]

# Download the model
huggingface-cli download onnxruntime/gpt-oss-20b-onnx \
  --include cuda/cuda-int4-kquant-block-32-mixed/* \
  --local-dir .

# Install ONNX Runtime GenAI CUDA
pip install onnxruntime-genai-cuda

# Download example script
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-chat.py -o model-chat.py

# Test the model
python model-chat.py -m cuda/cuda-int4-kquant-block-32-mixed -e follow_config
```

## Troubleshooting

### Issue: Hugging Face authentication required

If you encounter authentication errors, you may need to login:

```bash
huggingface-cli login
```

### Issue: CUDA not available

If you don't have CUDA, you can:
1. Use CPU version: `pip install onnxruntime-genai` (instead of `onnxruntime-genai-cuda`)
2. Download CPU model variant instead
3. Use `-e cpu` when running the script

### Issue: Model path not found

Make sure you're in the correct directory and the model was downloaded successfully. Check that the `cuda/cuda-int4-kquant-block-32-mixed/` directory exists.

### Issue: Out of disk space

The model requires approximately 10-20 GB of free disk space. Make sure you have enough space before downloading.

## Directory Structure After Download

After successful download, your directory should look like:

```
.
├── cuda/
│   └── cuda-int4-kquant-block-32-mixed/
│       ├── model.onnx
│       ├── genai_config.json
│       └── ... (other model files)
├── model-chat.py
└── ... (other project files)
```

## Next Steps

Once the model is downloaded and tested:

1. **Run the API server**: See `API_README.md` for instructions
2. **Test the API**: Use `python test_api.py` to test the endpoints
3. **Customize configuration**: Edit `api.py` to adjust model path and settings

## Additional Resources

- [ONNX Runtime GenAI Documentation](https://github.com/microsoft/onnxruntime-genai)
- [Hugging Face Model Page](https://huggingface.co/onnxruntime/gpt-oss-20b-onnx)
- [API Documentation](./API_README.md)
