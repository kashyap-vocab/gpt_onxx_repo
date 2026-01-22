# GPT-OSS-20B ONNX FastAPI Server

FastAPI endpoint for the GPT-OSS-20B ONNX model using `onnxruntime-genai`.

## Quick Start

### 1. Start the API Server

```bash
# Option 1: Run directly
python api.py

# Option 2: Use the run script
./run_api.sh

# Option 3: Use uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8001
```

The server will start on `http://0.0.0.0:8001`. The model will be loaded automatically on startup.

### 2. Access API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## API Endpoints

### Health Check

```bash
GET /health
```

Returns the health status of the API and model.

### Chat Completions (OpenAI-compatible)

```bash
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is AI?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "stream": false,
  "system_prompt": "You are a helpful AI assistant."
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-oss-20b-onnx",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "AI is..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Simple Text Generation

```bash
POST /generate
```

**Request Body:**
```json
{
  "prompt": "Explain transformers:",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true
}
```

**Response:**
```json
{
  "text": "Generated text here...",
  "tokens_generated": 150
}
```

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8001/health

# Chat completion
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is AI used for?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'

# Simple generation
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain transformers in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Using Python requests

```python
import requests

# Chat completion
response = requests.post(
    "http://localhost:8001/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### Using the test script

```bash
python test_api.py
```

## Configuration

The model path and execution provider can be configured in `api.py`:

```python
MODEL_PATH = "/path/to/model"
EXECUTION_PROVIDER = "follow_config"  # or "cuda", "cpu", "dml"
```

## Parameters

- **max_tokens**: Maximum number of tokens to generate (default: 256)
- **temperature**: Sampling temperature (0.0 = deterministic, higher = more random)
- **top_p**: Nucleus sampling parameter
- **top_k**: Top-k sampling parameter
- **repetition_penalty**: Penalty for repetition (1.0 = no penalty, >1.0 = less repetition)
- **do_sample**: Whether to use sampling (True) or greedy decoding (False)
- **stream**: Whether to stream the response (for chat completions)

## Notes

- The model is loaded once on startup, which may take some time
- The API uses the chat template from the model if available
- CUDA execution provider is used by default (from genai_config.json)
- Streaming is supported for chat completions

## Requirements

See `requirements_api.txt` for required packages:
- fastapi
- uvicorn
- onnxruntime-genai-cuda
- pydantic
