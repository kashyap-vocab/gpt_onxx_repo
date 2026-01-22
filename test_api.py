"""
Simple test script for the FastAPI endpoint
"""
import requests
import json

API_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_chat_completion():
    """Test chat completion endpoint"""
    print("Testing /v1/chat/completions endpoint...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "What is AI used for? Please give a brief answer."}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    response = requests.post(f"{API_URL}/v1/chat/completions", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Response: {result['choices'][0]['message']['content']}\n")
    else:
        print(f"Error: {response.text}\n")

def test_generate():
    """Test simple generate endpoint"""
    print("Testing /generate endpoint...")
    
    payload = {
        "prompt": "Explain transformers in simple terms:",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = requests.post(f"{API_URL}/generate", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text: {result['text']}")
        print(f"Tokens generated: {result['tokens_generated']}\n")
    else:
        print(f"Error: {response.text}\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing FastAPI GPT-OSS-20B ONNX API")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_chat_completion()
        test_generate()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running:")
        print("   python api.py")
        print("   or")
        print("   ./run_api.sh")
