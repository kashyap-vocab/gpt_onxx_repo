"""
FastAPI endpoint for GPT-OSS-20B ONNX model
"""
import os
import json
import time
import uuid
import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import onnxruntime_genai as og

# Model configuration
MODEL_PATH = "/media/server/be3a703c-2cfd-4e6e-adda-e6bd3c1aeee1/kashyap/gpt_20b_onxx/cuda/cuda-int4-kquant-block-32-mixed"
EXECUTION_PROVIDER = "follow_config"  # Uses CUDA from genai_config.json

# Initialize FastAPI app
app = FastAPI(
    title="GPT-OSS-20B ONNX API",
    description="FastAPI endpoint for GPT-OSS-20B ONNX model using onnxruntime-genai",
    version="1.0.0"
)

# Global model variables (loaded on startup)
model: Optional[og.Model] = None
tokenizer: Optional[og.Tokenizer] = None
template_str: Optional[str] = None


def extract_final_response(text: str) -> str:
    """
    Extract only the final channel response, removing analysis tokens and special formatting.
    
    The model may output responses with special tokens like:
    - <|channel|>analysis<|message|>...<|end|>
    - <|channel|>final<|message|>...<|end|>
    - <|start|>assistant<|channel|>final<|message|>...<|end|>
    
    This function extracts only the final channel content.
    """
    if not text:
        return text
    
    # Try to extract final channel content
    # Look for <|channel|>final<|message|> pattern
    final_pattern = "<|channel|>final<|message|>"
    if final_pattern in text:
        # Extract content after final channel marker
        start_idx = text.find(final_pattern)
        if start_idx != -1:
            # Get content after the marker
            content_start = start_idx + len(final_pattern)
            # Find the end marker
            end_idx = text.find("<|end|>", content_start)
            if end_idx != -1:
                return text[content_start:end_idx].strip()
            else:
                # No end marker, take everything after
                return text[content_start:].strip()
    
    # If no final channel, try to extract from analysis channel (fallback)
    analysis_pattern = "<|channel|>analysis<|message|>"
    if analysis_pattern in text:
        start_idx = text.find(analysis_pattern)
        if start_idx != -1:
            content_start = start_idx + len(analysis_pattern)
            end_idx = text.find("<|end|>", content_start)
            if end_idx != -1:
                return text[content_start:end_idx].strip()
    
    # Remove all special tokens if no channel markers found
    cleaned = text
    # Remove common special tokens
    tokens_to_remove = [
        "<|start|>",
        "<|end|>",
        "<|channel|>analysis<|message|>",
        "<|channel|>final<|message|>",
        "<|channel|>commentary<|message|>",
        "<|return|>",
        "<|message|>",
    ]
    
    for token in tokens_to_remove:
        cleaned = cleaned.replace(token, "")
    
    # Remove any remaining channel markers
    cleaned = re.sub(r'<\|channel\|>[^<]*<\|message\|>', '', cleaned)
    cleaned = re.sub(r'<\|[^|]*\|>', '', cleaned)
    
    return cleaned.strip()


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gpt-oss-20b-onnx"
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False
    system_prompt: Optional[str] = "You are a helpful AI assistant."


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = True


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup"""
    global model, tokenizer, template_str
    
    print("Loading model...")
    try:
        # Load model configuration
        config = og.Config(MODEL_PATH)
        if EXECUTION_PROVIDER != "follow_config":
            config.clear_providers()
            config.append_provider(EXECUTION_PROVIDER)
        
        # Load model
        model = og.Model(config)
        print("Model loaded successfully")
        
        # Load tokenizer
        tokenizer = og.Tokenizer(model)
        print("Tokenizer loaded successfully")
        
        # Load chat template if available
        jinja_path = os.path.join(MODEL_PATH, "chat_template.jinja")
        if os.path.exists(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                template_str = f.read()
            print("Chat template loaded")
        
        print("✅ Model ready for inference")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "gpt-oss-20b-onnx",
        "model_path": MODEL_PATH
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request),
            media_type="text/event-stream"
        )
    
    try:
        # Prepare messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        # Convert messages to JSON string (required by onnxruntime_genai)
        messages_json = json.dumps(messages)
        
        # Apply chat template
        if template_str:
            user_prompt = tokenizer.apply_chat_template(
                messages=messages_json,
                add_generation_prompt=True,
                template_str=template_str
            )
        else:
            user_prompt = tokenizer.apply_chat_template(
                messages=messages_json,
                add_generation_prompt=True
            )
        
        # Encode input
        input_tokens = tokenizer.encode(user_prompt)
        
        # Set up generator parameters
        params = og.GeneratorParams(model)
        search_options = {
            "do_sample": request.do_sample if request.do_sample is not None else True,
            "max_length": len(input_tokens) + (request.max_tokens or 256),
        }
        
        if request.temperature is not None:
            search_options["temperature"] = request.temperature
        if request.top_p is not None:
            search_options["top_p"] = request.top_p
        if request.top_k is not None:
            search_options["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            search_options["repetition_penalty"] = request.repetition_penalty
        
        params.set_search_options(**search_options)
        
        # Create generator
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        
        # Generate response
        tokenizer_stream = tokenizer.create_stream()
        output_text = ""
        
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(token)
        
        # Extract only the final response, removing analysis tokens and special formatting
        clean_output = extract_final_response(output_text)
        
        # Return response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "gpt-oss-20b-onnx",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=clean_output),
                    finish_reason="stop"
                )
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


async def stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion response"""
    if model is None or tokenizer is None:
        yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
        return
    
    try:
        # Prepare messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        # Convert messages to JSON string (required by onnxruntime_genai)
        messages_json = json.dumps(messages)
        
        # Apply chat template
        if template_str:
            user_prompt = tokenizer.apply_chat_template(
                messages=messages_json,
                add_generation_prompt=True,
                template_str=template_str
            )
        else:
            user_prompt = tokenizer.apply_chat_template(
                messages=messages_json,
                add_generation_prompt=True
            )
        
        # Encode input
        input_tokens = tokenizer.encode(user_prompt)
        
        # Set up generator parameters
        params = og.GeneratorParams(model)
        search_options = {
            "do_sample": request.do_sample if request.do_sample is not None else True,
            "max_length": len(input_tokens) + (request.max_tokens or 256),
        }
        
        if request.temperature is not None:
            search_options["temperature"] = request.temperature
        if request.top_p is not None:
            search_options["top_p"] = request.top_p
        if request.top_k is not None:
            search_options["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            search_options["repetition_penalty"] = request.repetition_penalty
        
        params.set_search_options(**search_options)
        
        # Create generator
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        
        # Generate and stream response
        tokenizer_stream = tokenizer.create_stream()
        chunk_id = uuid.uuid4().hex
        accumulated_text = ""
        last_clean_length = 0
        
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            token_text = tokenizer_stream.decode(token)
            accumulated_text += token_text
            
            # Extract clean response so far
            clean_text = extract_final_response(accumulated_text)
            
            # Only stream new clean content
            if len(clean_text) > last_clean_length:
                new_content = clean_text[last_clean_length:]
                if new_content:
                    chunk_data = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model or "gpt-oss-20b-onnx",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    last_clean_length = len(clean_text)
        
        # Send final chunk
        final_data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model or "gpt-oss-20b-onnx",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Simple text generation endpoint
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode prompt
        input_tokens = tokenizer.encode(request.prompt)
        
        # Set up generator parameters
        params = og.GeneratorParams(model)
        search_options = {
            "do_sample": request.do_sample if request.do_sample is not None else True,
            "max_length": len(input_tokens) + (request.max_tokens or 256),
        }
        
        if request.temperature is not None:
            search_options["temperature"] = request.temperature
        if request.top_p is not None:
            search_options["top_p"] = request.top_p
        if request.top_k is not None:
            search_options["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            search_options["repetition_penalty"] = request.repetition_penalty
        
        params.set_search_options(**search_options)
        
        # Create generator
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        
        # Generate response
        tokenizer_stream = tokenizer.create_stream()
        output_text = ""
        tokens_generated = 0
        
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(token)
            tokens_generated += 1
        
        return GenerateResponse(
            text=output_text.strip(),
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
