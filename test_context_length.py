"""
Context Length Testing Script for GPT-OSS-20B ONNX FastAPI
Tests performance across different context lengths and output lengths
"""
import requests
import json
import time
import statistics
import csv
import os
from typing import List, Dict, Optional
from datetime import datetime

API_URL = "http://localhost:8001"

# Test Matrix Configuration
CONTEXT_LENGTHS = [50, 100, 500, 1000, 2000, 4000, 8000, 16000]  # tokens
OUTPUT_LENGTHS = [50, 100, 256, 512]  # max_tokens
REPETITIONS = 5
COOLDOWN_DELAY = 3  # seconds between combinations
REPETITION_DELAY = 0.5  # seconds between repetitions

# Base question (extracted from prompt)
BASE_QUESTION = "What is artificial intelligence?"

# Base question (extracted from prompt)
BASE_QUESTION = "What is artificial intelligence?"

# Base prompt template for generating context
BASE_PROMPT = "What is artificial intelligence? Please provide a comprehensive explanation."
CONTEXT_PADDING = """
Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI systems can be categorized into narrow AI, which is designed for specific tasks, and general AI, which aims to replicate human cognitive abilities across a wide range of tasks. Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to analyze data and make decisions. Natural language processing allows AI systems to understand and generate human language. Computer vision enables AI to interpret and understand visual information from the world. AI applications are widespread, including virtual assistants, recommendation systems, autonomous vehicles, medical diagnosis, and many other domains. The field continues to evolve rapidly with advances in algorithms, computing power, and data availability.
"""


def generate_prompt(target_tokens: int) -> str:
    """
    Generate a prompt of approximately target_tokens length
    
    Args:
        target_tokens: Target number of tokens for the prompt
        
    Returns:
        Generated prompt string
    """
    # Approximate: 1 token ≈ 0.75 words
    target_words = int(target_tokens * 0.75)
    
    # Start with base prompt
    prompt = BASE_PROMPT
    current_words = len(prompt.split())
    
    # If we need more words, pad with context
    if current_words < target_words:
        padding_needed = target_words - current_words
        # Repeat padding text as needed
        padding_text = CONTEXT_PADDING.strip()
        padding_words = len(padding_text.split())
        
        # Calculate how many times to repeat
        repeats = (padding_needed // padding_words) + 1
        
        # Add padding
        full_padding = " ".join([padding_text] * repeats)
        prompt = f"{BASE_PROMPT} {full_padding}"
        
        # Trim to approximately target length
        words = prompt.split()
        if len(words) > target_words:
            prompt = " ".join(words[:target_words])
    
    return prompt


def test_combination(context_length: int, output_length: int, repetition: int, 
                     total_tests: int, current_test: int) -> Optional[Dict]:
    """
    Test a single combination (context length × output length × repetition)
    
    Args:
        context_length: Target context length in tokens
        output_length: Max tokens for output
        repetition: Current repetition number (1-5)
        total_tests: Total number of tests
        current_test: Current test number
        
    Returns:
        Dictionary with test results or None if failed
    """
    print(f"\n[{current_test}/{total_tests}] Testing: context={context_length}, "
          f"output={output_length}, repetition={repetition}/{REPETITIONS}")
    
    # Generate prompt
    prompt = generate_prompt(context_length)
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": output_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    # Measure total request time
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/v1/chat/completions",
            json=payload,
            timeout=600  # 10 minute timeout for large contexts
        )
        
        end_time = time.time()
        total_latency = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            tokens_generated = len(response_text.split())  # Approximate token count
            
            # Calculate tokens per second
            tokens_per_second = tokens_generated / total_latency if total_latency > 0 else 0
            
            print(f"  ✅ Latency: {total_latency:.3f}s, Throughput: {tokens_per_second:.2f} tokens/s, "
                  f"Output tokens: ~{tokens_generated}")
            
            return {
                "context_length": context_length,
                "output_length": output_length,
                "repetition": repetition,
                "status": "success",
                "question": BASE_QUESTION,
                "prompt": prompt,
                "answer": response_text,
                "latency": total_latency,
                "throughput": tokens_per_second,
                "output_tokens": tokens_generated,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"  ❌ Status: {response.status_code}, Error: {response.text[:100]}")
            return {
                "context_length": context_length,
                "output_length": output_length,
                "repetition": repetition,
                "status": "error",
                "question": BASE_QUESTION,
                "prompt": prompt,
                "answer": None,
                "status_code": response.status_code,
                "error": response.text[:200],
                "latency": total_latency,
                "timestamp": datetime.now().isoformat()
            }
            
    except requests.exceptions.Timeout:
        end_time = time.time()
        total_latency = end_time - start_time
        print(f"  ❌ Request timed out after {total_latency:.3f} seconds")
        return {
            "context_length": context_length,
            "output_length": output_length,
            "repetition": repetition,
            "status": "timeout",
            "question": BASE_QUESTION,
            "prompt": prompt,
            "answer": None,
            "latency": total_latency,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        end_time = time.time()
        total_latency = end_time - start_time
        print(f"  ❌ Exception: {str(e)}")
        return {
            "context_length": context_length,
            "output_length": output_length,
            "repetition": repetition,
            "status": "exception",
            "question": BASE_QUESTION,
            "prompt": prompt,
            "answer": None,
            "error": str(e),
            "latency": total_latency,
            "timestamp": datetime.now().isoformat()
        }


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate statistics from a list of test results (5 repetitions)
    
    Args:
        results: List of test result dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    successful_results = [r for r in results if r.get("status") == "success"]
    
    if not successful_results:
        return {
            "latency_mean": None,
            "latency_min": None,
            "latency_max": None,
            "latency_std": None,
            "throughput_mean": None,
            "throughput_min": None,
            "throughput_max": None,
            "throughput_std": None,
            "output_tokens_mean": None,
            "output_tokens_min": None,
            "output_tokens_max": None,
            "successful_tests": 0,
            "total_tests": len(results)
        }
    
    latencies = [r["latency"] for r in successful_results]
    throughputs = [r["throughput"] for r in successful_results]
    output_tokens = [r["output_tokens"] for r in successful_results]
    
    return {
        "latency_mean": statistics.mean(latencies),
        "latency_min": min(latencies),
        "latency_max": max(latencies),
        "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_mean": statistics.mean(throughputs),
        "throughput_min": min(throughputs),
        "throughput_max": max(throughputs),
        "throughput_std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "output_tokens_mean": statistics.mean(output_tokens),
        "output_tokens_min": min(output_tokens),
        "output_tokens_max": max(output_tokens),
        "successful_tests": len(successful_results),
        "total_tests": len(results)
    }


def save_context_csv(context_length: int, context_results: List[Dict], output_dir: str):
    """
    Save results for a specific context length to CSV file
    Each row represents one test repetition with question, prompt, answer, and metrics
    
    Args:
        context_length: The context length being saved
        context_results: List of result dictionaries for this context length
        output_dir: Directory to save CSV files
    """
    csv_path = os.path.join(output_dir, f"context_{context_length}.csv")
    
    # Organize results by output length, then by repetition
    output_results = {}
    for result in context_results:
        output_len = result["output_length"]
        if output_len not in output_results:
            output_results[output_len] = []
        output_results[output_len].append(result)
    
    # Sort results by output length, then by repetition
    for output_len in output_results:
        output_results[output_len].sort(key=lambda x: x["repetition"])
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "context_length",
            "output_length",
            "question",
            "prompt",
            "repetition",
            "answer",
            "latency",
            "throughput",
            "output_tokens",
            "status"
        ])
        
        # Write rows for each output length (sorted)
        for output_len in sorted(OUTPUT_LENGTHS):
            if output_len in output_results:
                results = output_results[output_len]
                # Get question from first result (should be same for all)
                question = results[0].get("question", BASE_QUESTION) if results else BASE_QUESTION
                
                # Write each repetition as a separate row
                for idx, result in enumerate(results):
                    # Show question only in first row of each group
                    question_value = question if idx == 0 else ""
                    
                    writer.writerow([
                        context_length,
                        output_len,
                        question_value,
                        result.get("prompt", ""),
                        result.get("repetition", ""),
                        result.get("answer", "") if result.get("status") == "success" else result.get("error", ""),
                        f"{result.get('latency', 0):.3f}" if result.get("latency") is not None else "N/A",
                        f"{result.get('throughput', 0):.2f}" if result.get("throughput") is not None else "N/A",
                        result.get("output_tokens", "") if result.get("output_tokens") is not None else "",
                        result.get("status", "")
                    ])
    
    print(f"\n💾 Saved results to {csv_path}")


def main():
    """
    Main function to run context length tests
    """
    print("="*70)
    print("🚀 GPT-OSS-20B ONNX Context Length Testing")
    print("="*70)
    print(f"\nAPI URL: {API_URL}")
    print(f"Context Lengths: {CONTEXT_LENGTHS}")
    print(f"Output Lengths: {OUTPUT_LENGTHS}")
    print(f"Repetitions per combination: {REPETITIONS}")
    print(f"Total combinations: {len(CONTEXT_LENGTHS) * len(OUTPUT_LENGTHS)}")
    print(f"Total tests: {len(CONTEXT_LENGTHS) * len(OUTPUT_LENGTHS) * REPETITIONS}")
    
    # Create test_context_length directory
    output_dir = "test_context_length"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Check if API is available
    try:
        print("\n🔍 Checking API health...")
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is healthy and ready")
        else:
            print(f"⚠️  API health check returned: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API!")
        print(f"   Make sure the server is running at {API_URL}")
        print("   Start it with: python api.py or ./run_api.sh")
        return
    except Exception as e:
        print(f"⚠️  Health check warning: {str(e)}")
    
    # Calculate total tests
    total_tests = len(CONTEXT_LENGTHS) * len(OUTPUT_LENGTHS) * REPETITIONS
    all_results = []
    current_test = 0
    
    overall_start = time.time()
    
    # Run tests
    print(f"\n🧪 Starting context length tests...")
    print(f"⏱️  Cooldown delay: {COOLDOWN_DELAY}s between combinations")
    print(f"⏱️  Repetition delay: {REPETITION_DELAY}s between repetitions\n")
    
    for context_length in CONTEXT_LENGTHS:
        print(f"\n{'='*70}")
        print(f"📊 Testing Context Length: {context_length} tokens")
        print(f"{'='*70}")
        
        context_results = []
        
        for output_length in OUTPUT_LENGTHS:
            print(f"\n  🔹 Output Length: {output_length} tokens")
            
            combination_results = []
            
            for repetition in range(1, REPETITIONS + 1):
                current_test += 1
                result = test_combination(
                    context_length, output_length, repetition,
                    total_tests, current_test
                )
                
                if result:
                    combination_results.append(result)
                    all_results.append(result)
                
                # Delay between repetitions (except after last)
                if repetition < REPETITIONS:
                    time.sleep(REPETITION_DELAY)
            
            # Add combination results to context results
            context_results.extend(combination_results)
            
            # Cooldown delay after each combination
            if output_length != OUTPUT_LENGTHS[-1]:  # Don't delay after last output length
                print(f"\n  ⏸️  Cooldown: waiting {COOLDOWN_DELAY} seconds...")
                time.sleep(COOLDOWN_DELAY)
        
        # Save CSV for this context length
        save_context_csv(context_length, context_results, output_dir)
        
        # Brief pause between context lengths
        if context_length != CONTEXT_LENGTHS[-1]:
            print(f"\n⏸️  Brief pause before next context length...")
            time.sleep(1)
    
    overall_end = time.time()
    total_test_time = overall_end - overall_start
    
    # Save JSON results
    json_path = "context_length_test_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "test_summary": {
                "total_tests": total_tests,
                "total_combinations": len(CONTEXT_LENGTHS) * len(OUTPUT_LENGTHS),
                "context_lengths": CONTEXT_LENGTHS,
                "output_lengths": OUTPUT_LENGTHS,
                "repetitions_per_combination": REPETITIONS,
                "total_test_time": total_test_time,
                "cooldown_delay": COOLDOWN_DELAY,
                "repetition_delay": REPETITION_DELAY
            },
            "raw_results": all_results
        }, f, indent=2)
    
    # Print summary
    successful = len([r for r in all_results if r.get("status") == "success"])
    failed = len(all_results) - successful
    
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Time: {total_test_time:.2f} seconds ({total_test_time/60:.2f} minutes)")
    print(f"\n💾 Results saved to:")
    print(f"   - JSON: {json_path}")
    print(f"   - CSV files: {output_dir}/context_*.csv ({len(CONTEXT_LENGTHS)} files)")
    print(f"   - Each CSV contains individual test results with question, prompt, and answers")
    print(f"\n{'='*70}")
    print("✅ Testing Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
