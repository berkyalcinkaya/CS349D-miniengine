MiniEngine is an educational LLM serving engine for CS349D. It serves an OpenAI-compatible HTTP API and is deliberately structured so the scheduler is the primary optimization target across a series of milestone assignments. See milestones/milestone1.md for the active assignment (batching / continuous batching).

## Common commands

First activate the correct conda environment (with deps installed) using 

```bash
# Install (editable, with benchmark deps) (no need if conda env activated)
pip install -e .[bench]

# Launch the server (downloads model on first run)
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16

# Serving benchmark — throughput/latency under concurrency sweep
python -m benchmark.bench_serving --input-len 512 --output-len 256 --concurrencies 1,4,8,16

# Accuracy benchmark — MMLU or GSM8K (verifies optimizations don't regress quality)
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200
python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100
```


## Testing

A minimal testing workflow is as follows:

```bash
# Launch, modify CLI args for correct objective
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": true
  }'
```

To assess code correctness/peformance under load, start with server and then run the bench serving script. 
