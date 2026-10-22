"""What a chat app sends to a model server: an OpenAI-compatible chat request with streaming.
Servers such as vLLM, SGLang or llama.cpp's llama-server expose this kind of endpoint.

This file is NOT executed when the site is built (it needs a running server).
Start one yourself, for example:   vllm serve Qwen/Qwen3-0.6B
then run:                           python 16_api_client.py      (needs: requests)
"""
import json
import requests

URL = "http://localhost:8000/v1/chat/completions"
request = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a transformer?"},
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 200,
    "stream": True,                      # ask for the answer piece by piece
}

with requests.post(URL, json=request, stream=True, timeout=60) as r:
    r.raise_for_status()
    for line in r.iter_lines():          # server-sent events: lines starting with "data: "
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: "):]
        if payload == b"[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)  # each chunk is the text of one or a few new tokens
print()
