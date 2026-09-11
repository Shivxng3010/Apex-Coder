import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from training.auto_learn import get_active_adapter_path
from run_chat import resolve_model_path, break_degenerate_repetition

if Path("E:/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "E:/hf_cache")
    os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
    os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")

model = None
tokenizer = None


def load_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return

    base_model_path = resolve_model_path()
    active_adapter = get_active_adapter_path()

    print("===========================================================")
    print("      APEX CODER - OPENAI COMPATIBLE FASTAPI SERVER        ")
    print("===========================================================")
    print(f"Base Model    : {base_model_path}")
    print(f"Active Adapter: {active_adapter.resolve()}")
    print(f"Device        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-----------------------------------------------------------\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    if (active_adapter / "adapter_model.safetensors").exists():
        print(f"[*] Attaching LoRA adapter: {active_adapter.name}")
        model = PeftModel.from_pretrained(base, str(active_adapter))
    else:
        model = base

    model.eval()
    vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"\n[Ready] Model online | VRAM Allocated: {vram:.2f} GB / 4.00 GB", flush=True)


app = FastAPI(
    title="Apex Coder API",
    description="OpenAI-compatible local inference API server for Apex Coder 3B with SDPA acceleration",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "apex-coder-3b"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.3, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")


@app.get("/")
async def root():
    return {
        "name": "Apex Coder Local API Server",
        "status": "online",
        "swagger_docs": "/docs",
        "openai_endpoint": "/v1/chat/completions",
        "models_endpoint": "/v1/models",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "apex-coder-3b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "apex-coder",
            }
        ]
    }


def _sync_generate(inputs, max_tokens, temperature, top_p):
    with torch.no_grad():
        return model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            do_sample=(temperature > 0.0),
        )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None or tokenizer is None:
        load_model_and_tokenizer()

    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]
    formatted_prompt = tokenizer.apply_chat_template(
        messages_payload, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    if not req.stream:
        outputs = await asyncio.to_thread(_sync_generate, inputs, req.max_tokens, req.temperature, req.top_p)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        clean_text = break_degenerate_repetition(raw_text)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": clean_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(generated_tokens),
                "total_tokens": inputs["input_ids"].shape[1] + len(generated_tokens)
            }
        }

    # Streaming Generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=1.15,
        do_sample=(req.temperature > 0.0),
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    async def event_generator():
        cmpl_id = f"chatcmpl-{int(time.time())}"
        for new_text in streamer:
            chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.001)

        final_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    load_model_and_tokenizer()
    port = 8000
    host = "127.0.0.1"
    print(f"\n[FastAPI Server Live on http://{host}:{port}]")
    print(f"Swagger Documentation : http://{host}:{port}/docs")
    print(f"OpenAI API Endpoint   : http://{host}:{port}/v1/chat/completions\n", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
