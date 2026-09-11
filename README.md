# Apex Coder 3B

> **Production-Ready Systems Coding & Continual Learning AI Model**

Apex Coder is a self-contained 3B parameter coding assistant equipped with active LoRA adapters, self-correction verification harness, DPO alignment, and a local OpenAI-compatible inference server.

---

## Quick Start

### 1. Requirements & Installation
```bash
pip install -r requirements.txt
```

### 2. Interactive CLI Chat
Run the full self-correcting interactive chat interface:
```bash
# Windows Batch
chat.bat

# Or Python directly
python run_chat.py
```

### 3. Local OpenAI-Compatible API Server
Start the FastAPI server on port 8000 (compatible with Cursor, Continue.dev, LangChain, etc.):
```bash
# Windows Batch
serve_local_api.bat

# Or Python directly
python serve_api.py
```
- **Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Chat Completions Endpoint:** `http://127.0.0.1:8000/v1/chat/completions`

### 4. Standalone GGUF Inference
The quantized model `models/apex-coder-3b-q4_k_m.gguf` is fully compatible with `llama.cpp`, `Ollama`, and `LM Studio`.

---

## Architecture & Features

- **Harness & Verification (`/harness`):** TypeScript & Python test runners with automated runtime AST check and self-correction engine.
- **Continual Learning Flywheel (`/training`):** Auto-learning replay buffer, DPO pair generation, and live adapter promotions (`outputs/apex_coder_3b_lora_v*`).
- **Data Curation Pipeline (`/pipeline`):** Synthetic dataset generators, concurrency problem banks, and token budget management.
- **Containerization (`/docker`):** Production Dockerfiles for isolated test execution and training environments.

---

## License
Proprietary / Private repository.
