# Hierarchical Meta-Agent Architecture (HMA)

This repository contains the research codebase for testing the efficiency of **Hierarchical Meta-Agent Systems** versus flat multi-agent structures.

## Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency (quality of final output per unit of cost) of a multi-agent system will be maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting** and **Interruption Synthesis**, rather than just task decomposition.

## Architecture & Management Theory

This system implements key industrial management theories to optimize AI Token Efficiency:

1.  **Analyst Node (Context Compression):**
    *   **Theory:** MECE (Mutually Exclusive, Collectively Exhaustive).
    *   **Function:** Rewrites verbose user prompts into strict technical briefs and plans **Parallel Execution Waves** (Dependency-Aware DAG).
    *   **Benefit:** Reduces noise and hallucination risk for downstream workers.

2.  **Parallel Execution Engine (HPC Optimization):**
    *   **Theory:** Amdahl's Law & Batch Processing.
    *   **Function:** Uses `asyncio.gather` to fire all tasks in a wave simultaneously.
    *   **Benefit:** Synergizes with vLLM's continuous batching to saturate 8x3090 GPUs, reducing total wall-clock time by N-fold.

3.  **Dynamic Sizing (Project Management):**
    *   **Theory:** Complexity-Based Budgeting.
    *   **Function:** Automatically rates task complexity (LOW/MEDIUM/HIGH) and adjusts the token budget (0.8x - 1.5x).
    *   **Routing:** Dynamically enables **Chain-of-Thought (CoT)** for complex tasks while using fast MoE for simple ones.

4.  **Poka-Yoke Worker (Quality Engineering):**
    *   **Theory:** Mistake Proofing.
    *   **Function:** Enforces **Strict JSON Structured Output** from worker agents.
    *   **Benefit:** Zero-token validation (syntax check) prevents the Manager from wasting tokens reading invalid responses.

## Enterprise Operations (Ops)

This repository includes production-grade resilience features:

1.  **Fault Tolerance:** Exponential backoff retries (via `tenacity`) for API failures.
2.  **Rate Limiting:** `asyncio.Semaphore` prevents overloading the vLLM server.
    *   Configure via `MAX_CONCURRENT_WORKERS` (Default: 20).
3.  **Observability:** Prometheus metrics exported for Grafana dashboards.
    *   `hma_tokens_total`: Total token consumption.
    *   `hma_worker_errors_total`: Error rates.
    *   `hma_task_duration_seconds`: Latency distribution.

## Structure

*   **`hma_orchestrator.py`**: The Async LangGraph engine implementing the HMA logic (Analyst -> Parallel Waves -> Reduce -> Loop).
*   **`service.py`**: FastAPI wrapper exposing the HMA as a scalable "Agent-as-a-Service" (AaaS).
*   **`docker-compose.yml`**: Full-stack deployment (HMA Service + vLLM Worker).
*   **`hma_benchmark_logs.csv`**: Automatically generated metrics file tracking Token ROI and Latency.

## Quick Start (Docker Compose)

The easiest way to run the entire HMA cluster (Service + Worker + vLLM Inference) is with Docker Compose.

### 1. Prerequisites
*   **Docker & Docker Compose** installed.
*   **NVIDIA Drivers & NVIDIA Container Toolkit** (for GPU support).

### 2. Configure Hardware (Optional)
Open `docker-compose.yml` and update the `vllm-worker` service to match your GPU count:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all # Use all 8 GPUs!
          capabilities: [gpu]
```

### 3. Launch
```bash
docker-compose up --build
```
This starts the vLLM Inference Server on port 8000 and the HMA Orchestrator on port 8080.

### 4. Submit a Task
```bash
curl -X POST "http://localhost:8080/submit" \
     -H "Content-Type: application/json" \
     -d '{"task_description": "Analyze the token efficiency of HMA vs Flat agents.", "total_budget_tokens": 5000}'
```

### 5. Check Metrics
View real-time efficiency logs:
```bash
curl "http://localhost:8080/metrics"
```
