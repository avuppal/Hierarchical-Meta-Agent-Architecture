# Hierarchical Meta-Agent Architecture (HMA)

This repository contains the research codebase for testing the efficiency of **Hierarchical Meta-Agent Systems** versus flat multi-agent structures.

## Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency (quality of final output per unit of cost) of a multi-agent system will be maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting** and **Interruption Synthesis**, rather than just task decomposition.

## Architecture & Management Theory

This system implements key industrial management theories to optimize AI Token Efficiency:

1.  **Analyst Node (Context Compression):**
    *   **Theory:** MECE (Mutually Exclusive, Collectively Exhaustive).
    *   **Function:** Rewrites verbose user prompts into strict technical briefs *before* any expensive work begins.
    *   **Benefit:** Reduces noise and hallucination risk for downstream workers.

2.  **Dynamic Sizing (Project Management):**
    *   **Theory:** Complexity-Based Budgeting.
    *   **Function:** Automatically rates task complexity (LOW/MEDIUM/HIGH) and adjusts the token budget (0.8x - 1.5x) to prevent waste on simple tasks and starvation on complex ones.

3.  **Poka-Yoke Worker (Quality Engineering):**
    *   **Theory:** Mistake Proofing.
    *   **Function:** Enforces **Strict JSON Structured Output** from worker agents.
    *   **Benefit:** Zero-token validation (syntax check) prevents the Manager from wasting tokens reading invalid or malformed responses.

## Structure

*   **`hma_orchestrator.py`**: The LangGraph-based engine implementing the HMA logic (Analyst -> Budget -> Execute -> Review -> Loop).
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
