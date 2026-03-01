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

3.  **Vector Semantic Cache (Knowledge Management):**
    *   **Theory:** Memoization & Knowledge Reuse.
    *   **Function:** Uses **ChromaDB** to store vector embeddings of prompts. If a new request is semantically similar (e.g., "What is SSM?" vs "Define State Space Model"), it returns the cached answer instantly.
    *   **Benefit:** Reduces redundant computation to **Zero Tokens** and **Zero Latency**.

4.  **Worker Reflexion (Quality Assurance):**
    *   **Theory:** Deming Cycle (Plan-Do-Check-Act).
    *   **Function:** Workers execute a "Draft -> Critique -> Refine" loop internally before submitting work.
    *   **Benefit:** Catches errors locally, preventing expensive re-work loops at the Manager level.

5.  **Dynamic Sizing & Routing:**
    *   **Theory:** Complexity-Based Budgeting.
    *   **Function:** Automatically rates task complexity. Uses **Chain-of-Thought (CoT)** for high-complexity reasoning and fast **MoE (Mixture of Experts)** for standard tasks.

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
This starts:
*   **vLLM Inference Server** (Port 8000)
*   **HMA Orchestrator** (Port 8080)
*   **Prometheus Metrics** (Port 8001 internal)

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
