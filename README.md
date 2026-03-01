# Hierarchical Meta-Agent Architecture (HMA)

This repository contains the research codebase for testing the efficiency of **Hierarchical Meta-Agent Systems** versus flat multi-agent structures.

## Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency (quality of final output per unit of cost) of a multi-agent system will be maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting**, **Interruption Synthesis**, and **Tool Creation**, rather than just task decomposition.

## Architecture & Management Theory

This system implements key industrial management theories to optimize AI Token Efficiency:

1.  **Analyst Node (Context Compression & Planning):**
    *   **Theory:** MECE (Mutually Exclusive, Collectively Exhaustive).
    *   **Function:** Rewrites user prompts into strict technical briefs and plans **Parallel Execution Waves** (Dependency-Aware DAG).
    *   **Benefit:** Reduces noise and plans optimal parallel routes.

2.  **The Engineer (Dynamic Tool Creation):**
    *   **Theory:** Make vs. Buy Decision.
    *   **Function:** Automatically detects computational tasks (Math/Data) and writes/executes **Python Code** in a sandbox instead of using LLM reasoning.
    *   **Benefit:** 100x efficiency for math/logic. Solves problems the model cannot hallucinate through.

3.  **Vector Semantic Cache (Collective Memory):**
    *   **Theory:** Knowledge Management (KM).
    *   **Function:** Uses **ChromaDB** to store vector embeddings of *every* sub-agent prompt and result.
    *   **Benefit:** "Skill Acquisition." If an agent solves a problem once, the solution is memorized forever. Subsequent similar requests cost **Zero Tokens**.

4.  **Parallel Execution Engine (HPC Optimization):**
    *   **Theory:** Amdahl's Law.
    *   **Function:** Uses `asyncio.gather` to fire all tasks in a wave simultaneously, leveraging vLLM's continuous batching on 8x GPUs.

5.  **Reflexion & Poka-Yoke:**
    *   **Theory:** Quality Engineering.
    *   **Function:** Workers self-correct (Draft -> Critique -> Refine) and output strictly validated JSON.

## Structure

*   **`hma_orchestrator.py`**: The Async LangGraph engine implementing the HMA logic (Analyst -> Parallel Waves -> Engineer/Worker -> Reduce).
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
     -d '{"task_description": "Calculate the sum of the first 100 prime numbers using Python.", "total_budget_tokens": 5000}'
```

### 5. Check Metrics
View real-time efficiency logs:
```bash
curl "http://localhost:8080/metrics"
```
