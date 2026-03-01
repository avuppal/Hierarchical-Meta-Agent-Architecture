# Hierarchical Meta-Agent Architecture (HMA)

This repository contains the research codebase for testing the efficiency of **Hierarchical Meta-Agent Systems** versus flat multi-agent structures.

## Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency (quality of final output per unit of cost) of a multi-agent system will be maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting**, **Interruption Synthesis**, and **Tool Creation**, rather than just task decomposition.

## Architecture & Management Theory

This system implements key industrial management theories to optimize AI Token Efficiency:

1.  **DSPy Cognitive Layer (Self-Improving Prompts):**
    *   **Theory:** Optimization over Heuristics.
    *   **Function:** Replaces brittle hand-written prompts with **DSPy Signatures** (Analyst, Worker, Engineer). Allows the system to *compile* optimized prompts based on performance metrics.
    *   **Benefit:** Higher accuracy with fewer tokens (Prompt Optimization).

2.  **The Librarian (RAG / Knowledge Management):**
    *   **Theory:** Grounded Truth.
    *   **Function:** Automatically indexes local documents (PDFs, Text) in the `data/` folder into ChromaDB. Agents can trigger a "RESEARCH" task to query this private knowledge base.
    *   **Benefit:** Eliminates hallucination on internal data.

3.  **The Engineer (Dynamic Tool Creation):**
    *   **Theory:** Make vs. Buy Decision.
    *   **Function:** Automatically detects computational tasks (Math/Data) and writes/executes **Python Code** in a sandbox instead of using LLM reasoning.
    *   **Benefit:** 100x efficiency for math/logic.

4.  **Vector Semantic Cache (Collective Memory):**
    *   **Theory:** Memoization.
    *   **Function:** Stores vector embeddings of every sub-agent interaction. If a similar problem arises, the solution is retrieved instantly.
    *   **Benefit:** "Skill Acquisition." Cost drops to **Zero Tokens** for repeated tasks.

5.  **Parallel Execution Engine (HPC Optimization):**
    *   **Theory:** Amdahl's Law.
    *   **Function:** Uses `asyncio.gather` to fire all tasks in a wave simultaneously, leveraging vLLM's continuous batching on 8x GPUs.

## Structure

*   **`hma_orchestrator.py`**: The Async LangGraph engine implementing the HMA logic with DSPy modules.
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

### 3. Add Data (Optional)
Place any PDFs or text files you want the agents to reference into the `agent_budget_research/data/` folder. The **Librarian** will index them on startup.

### 4. Launch
```bash
docker-compose up --build
```
This starts:
*   **vLLM Inference Server** (Port 8000)
*   **HMA Orchestrator** (Port 8080)
*   **Prometheus Metrics** (Port 8001 internal)

### 5. Submit a Task
```bash
curl -X POST "http://localhost:8080/submit" \
     -H "Content-Type: application/json" \
     -d '{"task_description": "Analyze the 'quarterly_report.pdf' in the data folder.", "total_budget_tokens": 5000}'
```

### 6. Check Metrics
View real-time efficiency logs:
```bash
curl "http://localhost:8080/metrics"
```
