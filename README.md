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
*   **`Dockerfile`**: Containerized environment with vLLM integration and OpenAI client compatibility.
*   **`hma_benchmark_logs.csv`**: Automatically generated metrics file tracking Token ROI and Latency.

## Running the Service

### 1. Build
```bash
docker build -t hma-service .
```

### 2. Run (Connecting to vLLM)
Ensure you have a vLLM server running (e.g., on host port 8000).

```bash
docker run -p 8080:8080 \
  -e VLLM_API_BASE="http://host.docker.internal:8000/v1" \
  -e HMA_API_BASE="http://host.docker.internal:8000/v1" \
  hma-service
```

### 3. Submit a Task
```bash
curl -X POST "http://localhost:8080/submit" \
     -H "Content-Type: application/json" \
     -d '{"task_description": "Explain the difference between SSM and Transformer architectures.", "total_budget_tokens": 5000}'
```
