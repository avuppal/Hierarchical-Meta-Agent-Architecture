# Hierarchical Meta-Agent Architecture (HMA)

The next-generation infrastructure for **Token-Efficient, Latency-Optimized Agent Swarms**.

This repository implements the **Hierarchical Meta-Agent Architecture (HMA)**, a modular system designed to separate execution from cognition, enabling dynamic swarm composition, just-in-time skill loading, and massive parallel execution.

---

## 🚀 Key Features

### 1. The Cortex (Dynamic Planning)
*   **What it does:** Replaces static task lists with dynamic strategy generation.
*   **Component:** `orchestrator/AgentArchitect.py`
*   **Benefit:** The system doesn't just "do tasks"—it **architects a team** specifically for your request (e.g., "Spawn 1 Researcher + 1 Python Engineer").

### 2. The Core (Efficient Execution)
*   **What it does:** Provides a sandboxed, state-isolated runtime for agents.
*   **Component:** `core/SessionManager.py`
*   **Benefit:** **Zero Context Bloat.** Sub-agents start with empty context, reducing token costs by ~80% compared to shared-history swarms.

### 3. The Universal Registry (Infinite Skills)
*   **What it does:** An "App Store" for agent tools. Supports local skills (`skills/`) and imports directly from **LangChain**, **ClawHub**, and remote repositories.
*   **Component:** `skills/SkillRegistry.py`
*   **Benefit:** Your agents have access to 10,000+ tools without bloating the codebase.

### 4. HPC Optimization (Parallelization)
*   **What it does:** Fires all sub-agents simultaneously using `asyncio.gather`.
*   **Benefit:** Saturates GPU batching (via vLLM) for **5x faster throughput** than sequential chains.

### 5. Queuing & Forecasting (Flow Control)
*   **What it does:** Manages job queues with priorities (`QueueManager`) and predicts costs/times (`ForecastingEngine`).
*   **Benefit:** Prevents overload and enables proactive budgeting (Queuing Theory + Forecasting).

### 6. Bottleneck Detection & Resilience (TOC-Inspired)
*   **What it does:** Identifies slow agents (`BottleneckDetector`) and handles failures/retry.
*   **Benefit:** Optimizes throughput by reallocating resources and ensuring reliability.

---

## 📂 Architecture Overview

The system is split into three distinct layers:

| Layer | Component | Role | Responsibility |
| :--- | :--- | :--- | :--- |
| **Cognition** | `AgentArchitect` | Strategy | Analyzes user intent, designs swarm manifests. |
| **Cognition** | `ForecastingEngine` | Prediction | Forecasts costs/times using historical data. |
| **Memory** | `SkillRegistry` | Knowledge | Indexes capabilities, RAG search for tools. |
| **Execution** | `SessionManager` | Runtime | Spawns processes, enforces sandboxes, manages lifecycle. |
| **Execution** | `QueueManager` | Flow Control | Manages job queues with priorities. |
| **Execution** | `BottleneckDetector` | Optimization | Detects bottlenecks, suggests reallocations. |

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/avuppal/Hierarchical-Meta-Agent-Architecture.git
cd Hierarchical-Meta-Agent-Architecture
pip install -r requirements.txt
```

### 2. Run the Service (API)
Start the HMA-AaaS (Agent-as-a-Service) API server:
```bash
python3 service.py
```
*   **API:** `http://localhost:8080`
*   **Metrics:** `http://localhost:8080/metrics`

### 3. Submit a Job
```bash
curl -X POST "http://localhost:8080/submit" \
     -H "Content-Type: application/json" \
     -d '{
           "task_description": "Analyze the Q3 financial report using Python and search for competitor data.",
           "total_budget_tokens": 5000
         }'
```

### 4. Check Status
```bash
curl "http://localhost:8080/status/{job_id}"
```
Response includes the **Swarm Plan** (which agents were spawned) and their individual outputs.

---

## 🧩 Adding Skills

### Option A: Local Skill
Create a folder in `skills/`:
```text
skills/
  my-custom-tool/
    SKILL.md       # The system prompt / instructions
    manifest.json  # Metadata (tools required, permissions)
```

### Option B: Import from LangChain
The `SkillRegistry` can dynamically import any LangChain tool:
```python
from skills.SkillRegistry import SkillRegistry
registry = SkillRegistry()
registry.import_langchain_tool("GoogleSerperRun")
```

---

## 📊 Deployment (Docker)

For full-stack deployment with a local vLLM inference server:

```bash
docker-compose up --build
```

This launches:
1.  **HMA Service** (Port 8080)
2.  **vLLM Worker** (Port 8000) - utilizing all available GPUs.
3.  **Prometheus Metrics** (Port 8001)

---

## 🔬 Research Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency of a multi-agent system is maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting** and **Tool Creation**, rather than just task decomposition.

**Benchmarks:** See `hma_benchmark_logs.csv` for real-time token ROI and latency metrics.
