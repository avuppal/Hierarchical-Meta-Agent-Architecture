# Hierarchical Meta-Agent Architecture (HMA)

This repository contains the research codebase for testing the efficiency of **Hierarchical Meta-Agent Systems** versus flat multi-agent structures.

## Hypothesis

> For a fixed computational budget (time/energy), the overall efficiency (quality of final output per unit of cost) of a multi-agent system will be maximized by introducing a **Hierarchical Meta-Agent** whose primary function is **Dynamic Resource Budgeting** and **Interruption Synthesis**, rather than just task decomposition.

## Structure

*   **`hma_orchestrator.py`**: The core LangGraph-based orchestrator that implements the HMA logic (Budget Check -> Execute -> Review -> Loop).
*   **`Dockerfile`**: Environment setup for reproducible research.

## Running the Simulation

1.  Build the Docker image: `docker build -t hma-research .`
2.  Run the simulation: `docker run hma-research`
