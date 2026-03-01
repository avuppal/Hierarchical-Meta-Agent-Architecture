# Human-Machine Architecture (HMA) - Modular Design

This directory contains the modular implementation of HMA, separating **Execution (Core)** from **Cognition (Cortex)**.

## Core Concepts

### 1. The Core (Mechanical Layer)
- **Role:** Dumb, fast, reliable execution.
- **Responsibility:** Spawns processes, manages lifecycles, enforces sandboxes.
- **Location:** `core/`
- **Key Components:** `SessionManager`

### 2. The Cortex (Cognitive Layer)
- **Role:** Strategy, planning, and dynamic team composition.
- **Responsibility:** Analyzes user intent, designs swarms, and generates manifests.
- **Location:** `orchestrator/`
- **Key Components:** `AgentArchitect`

### 3. The Registry (Skill Plugin System)
- **Role:** The "App Store" for agent capabilities.
- **Responsibility:** Indexes local/remote skills, performs semantic search (RAG), and fetches code on demand.
- **Location:** `skills/`
- **Key Components:** `SkillRegistry`

## Usage

### Adding a New Skill
1. Create a folder in `skills/` (or a remote repo).
2. Add a `SKILL.md` (the prompt) and `manifest.json` (metadata).
3. The `SkillRegistry` will automatically index it.

### Running a Swarm
```python
from orchestrator.AgentArchitect import AgentArchitect

architect = AgentArchitect()
plan = architect.design_swarm("Build a fraud detection pipeline on AWS")
architect.deploy_swarm(plan)
```

## Integration with Existing HMA
This modular design can be integrated with the existing `hma_orchestrator.py` by having the orchestrator use `AgentArchitect` to resolve skills and `SessionManager` to execute tasks.
