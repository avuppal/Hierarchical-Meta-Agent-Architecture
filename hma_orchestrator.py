from typing import Annotated, Sequence, TypedDict, Callable, Any, Optional, List, Dict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
import re
import os
import time
import hashlib
import pandas as pd
import asyncio
import subprocess
import sys
import chromadb
from chromadb.utils import embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import dspy
from pypdf import PdfReader

# --- New Modular HMA Imports ---
try:
    from orchestrator.AgentArchitect import AgentArchitect
    from core.SessionManager import SessionManager
except ImportError:
    # Fallback for when running without the full package structure
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from orchestrator.AgentArchitect import AgentArchitect
    from core.SessionManager import SessionManager

# --- Configuration Constants ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") 
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") 

# Configure DSPy
lm = dspy.LM(model=f"openai/{VLLM_MODEL_NAME}", api_base=VLLM_API_BASE, api_key=VLLM_API_KEY)
dspy.settings.configure(lm=lm)

BENCHMARK_FILE = "hma_benchmark_logs.csv"
CHROMA_DB_PATH = "hma_semantic_cache_db"

# Rate Limiting
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "20"))
worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

# --- Metrics ---
HMA_TOKENS_TOTAL = Counter('hma_tokens_total', 'Total tokens', ['model_name', 'step_type'])
HMA_TASK_DURATION = Histogram('hma_task_duration_seconds', 'Task duration', ['step_type'])

# --- Logger ---
class BenchmarkLogger:
    def __init__(self, filename=BENCHMARK_FILE):
        self.filename = filename
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=["timestamp", "step_type", "duration_ms", "status"])
            df.to_csv(self.filename, index=False)

    def log(self, step_type: str, duration_ms: float, status: str):
        new_row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "step_type": step_type, "duration_ms": duration_ms, "status": status}
        df = pd.DataFrame([new_row])
        df.to_csv(self.filename, mode='a', header=False, index=False)

logger = BenchmarkLogger()

# --- Semantic Cache ---
class VectorSemanticCache:
    def __init__(self, path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name="hma_cache", embedding_function=self.ef)

    def get(self, query: str) -> Optional[dict]:
        try:
            res = self.collection.query(query_texts=[query], n_results=1)
            if res['documents'] and res['documents'][0]:
                return json.loads(res['metadatas'][0][0]['response_json'])
        except: pass
        return None

    def set(self, query: str, data: dict):
        try:
            doc_id = hashlib.sha256(query.encode()).hexdigest()
            self.collection.upsert(documents=[query], metadatas=[{"response_json": json.dumps(data)}], ids=[doc_id])
        except: pass

semantic_cache = VectorSemanticCache()

# --- DSPy Signatures (Legacy/Fallback) ---
class WorkerSignature(dspy.Signature):
    """Execute a task."""
    sub_task = dspy.InputField()
    context = dspy.InputField()
    response_json = dspy.OutputField()

# --- State ---
class GraphState(TypedDict):
    task_description: str
    total_budget_tokens: int
    task_waves: List[List[Dict[str, str]]]
    current_wave_index: int
    worker_outputs: Dict[str, str]
    worker_status: str

# --- Nodes ---
async def analyst_wave_planner(state: GraphState) -> dict:
    """
    Uses the AgentArchitect (Cortex) to design the swarm.
    """
    print("--- [Architect]: Designing Swarm ---")
    
    # Cache Check
    cache_key = f"ARCHITECT:{state['task_description']}"
    cached = semantic_cache.get(cache_key)
    if cached: 
        print("  [Cache Hit]")
        return cached

    start = time.time()
    
    # NEW: Use the Modular AgentArchitect
    architect = AgentArchitect()
    swarm_plan = architect.design_swarm(state['task_description'])
    
    duration = (time.time() - start) * 1000
    logger.log("ARCHITECT", duration, "SUCCESS")
    
    # Convert Swarm Plan to "Waves" format for compatibility with existing graph
    # The Architect returns {"strategy": "parallel", "agents": [...]}
    # We treat the list of agents as a single parallel wave for now.
    
    agents = swarm_plan.get("agents", [])
    current_wave = []
    
    for agent in agents:
        current_wave.append({
            "description": agent["task"],
            "type": agent["skill"], # Use skill ID as type
            "role": agent["role"],
            "tools": agent.get("tools", [])
        })
        
    # If strategy is sequential, we might want to split into multiple waves, 
    # but for this integration we'll just queue them in one wave list for dispatch
    waves = [current_wave]
    
    new_state = {"task_waves": waves, "current_wave_index": 0}
    semantic_cache.set(cache_key, new_state)
    return new_state

async def dispatch_wave(state: GraphState) -> dict:
    if state['current_wave_index'] >= len(state['task_waves']): return {"worker_status": "COMPLETE"}
    return {"worker_status": "EXECUTING_WAVE"}

async def execute_wave_tasks(state: GraphState) -> dict:
    current_wave = state['task_waves'][state['current_wave_index']]
    print(f"--- Wave {state['current_wave_index']} ({len(current_wave)} tasks) ---")
    
    # NEW: Use SessionManager for execution
    session_manager = SessionManager()
    
    # For simulation/asyncio compatibility, we'll wrap the SessionManager calls
    # In a real deployed version, SessionManager might handle async internally or via API
    
    async def run_agent_task(agent_def):
        role = agent_def.get("role", "worker")
        task_desc = agent_def.get("description")
        tools = agent_def.get("tools")
        
        # Spawn the session (Virtual)
        session_id = session_manager.spawn(role, task_desc, tools=tools)
        
        # Execute (Simulated for now, would be an API call to the agent)
        # Here we still use the local DSPy worker for the "thinking" part, 
        # but logically it runs inside the SessionManager's scope.
        
        if "python" in tools or "exec" in tools:
            # It's an engineer
             # Simple mock for demo since we replaced the specific EngineerSignature
            return f"[Session {session_id}] Executed code for: {task_desc}"
        
        elif "web_search" in tools:
            # Research agent
            return f"[Session {session_id}] Researched: {task_desc}"
            
        else:
            # General worker
            prog = dspy.ChainOfThought(WorkerSignature)
            # We run this in a thread to keep it async
            pred = await asyncio.to_thread(prog, sub_task=task_desc, context=str(state.get('worker_outputs', {})))
            return f"[Session {session_id}] {pred.response_json}"

    tasks = [run_agent_task(agent) for agent in current_wave]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    outputs = {}
    for i, res in enumerate(results):
        desc = current_wave[i]['description']
        outputs[desc] = str(res)
            
    return {
        "worker_outputs": {**state.get('worker_outputs', {}), **outputs},
        "current_wave_index": state['current_wave_index'] + 1,
        "worker_status": "READY_TO_DISPATCH"
    }

async def final_review(state: GraphState) -> dict:
    print("--- [Review]: Swarm Execution Complete ---")
    return {"worker_status": "DONE"}

# --- Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("analyst", analyst_wave_planner)
workflow.add_node("dispatch", dispatch_wave)
workflow.add_node("execute", execute_wave_tasks)
workflow.add_node("review", final_review)

workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "dispatch")
workflow.add_conditional_edges("dispatch", lambda x: x['worker_status'], {"EXECUTING_WAVE": "execute", "COMPLETE": "review"})
workflow.add_edge("execute", "dispatch")
workflow.add_edge("review", END)
app = workflow.compile()

if __name__ == "__main__":
    print("Starting Modular HMA.")
    async def main():
        async for step in app.astream({"task_description": "Deploy a fraud detection system on AWS", "total_budget_tokens": 1000}):
            pass # Output is handled by print statements in nodes
    asyncio.run(main())
