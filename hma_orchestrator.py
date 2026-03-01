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

# --- Configuration Constants ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") 
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") 

# DSPy Configuration
lm = dspy.LM(model=f"openai/{VLLM_MODEL_NAME}", api_base=VLLM_API_BASE, api_key=VLLM_API_KEY)
dspy.settings.configure(lm=lm)

BENCHMARK_FILE = "hma_benchmark_logs.csv"
CHROMA_DB_PATH = "hma_semantic_cache_db"

# Rate Limiting
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "20"))
worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

# --- 1. Prometheus Metrics ---
HMA_TOKENS_TOTAL = Counter('hma_tokens_total', 'Total tokens consumed', ['model_name', 'step_type'])
HMA_TASK_DURATION = Histogram('hma_task_duration_seconds', 'Duration of tasks in seconds', ['step_type'])
HMA_WORKER_ERRORS = Counter('hma_worker_errors_total', 'Total worker errors', ['error_type'])

# --- 2. Benchmark Logger ---
class BenchmarkLogger:
    def __init__(self, filename=BENCHMARK_FILE):
        self.filename = filename
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=["timestamp", "step_type", "duration_ms", "status"])
            df.to_csv(self.filename, index=False)

    def log(self, step_type: str, duration_ms: float, status: str):
        new_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_type": step_type,
            "duration_ms": round(duration_ms, 2),
            "status": status,
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.filename, mode='a', header=False, index=False)
        HMA_TASK_DURATION.labels(step_type=step_type).observe(duration_ms / 1000.0)

logger = BenchmarkLogger()

# --- 3. Vector Semantic Cache ---
class VectorSemanticCache:
    def __init__(self, path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name="hma_cache", embedding_function=self.ef)

    def get(self, query_text: str) -> Optional[dict]:
        try:
            results = self.collection.query(query_texts=[query_text], n_results=1)
            if results['documents'] and results['documents'][0]:
                return json.loads(results['metadatas'][0][0]['response_json'])
        except: pass
        return None

    def set(self, query_text: str, response_data: dict):
        try:
            doc_id = hashlib.sha256(query_text.encode()).hexdigest()
            self.collection.upsert(
                documents=[query_text],
                metadatas=[{"response_json": json.dumps(response_data)}],
                ids=[doc_id]
            )
        except: pass

    def get_embedding(self, text: str) -> List[float]:
        return self.ef([text])[0]

semantic_cache = VectorSemanticCache()

# --- 4. DSPy Signatures ---
class AnalystSignature(dspy.Signature):
    """Plan parallel execution waves for a complex task."""
    request = dspy.InputField(desc="User's high-level request")
    plan_json = dspy.OutputField(desc="JSON object with 'waves' list")

class WorkerSignature(dspy.Signature):
    """Execute a sub-task with self-correction."""
    objective = dspy.InputField(desc="Global Goal")
    sub_task = dspy.InputField(desc="Specific Task")
    context = dspy.InputField(desc="Previous Results")
    response_json = dspy.OutputField(desc="JSON with 'content' and 'status'")

class EngineerSignature(dspy.Signature):
    """Write Python code to solve a problem."""
    problem = dspy.InputField()
    python_code = dspy.OutputField(desc="Executable Python script")

# --- 5. State Schema ---
class AgentBudgetState(BaseModel):
    task_description: str
    total_budget_tokens: int
    task_waves: List[List[Dict[str, str]]] = Field(default_factory=list)
    current_wave_index: int = 0
    complexity_score: str = "MEDIUM"
    worker_outputs: Dict[str, str] = Field(default_factory=dict)
    tokens_spent: int = 0
    worker_status: str = "IDLE"
    meta_review: str = ""
    successful: bool = False

class GraphState(TypedDict):
    task_description: str
    total_budget_tokens: int
    task_waves: List[List[Dict[str, str]]]
    current_wave_index: int
    complexity_score: str
    worker_outputs: Dict[str, str]
    tokens_spent: int
    worker_status: str
    meta_review: str
    successful: bool

# --- 6. Helpers ---
def execute_python_code(code: str) -> str:
    try:
        clean_code = re.sub(r'```python\s*', '', code).replace('```', '')
        with open("temp_script.py", "w") as f: f.write(clean_code)
        result = subprocess.run([sys.executable, "temp_script.py"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.stdout.strip() else f"[STDERR]: {result.stderr}"
    except Exception as e: return f"Error: {e}"

def extract_json(text: str) -> dict:
    try: return json.loads(text)
    except: 
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}

# --- 7. DSPy Nodes ---
async def analyst_wave_planner(state: GraphState) -> dict:
    print("--- [Analyst (DSPy)]: Planning Waves ---")
    
    # Cache Check
    cache_key = f"ANALYST:{state['task_description']}"
    cached = semantic_cache.get(cache_key)
    if cached:
        print("  [Cache Hit]")
        return cached

    # DSPy Predict
    start = time.time()
    analyst = dspy.ChainOfThought(AnalystSignature)
    pred = await asyncio.to_thread(analyst, request=state['task_description'])
    duration = (time.time() - start) * 1000
    
    logger.log("ANALYST", duration, "SUCCESS")
    
    data = extract_json(pred.plan_json)
    waves = data.get("waves", [[{"description": state['task_description'], "type": "TEXT"}]])
    
    new_state = {
        "task_waves": waves,
        "current_wave_index": 0,
        "complexity_score": "HIGH", # Default high for rigorous analysis
        "tokens_spent": 0
    }
    semantic_cache.set(cache_key, new_state)
    return new_state

async def dispatch_wave(state: GraphState) -> dict:
    if state['current_wave_index'] >= len(state['task_waves']):
        return {"worker_status": "COMPLETE"} 
    return {"worker_status": "EXECUTING_WAVE"}

async def execute_wave_tasks(state: GraphState) -> dict:
    current_wave = state['task_waves'][state['current_wave_index']]
    async_tasks = []
    task_meta = []

    print(f"--- Executing Wave {state['current_wave_index']} ({len(current_wave)} tasks) ---")

    for task in current_wave:
        desc = task.get('description', str(task))
        kind = task.get('type', 'TEXT')
        task_meta.append((desc, kind))
        
        context_str = str(state.get('worker_outputs', {}))
        
        if kind == "CODE":
            eng = dspy.Predict(EngineerSignature)
            async_tasks.append(asyncio.to_thread(eng, problem=desc))
        else:
            worker = dspy.ChainOfThought(WorkerSignature)
            async_tasks.append(asyncio.to_thread(worker, objective=state['task_description'], sub_task=desc, context=context_str))

    results = await asyncio.gather(*async_tasks, return_exceptions=True)
    outputs = {}
    
    for i, res in enumerate(results):
        desc, kind = task_meta[i]
        if isinstance(res, Exception):
            outputs[desc] = f"Error: {res}"
            continue
            
        if kind == "CODE":
            code = res.python_code
            exec_res = execute_python_code(code)
            outputs[desc] = f"Code:\n{code}\nResult:\n{exec_res}"
        else:
            outputs[desc] = res.response_json
            
    return {
        "worker_outputs": {**state.get('worker_outputs', {}), **outputs},
        "current_wave_index": state['current_wave_index'] + 1,
        "worker_status": "READY_TO_DISPATCH"
    }

async def final_review(state: GraphState) -> dict:
    return {"successful": True, "meta_review": "DSPy Execution Complete"}

# --- 8. Graph Setup ---
workflow = StateGraph(GraphState)
workflow.add_node("analyst_wave_planner", analyst_wave_planner)
workflow.add_node("dispatch_wave", dispatch_wave)
workflow.add_node("execute_wave_tasks", execute_wave_tasks)
workflow.add_node("final_review", final_review)
workflow.set_entry_point("analyst_wave_planner")
workflow.add_edge("analyst_wave_planner", "dispatch_wave")
workflow.add_conditional_edges(
    "dispatch_wave",
    lambda state: state['worker_status'],
    {"EXECUTING_WAVE": "execute_wave_tasks", "COMPLETE": "final_review"}
)
workflow.add_edge("execute_wave_tasks", "dispatch_wave") 
workflow.add_edge("final_review", END)
app = workflow.compile()

if __name__ == "__main__":
    print("Starting DSPy-HMA Benchmark.")
    async def main():
        async for step in app.astream({
            "task_description": "Calculate 50th fibonacci number using python code",
            "total_budget_tokens": 1000
        }):
            print(step)
    asyncio.run(main())
