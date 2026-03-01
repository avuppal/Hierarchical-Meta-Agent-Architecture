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

# --- Configuration Constants ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") 
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") 

# Configure DSPy
lm = dspy.LM(model=f"openai/{VLLM_MODEL_NAME}", api_base=VLLM_API_BASE, api_key=VLLM_API_KEY)
dspy.settings.configure(lm=lm)

BENCHMARK_FILE = "hma_benchmark_logs.csv"
CHROMA_DB_PATH = "hma_semantic_cache_db"
LIBRARIAN_DB_PATH = "hma_librarian_db"
DATA_DIR = "data"

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

# --- THE LIBRARIAN (RAG) ---
class LibrarianRAG:
    def __init__(self, data_dir=DATA_DIR, db_path=LIBRARIAN_DB_PATH):
        self.data_dir = data_dir
        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name="docs", embedding_function=self.ef)
        self._ingest_data()

    def _ingest_data(self):
        if not os.path.exists(self.data_dir): return
        print("--- [Librarian]: Indexing Documents ---")
        for f in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, f)
            text = ""
            if f.endswith(".txt") or f.endswith(".md"):
                with open(path, "r") as file: text = file.read()
            elif f.endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    for page in reader.pages: text += page.extract_text() + "\n"
                except: pass
            
            if text:
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                ids = [f"{f}_{i}" for i in range(len(chunks))]
                if chunks:
                    self.collection.upsert(documents=chunks, ids=ids, metadatas=[{"source": f}] * len(chunks))
                    print(f"  -> Indexed {f} ({len(chunks)} chunks)")

    def query(self, query_text: str, n=3) -> str:
        try:
            results = self.collection.query(query_texts=[query_text], n_results=n)
            if not results['documents'] or not results['documents'][0]: return "No relevant documents found."
            return "\n---\n".join(results['documents'][0])
        except: return "Error querying docs."

librarian = LibrarianRAG()

# --- DSPy Signatures ---
class AnalystSignature(dspy.Signature):
    """Plan parallel execution waves. Determine if tasks are TEXT, CODE, or RESEARCH."""
    request = dspy.InputField()
    plan_json = dspy.OutputField(desc="JSON with 'waves'. Task types: TEXT, CODE, RESEARCH")

class WorkerSignature(dspy.Signature):
    """Execute a task."""
    sub_task = dspy.InputField()
    context = dspy.InputField()
    response_json = dspy.OutputField()

class EngineerSignature(dspy.Signature):
    """Write Python code."""
    problem = dspy.InputField()
    python_code = dspy.OutputField()

class LibrarianSignature(dspy.Signature):
    """Answer using retrieved context."""
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

# --- State ---
class GraphState(TypedDict):
    task_description: str
    total_budget_tokens: int
    task_waves: List[List[Dict[str, str]]]
    current_wave_index: int
    worker_outputs: Dict[str, str]
    worker_status: str

# --- Helpers ---
def execute_python_code(code: str) -> str:
    try:
        clean_code = re.sub(r'```python\s*', '', code).replace('```', '')
        with open("temp_script.py", "w") as f: f.write(clean_code)
        res = subprocess.run([sys.executable, "temp_script.py"], capture_output=True, text=True, timeout=10)
        return res.stdout.strip() if res.stdout.strip() else res.stderr
    except Exception as e: return str(e)

def extract_json(text: str) -> dict:
    try: return json.loads(text)
    except: 
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}

# --- Nodes ---
async def analyst_wave_planner(state: GraphState) -> dict:
    print("--- [Analyst]: Planning ---")
    
    # Cache Check (Manual)
    cache_key = f"ANALYST:{state['task_description']}"
    cached = semantic_cache.get(cache_key)
    if cached: 
        print("  [Cache Hit]")
        return cached

    start = time.time()
    analyst = dspy.ChainOfThought(AnalystSignature)
    pred = await asyncio.to_thread(analyst, request=state['task_description'])
    duration = (time.time() - start) * 1000
    logger.log("ANALYST", duration, "SUCCESS")
    
    data = extract_json(pred.plan_json)
    waves = data.get("waves", [[{"description": state['task_description'], "type": "TEXT"}]])
    
    new_state = {"task_waves": waves, "current_wave_index": 0}
    semantic_cache.set(cache_key, new_state)
    return new_state

async def dispatch_wave(state: GraphState) -> dict:
    if state['current_wave_index'] >= len(state['task_waves']): return {"worker_status": "COMPLETE"}
    return {"worker_status": "EXECUTING_WAVE"}

async def execute_wave_tasks(state: GraphState) -> dict:
    current_wave = state['task_waves'][state['current_wave_index']]
    print(f"--- Wave {state['current_wave_index']} ({len(current_wave)} tasks) ---")
    
    async_tasks = []
    task_meta = []
    
    for task in current_wave:
        desc = task.get('description', str(task))
        kind = task.get('type', 'TEXT')
        task_meta.append((desc, kind))
        context = str(state.get('worker_outputs', {}))

        if kind == "CODE":
            prog = dspy.Predict(EngineerSignature)
            async_tasks.append(asyncio.to_thread(prog, problem=desc))
        elif kind == "RESEARCH":
            print(f"  -> [Librarian] Searching docs for: {desc}")
            retrieved_docs = librarian.query(desc)
            prog = dspy.ChainOfThought(LibrarianSignature)
            async_tasks.append(asyncio.to_thread(prog, question=desc, context=retrieved_docs))
        else:
            prog = dspy.ChainOfThought(WorkerSignature)
            async_tasks.append(asyncio.to_thread(prog, sub_task=desc, objective=state['task_description'], context=context))

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
        elif kind == "RESEARCH":
            outputs[desc] = res.answer
        else:
            outputs[desc] = res.response_json
            
    return {
        "worker_outputs": {**state.get('worker_outputs', {}), **outputs},
        "current_wave_index": state['current_wave_index'] + 1,
        "worker_status": "READY_TO_DISPATCH"
    }

async def final_review(state: GraphState) -> dict:
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
    print("Starting HMA with Librarian.")
    async def main():
        async for step in app.astream({"task_description": "Calculate 10th fibonacci and research transformers in docs", "total_budget_tokens": 1000}):
            print(step)
    asyncio.run(main())
