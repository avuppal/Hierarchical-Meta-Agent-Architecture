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
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration Constants ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") 
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") 

HMA_API_BASE = os.getenv("HMA_API_BASE", "http://localhost:8000/v1")
HMA_API_KEY = os.getenv("HMA_API_KEY", "EMPTY")
HMA_MODEL_NAME = os.getenv("HMA_MODEL_NAME", "facebook/opt-125m") 

BENCHMARK_FILE = "hma_benchmark_logs.csv"
CHROMA_DB_PATH = "hma_semantic_cache_db"

# Rate Limiting (Semaphore)
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "20"))
worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

# --- 1. Prometheus Metrics ---
HMA_TOKENS_TOTAL = Counter('hma_tokens_total', 'Total tokens consumed', ['model_name', 'step_type'])
HMA_TASK_DURATION = Histogram('hma_task_duration_seconds', 'Duration of tasks in seconds', ['step_type'])
HMA_WORKER_ERRORS = Counter('hma_worker_errors_total', 'Total worker errors', ['error_type'])
HMA_ACTIVE_JOBS = Gauge('hma_active_jobs', 'Number of currently running jobs')

# --- 2. Benchmark Logger ---
class BenchmarkLogger:
    def __init__(self, filename=BENCHMARK_FILE):
        self.filename = filename
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=[
                "timestamp", "step_type", "duration_ms", "tokens_in", "tokens_out", 
                "total_tokens", "cost_estimate", "status", "output_snippet"
            ])
            df.to_csv(self.filename, index=False)

    def log(self, step_type: str, duration_ms: float, usage: dict, status: str, output: str):
        new_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_type": step_type,
            "duration_ms": round(duration_ms, 2),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_estimate": 0.0,
            "status": status,
            "output_snippet": output[:50].replace("\n", " ") + "..."
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.filename, mode='a', header=False, index=False)
        HMA_TOKENS_TOTAL.labels(model_name="unknown", step_type=step_type).inc(usage.get('total_tokens', 0))
        HMA_TASK_DURATION.labels(step_type=step_type).observe(duration_ms / 1000.0)
        print(f"  [Log] Saved metric: {step_type} ({duration_ms}ms, {usage.get('total_tokens', 0)} tokens)")

logger = BenchmarkLogger()

# --- 3. Vector Semantic Cache ---
class VectorSemanticCache:
    def __init__(self, path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="hma_cache", 
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def get(self, query_text: str, threshold: float = 0.92) -> Optional[dict]:
        try:
            results = self.collection.query(query_texts=[query_text], n_results=1)
            if not results['documents'] or not results['documents'][0]: return None
            distance = results['distances'][0][0]
            if distance < (1 - threshold): 
                cached_data = json.loads(results['metadatas'][0][0]['response_json'])
                print(f"  [Vector Cache Hit] Distance: {distance:.4f}")
                return cached_data
        except Exception as e:
            print(f"Cache Query Error: {e}")
        return None

    def set(self, query_text: str, response_data: dict):
        try:
            doc_id = hashlib.sha256(query_text.encode()).hexdigest()
            self.collection.upsert(
                documents=[query_text],
                metadatas=[{"response_json": json.dumps(response_data)}],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"Cache Set Error: {e}")

semantic_cache = VectorSemanticCache()

# --- 4. State Schema ---
class TaskItem(BaseModel):
    description: str
    type: str = "TEXT" # TEXT or CODE

class AgentBudgetState(BaseModel):
    task_description: str
    total_budget_tokens: int
    
    # Updated: List of Lists of Dicts (Task objects)
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

# --- 5. Helpers ---
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_llm_async(api_base, api_key, model, prompt, system_prompt="You are a helpful assistant."):
    semantic_key = f"{system_prompt}\n\n{prompt}"
    cached = semantic_cache.get(semantic_key)
    if cached:
        return cached['content'], cached['usage'], 0.0 
    
    async with worker_semaphore:
        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            duration = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            usage = response.usage.model_dump()
            semantic_cache.set(semantic_key, {'content': content, 'usage': usage})
            return content, usage, duration
        except Exception as e:
            HMA_WORKER_ERRORS.labels(error_type=type(e).__name__).inc()
            raise e 

def extract_json(text: str) -> dict:
    try: return json.loads(text)
    except:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: 
            try: return json.loads(match.group(1))
            except: pass
    return {}

def execute_python_code(code: str) -> str:
    """Executes generated Python code in a subprocess."""
    print("    [Engineer]: Executing Code...")
    try:
        # Clean code (remove markdown code blocks if present)
        clean_code = re.sub(r'```python\s*', '', code).replace('```', '')
        
        with open("temp_script.py", "w") as f:
            f.write(clean_code)
        
        result = subprocess.run(
            [sys.executable, "temp_script.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        return output.strip() if output.strip() else "[No Output]"
    except Exception as e:
        return f"Execution Error: {str(e)}"

# --- 6. Node Functions ---

async def analyst_wave_planner(state: GraphState) -> dict:
    print("--- [Analyst Node]: Planning Task Waves ---")
    raw_task = state['task_description']
    
    prompt = f"""
    You are an Expert AI Project Manager.
    Raw Request: "{raw_task}"
    
    1. Break this into distinct sub-tasks.
    2. Determine if each task is "TEXT" (Research/Writing) or "CODE" (Calculation/Data).
    3. Group them into PARALLEL WAVES.
    
    OUTPUT JSON FORMAT:
    {{
      "complexity": "MEDIUM",
      "waves": [
        [{{"description": "Task 1", "type": "TEXT"}}, {{"description": "Calc 1", "type": "CODE"}}],
        [{{"description": "Task 2", "type": "TEXT"}}]
      ]
    }}
    """
    
    try:
        content, usage, duration = await call_llm_async(
            HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME, 
            prompt=prompt,
            system_prompt="You are a precise technical analyst."
        )
        logger.log("ANALYST_PLANNING", duration, usage, "SUCCESS", content)
        
        data = extract_json(content)
        # Fallback to simple format if parsing complex format fails
        waves_raw = data.get("waves", [])
        # Normalize waves to Dict format if they came back as strings (legacy cache)
        waves_normalized = []
        for wave in waves_raw:
            new_wave = []
            for task in wave:
                if isinstance(task, str):
                    new_wave.append({"description": task, "type": "TEXT"})
                else:
                    new_wave.append(task)
            waves_normalized.append(new_wave)
            
        if not waves_normalized:
             waves_normalized = [[{"description": raw_task, "type": "TEXT"}]]
             
        complexity = data.get("complexity", "MEDIUM")
    except Exception as e:
        print(f"Analyst Failed: {e}. Fallback.")
        waves_normalized = [[{"description": raw_task, "type": "TEXT"}]]
        complexity = "MEDIUM"
        usage = {'total_tokens': 0}
    
    print(f"Analyst Plan: {len(waves_normalized)} Waves. Complexity: {complexity}")
    
    base_budget = state['total_budget_tokens']
    multipliers = {"LOW": 0.8, "MEDIUM": 1.0, "HIGH": 1.5}
    adjusted_budget = int(base_budget * multipliers.get(complexity, 1.0))

    return {
        "task_waves": waves_normalized,
        "current_wave_index": 0,
        "complexity_score": complexity,
        "total_budget_tokens": adjusted_budget,
        "tokens_spent": state['tokens_spent'] + usage.get('total_tokens', 0)
    }

async def dispatch_wave(state: GraphState) -> dict:
    if state['current_wave_index'] >= len(state['task_waves']):
        return {"worker_status": "COMPLETE"} 
    current_wave = state['task_waves'][state['current_wave_index']]
    print(f"--- [Orchestrator]: Dispatching Wave {state['current_wave_index'] + 1}/{len(state['task_waves'])}: {len(current_wave)} parallel tasks ---")
    return {"worker_status": "EXECUTING_WAVE"}

async def execute_wave_tasks(state: GraphState) -> dict:
    current_wave = state['task_waves'][state['current_wave_index']]
    async_tasks = []
    task_metadata = [] # Store (name, type)

    for task_obj in current_wave:
        description = task_obj['description']
        task_type = task_obj.get('type', 'TEXT')
        task_metadata.append((description, task_type))
        
        print(f"  -> Queueing {task_type} Sub-Task: {description[:30]}...")
        
        context_str = "\n".join([f"- {k}: {v[:100]}..." for k, v in state.get('worker_outputs', {}).items()])
        
        if task_type == "CODE":
            prompt = f"""
            Task: {description}
            Context: {context_str}
            
            Write a Python script to solve this. 
            OUTPUT FORMAT: STRICT JSON ONLY.
            {{
                "code": "print('hello')",
                "status": "complete"
            }}
            """
            system_prompt = "You are a Python Engineer. Write code to solve problems."
        else:
            # TEXT Task (with Reflexion)
            use_cot = state['complexity_score'] == "HIGH"
            instruction = "Think step-by-step." if use_cot else "Be concise."
            prompt = f"""
            Task: {description}
            Context: {context_str}
            Instruction: {instruction} Draft, critique, and refine.
            OUTPUT FORMAT: STRICT JSON ONLY.
            {{
                "critique": "...",
                "content": "...",
                "status": "complete"
            }}
            """
            system_prompt = "You are a precise Worker Agent."
        
        async_tasks.append(call_llm_async(
            VLLM_API_BASE, VLLM_API_KEY, VLLM_MODEL_NAME, 
            prompt=prompt,
            system_prompt=system_prompt
        ))
    
    print(f"  -> Firing {len(async_tasks)} concurrent requests...")
    results = await asyncio.gather(*async_tasks, return_exceptions=True)
    
    outputs = {}
    total_cost = 0
    
    for i, result in enumerate(results):
        desc, task_type = task_metadata[i]
        if isinstance(result, Exception):
            outputs[desc] = f"ERROR: {str(result)}"
            continue
            
        content, usage, duration = result
        data = extract_json(content)
        total_cost += usage.get('total_tokens', 0)
        logger.log("WORKER_EXECUTION", duration, usage, "SUCCESS", content)
        
        if task_type == "CODE":
            code = data.get("code", "")
            if code:
                exec_result = execute_python_code(code)
                outputs[desc] = f"[CODE EXECUTED]\nCode:\n{code}\nResult:\n{exec_result}"
            else:
                outputs[desc] = "[ERROR] No code generated."
        else:
            outputs[desc] = data.get("content", content)

    print(f"Wave {state['current_wave_index'] + 1} Complete. Cost: {total_cost} tokens.")
    
    return {
        "worker_outputs": {**state.get('worker_outputs', {}), **outputs},
        "tokens_spent": state['tokens_spent'] + total_cost,
        "current_wave_index": state['current_wave_index'] + 1,
        "worker_status": "READY_TO_DISPATCH"
    }

async def final_review(state: GraphState) -> dict:
    print("--- [HMA Controller]: Final Review ---")
    all_work = json.dumps(state['worker_outputs'], indent=2)
    prompt = f"""
    Original Goal: {state['task_description']}
    Completed Work:
    {all_work}
    Synthesize this into a final report.
    """
    try:
        content, usage, duration = await call_llm_async(
            HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME,
            prompt=prompt,
            system_prompt="You are a strict Project Manager."
        )
        logger.log("HMA_FINAL_REVIEW", duration, usage, "SUCCESS", content)
        return {
            "meta_review": content,
            "tokens_spent": state['tokens_spent'] + usage.get('total_tokens', 0),
            "successful": True
        }
    except Exception as e:
        return {"meta_review": "Failed", "successful": False}

# --- 7. Graph Setup ---
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
    print(f"Starting HMA Benchmark.")
    initial_state = {
        "task_description": "Calculate the sum of the first 100 prime numbers and then explain the distribution of primes.",
        "total_budget_tokens": 5000, 
    }
    async def main():
        try:
            async for step in app.astream(initial_state, config={"configurable": {"recursion_limit": 20}}):
                for key in step: print(f"  -> Finished Node: {key}")
        except Exception as e: print(f"Error: {e}")
    asyncio.run(main())
