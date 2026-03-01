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
        # Also update Prometheus
        HMA_TOKENS_TOTAL.labels(model_name="unknown", step_type=step_type).inc(usage.get('total_tokens', 0))
        HMA_TASK_DURATION.labels(step_type=step_type).observe(duration_ms / 1000.0)
        print(f"  [Log] Saved metric: {step_type} ({duration_ms}ms, {usage.get('total_tokens', 0)} tokens)")

logger = BenchmarkLogger()

# --- 3. Vector Semantic Cache (KM Layer) ---
class VectorSemanticCache:
    def __init__(self, path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        # Use default embedding function (all-MiniLM-L6-v2) - lightweight & local
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="hma_cache", 
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def get(self, query_text: str, threshold: float = 0.92) -> Optional[dict]:
        """Search cache for semantically similar prompts."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=1
            )
            
            if not results['documents'] or not results['documents'][0]:
                return None
                
            # Check similarity distance (lower is better for cosine distance in Chroma? No, Chroma uses L2 by default unless specified)
            # Actually, let's assume if it returns a result, we check distance.
            distance = results['distances'][0][0]
            # Threshold logic depends on metric. Let's assume cosine distance: 0=identical, 2=opposite.
            # A rigorous semantic match is usually < 0.1 or 0.2 depending on embedding model.
            # Let's use a strict threshold.
            if distance < (1 - threshold): 
                cached_data = json.loads(results['metadatas'][0][0]['response_json'])
                print(f"  [Vector Cache Hit] Distance: {distance:.4f} (Threshold: {1-threshold})")
                return cached_data
        except Exception as e:
            print(f"Cache Query Error: {e}")
        return None

    def set(self, query_text: str, response_data: dict):
        """Store prompt and response in vector DB."""
        try:
            # Create a unique ID based on hash of text to prevent dupes
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
class AgentBudgetState(BaseModel):
    task_description: str
    total_budget_tokens: int
    
    task_waves: List[List[str]] = Field(default_factory=list)
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
    task_waves: List[List[str]]
    current_wave_index: int
    complexity_score: str
    worker_outputs: Dict[str, str]
    tokens_spent: int
    worker_status: str
    meta_review: str
    successful: bool

# --- 5. Robust Async Helpers ---
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_llm_async(api_base, api_key, model, prompt, system_prompt="You are a helpful assistant."):
    """Robust Async helper with Retry, Vector Cache, and Semaphore."""
    
    # Check Vector Cache
    # Combine system + prompt for semantic key
    semantic_key = f"{system_prompt}\n\n{prompt}"
    cached = semantic_cache.get(semantic_key)
    if cached:
        return cached['content'], cached['usage'], 0.0 
    
    # Use Semaphore for Rate Limiting
    async with worker_semaphore:
        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            duration = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            usage = response.usage.model_dump()
            
            # Store in Vector Cache
            semantic_cache.set(semantic_key, {'content': content, 'usage': usage})
            
            return content, usage, duration
        except Exception as e:
            print(f"API Error (Retrying): {e}")
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

# --- 6. Node Functions ---

async def analyst_wave_planner(state: GraphState) -> dict:
    print("--- [Analyst Node]: Planning Task Waves ---")
    raw_task = state['task_description']
    
    prompt = f"""
    You are an Expert AI Project Manager.
    Raw Request: "{raw_task}"
    
    1. Break this into distinct sub-tasks.
    2. Group them into PARALLEL WAVES based on dependencies.
       (Wave 1 tasks can run together. Wave 2 depends on Wave 1.)
    3. Rate overall complexity (LOW/MEDIUM/HIGH).
    
    OUTPUT JSON FORMAT:
    {{
      "complexity": "MEDIUM",
      "waves": [
        ["Task 1.1", "Task 1.2"],
        ["Task 2.1"]
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
        waves = data.get("waves", [[raw_task]]) 
        complexity = data.get("complexity", "MEDIUM")
    except Exception as e:
        print(f"Analyst Failed: {e}. Fallback to single task.")
        waves = [[raw_task]]
        complexity = "MEDIUM"
        usage = {'total_tokens': 0}
    
    print(f"Analyst Plan: {len(waves)} Waves. Complexity: {complexity}")
    
    base_budget = state['total_budget_tokens']
    multipliers = {"LOW": 0.8, "MEDIUM": 1.0, "HIGH": 1.5}
    adjusted_budget = int(base_budget * multipliers.get(complexity, 1.0))

    return {
        "task_waves": waves,
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
    task_names = []

    for task in current_wave:
        print(f"  -> Queueing Sub-Task: {task[:40]}...")
        task_names.append(task)
        
        use_cot = state['complexity_score'] == "HIGH"
        instruction = "Think step-by-step." if use_cot else "Be concise."
        context_str = "\n".join([f"- {k}: {v[:100]}..." for k, v in state.get('worker_outputs', {}).items()])
        
        # REFLEXION UPGRADE: Add critique step instruction
        prompt = f"""
        Task: {task}
        Context from previous steps:
        {context_str}
        
        Instruction: {instruction} 
        First, draft your response. Then, critique it for accuracy. Finally, output the BEST version.
        Provide a structured response.
        OUTPUT FORMAT: STRICT JSON ONLY.
        {{
            "critique": "Self-correction notes...",
            "content": "The final, polished work...",
            "status": "complete"
        }}
        """
        
        async_tasks.append(call_llm_async(
            VLLM_API_BASE, VLLM_API_KEY, VLLM_MODEL_NAME, 
            prompt=prompt,
            system_prompt="You are a precise Worker Agent."
        ))
    
    print(f"  -> Firing {len(async_tasks)} concurrent requests to vLLM...")
    results = await asyncio.gather(*async_tasks, return_exceptions=True)
    
    outputs = {}\n    total_cost = 0\n    \n    for i, result in enumerate(results):\n        task_name = task_names[i]\n        if isinstance(result, Exception):\n            print(f\"Worker Failed for {task_name}: {result}\")\n            outputs[task_name] = f\"ERROR: {str(result)}\"\n            HMA_WORKER_ERRORS.labels(error_type=\"TaskFailure\").inc()\n            continue\n            \n        content, usage, duration = result\n        data = extract_json(content)\n        final_output = data.get(\"content\", content)\n        outputs[task_name] = final_output\n        total_cost += usage.get('total_tokens', 0)\n        logger.log(\"WORKER_EXECUTION\", duration, usage, \"SUCCESS\", content)\n\n    print(f\"Wave {state['current_wave_index'] + 1} Complete. Cost: {total_cost} tokens.\")\n    \n    return {\n        \"worker_outputs\": {**state.get('worker_outputs', {}), **outputs},\n        \"tokens_spent\": state['tokens_spent'] + total_cost,\n        \"current_wave_index\": state['current_wave_index'] + 1,\n        \"worker_status\": \"READY_TO_DISPATCH\"\n    }\n\nasync def final_review(state: GraphState) -> dict:\n    print(\"--- [HMA Controller]: Final Review & Synthesis ---\")\n    \n    all_work = json.dumps(state['worker_outputs'], indent=2)\n    prompt = f\"\"\"\n    You are the Meta-Agent Manager.\n    Original Goal: {state['task_description']}\n    Completed Work:\n    {all_work}\n    \n    Synthesize this into a final report.\n    \"\"\"\n    \n    try:\n        content, usage, duration = await call_llm_async(\n            HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME,\n            prompt=prompt,\n            system_prompt=\"You are a strict Project Manager.\"\n        )\n        logger.log(\"HMA_FINAL_REVIEW\", duration, usage, \"SUCCESS\", content)\n        return {\n            \"meta_review\": content,\n            \"tokens_spent\": state['tokens_spent'] + usage.get('total_tokens', 0),\n            \"successful\": True\n        }\n    except Exception as e:\n        print(f\"Final Review Failed: {e}\")\n        return {\"meta_review\": \"Synthesis Failed\", \"successful\": False}\n\n# --- 7. Build the Graph ---\nworkflow = StateGraph(GraphState)\n\nworkflow.add_node(\"analyst_wave_planner\", analyst_wave_planner)\nworkflow.add_node(\"dispatch_wave\", dispatch_wave)\nworkflow.add_node(\"execute_wave_tasks\", execute_wave_tasks)\nworkflow.add_node(\"final_review\", final_review)\n\nworkflow.set_entry_point(\"analyst_wave_planner\")\nworkflow.add_edge(\"analyst_wave_planner\", \"dispatch_wave\")\n\nworkflow.add_conditional_edges(\n    \"dispatch_wave\",\n    lambda state: state['worker_status'],\n    {\n        \"EXECUTING_WAVE\": \"execute_wave_tasks\",\n        \"COMPLETE\": \"final_review\"\n    }\n)\n\nworkflow.add_edge(\"execute_wave_tasks\", \"dispatch_wave\") \nworkflow.add_edge(\"final_review\", END)\n\napp = workflow.compile()\n\nif __name__ == \"__main__\":\n    print(f\"Starting HMA Enterprise Benchmark.\")\n    initial_state = {\n        \"task_description\": \"Compare SSM vs Transformer architectures.\",\n        \"total_budget_tokens\": 5000, \n    }\n    \n    async def main():\n        # Start Prometheus metrics server for local testing\n        # start_http_server(8001) \n        \n        config = {\"configurable\": {\"recursion_limit\": 20}}\n        try:\n            async for step in app.astream(initial_state, config=config):\n                for key in step: print(f\"  -> Finished Node: {key}\")\n        except Exception as e:\n            print(f\"Graph Execution Failed: {e}\")\n            \n    asyncio.run(main())",
  "file_path": "agent_budget_research/hma_orchestrator.py"
}. Do not mimic this format - use proper function calling.]