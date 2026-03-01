from typing import Annotated, Sequence, TypedDict, Callable, Any, Optional, List, Dict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
import re
import os
import time
import hashlib
import pandas as pd
from openai import OpenAI

# --- Configuration Constants ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") 
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") # Worker (MoE recommended)

HMA_API_BASE = os.getenv("HMA_API_BASE", "http://localhost:8000/v1")
HMA_API_KEY = os.getenv("HMA_API_KEY", "EMPTY")
HMA_MODEL_NAME = os.getenv("HMA_MODEL_NAME", "facebook/opt-125m") # Analyst (Dense/CoT recommended)

BENCHMARK_FILE = "hma_benchmark_logs.csv"
CACHE_FILE = "hma_semantic_cache.json"

# --- 1. Benchmark Logger ---
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
        print(f"  [Log] Saved metric: {step_type} ({duration_ms}ms, {usage.get('total_tokens', 0)} tokens)")

logger = BenchmarkLogger()

# --- 2. Semantic Cache ---
class SemanticCache:
    def __init__(self, filename=CACHE_FILE):
        self.filename = filename
        self.cache = {}
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}

    def get(self, key: str) -> Optional[dict]:
        return self.cache.get(key)

    def set(self, key: str, value: dict):
        self.cache[key] = value
        with open(self.filename, 'w') as f:
            json.dump(self.cache, f, indent=2)

semantic_cache = SemanticCache()

# --- 3. Define State Schema ---
class AgentBudgetState(BaseModel):
    task_description: str
    total_budget_tokens: int
    
    # Wave Planning State
    task_waves: List[List[str]] = Field(default_factory=list, description="Ordered list of parallel task groups")
    current_wave_index: int = Field(default=0, description="Current wave being executed")
    
    complexity_score: str = "MEDIUM"
    
    # Execution State
    worker_outputs: Dict[str, str] = Field(default_factory=dict, description="Map of sub-task -> output")
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

# --- 4. Helpers ---
def call_llm(api_base, api_key, model, prompt, system_prompt="You are a helpful assistant."):
    cache_key = hashlib.sha256(f"{model}:{system_prompt}:{prompt}".encode()).hexdigest()
    cached = semantic_cache.get(cache_key)
    if cached:
        print(f"  [Cache Hit] Serving response from {CACHE_FILE}")
        return cached['content'], cached['usage'], 0.0 
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    start_time = time.time()
    try:
        response = client.chat.completions.create(
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
        semantic_cache.set(cache_key, {'content': content, 'usage': usage})
        return content, usage, duration
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {str(e)}", {"total_tokens": 0}, 0

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

# --- 5. Node Functions ---

def analyst_wave_planner(state: GraphState) -> dict:
    """Plans the execution waves based on dependencies."""
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
    
    content, usage, duration = call_llm(
        HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME, 
        prompt=prompt,
        system_prompt="You are a precise technical analyst."
    )
    logger.log("ANALYST_PLANNING", duration, usage, "SUCCESS", content)
    
    data = extract_json(content)
    waves = data.get("waves", [[raw_task]]) # Fallback to single task
    complexity = data.get("complexity", "MEDIUM")
    
    print(f"Analyst Plan: {len(waves)} Waves. Complexity: {complexity}")
    
    # Sizing Logic
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

def dispatch_wave(state: GraphState) -> dict:
    """Router: Checks if there are waves left to run."""
    if state['current_wave_index'] >= len(state['task_waves']):
        return {"worker_status": "COMPLETE"} # All waves done, go to Final Review
    
    current_wave = state['task_waves'][state['current_wave_index']]
    print(f"--- [Orchestrator]: Dispatching Wave {state['current_wave_index'] + 1}/{len(state['task_waves'])}: {len(current_wave)} parallel tasks ---")
    return {"worker_status": "EXECUTING_WAVE"}

def execute_wave_tasks(state: GraphState) -> dict:
    """Executes all tasks in the current wave (Simulated Parallelism)."""
    current_wave = state['task_waves'][state['current_wave_index']]
    outputs = {}
    total_cost = 0
    
    # In a real async runtime, these would run in parallel threads.
    for task in current_wave:
        print(f"  -> Starting Sub-Task: {task[:40]}...")
        
        # Hybrid Routing: Inject CoT if High Complexity
        use_cot = state['complexity_score'] == "HIGH"
        instruction = "Think step-by-step." if use_cot else "Be concise."
        
        # Context Injection: Include previous wave outputs for context
        context_str = "\n".join([f"- {k}: {v[:100]}..." for k, v in state.get('worker_outputs', {}).items()])
        
        prompt = f"""
        Task: {task}
        Context from previous steps:
        {context_str}
        
        Instruction: {instruction} Provide a structured response.
        OUTPUT FORMAT: STRICT JSON ONLY.
        {{
            "content": "...",
            "status": "complete"
        }}
        """
        
        content, usage, duration = call_llm(
            VLLM_API_BASE, VLLM_API_KEY, VLLM_MODEL_NAME, 
            prompt=prompt,
            system_prompt="You are a precise Worker Agent."
        )
        
        data = extract_json(content)
        final_output = data.get("content", content)
        outputs[task] = final_output
        total_cost += usage.get('total_tokens', 0)
        
        logger.log("WORKER_EXECUTION", duration, usage, "SUCCESS", content)

    print(f"Wave {state['current_wave_index'] + 1} Complete. Cost: {total_cost} tokens.")
    
    return {
        "worker_outputs": {**state.get('worker_outputs', {}), **outputs},
        "tokens_spent": state['tokens_spent'] + total_cost,
        "current_wave_index": state['current_wave_index'] + 1,
        "worker_status": "READY_TO_DISPATCH" # Loop back to check next wave
    }

def final_review(state: GraphState) -> dict:
    """Synthesizes all outputs."""
    print("--- [HMA Controller]: Final Review & Synthesis ---")
    
    all_work = json.dumps(state['worker_outputs'], indent=2)
    prompt = f"""
    You are the Meta-Agent Manager.
    Original Goal: {state['task_description']}
    Completed Work:
    {all_work}
    
    Synthesize this into a final report.
    """
    
    content, usage, duration = call_llm(
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

# --- 6. Build the Graph ---
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
    {
        "EXECUTING_WAVE": "execute_wave_tasks",
        "COMPLETE": "final_review"
    }
)

workflow.add_edge("execute_wave_tasks", "dispatch_wave") # Loop back
workflow.add_edge("final_review", END)

app = workflow.compile()

if __name__ == "__main__":
    print(f"Starting HMA Wave Benchmark.")
    initial_state = {
        "task_description": "Compare SSM vs Transformer architectures. Cover: 1. Theory, 2. Performance, 3. Use Cases.",
        "total_budget_tokens": 5000, 
    }
    
    config = {"configurable": {"recursion_limit": 20}}
    try:
        for step in app.stream(initial_state, config=config):
            for key in step: print(f"  -> Finished Node: {key}")
    except Exception as e:
        print(f"Graph Execution Failed: {e}")
