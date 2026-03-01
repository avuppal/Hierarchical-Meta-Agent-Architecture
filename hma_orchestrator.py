from typing import Annotated, Sequence, TypedDict, Callable, Any, Optional
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
# vLLM Configuration (Assumes vLLM running locally or in adjacent container)
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") # vLLM often uses 'EMPTY'
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "facebook/opt-125m") # Default small model

# HMA Controller Configuration (Using OpenAI-compatible endpoint, could be vLLM or external)
HMA_API_BASE = os.getenv("HMA_API_BASE", "http://localhost:8000/v1")
HMA_API_KEY = os.getenv("HMA_API_KEY", "EMPTY")
HMA_MODEL_NAME = os.getenv("HMA_MODEL_NAME", "facebook/opt-125m")

# Benchmark Logging
BENCHMARK_FILE = "hma_benchmark_logs.csv"
CACHE_FILE = "hma_semantic_cache.json"

# --- 1. Benchmark Logger ---
class BenchmarkLogger:
    def __init__(self, filename=BENCHMARK_FILE):
        self.filename = filename
        # Create file with header if not exists
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=[
                "timestamp", "step_type", "duration_ms", "tokens_in", "tokens_out", 
                "total_tokens", "cost_estimate", "status", "output_snippet"
            ])
            df.to_csv(self.filename, index=False)

    def log(self, step_type: str, duration_ms: float, usage: dict, status: str, output: str):
        """Logs a single execution step to CSV."""
        new_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_type": step_type,
            "duration_ms": round(duration_ms, 2),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_estimate": 0.0, # Placeholder for dollar cost calculation
            "status": status,
            "output_snippet": output[:50].replace("\n", " ") + "..."
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.filename, mode='a', header=False, index=False)
        print(f"  [Log] Saved metric: {step_type} ({duration_ms}ms, {usage.get('total_tokens', 0)} tokens)")

logger = BenchmarkLogger()

# --- 2. Semantic Cache (KM Layer) ---
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
        # Write-through (simple consistency)
        with open(self.filename, 'w') as f:
            json.dump(self.cache, f, indent=2)

semantic_cache = SemanticCache()

# --- 3. Define State Schema ---
class AgentBudgetState(BaseModel):
    # Input State
    task_description: str = Field(description="The overall goal for the agents.")
    total_budget_tokens: int = Field(description="The hard limit for token expenditure.")
    
    # Enhanced State (Analyst Node)
    refined_task: str = Field(default="", description="Optimized technical brief")
    complexity_score: str = Field(default="MEDIUM", description="LOW/MEDIUM/HIGH")

    # Coordination (Blackboard Pattern)
    claimed_tasks: list[str] = Field(default_factory=list, description="List of sub-tasks currently being worked on")

    # Runtime State
    tokens_spent: int = Field(default=0, description="Accumulated tokens spent so far.")
    worker_status: str = Field(default="IDLE", description="IDLE, RUNNING, REVIEWING, RESTART, COMPLETE")
    worker_output: str = ""
    meta_review: str = ""
    successful: bool = False

# Type Hint for LangGraph State
class GraphState(TypedDict):
    task_description: str
    total_budget_tokens: int
    refined_task: str
    complexity_score: str
    claimed_tasks: list[str]
    tokens_spent: int
    worker_status: str
    worker_output: str
    meta_review: str
    successful: bool

# --- 4. Helpers ---
def call_llm(api_base, api_key, model, prompt, system_prompt="You are a helpful assistant."):
    """Generic helper with Semantic Caching."""
    
    # Generate Cache Key (Hash of inputs)
    cache_key = hashlib.sha256(f"{model}:{system_prompt}:{prompt}".encode()).hexdigest()
    
    # 1. Check Cache
    cached = semantic_cache.get(cache_key)
    if cached:
        print(f"  [Cache Hit] Serving response from {CACHE_FILE}")
        # Return cached content with 0 duration
        return cached['content'], cached['usage'], 0.0 
    
    # 2. Call API (Cache Miss)
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
        usage = response.usage.model_dump() # dict: prompt_tokens, completion_tokens, total_tokens
        
        # 3. Store in Cache
        semantic_cache.set(cache_key, {
            'content': content,
            'usage': usage
        })
        
        return content, usage, duration
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {str(e)}", {"total_tokens": 0}, 0

def extract_json(text: str) -> dict:
    """Tries to extract JSON from text (handles markdown blocks)."""
    try:
        return json.loads(text)
    except:
        # Try finding code block
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try finding raw braces
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    return {}

# --- 5. Define Node Functions ---

def analyst_sizing_heuristic(state: GraphState) -> dict:
    """Refines the task and estimates complexity (Sizing)."""
    print("--- [Analyst Node]: Refining Task & Sizing Project ---")
    
    raw_task = state['task_description']
    
    # Prompt for rewriting and sizing
    prompt = f"""
    You are an Expert AI Project Manager.
    Your goal is to prepare a task for an autonomous agent.
    
    Raw Request: "{raw_task}"
    
    1. Rewrite this task into a strict, technical, MECE-structured brief. Remove fluff.
    2. Rate the complexity as LOW, MEDIUM, or HIGH.
    
    Output Format:
    COMPLEXITY: [LOW/MEDIUM/HIGH]
    BRIEF: [The refined text]
    """
    
    content, usage, duration = call_llm(
        HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME, 
        prompt=prompt,
        system_prompt="You are a precise technical analyst."
    )
    
    logger.log("ANALYST_SIZING", duration, usage, "SUCCESS", content)
    
    # Parse the output (simple parsing)
    complexity = "MEDIUM"
    refined_brief = raw_task # Fallback
    
    if "COMPLEXITY:" in content and "BRIEF:" in content:
        try:
            parts = content.split("BRIEF:")
            complexity_part = parts[0].split("COMPLEXITY:")[1].strip().split()[0].upper() # Grab first word
            refined_brief = parts[1].strip()
            if complexity_part in ["LOW", "MEDIUM", "HIGH"]:
                complexity = complexity_part
        except:
            pass
            
    # Apply Sizing Heuristic to Budget
    # If user gave a budget, we treat it as a 'baseline' and adjust.
    base_budget = state['total_budget_tokens']
    multipliers = {"LOW": 0.8, "MEDIUM": 1.0, "HIGH": 1.5} # Low complexity saves budget!
    
    adjusted_budget = int(base_budget * multipliers.get(complexity, 1.0))
    
    print(f"Analyst Result: {complexity} Complexity. Budget adjusted: {base_budget} -> {adjusted_budget}")
    
    return {
        "refined_task": refined_brief,
        "complexity_score": complexity,
        "total_budget_tokens": adjusted_budget,
        "tokens_spent": state['tokens_spent'] + usage.get('total_tokens', 0)
    }

def budget_check(state: GraphState) -> dict:
    """Checks if the remaining budget is sufficient to proceed."""
    print("--- [HMA Controller]: Checking Budget ---")
    
    remaining = state['total_budget_tokens'] - state['tokens_spent']
    
    if remaining < 500: 
        print(f"Budget critical! Remaining: {remaining} tokens.")
        return {"worker_status": "HALT_BUDGET_EXCEEDED"}
    
    print(f"Budget OK. Remaining: {remaining} tokens.")
    return {"worker_status": "READY_TO_EXECUTE"}

def execute_worker_step(state: GraphState) -> dict:
    """Calls the local Worker Agent (vLLM)."""
    print(f"--- [Worker Agent]: Executing Task on {VLLM_MODEL_NAME} ---")
    
    task_to_use = state.get('refined_task') or state['task_description']
    
    # Blackboard Pattern Check (Future Proofing)
    claimed = state.get('claimed_tasks', [])
    # In a parallel version, we would verify task_to_use isn't in claimed here.
    
    prompt = f"""
    Task: {task_to_use}
    Previous Output: {state.get('worker_output', 'None')}
    
    Instruction: Continue the work. Provide a structured response.
    OUTPUT FORMAT: STRICT JSON ONLY. No markdown, no conversation.
    {{
        "thought_process": "Brief reasoning...",
        "content": "The actual work/code/text...",
        "status": "partial" | "complete"
    }}
    """
    
    content, usage, duration = call_llm(
        VLLM_API_BASE, VLLM_API_KEY, VLLM_MODEL_NAME, 
        prompt=prompt,
        system_prompt="You are a precise JSON-speaking Worker Agent."
    )
    
    # Validate JSON (Zero Token Cost)
    data = extract_json(content)
    valid_json = bool(data and "content" in data)
    
    status = "REVIEWING"
    final_output = content
    
    if valid_json:
        print("  -> Valid JSON output received.")
        final_output = json.dumps(data) # Normalized
        # Check self-reported status
        if data.get("status") == "complete":
            status = "REVIEWING" # Ready for manager
    else:
        print("  -> Invalid JSON. Triggering automatic retry loop (managed by graph).")
        final_output = f"ERROR: Invalid JSON received. content='{content[:50]}...'"
        # We could add a 'RETRY' status here, but for simplicity we let manager catch it or retry logic handle it.
        # Actually, let's be strict:
        # status = "RETRY_JSON" # We could add a loop back to worker immediately?
        pass

    logger.log("WORKER_EXECUTION", duration, usage, "SUCCESS" if valid_json else "INVALID_JSON", content)
    
    new_spent = state['tokens_spent'] + usage.get('total_tokens', 0)
    
    print(f"Worker finished in {duration:.2f}ms. Cost: {usage.get('total_tokens', 0)} tokens.")
    
    return {
        "tokens_spent": new_spent,
        "worker_output": final_output,
        "worker_status": status,
        "claimed_tasks": claimed + [task_to_use[:20]] # Mark as worked on
    }

def review_and_decide_by_hma(state: GraphState) -> dict:
    """HMA Controller uses a high-reasoning model to review output and decide next step."""
    print("--- [HMA Controller]: Reviewing Output & Deciding ---")
    
    task_to_use = state.get('refined_task') or state['task_description']

    # Prompt the HMA to evaluate progress
    prompt = f"""
    You are the Meta-Agent Manager.
    Original Task: {task_to_use}
    Current Budget Used: {state['tokens_spent']} / {state['total_budget_tokens']}
    Latest Worker Output (JSON):
    ---
    {state['worker_output']}
    ---
    
    Evaluate the quality and completion status.
    If satisfactory and complete, respond with specific keyword 'COMPLETE'.
    If good but needs more work, respond with 'CONTINUE'.
    If bad/off-track, respond with 'RETRY'.
    Provide a brief reasoning first.
    """
    
    content, usage, duration = call_llm(
        HMA_API_BASE, HMA_API_KEY, HMA_MODEL_NAME,
        prompt=prompt,
        system_prompt="You are a strict Project Manager managing AI agents."
    )
    
    # Log the HMA's own cost
    logger.log("HMA_REVIEW", duration, usage, "SUCCESS", content)
    
    new_spent = state['tokens_spent'] + usage.get('total_tokens', 0) # HMA costs money too!
    
    # Naive parsing of decision
    if "COMPLETE" in content:
        status = "COMPLETE"
        success = True
    elif "RETRY" in content:
        status = "READY_TO_EXECUTE" # Loop back (could add specific retry logic)
        success = False
    else:
        status = "READY_TO_EXECUTE" # Default to continue
        success = False
        
    print(f"HMA Decision: {status}. Reason: {content[:50]}...")
    
    return {
        "tokens_spent": new_spent,
        "meta_review": content,
        "worker_status": status,
        "successful": success
    }

# --- 5. Build the Graph ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("analyst_sizing", analyst_sizing_heuristic)
workflow.add_node("budget_check", budget_check)
workflow.add_node("execute_worker", execute_worker_step)
workflow.add_node("review_and_decide", review_and_decide_by_hma)

# Define the edges (flow)
# ENTRY POINT: Analyst Node First
workflow.set_entry_point("analyst_sizing")

# Analyst -> Budget Check (Start the loop)
workflow.add_edge("analyst_sizing", "budget_check")

workflow.add_conditional_edges(
    "budget_check",
    lambda state: state['worker_status'],
    {
        "READY_TO_EXECUTE": "execute_worker",
        "HALT_BUDGET_EXCEEDED": END 
    }
)

workflow.add_edge("execute_worker", "review_and_decide")

workflow.add_conditional_edges(
    "review_and_decide",
    lambda state: state['worker_status'],
    {
        "READY_TO_EXECUTE": "execute_worker", # Loop back
        "COMPLETE": END, 
        "HALT_BUDGET_EXCEEDED": END # Fallback
    }
)

# Compile the graph
app = workflow.compile()

# --- 6. Initial Run Simulation ---
if __name__ == "__main__":
    print(f"Starting HMA Benchmark. Logging to {BENCHMARK_FILE}")
    print(f"Connecting to vLLM at {VLLM_API_BASE} (Worker) and {HMA_API_BASE} (HMA)")
    
    initial_state = {
        "task_description": "I want to know the difference between ssm and transformer but make it quick and focus on long context.",
        "total_budget_tokens": 5000, 
        "tokens_spent": 0,
        "worker_status": "IDLE",
        "successful": False
    }

    # Run the loop
    config = {"configurable": {"recursion_limit": 10}}
    
    final_state = None
    try:
        for step in app.stream(initial_state, config=config):
            final_state = step
            # Optional: pretty print active node
            for key in step:
                print(f"  -> Finished Node: {key}")
    except Exception as e:
        print(f"Graph Execution Failed: {e}")
        
    print("\\n--- BENCHMARK END ---")
    if final_state:
        last_node_state = list(final_state.values())[0]
        print(f"Final Tokens Spent: {last_node_state.get('tokens_spent', 'Unknown')}")
        print(f"Refined Task: {last_node_state.get('refined_task', 'None')}")
        print(f"Complexity: {last_node_state.get('complexity_score', 'None')}")
