from typing import Annotated, Sequence, TypedDict, Callable, Any, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
import os
import time
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

# --- 2. Define State Schema ---
class AgentBudgetState(BaseModel):
    # Input State
    task_description: str = Field(description="The overall goal for the agents.")
    total_budget_tokens: int = Field(description="The hard limit for token expenditure.")
    
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
    tokens_spent: int
    worker_status: str
    worker_output: str
    meta_review: str
    successful: bool

# --- 3. OpenAI Client Helper ---
def call_llm(api_base, api_key, model, prompt, system_prompt="You are a helpful assistant."):
    """Generic helper to call an OpenAI-compatible endpoint (vLLM/TGI/OpenAI)."""
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
        
        return content, usage, duration
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {str(e)}", {"total_tokens": 0}, 0

# --- 4. Define Node Functions ---

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
    
    prompt = f"Task: {state['task_description']}\nPrevious Output: {state.get('worker_output', 'None')}\nInstruction: Continue the work."
    
    # Call the actual vLLM endpoint
    content, usage, duration = call_llm(
        VLLM_API_BASE, VLLM_API_KEY, VLLM_MODEL_NAME, 
        prompt=prompt,
        system_prompt="You are a capable Worker Agent. Execute the task step-by-step."
    )
    
    # Log the benchmark data
    logger.log("WORKER_EXECUTION", duration, usage, "SUCCESS", content)
    
    new_spent = state['tokens_spent'] + usage.get('total_tokens', 0)
    
    print(f"Worker finished in {duration:.2f}ms. Cost: {usage.get('total_tokens', 0)} tokens.")
    
    return {
        "tokens_spent": new_spent,
        "worker_output": content,
        "worker_status": "REVIEWING"
    }

def review_and_decide_by_hma(state: GraphState) -> dict:
    """HMA Controller uses a high-reasoning model to review output and decide next step."""
    print("--- [HMA Controller]: Reviewing Output & Deciding ---")
    
    # Prompt the HMA to evaluate progress
    prompt = f"""
    You are the Meta-Agent Manager.
    Original Task: {state['task_description']}
    Current Budget Used: {state['tokens_spent']} / {state['total_budget_tokens']}
    Latest Worker Output:
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
workflow.add_node("budget_check", budget_check)
workflow.add_node("execute_worker", execute_worker_step)
workflow.add_node("review_and_decide", review_and_decide_by_hma)

# Define the edges (flow)
workflow.set_entry_point("budget_check")

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
        "task_description": "Summarize the key differences between Transformer and SSM architectures.",
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
        print(f"Final Tokens Spent: {final_state.get('review_and_decide', {}).get('tokens_spent', 'Unknown')}")
