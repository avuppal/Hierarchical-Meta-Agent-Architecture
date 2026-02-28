from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json

# --- 1. Define State Schema using Pydantic ---
class AgentBudgetState(BaseModel):
    # Input State
    task_description: str = Field(description="The overall goal for the agents.")
    total_budget_tokens: int = Field(description="The hard limit for token expenditure.")
    
    # Runtime State
    tokens_spent: int = Field(default=0, description="Accumulated tokens spent so far.")
    worker_status: str = Field(default="IDLE", description="IDLE, RUNNING, REVIEWING, RESTART, COMPLETE")
    worker_output: str = ""
    meta_review: str = ""
    successful: bool = Field(default=False)

# Type Hint for LangGraph State
class GraphState(TypedDict):
    task_description: str
    total_budget_tokens: int
    tokens_spent: int
    worker_status: str
    worker_output: str
    meta_review: str
    successful: bool

# --- 2. Define Nodes (Functions) ---

def budget_check(state: GraphState) -> dict:
    """Checks if the remaining budget is sufficient to proceed."""
    print("--- [Meta-Agent]: Checking Budget ---")
    
    remaining = state['total_budget_tokens'] - state['tokens_spent']
    
    if remaining < 1000: # Arbitrary small threshold for a single step
        print(f"Budget critical! Remaining: {remaining} tokens.")
        return {"worker_status": "HALT_BUDGET_EXCEEDED"}
    
    print(f"Budget OK. Remaining: {remaining} tokens.")
    return {"worker_status": "READY_TO_EXECUTE"}

def execute_worker(state: GraphState) -> dict:
    """Simulates calling a Worker Agent (e.g., a local model running on 3090s)."""
    print("--- [Meta-Agent]: Executing Worker Task ---")
    
    # *** In a real scenario, this function would call sessions_spawn or run a local model. ***
    # For this example, we simulate work and cost.
    
    simulated_cost = 2500 # Assume the worker used 2.5k tokens for this step
    new_spent = state['tokens_spent'] + simulated_cost
    
    # Simulate worker output and status update
    worker_output = f"Worker successfully processed the chunk. Actual cost: {simulated_cost} tokens."
    
    print(f"Worker finished. Total spent: {new_spent}")
    return {
        "tokens_spent": new_spent,
        "worker_output": worker_output,
        "worker_status": "REVIEWING"
    }

def review_and_decide(state: GraphState) -> dict:
    """Meta-Agent reviews output and decides next step (Loop/Restart/Finish)."""
    print("--- [Meta-Agent]: Reviewing Output & Deciding ---")
    
    # *** In a real scenario, this calls the high-reasoning HMA model (e.g., gemini-flash-lite-latest or a better one) ***
    # The prompt would include the HMA Hypothesis and the worker_output.
    
    # For simulation: If spent is less than 75% of total, we loop.
    progress = state['tokens_spent'] / state['total_budget_tokens']
    
    if progress < 0.75:
        meta_review = "Output looks promising but incomplete. Directing worker to continue to next step."
        print("Decision: Continuing loop.")
        return {
            "meta_review": meta_review,
            "worker_status": "READY_TO_EXECUTE" # Loop back to execution
        }
    else:
        meta_review = "Budget nearly exhausted or convergence reached. Recommending completion."
        print("Decision: Completing task.")
        return {
            "meta_review": meta_review,
            "worker_status": "COMPLETE",
            "successful": True
        }

# --- 3. Build the Graph ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("budget_check", budget_check)
workflow.add_node("execute_worker", execute_worker)
workflow.add_node("review_and_decide", review_and_decide)

# Define the edges (flow)
workflow.set_entry_point("budget_check")

workflow.add_conditional_edges(
    "budget_check",
    lambda state: state['worker_status'],
    {
        "READY_TO_EXECUTE": "execute_worker",
        "HALT_BUDGET_EXCEEDED": END # Exit condition 1
    }
)

workflow.add_edge("execute_worker", "review_and_decide")

workflow.add_conditional_edges(
    "review_and_decide",
    lambda state: state['worker_status'],
    {
        "READY_TO_EXECUTE": "execute_worker", # Loop back
        "COMPLETE": END, # Exit condition 2
        "RESTART": "execute_worker" # Restart node (for a future implementation)
    }
)

# Compile the graph
app = workflow.compile()

# --- 4. Initial Run Simulation ---
initial_state = {
    "task_description": "Analyze the novel research areas in Agentic Efficiency.",
    "total_budget_tokens": 10000, # 10k token budget for simulation
    "tokens_spent": 0,
    "worker_status": "IDLE",
    "successful": False
}

print("\\n--- STARTING HMA SIMULATION (2 CYCLES) ---")
# We'll run it for two cycles manually to show the loop
config = {"configurable": {"recursion_limit": 2}}
steps = app.stream(initial_state, config=config)

final_state = None
for i, step in enumerate(steps):
    print(f"Cycle {i+1} Updates:")
    for key, value in step.items():
        print(f"  -> Node: {key}")
        final_state = value
    if i >= 1: # Stop after 2 explicit steps for demonstration
        break

print("\\n--- SIMULATION END ---")
print(f"Final State: {json.dumps(final_state, indent=2)}")