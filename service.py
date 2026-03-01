from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
import time
import pandas as pd
from typing import Dict, Any, Optional

# Import the HMA logic
from hma_orchestrator import app as hma_app, GraphState
from core.SecurityManager import security_manager

app = FastAPI(title="Hierarchical Meta-Agent Service (HMA-AaaS)")

# In-memory store for job status (Use Redis/DB in production)
JOB_STORE: Dict[str, Dict[str, Any]] = {}

class TaskRequest(BaseModel):
    task_description: str
    total_budget_tokens: int = 5000
    callback_url: Optional[str] = None # For webhook notifications

class TaskResponse(BaseModel):
    job_id: str
    status: str
    eta_seconds: int = 30

def run_hma_background(job_id: str, task_desc: str, budget: int):
    """Executes the HMA workflow in the background and updates store."""
    print(f"--- [Service]: Starting Job {job_id} ---")
    JOB_STORE[job_id]["status"] = "RUNNING"
    JOB_STORE[job_id]["start_time"] = time.time()
    
    initial_state = {
        "task_description": task_desc,
        "total_budget_tokens": budget,
        "task_waves": [],        # Initialize empty waves for the Architect to fill
        "current_wave_index": 0, # Start at wave 0
        "worker_outputs": {},    # Store outputs
        "worker_status": "IDLE"
    }
    
    try:
        # Run the LangGraph application
        # The Architect will populate 'task_waves' in the first step
        config = {"configurable": {"recursion_limit": 20}}
        final_state = hma_app.invoke(initial_state, config=config)
        
        # Extract results
        # We now include the 'task_waves' to show WHAT the swarm did
        result_summary = {
            "final_status": final_state.get("worker_status", "UNKNOWN"),
            "swarm_plan": final_state.get("task_waves", []),  # <--- VISIBILITY
            "outputs": final_state.get("worker_outputs", {})   # <--- RESULTS
        }
        
        JOB_STORE[job_id]["status"] = "COMPLETED"
        JOB_STORE[job_id]["result"] = result_summary
        JOB_STORE[job_id]["end_time"] = time.time()
        print(f"--- [Service]: Job {job_id} Completed Successfully ---")
        
    except Exception as e:
        JOB_STORE[job_id]["status"] = "FAILED"
        JOB_STORE[job_id]["error"] = str(e)
        print(f"--- [Service]: Job {job_id} Failed: {e} ---")

@app.post("/submit", response_model=TaskResponse)
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a task to the HMA cluster."""
    # Security checks: Rate limiting and input sanitization
    client_ip = "127.0.0.1"  # In production, extract from request headers
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    sanitized_task = security_manager.sanitize_input(request.task_description)
    
    job_id = str(uuid.uuid4())
    
    # Initialize job in store
    JOB_STORE[job_id] = {
        "id": job_id,
        "status": "QUEUED",
        "task": sanitized_task,
        "budget": request.total_budget_tokens,
        "submitted_at": time.time()
    }
    
    # Trigger execution in background
    background_tasks.add_task(
        run_hma_background, 
        job_id, 
        sanitized_task, 
        request.total_budget_tokens
    )
    
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "eta_seconds": 60 # Rough estimate
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check the status of a submitted job."""
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    return JOB_STORE[job_id]

@app.get("/metrics")
async def get_metrics():
    """Return system-wide utilization metrics."""
    # Read the benchmark log file if it exists
    try:
        df = pd.read_csv("hma_benchmark_logs.csv")
        return {
            "total_jobs_processed": len(JOB_STORE),
            "total_tokens_consumed": int(df["total_tokens"].sum()) if not df.empty else 0,
            "average_duration_ms": float(df["duration_ms"].mean()) if not df.empty else 0.0,
            "recent_logs": df.tail(5).to_dict(orient="records") if not df.empty else []
        }
    except FileNotFoundError:
        return {"error": "No benchmark logs found yet."}

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for debug
    uvicorn.run(app, host="0.0.0.0", port=8080)
