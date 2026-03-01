import asyncio
from typing import Dict, Any, Optional

class HITLManager:
    """
    Human-in-the-Loop manager for critical tasks.
    Lightweight: Pauses execution for approval without complex UI.
    """
    def __init__(self):
        self.pending_approvals: Dict[str, asyncio.Future] = {}

    async def request_approval(self, job_id: str, task_description: str, swarm_plan: Dict[str, Any]) -> bool:
        """
        Request human approval for a task.
        In production, this could send a notification/email.
        For simplicity, we simulate approval after a delay.
        """
        print(f"[HITL] Approval needed for job {job_id}: {task_description}")
        print(f"Swarm Plan: {swarm_plan}")
        
        # Simulate human review (in real use, this would be async webhook/email)
        await asyncio.sleep(2)  # Simulate delay
        
        # For demo: Auto-approve if not critical; otherwise, deny
        if "deploy" in task_description.lower() or "production" in task_description.lower():
            # Critical: Require real approval
            future = asyncio.Future()
            self.pending_approvals[job_id] = future
            print(f"[HITL] Waiting for manual approval for job {job_id}")
            try:
                approved = await asyncio.wait_for(future, timeout=300)  # 5 min timeout
                return approved
            except asyncio.TimeoutError:
                print(f"[HITL] Approval timeout for {job_id}, denying")
                return False
        else:
            print(f"[HITL] Auto-approved non-critical task for {job_id}")
            return True

    def approve_job(self, job_id: str, approved: bool):
        """Manually approve/deny a pending job."""
        if job_id in self.pending_approvals:
            self.pending_approvals[job_id].set_result(approved)
            del self.pending_approvals[job_id]

    def check_critical_task(self, task_description: str, skill: str) -> bool:
        """
        Determine if a task requires HITL based on keywords/skills.
        Keeps logic simple and configurable.
        """
        critical_keywords = ["deploy", "delete", "production", "finance", "security"]
        critical_skills = ["scalability-manager", "queue-optimization"]  # Example
        
        return any(word in task_description.lower() for word in critical_keywords) or skill in critical_skills

# Global instance
hitl_manager = HITLManager()
