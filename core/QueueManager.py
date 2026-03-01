import heapq
import asyncio
from typing import Dict, List, Tuple, Any
from enum import Enum

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class QueueManager:
    """
    Manages job queues with priorities for load balancing and flow control.
    Implements Queuing Theory principles to prevent agent overload.
    """
    def __init__(self, max_concurrent_jobs=10):
        self.queue = []  # Priority queue: (priority, timestamp, job_id, job_data)
        self.active_jobs = {}  # job_id -> job_data
        self.max_concurrent = max_concurrent_jobs
        self.lock = asyncio.Lock()

    async def enqueue(self, job_id: str, job_data: Dict[str, Any], priority: Priority = Priority.MEDIUM):
        """Add a job to the priority queue."""
        async with self.lock:
            heapq.heappush(self.queue, (-priority.value, asyncio.get_event_loop().time(), job_id, job_data))
            print(f"[QueueManager] Enqueued job {job_id} with priority {priority.name}")

    async def dequeue(self) -> Tuple[str, Dict[str, Any]]:
        """Retrieve the highest-priority job if under concurrency limit."""
        async with self.lock:
            if len(self.active_jobs) >= self.max_concurrent or not self.queue:
                return None, None
            
            _, _, job_id, job_data = heapq.heappop(self.queue)
            self.active_jobs[job_id] = job_data
            print(f"[QueueManager] Dequeued job {job_id}")
            return job_id, job_data

    async def complete_job(self, job_id: str):
        """Mark a job as complete and remove from active."""
        async with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                print(f"[QueueManager] Completed job {job_id}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Return current queue stats for monitoring."""
        return {
            "queued": len(self.queue),
            "active": len(self.active_jobs),
            "max_concurrent": self.max_concurrent
        }

# Global instance
queue_manager = QueueManager()
