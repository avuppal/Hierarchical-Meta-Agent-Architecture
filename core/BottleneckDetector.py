from collections import defaultdict
import time
from typing import Dict, List, Any

class BottleneckDetector:
    """
    Detects bottlenecks in agent execution and suggests reallocations.
    Inspired by Theory of Constraints (TOC) for optimizing throughput.
    """
    def __init__(self):
        self.agent_performance = defaultdict(list)  # agent_role -> [durations]
        self.bottlenecks = []

    def log_execution(self, agent_role: str, duration: float):
        """Log execution time for an agent role."""
        self.agent_performance[agent_role].append(duration)
        # Keep only last 100 executions
        if len(self.agent_performance[agent_role]) > 100:
            self.agent_performance[agent_role].pop(0)

    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify slow agents as bottlenecks."""
        self.bottlenecks = []
        for role, durations in self.agent_performance.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                if avg_duration > 60:  # Threshold: >60 seconds
                    self.bottlenecks.append({
                        "role": role,
                        "avg_duration": avg_duration,
                        "recommendation": f"Increase resources for {role} or parallelize tasks."
                    })
        return self.bottlenecks

    def suggest_reallocation(self) -> Dict[str, Any]:
        """Suggest resource reallocations based on bottlenecks."""
        if not self.bottlenecks:
            return {"status": "No bottlenecks detected."}

        # Simple suggestion: allocate more to bottlenecks
        suggestions = {}
        for bottleneck in self.bottlenecks:
            suggestions[bottleneck["role"]] = "Increase concurrency limit or assign dedicated GPU."

        return {
            "bottlenecks": self.bottlenecks,
            "suggestions": suggestions
        }

# Global instance
bottleneck_detector = BottleneckDetector()
