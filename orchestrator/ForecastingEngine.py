import pandas as pd
import os
from typing import Dict, Any, Optional

class ForecastingEngine:
    """
    Predicts token costs and execution times using historical data.
    Implements forecasting for better resource budgeting.
    """
    def __init__(self, log_file="hma_benchmark_logs.csv"):
        self.log_file = log_file
        self._load_history()

    def _load_history(self):
        """Load and preprocess historical logs."""
        if os.path.exists(self.log_file):
            self.history = pd.read_csv(self.log_file)
        else:
            self.history = pd.DataFrame(columns=["step_type", "duration_ms", "status"])

    def forecast_cost(self, task_description: str, task_type: str = "general") -> float:
        """Predict token cost based on similar tasks."""
        if self.history.empty:
            return 100.0  # Default estimate

        # Simple heuristic: average cost for similar step types
        similar_tasks = self.history[self.history["step_type"].str.contains(task_type, case=False, na=False)]
        if not similar_tasks.empty:
            avg_cost = similar_tasks["duration_ms"].mean() * 0.01  # Rough token proxy
            return max(avg_cost, 50.0)  # Minimum estimate
        return 100.0

    def forecast_duration(self, task_description: str) -> float:
        """Predict execution time in seconds."""
        if self.history.empty:
            return 30.0

        # Average duration for recent tasks
        recent = self.history.tail(10)
        if not recent.empty:
            avg_duration = recent["duration_ms"].mean() / 1000
            return max(avg_duration, 5.0)
        return 30.0

    def update_history(self, step_type: str, duration_ms: float, status: str):
        """Log new data for future forecasting."""
        new_row = {"step_type": step_type, "duration_ms": duration_ms, "status": status}
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)
        self.history.to_csv(self.log_file, index=False)

# Global instance
forecasting_engine = ForecastingEngine()
