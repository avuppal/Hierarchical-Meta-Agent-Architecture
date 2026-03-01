import os
import sys
from typing import Dict, Any, List

# Ensure relative imports work
try:
    from ForecastingEngine import forecasting_engine
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from orchestrator.ForecastingEngine import forecasting_engine

class ModelSelector:
    """
    Selects optimal open-source models for tasks, prioritizing privacy (local execution only).
    Uses cost-performance heuristics; no API calls or data sharing.
    """
    def __init__(self):
        # Pool of available local models (add via env or config)
        self.available_models = {
            "facebook/opt-125m": {"size": "small", "speed": "fast", "cost": "low", "accuracy": "low"},
            "meta-llama/Llama-3-8B": {"size": "medium", "speed": "medium", "cost": "medium", "accuracy": "high"},
            "meta-llama/Llama-3-70B": {"size": "large", "speed": "slow", "cost": "high", "accuracy": "very_high"}
        }
        # Load from env if specified
        env_models = os.getenv("AVAILABLE_MODELS", "").split(",")
        if env_models and env_models != [""]:
            self.available_models = {m: {"size": "unknown"} for m in env_models}

    def select_model(self, task_description: str, budget_tokens: int = 1000) -> str:
        """
        Selects model based on task complexity, budget, and performance needs.
        Prioritizes privacy: only local open-source models.
        """
        # Estimate task complexity (simple heuristic)
        complexity = self._estimate_complexity(task_description)
        
        # Forecast cost for task
        predicted_cost = forecasting_engine.forecast_cost(task_description)
        
        # Select model
        if complexity == "low" or predicted_cost < 200 or budget_tokens < 500:
            return "facebook/opt-125m"  # Fast, cheap, private
        elif complexity == "medium" or predicted_cost < 500:
            return "meta-llama/Llama-3-8B"  # Balanced
        else:
            return "meta-llama/Llama-3-70B"  # High accuracy, but costly
        
        # Fallback
        return "facebook/opt-125m"

    def _estimate_complexity(self, task: str) -> str:
        """Simple keyword-based complexity estimation."""
        low_keywords = ["add", "sum", "simple", "basic"]
        high_keywords = ["analyze", "design", "complex", "research", "optimize"]
        
        task_lower = task.lower()
        if any(k in task_lower for k in high_keywords):
            return "high"
        elif any(k in task_lower for k in low_keywords):
            return "low"
        return "medium"

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Returns vLLM config for the selected model."""
        return {
            "model": model_name,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 2048,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "enforce_eager": True
        }

# Global instance
model_selector = ModelSelector()
