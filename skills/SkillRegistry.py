import json
import os
import sys

# Try importing LangChain components if available
try:
    from langchain.tools import BaseTool
    from langchain_community.tools import GoogleSerperRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainAdapter:
    """
    Adapter to convert LangChain tools into HMA Skill format.
    Allows importing capabilities from the massive LangChain ecosystem.
    """
    @staticmethod
    def convert_tool_to_skill(tool_instance):
        """
        Converts a LangChain tool instance into an HMA skill dictionary.
        Extracts name, description, and argument schema.
        """
        if not LANGCHAIN_AVAILABLE:
            return None
            
        skill_id = tool_instance.name.replace(" ", "-").lower()
        description = tool_instance.description
        
        # Extract argument schema if available
        args_schema = {}
        if hasattr(tool_instance, "args"):
            args_schema = tool_instance.args
            
        # Generate a prompt-friendly representation
        system_prompt = f"""
You are an expert agent with the capability: {tool_instance.name}.
Description: {description}

To use this skill, you must output a JSON function call matching this schema:
{json.dumps(args_schema, indent=2)}

Your task is to execute this tool effectively to solve user problems.
"""
        
        return {
            "id": skill_id,
            "name": tool_instance.name,
            "description": description,
            "system_prompt": system_prompt,
            "tools": ["python", "exec"], # LangChain tools often run via python/exec
            "origin": "langchain"
        }

class SkillRegistry:
    """
    The 'App Store' for skills.
    Manages discovering, indexing, and fetching skills for agents.
    Now supports LangChain integration as a Universal Adapter.
    """
    def __init__(self, index_path="skills/index.json"):
        self.index_path = index_path
        self.skills_cache = {}
        
        # Base Local Index
        self.local_index = {
            "fraud-detection": {
                "name": "fraud-detection",
                "description": "Analyze transaction logs for anomalies using Isolation Forest.",
                "tools": ["python", "read", "write"],
                "repo": "local/skills/fraud"
            },
            "aws-deploy": {
                "name": "aws-deploy",
                "description": "Deploy infrastructure using Terraform or AWS CLI.",
                "tools": ["exec", "read", "write"],
                "repo": "https://github.com/company/skills/aws-deploy"
            },
            "queue-optimization": {
                "name": "queue-optimization",
                "description": "Manage job queues with priorities and load balancing.",
                "tools": ["python", "exec"],
                "repo": "local/skills/queue"
            },
            "forecasting": {
                "name": "forecasting",
                "description": "Predict token costs and execution times using historical data.",
                "tools": ["python", "read"],
                "repo": "local/skills/forecast"
            },
            "bottleneck-analysis": {
                "name": "bottleneck-analysis",
                "description": "Detect execution bottlenecks and suggest reallocations.",
                "tools": ["python", "read"],
                "repo": "local/skills/bottleneck"
            },
            "scalability-manager": {
                "name": "scalability-manager",
                "description": "Handle dynamic resource scaling and failure resilience.",
                "tools": ["exec", "read"],
                "repo": "local/skills/scalability"
            }
        }

    def import_langchain_tool(self, tool_name):
        """
        Dynamically imports a LangChain tool and registers it as an HMA skill.
        Example: import_langchain_tool("GoogleSerperRun")
        """
        if not LANGCHAIN_AVAILABLE:
            print("[REGISTRY] LangChain not available. Please install langchain-community.")
            return None

        print(f"[REGISTRY] Attempting to import LangChain tool: {tool_name}")
        
        try:
            # Dynamic loading simulation (mapping names to classes)
            # In a full implementation, we'd use importlib to load any class
            tool_instance = None
            if tool_name == "GoogleSerperRun":
                # Mock wrapper for serper if API key missing, or try real init
                from langchain_community.utilities import GoogleSerperAPIWrapper
                wrapper = GoogleSerperAPIWrapper(serper_api_key="mock-key") 
                tool_instance = GoogleSerperRun(api_wrapper=wrapper)
            
            if tool_instance:
                skill_data = LangChainAdapter.convert_tool_to_skill(tool_instance)
                self.local_index[skill_data["id"]] = {
                    "name": skill_data["name"],
                    "description": skill_data["description"],
                    "tools": skill_data["tools"],
                    "repo": "langchain-adapter",
                    "system_prompt": skill_data["system_prompt"]
                }
                print(f"[REGISTRY] Successfully imported '{skill_data['id']}' from LangChain.")
                return skill_data["id"]
                
        except Exception as e:
            print(f"[REGISTRY] Failed to import tool '{tool_name}': {e}")
            return None

    def search_skills(self, query):
        """
        Simulates a semantic search (RAG) against the skill index.
        """
        print(f"[REGISTRY] Searching skills for query: '{query}'")
        
        results = []
        query_words = query.lower().split()
        
        for skill_id, meta in self.local_index.items():
            score = 0
            desc = meta["description"].lower()
            name = meta["name"].lower()
            
            for word in query_words:
                if word in name: score += 5
                if word in desc: score += 2
            
            if score > 0:
                results.append({"skill": skill_id, "score": score, "meta": meta})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def fetch_skill_content(self, skill_id):
        """
        Retrieves the prompt content. Supports both local repo simulation and 
        dynamic adapter prompts.
        """
        if skill_id in self.local_index:
            meta = self.local_index[skill_id]
            
            # If it's an imported skill, return the generated prompt
            if "system_prompt" in meta:
                return meta["system_prompt"]
                
            repo = meta["repo"]
            print(f"[REGISTRY] Fetching skill '{skill_id}' from {repo}")
            
            # Simulated fetching logic for local skills
            if "aws" in skill_id:
                return "You are an AWS infrastructure expert. Use Terraform to provision resources."
            elif "fraud" in skill_id:
                return "You are a data scientist specializing in anomaly detection."
            elif "queue" in skill_id:
                return "You are a queue optimization expert. Manage priorities and load balance tasks."
            elif "forecast" in skill_id:
                return "You are a forecasting analyst. Predict costs and times based on historical data."
            elif "bottleneck" in skill_id:
                return "You are a bottleneck analyst. Identify constraints and suggest improvements."
            elif "scalability" in skill_id:
                return "You are a scalability manager. Scale resources and handle failures."
            
        return "You are a helpful assistant."
