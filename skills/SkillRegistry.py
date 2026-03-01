import json
import os

class SkillRegistry:
    """
    The 'App Store' for skills.
    Manages discovering, indexing, and fetching skills for agents.
    """
    def __init__(self, index_path="skills/index.json"):
        self.index_path = index_path
        self.skills_cache = {}
        
        # Simulating a local index for demonstration
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
            "sales-outreach": {
                "name": "sales-outreach",
                "description": "Research leads and draft personalized emails.",
                "tools": ["web_search", "browser", "message"],
                "repo": "https://github.com/company/skills/sales"
            }
        }

    def search_skills(self, query):
        """
        Simulates a semantic search (RAG) against the skill index.
        In a real implementation, this would use embeddings (e.g., LanceDB).
        """
        print(f"[REGISTRY] Searching skills for query: '{query}'")
        
        # Simple keyword matching for demo
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
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def fetch_skill_content(self, skill_id):
        """
        Retrieves the actual prompt/skill content (SKILL.md).
        Simulates fetching from a repo.
        """
        if skill_id in self.local_index:
            repo = self.local_index[skill_id]["repo"]
            print(f"[REGISTRY] Fetching skill '{skill_id}' from {repo}")
            
            # Simulated fetching logic
            if "aws" in skill_id:
                return "You are an AWS infrastructure expert. Use Terraform to provision resources. Always validate plans before applying."
            elif "fraud" in skill_id:
                return "You are a data scientist specializing in anomaly detection. Use pandas and sklearn to identify fraudulent transactions."
            elif "sales" in skill_id:
                return "You are a sales development representative. Find leads and draft emails."
            
        return "You are a helpful assistant."
