import json
import time
from HMA.core.SessionManager import SessionManager
from HMA.skills.SkillRegistry import SkillRegistry

class AgentArchitect:
    """
    The 'Cognitive Cortex' of HMA.
    Uses RAG to find skills and designs the optimal agent swarm.
    """
    def __init__(self):
        self.registry = SkillRegistry()
        self.session_manager = SessionManager()

    def design_swarm(self, user_prompt):
        """
        Takes a high-level user request, reasons about required skills,
        and generates a deployment plan.
        """
        print(f"[ARCHITECT] Reasoning on prompt: '{user_prompt}'")
        
        # 1. Skill Discovery (RAG)
        # Search for potential skills based on keywords in the prompt
        skills_found = self.registry.search_skills(user_prompt)
        
        # In a real system, an LLM would select the best skills.
        # Here, we simulate simple logic: top 2 matches if score > 3.
        active_skills = [s for s in skills_found if s["score"] > 3][:2]
        
        if not active_skills:
            print("[ARCHITECT] No specialized skills found. Falling back to general agent.")
            swarm_plan = {
                "strategy": "sequential",
                "agents": [
                    {
                        "role": "general-assistant",
                        "skill": "general-reasoning",
                        "task": f"Assist with: {user_prompt}"
                    }
                ]
            }
        else:
            print(f"[ARCHITECT] Found relevant skills: {[s['skill'] for s in active_skills]}")
            
            # 2. Swarm Planning (LLM Simulation)
            # The Architect decides roles based on the skills found.
            agents = []
            for item in active_skills:
                skill_id = item["skill"]
                meta = item["meta"]
                
                # Fetch the detailed prompt/content
                skill_content = self.registry.fetch_skill_content(skill_id)
                
                # Create an agent definition
                agent_def = {
                    "role": f"agent-{skill_id}",
                    "skill": skill_id,
                    "system_prompt": skill_content,
                    "task": f"Execute using skill: {skill_id}",
                    "tools": meta["tools"]
                }
                agents.append(agent_def)
            
            swarm_plan = {
                "strategy": "parallel",
                "agents": agents
            }
        
        return swarm_plan

    def deploy_swarm(self, swarm_plan):
        """
        Executes the plan by spawning sessions via the SessionManager.
        """
        print(f"[ARCHITECT] Deploying swarm with strategy: {swarm_plan['strategy']}")
        
        session_keys = []
        for agent in swarm_plan["agents"]:
            key = self.session_manager.spawn(
                role=agent["role"],
                task=agent["task"],
                tools=agent.get("tools")
            )
            session_keys.append(key)
            
            # In a real implementation, we would inject the specific system prompt here
            # e.g., openclaw sessions update --session {key} --system "{agent['system_prompt']}"
            
        print(f"[ARCHITECT] Swarm active: {session_keys}")
        return session_keys

# Example usage (for testing)
if __name__ == "__main__":
    architect = AgentArchitect()
    
    # Simulate a request
    prompt = "I need to deploy a fraud detection system on AWS."
    
    plan = architect.design_swarm(prompt)
    architect.deploy_swarm(plan)
