import json
import subprocess
import time
import os

class SessionManager:
    """
    The 'Kernel' of HMA. Responsible for spawning sub-agents,
    sending messages, and managing their lifecycle.
    """
    def __init__(self):
        self.active_sessions = {}

    def spawn(self, role, task, model=None, tools=None):
        """
        Spawns a sub-agent session with a specific role and task.
        In a real HMA, this would call `sessions_spawn`.
        Since we are simulating the architecture, we represent the call.
        """
        session_key = f"agent:{role}:{int(time.time())}"
        
        # In a real implementation, we would call the OpenClaw API or CLI here.
        # Example: openclaw sessions spawn --task "{task}" --model "{model}"
        
        print(f"[CORE] Spawning session '{session_key}' for role '{role}'")
        print(f"       Task: {task}")
        print(f"       Tools Allowed: {tools if tools else 'ALL'}")
        
        self.active_sessions[session_key] = {
            "role": role,
            "status": "running",
            "task": task,
            "tools": tools
        }
        return session_key

    def kill(self, session_key):
        """Terminates a sub-agent session."""
        if session_key in self.active_sessions:
            print(f"[CORE] Killing session '{session_key}'")
            self.active_sessions[session_key]["status"] = "killed"
            # Actual implementation: openclaw subagents kill --target {session_key}
            return True
        return False

    def list_sessions(self):
        return self.active_sessions
