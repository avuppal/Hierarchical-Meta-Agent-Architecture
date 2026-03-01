import re
import asyncio
from typing import Dict, Any

class SecurityManager:
    """
    Lightweight security layer for HMA.
    Focuses on prompt injection prevention and DDoS mitigation
    without adding significant overhead.
    """
    def __init__(self):
        self.rate_limits: Dict[str, list] = {}  # client_ip -> timestamps
        self.max_requests_per_minute = 60

    def sanitize_input(self, input_text: str) -> str:
        """
        Basic sanitization: Remove common injection patterns.
        Keeps it simple and fast.
        """
        # Remove attempts to override instructions
        patterns = [
            r"(?i)ignore.*previous.*instructions",
            r"(?i)override.*system.*prompt",
            r"(?i)act.*as.*different.*agent"
        ]
        for pattern in patterns:
            input_text = re.sub(pattern, "[FILTERED]", input_text)
        
        # Limit length to prevent DoS
        return input_text[:5000]  # Max 5k chars

    def check_rate_limit(self, client_ip: str) -> bool:
        """
        Simple rate limiting: Max 60 requests/min per IP.
        Uses in-memory tracking for simplicity.
        """
        now = asyncio.get_event_loop().time()
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old timestamps
        self.rate_limits[client_ip] = [t for t in self.rate_limits[client_ip] if now - t < 60]
        
        if len(self.rate_limits[client_ip]) >= self.max_requests_per_minute:
            return False  # Rate limited
        
        self.rate_limits[client_ip].append(now)
        return True

    def validate_skill(self, skill_manifest: Dict[str, Any]) -> bool:
        """
        Basic validation for imported skills: Check for required fields.
        Prevents malicious skills from being loaded.
        """
        required = ["name", "description", "tools"]
        return all(key in skill_manifest for key in required)

# Global instance
security_manager = SecurityManager()
