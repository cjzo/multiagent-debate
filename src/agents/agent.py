from typing import List, Dict, Optional
from ..llm.base import LLMProvider

class DebaterAgent:
    def __init__(self, name: str, provider: LLMProvider, system_prompt: str):
        self.name = name
        self.provider = provider
        self.system_prompt = system_prompt
        self.memory: List[Dict[str, str]] = []

    def speak(self, context: str) -> str:
        """Generates a response based on the provided context."""
        # In a real debate, we might append the context to memory or just use it as the prompt
        # For now, we'll treat 'context' as the immediate prompt, but we could also build a history.
        full_prompt = f"{context}\n\n(You are {self.name}. Respond accordingly.)"
        response = self.provider.generate(full_prompt, system_prompt=self.system_prompt)
        self.memory.append({"role": "user", "content": context})
        self.memory.append({"role": "assistant", "content": response})
        return response

    def speak_stream(self, context: str):
        """Generates a streaming response."""
        full_prompt = f"{context}\n\n(You are {self.name}. Respond accordingly.)"
        # We don't update memory here immediately, or we handle it after consumption
        return self.provider.generate_stream(full_prompt, system_prompt=self.system_prompt)

    def listen(self, content: str):
        """Updates the agent's memory with what others have said."""
        self.memory.append({"role": "user", "content": content})
