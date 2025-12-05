from abc import ABC, abstractmethod
from typing import Generator, Optional

class LLMProvider(ABC):
    """Abstract base class for LLM providers with token tracking."""

    def __init__(self):
        # Global cumulative usage for this provider
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a complete response for the given prompt."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Generates a streaming response for the given prompt."""
        pass

    # --- Token tracking helpers ---
    def add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def get_usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def reset_usage(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
