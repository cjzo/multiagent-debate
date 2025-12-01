from typing import Generator, Optional
from .base import LLMProvider

class MockProvider(LLMProvider):
    def __init__(self):
        pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "This is a mock response from the MockProvider."

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        response = "This is a mock streaming response."
        for word in response.split():
            yield word + " "
