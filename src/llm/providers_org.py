import os
from typing import Generator, Optional
from .base import LLMProvider

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Please run `pip install openai`.")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        if Anthropic is None:
            raise ImportError("Anthropic library not installed. Please run `pip install anthropic`.")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("Google Generative AI library not installed. Please run `pip install google-generativeai`.")
        
        # Configure the API key
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Gemini handles system prompts differently (often just prepended or via specific config if available in newer SDKs)
        # For simplicity/compatibility, we can prepend it or use the system_instruction if supported by the specific model version instantiation.
        # However, the simplest robust way across versions is often prepending.
        # Let's check if we can pass it in generation config or just prepend.
        # For now, prepending is safest.
        
        full_prompt = prompt
        if system_prompt:
            # Some models support system_instruction in constructor, but we are re-using the model instance.
            # We'll just prepend for now.
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = self.model.generate_content(full_prompt)
        return response.text

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

class LocalProvider(LLMProvider):
    def __init__(self, model_path: str):
        # Placeholder for local model implementation (e.g., using transformers or llama.cpp)
        self.model_path = model_path
        print(f"Warning: LocalProvider for {model_path} is not yet fully implemented.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError("Local generation not yet implemented.")

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        raise NotImplementedError("Local streaming not yet implemented.")
