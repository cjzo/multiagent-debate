import os
from typing import Generator, Optional
from .base import LLMProvider

#
# SAFE, STABLE, TOKEN-TRACKED PROVIDERS
#

# -------------- OPENAI PROVIDER ---------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None):
        super().__init__()
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # --- Build messages ---
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # --- Call non-streaming completion ---
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # --- SAFE TOKEN COUNTING ---
        if hasattr(response, "usage") and response.usage:
            self.add_usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

        return response.choices[0].message.content

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """
        SAFE STREAMING IMPLEMENTATION:
        - Never modifies token counters during streaming
        - After stream ends, makes one followup non-stream call to get usage
        """

        # --- Prepare messages ---
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ---- First pass: streaming text to user ---
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        full_text = ""  # reconstruct for completeness

        for event in stream:
            # SAFETY: skip malformed chunks
            if not hasattr(event, "choices") or not event.choices:
                continue
            delta = event.choices[0].delta
            if delta is None:
                continue
            if not hasattr(delta, "content"):
                continue
            if delta.content is None:
                continue
            if not isinstance(delta.content, str):
                continue

            token = delta.content
            full_text += token
            yield token

        # ---- Second pass: non-streaming call to retrieve usage safely ----
        followup = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        if hasattr(followup, "usage") and followup.usage:
            self.add_usage(
                prompt_tokens=followup.usage.prompt_tokens or 0,
                completion_tokens=followup.usage.completion_tokens or 0,
            )


# -------------- ANTHROPIC PROVIDER ---------------
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        super().__init__()
        if Anthropic is None:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")

        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        # --- SAFE TOKEN COUNT ---
        if hasattr(response, "usage") and response.usage:
            self.add_usage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
            )

        return response.content[0].text

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:

        stream = self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        full_text = ""

        for text in stream.text_stream:
            if not isinstance(text, str):
                continue
            if text.strip() == "":
                continue

            full_text += text
            yield text

        # Get usage from final info
        usage = stream.get_final_response().usage
        if usage:
            self.add_usage(
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0
            )


# -------------- GEMINI PROVIDER ---------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__()
        if genai is None:
            raise ImportError("google-generativeai not installed")

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found")
        genai.configure(api_key=key)

        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt

        response = self.model.generate_content(full_prompt)

        # SAFE TOKEN COUNT (Gemini API supports usage tokens)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.add_usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        return response.text

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:

        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt

        stream = self.model.generate_content(full_prompt, stream=True)

        full_text = ""

        for chunk in stream:
            if not hasattr(chunk, "text"):
                continue
            if chunk.text is None:
                continue
            if not isinstance(chunk.text, str):
                continue
            if chunk.text.strip() == "":
                continue

            full_text += chunk.text
            yield chunk.text

        # After stream ends, Gemini also provides usage metadata:
        if hasattr(stream, "usage_metadata") and stream.usage_metadata:
            self.add_usage(
                prompt_tokens=stream.usage_metadata.prompt_token_count or 0,
                completion_tokens=stream.usage_metadata.candidates_token_count or 0,
            )
