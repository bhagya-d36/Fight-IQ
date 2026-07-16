"""llm.py — chat-provider abstraction. Pick a provider with LLM_PROVIDER in
.env; each SDK is imported lazily so you only need the package for the
provider you actually use.
"""

import os
from collections.abc import Iterator

import config


class LLMError(Exception):
    """Provider-agnostic error raised when a chat-completion call fails."""


class ChatProvider:
    def chat(self, messages: list[dict], system: str | None = None, temperature: float = 0.2) -> str:
        raise NotImplementedError

    def stream_chat(
        self, messages: list[dict], system: str | None = None, temperature: float = 0.2
    ) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=None, temperature=temperature)


class GeminiChat(ChatProvider):
    def __init__(self, api_key: str, model: str, timeout_ms: int, retry_attempts: int) -> None:
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,  # milliseconds; per-read for streams
                retry_options=types.HttpRetryOptions(attempts=retry_attempts),
            ),
        )

    def _contents(self, messages: list[dict]) -> list[dict]:
        return [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]

    def chat(self, messages, system=None, temperature=0.2):
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._contents(messages),
                config=self._types.GenerateContentConfig(system_instruction=system, temperature=temperature),
            )
            return response.text or ""
        except self._errors.APIError as err:
            raise LLMError(str(err)) from err

    def stream_chat(self, messages, system=None, temperature=0.2):
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model,
                contents=self._contents(messages),
                config=self._types.GenerateContentConfig(system_instruction=system, temperature=temperature),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except self._errors.APIError as err:
            raise LLMError(str(err)) from err


class OpenAIChat(ChatProvider):
    """Covers OpenAI and any OpenAI-compatible endpoint (DeepSeek, Kimi/Moonshot,
    Groq, OpenRouter, Ollama, ...) via base_url.
    """

    def __init__(
        self, api_key: str, model: str, base_url: str | None, timeout_ms: int, retry_attempts: int
    ) -> None:
        from openai import APIError, OpenAI

        self._errors = APIError
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_ms / 1000,
            max_retries=retry_attempts,
        )

    def _messages(self, messages: list[dict], system: str | None) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        out.extend({"role": m["role"], "content": m["content"]} for m in messages)
        return out

    def chat(self, messages, system=None, temperature=0.2):
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(messages, system),
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except self._errors as err:
            raise LLMError(str(err)) from err

    def stream_chat(self, messages, system=None, temperature=0.2):
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(messages, system),
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except self._errors as err:
            raise LLMError(str(err)) from err


class AnthropicChat(ChatProvider):
    def __init__(
        self, api_key: str, model: str, timeout_ms: int, retry_attempts: int, max_tokens: int
    ) -> None:
        from anthropic import Anthropic, APIError

        self._errors = APIError
        self._model = model
        self._max_tokens = max_tokens
        self._client = Anthropic(
            api_key=api_key,
            timeout=timeout_ms / 1000,
            max_retries=retry_attempts,
        )

    def chat(self, messages, system=None, temperature=0.2):
        try:
            response = self._client.messages.create(
                model=self._model,
                system=system or "",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except self._errors as err:
            raise LLMError(str(err)) from err

    def stream_chat(self, messages, system=None, temperature=0.2):
        try:
            with self._client.messages.stream(
                model=self._model,
                system=system or "",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=self._max_tokens,
                temperature=temperature,
            ) as stream:
                yield from stream.text_stream
        except self._errors as err:
            raise LLMError(str(err)) from err


# provider name -> (API key env var, default model, optional base_url)
_PROVIDERS = {
    "gemini": {"env_key": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash"},
    "openai": {"env_key": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "kimi": {
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.ai/v1",
    },
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-5"},
}


def make_chat_provider() -> ChatProvider:
    provider = config.LLM_PROVIDER
    info = _PROVIDERS.get(provider)
    if info is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={provider!r}. Choose one of: {', '.join(_PROVIDERS)}."
        )
    api_key = os.environ.get(info["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing {info['env_key']}. Create a .env file (see .env.example).")
    model = config.CHAT_MODEL or info["default_model"]

    if provider == "gemini":
        return GeminiChat(api_key, model, config.LLM_TIMEOUT_MS, config.LLM_RETRY_ATTEMPTS)
    if provider == "anthropic":
        return AnthropicChat(
            api_key, model, config.LLM_TIMEOUT_MS, config.LLM_RETRY_ATTEMPTS, config.MAX_OUTPUT_TOKENS
        )
    return OpenAIChat(api_key, model, info.get("base_url"), config.LLM_TIMEOUT_MS, config.LLM_RETRY_ATTEMPTS)


