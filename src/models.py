"""
API client and model configurations for LLM experiments.

This module provides:
- OpenRouter API client wrapper
- Model name constants
- Chat function with conversation history support
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Model Constants
# =============================================================================

# Models used in the paper (via OpenRouter)
MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "mistral-small-3": "mistralai/mistral-small-3.1-24b-instruct",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
}

# Shorter aliases for convenience
MODEL_ALIASES = {
    "gpt": "gpt-4o-mini",
    "mistral": "mistral-small-3",
    "gemma": "gemma-3-12b",
    "llama8b": "llama-3.1-8b",
    "llama3b": "llama-3.2-3b",
}


# =============================================================================
# API Client
# =============================================================================

def get_client(api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1") -> OpenAI:
    """
    Get an OpenAI-compatible client for OpenRouter.
    
    Args:
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
        base_url: API base URL. Default is OpenRouter.
    
    Returns:
        OpenAI client configured for OpenRouter
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    return OpenAI(base_url=base_url, api_key=api_key)


# Global client instance (initialized lazily)
_client: Optional[OpenAI] = None


def get_default_client() -> OpenAI:
    """Get or create the default API client."""
    global _client
    if _client is None:
        _client = get_client()
    return _client


# =============================================================================
# Chat Function
# =============================================================================

def chat(
    model: str,
    prompt: str,
    *,
    history: Optional[list] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    client: Optional[OpenAI] = None,
) -> tuple[str, list]:
    """
    Send a chat completion request and optionally maintain conversation history.
    
    Args:
        model: Model identifier (can be a key from MODELS dict or full model string)
        prompt: User message to send
        history: List of previous messages [{"role": "...", "content": "..."}]
                 If provided, will be updated in place with the new exchange.
        system: Optional system prompt
        temperature: Sampling temperature (default: 0 for reproducibility)
        max_tokens: Maximum tokens in response
        client: Optional custom client. If None, uses default client.
    
    Returns:
        Tuple of (reply_text, updated_history)
    
    Example:
        >>> history = []
        >>> reply, history = chat("gpt-4o-mini", "Hello!", history=history)
        >>> reply, history = chat("gpt-4o-mini", "Follow up question", history=history)
    """
    # Resolve model name
    if model in MODEL_ALIASES:
        model = MODEL_ALIASES[model]
    if model in MODELS:
        model = MODELS[model]
    
    # Get client
    if client is None:
        client = get_default_client()
    
    # Build messages
    msgs = []
    
    # Add history if provided
    if history is not None:
        msgs = list(history)
    
    # Add system prompt if provided and not already present
    if system and not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": system})
    
    # Add current user message
    msgs.append({"role": "user", "content": prompt})
    
    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content
    
    # Update history if provided
    if history is not None:
        if system and not any(m.get("role") == "system" for m in history):
            history.insert(0, {"role": "system", "content": system})
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
    
    return reply, history if history is not None else []


def resolve_model_name(model: str) -> str:
    """
    Resolve a model name to its full OpenRouter identifier.
    
    Args:
        model: Model key, alias, or full identifier
    
    Returns:
        Full model identifier string
    """
    if model in MODEL_ALIASES:
        model = MODEL_ALIASES[model]
    if model in MODELS:
        return MODELS[model]
    return model
