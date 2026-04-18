"""Module 2: prompt-engineered intelligence extraction.

This package uses pre-trained LLMs as-is through Ollama or Groq.
It does not perform fine-tuning, LoRA, adapter training, or dataset updates.
"""

from .intelligence_extractor import IntelligenceExtractor
from .llm_client import BaseLLMClient, GroqClient, OllamaClient, create_llm_client

__all__ = [
    "BaseLLMClient",
    "GroqClient",
    "IntelligenceExtractor",
    "OllamaClient",
    "create_llm_client",
]
