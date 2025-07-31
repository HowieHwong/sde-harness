"""Backward compatibility module for LLM functions."""
from src.utils.llm_interface import complete_text, complete_text_claude

__all__ = ["complete_text", "complete_text_claude"]