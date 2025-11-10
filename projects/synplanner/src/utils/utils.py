from typing import List, Dict, Any, Tuple
import os
import logging
from litellm import completion


def setup_logging(logging_path: str) -> None:
    """
    Setup file and console logging.
    Adapted from https://github.com/schwallergroup/saturn/blob/master/utils/utils.py.
    """
    logging.basicConfig(
        filename=logging_path, 
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)
    
    # Silence LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def get_full_model_name(model: str) -> str:
    """Get the full model name from the short model name."""
    if model == "gpt-4o": return "openai/gpt-4o"
    if model == "gpt-5": return "openai/gpt-5"
    if model == "gpt-5-mini": return "openai/gpt-5-mini"
    if model == "claude-sonnet-4-5": return "anthropic/claude-sonnet-4-5"
    if model == "grok-4": return "xai/grok-4"
    if model == "deepseek-reasoner": return "deepseek/deepseek-reasoner"

def query_LLM(
    query: str,
    # The arguments below are the default in the original code
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 8192
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Query LLM for retrosynthesis planning and return query cost."""
    messages = [
        {"role": "system", "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."},
        {"role": "user", "content": query},
    ]

    response = completion(
        model=get_full_model_name(model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    cost = response._hidden_params["response_cost"]
    return messages, content, cost
