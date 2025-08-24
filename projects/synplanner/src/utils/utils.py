from typing import List, Dict, Any, Tuple
import os
import logging
from openai import OpenAI


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
    
    # Silence OpenAI HTTP request logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def query_LLM(
    query: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 8192
) -> Tuple[List[Dict[str, Any]], str]:
    """Query OpenAI API for retrosynthesis planning."""
    messages = [
        {"role": "system", "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."},
        {"role": "user", "content": query},
    ]

    resp = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    return messages, content
