import os
from typing import Optional, Union, Dict, Any, List

import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Generation:
    def __init__(
        self,
        api_key: Optional[str] = None,
        hf_model_name: Optional[str] = None,
        device: Optional[Union[str, int]] = None,
        max_workers: int = 4,
    ):
        """
        A unified generator supporting both OpenAI API-based models and Hugging Face models,
        with optional concurrency.

        Args:
            api_key: OpenAI API key. If provided, enabling API-based generation.
            hf_model_name: Hugging Face model identifier (e.g., "meta-llama/Llama-3-7B").
            device: Device identifier for HF models (e.g., "cuda" or 0). If None, will auto-detect.
            max_workers: Number of threads for concurrent generation.
        """
        # Setup OpenAI
        self.use_api = False
        if api_key or os.getenv("OPENAI_API_KEY"):
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.use_api = True

        # Setup HF
        self.hf_model = None
        self.hf_tokenizer = None
        self.generator = None
        if hf_model_name:
            self.hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            # move model to appropriate device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            self.hf_model.to(self.device)
            self.generator = pipeline(
                "text-generation",
                model=self.hf_model,
                tokenizer=self.hf_tokenizer,
                device=0 if isinstance(self.device, int) or self.device == "cuda" else -1,
            )

        # Thread pool for concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from the given prompt using the configured model.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            **kwargs: Additional model-specific arguments.

        Returns:
            A dict containing the generated text and metadata.
        """
        if self.use_api:
            return self._generate_api(prompt, max_tokens, temperature, top_p, **kwargs)
        elif self.generator:
            return self._generate_hf(prompt, max_tokens, temperature, top_p, **kwargs)
        else:
            raise RuntimeError("No model is configured. Provide an API key or HF model name.")

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronous wrapper for generate using ThreadPoolExecutor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate(prompt, max_tokens, temperature, top_p, **kwargs)
        )

    async def generate_batch_async(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of prompts concurrently.

        Args:
            prompts: List of input prompt strings.
            max_tokens: Maximum tokens to generate for each.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            **kwargs: Additional model-specific arguments.

        Returns:
            A list of dicts containing generated text and metadata for each prompt.
        """
        tasks = [self.generate_async(p, max_tokens, temperature, top_p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to call OpenAI API completion endpoint.
        """
        response = openai.Completion.create(
            model=kwargs.get("model", "gpt-4"),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **{k: v for k, v in kwargs.items() if k not in ["model"]},
        )
        text = response.choices[0].text.strip()
        return {
            "model": response.model,
            "text": text,
            "usage": response.usage.to_dict() if hasattr(response, 'usage') else None,
        }

    def _generate_hf(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using a Hugging Face model.
        """
        outputs = self.generator(
            prompt,
            max_length=len(self.hf_tokenizer(prompt).input_ids) + max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        generated = outputs[0]["generated_text"]
        return {
            "model": self.hf_model.config._name_or_path,
            "text": generated,
        }


if __name__ == "__main__":
    # Example usage
    import asyncio

    gen = Generation(api_key="YOUR_OPENAI_KEY", hf_model_name="meta-llama/Llama-3-7B", max_workers=8)
    prompts = ["Hello world!", "What is AI?", "Science discovery?"]

    # Concurrent batch generation
    results = asyncio.run(gen.generate_batch_async(prompts, max_tokens=50))
    for i, res in enumerate(results):
        print(f"Prompt {i}: {res['text']}")
