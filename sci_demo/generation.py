import os
from typing import Optional, Union, Dict, Any, List
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# API clients
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Optional imports for additional APIs
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class ModelProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    HUGGINGFACE = "huggingface"


class Generation:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        hf_model_name: Optional[str] = None,
        device: Optional[Union[str, int]] = None,
        max_workers: int = 4,
    ):
        """
        A unified generator supporting OpenAI (GPT-4o, GPT-4, etc.), Google Gemini, 
        Anthropic Claude, and Hugging Face models with optional concurrency.

        Args:
            openai_api_key: OpenAI API key. If provided, enables OpenAI models.
            gemini_api_key: Google Gemini API key. If provided, enables Gemini models.
            claude_api_key: Anthropic Claude API key. If provided, enables Claude models.
            hf_model_name: Hugging Face model identifier (e.g., "meta-llama/Llama-3-7B").
            device: Device identifier for HF models (e.g., "cuda" or 0). If None, will auto-detect.
            max_workers: Number of threads for concurrent generation.
        """
        
        # Setup OpenAI
        self.openai_client = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )

        # Setup Gemini
        self.gemini_client = None
        if GEMINI_AVAILABLE and (gemini_api_key or os.getenv("GEMINI_API_KEY")):
            genai.configure(api_key=gemini_api_key or os.getenv("GEMINI_API_KEY"))
            self.gemini_client = genai

        # Setup Claude
        self.claude_client = None
        if CLAUDE_AVAILABLE and (claude_api_key or os.getenv("CLAUDE_API_KEY")):
            self.claude_client = anthropic.Anthropic(
                api_key=claude_api_key or os.getenv("CLAUDE_API_KEY")
            )

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

        # Model mappings
        self.openai_models = {
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
            "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-1106-preview"
        }
        
        self.gemini_models = {
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", 
            "gemini-pro", "gemini-pro-vision"
        }
        
        self.claude_models = {
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307", "claude-2.1", "claude-2.0"
        }

    def _get_provider(self, model: str) -> ModelProvider:
        """Determine the provider based on model name."""
        if model in self.openai_models:
            return ModelProvider.OPENAI
        elif model in self.gemini_models:
            return ModelProvider.GEMINI
        elif model in self.claude_models:
            return ModelProvider.CLAUDE
        else:
            return ModelProvider.HUGGINGFACE

    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from the given prompt using the specified model.

        Args:
            prompt: Input prompt string.
            model: Model identifier (e.g., "gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            **kwargs: Additional model-specific arguments.

        Returns:
            A dict containing the generated text and metadata.
        """
        provider = self._get_provider(model)
        
        if provider == ModelProvider.OPENAI:
            return self._generate_openai(prompt, model, max_tokens, temperature, top_p, **kwargs)
        elif provider == ModelProvider.GEMINI:
            return self._generate_gemini(prompt, model, max_tokens, temperature, top_p, **kwargs)
        elif provider == ModelProvider.CLAUDE:
            return self._generate_claude(prompt, model, max_tokens, temperature, top_p, **kwargs)
        elif provider == ModelProvider.HUGGINGFACE:
            return self._generate_hf(prompt, max_tokens, temperature, top_p, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def generate_async(
        self,
        prompt: str,
        model: str = "gpt-4o",
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
            lambda: self.generate(prompt, model, max_tokens, temperature, top_p, **kwargs)
        )

    async def generate_batch_async(
        self,
        prompts: List[str],
        model: str = "gpt-4o",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of prompts concurrently.

        Args:
            prompts: List of input prompt strings.
            model: Model identifier.
            max_tokens: Maximum tokens to generate for each.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            **kwargs: Additional model-specific arguments.

        Returns:
            A list of dicts containing generated text and metadata for each prompt.
        """
        tasks = [
            self.generate_async(p, model, max_tokens, temperature, top_p, **kwargs) 
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    def _generate_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to call OpenAI Chat Completions API.
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured. Provide an OpenAI API key.")
        
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **{k: v for k, v in kwargs.items() if k not in ["messages"]},
        )
        
        text = response.choices[0].message.content
        return {
            "provider": "openai",
            "model": response.model,
            "text": text,
            "usage": response.usage.model_dump() if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }

    def _generate_gemini(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using Google Gemini API.
        """
        if not self.gemini_client:
            raise RuntimeError("Gemini client not configured. Provide a Gemini API key and install google-generativeai.")
        
        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        
        # Create model instance
        gemini_model = self.gemini_client.GenerativeModel(model)
        
        # Generate content
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return {
            "provider": "gemini",
            "model": model,
            "text": response.text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else None,
                "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else None,
            },
            "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
        }

    def _generate_claude(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using Anthropic Claude API.
        """
        if not self.claude_client:
            raise RuntimeError("Claude client not configured. Provide a Claude API key and install anthropic.")
        
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        response = self.claude_client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **{k: v for k, v in kwargs.items() if k not in ["messages"]},
        )
        
        text = response.content[0].text if response.content else ""
        return {
            "provider": "claude",
            "model": response.model,
            "text": text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "finish_reason": response.stop_reason,
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
        if not self.generator:
            raise RuntimeError("Hugging Face model not configured. Provide a model name.")
        
        outputs = self.generator(
            prompt,
            max_length=len(self.hf_tokenizer(prompt).input_ids) + max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs,
        )
        generated = outputs[0]["generated_text"]
        
        # Extract only the newly generated part
        new_text = generated[len(prompt):].strip()
        
        return {
            "provider": "huggingface",
            "model": self.hf_model.config._name_or_path,
            "text": new_text,
            "usage": None,
        }

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available models by provider.
        """
        available = {}
        
        if self.openai_client:
            available["openai"] = list(self.openai_models)
        
        if self.gemini_client:
            available["gemini"] = list(self.gemini_models)
        
        if self.claude_client:
            available["claude"] = list(self.claude_models)
        
        if self.generator:
            available["huggingface"] = [self.hf_model.config._name_or_path]
        
        return available


if __name__ == "__main__":
    # Example usage
    import asyncio

    # Initialize with multiple providers
    gen = Generation(
        openai_api_key="YOUR_OPENAI_KEY",
        gemini_api_key="YOUR_GEMINI_KEY", 
        claude_api_key="YOUR_CLAUDE_KEY",
        max_workers=8
    )
    
    # List available models
    print("Available models:", gen.list_available_models())
    
    # Test different models
    prompts = ["Hello world!", "What is AI?", "Science discovery?"]
    models = ["gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet-20241022"]
    
    async def test_models():
        for model in models:
            try:
                print(f"\nTesting {model}:")
                results = await gen.generate_batch_async(prompts, model=model, max_tokens=50)
                for i, res in enumerate(results):
                    print(f"  Prompt {i}: {res['text'][:100]}...")
            except Exception as e:
                print(f"  Error with {model}: {e}")
    
    # Run async test
    asyncio.run(test_models())
