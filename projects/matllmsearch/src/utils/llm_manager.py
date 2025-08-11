"""Simplified LLM Manager for SDE-harness integration"""

import os
import json
import torch
from typing import List, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMManager:
    """Simplified LLM manager for materials structure generation"""
    
    def __init__(self, base_model: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.8, temperature: float = 1.0, 
                 max_tokens: int = 4000, seed: int = 42):
        
        self.base_model = base_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        
        self.llm, self.tokenizer = self._initialize_llm(
            tensor_parallel_size, gpu_memory_utilization
        )
        
        if self.tokenizer is not None:
            self.sampling_params = SamplingParams(
                temperature=temperature, 
                top_p=0.9, 
                max_tokens=max_tokens
            )
    
    def _initialize_llm(self, tensor_parallel_size: int, gpu_memory_utilization: float):
        """Initialize LLM based on model type"""
        
        if "gpt" in self.base_model.lower():
            from openai import OpenAI
            # OpenAI GPT models
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), None
            
        elif "deepseek" in self.base_model.lower():
            from openai import OpenAI
            # DeepSeek models via OpenRouter
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            ), None
            
        else:
            # Local models via vLLM
            model = LLM(
                model=self.base_model,
                dtype=torch.float16,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                trust_remote_code=True,
                max_num_seqs=8,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            return model, tokenizer
    
    def _get_max_token_length(self) -> int:
        """Get maximum token length based on model"""
        if '70b' in self.base_model.lower():
            return 11000
        elif 'mistral' in self.base_model.lower():
            return 32000
        return 11000
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for given prompts"""
        
        # Format prompts as chat messages
        chat_prompts = [
            [{"role": "user", "content": prompt}] 
            for prompt in prompts
        ]
        
        if self.tokenizer is not None:
            # Apply chat template for local models
            formatted_prompts = [
                self.tokenizer.apply_chat_template(prompt, tokenize=False) 
                for prompt in chat_prompts
            ]
        else:
            formatted_prompts = chat_prompts
        
        if "gpt" in self.base_model.lower():
            return self._generate_openai(formatted_prompts)
        elif "deepseek" in self.base_model.lower():
            return self._generate_deepseek(formatted_prompts)
        else:
            return self._generate_vllm(formatted_prompts)
    
    def _generate_openai(self, prompts: List) -> List[str]:
        """Generate using OpenAI API"""
        results = []
        for prompt in prompts:
            try:
                response = self.llm.chat.completions.create(
                    model=self.base_model,
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                results.append(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"OpenAI generation error: {e}")
                results.append("")
        return results
    
    def _generate_deepseek(self, prompts: List) -> List[str]:
        """Generate using DeepSeek via OpenRouter"""
        results = []
        for prompt in prompts:
            try:
                response = self.llm.chat.completions.create(
                    model="deepseek/deepseek-chat",
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                results.append(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"DeepSeek generation error: {e}")
                results.append("")
        return results
    
    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generate using vLLM"""
        try:
            results = self.llm.generate(prompts, self.sampling_params)
            return [output.text for result in results for output in result.outputs]
        except Exception as e:
            print(f"vLLM generation error: {e}")
            return [""] * len(prompts)