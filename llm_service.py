from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
from typing import List, Dict, Any
from config import Config
import json
import os

class LLMService:
    def __init__(self):
        self.model_name = Config.LLM_MODEL
        self.max_length = Config.MAX_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set cache directories for volume mounting
        os.environ["TRANSFORMERS_CACHE"] = Config.TRANSFORMERS_CACHE
        os.environ["HF_HOME"] = Config.HF_HOME
        os.environ["TORCH_HOME"] = Config.TORCH_HOME
        
        # Ensure cache directories exist
        os.makedirs(Config.TRANSFORMERS_CACHE, exist_ok=True)
        os.makedirs(Config.HF_HOME, exist_ok=True)
        os.makedirs(Config.TORCH_HOME, exist_ok=True)
        
        print(f"Loading LLM model: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Cache directories: {Config.TRANSFORMERS_CACHE}, {Config.HF_HOME}, {Config.TORCH_HOME}")
        
        # Load tokenizer and model
        try:
            # Try Llama-specific tokenizer first
            if "llama" in self.model_name.lower():
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=Config.TRANSFORMERS_CACHE
                )
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=Config.TRANSFORMERS_CACHE,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
            else:
                # Fallback to Auto classes
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=Config.TRANSFORMERS_CACHE
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=Config.TRANSFORMERS_CACHE,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to DialoGPT-small...")
            self.model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=Config.TRANSFORMERS_CACHE
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=Config.TRANSFORMERS_CACHE
            )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("LLM model loaded successfully")
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context"""
        
        # Prepare context
        context_text = self._prepare_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(question, context_text)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": Config.LLAMA_MAX_NEW_TOKENS,
                "num_return_sequences": 1,
                "temperature": Config.LLAMA_TEMPERATURE,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": torch.ones_like(inputs)
            }
            
            # Add top_p for Llama models
            if "llama" in self.model_name.lower():
                generation_kwargs["top_p"] = Config.LLAMA_TOP_P
                generation_kwargs["repetition_penalty"] = 1.1
            
            outputs = self.model.generate(inputs, **generation_kwargs)
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from response
        answer = self._extract_answer(response, question)
        
        return answer
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(context_chunks[:5]):  # Limit to top 5 chunks
            content = chunk.get('content', '')
            codex_name = chunk.get('codex_name', '')
            score = chunk.get('score', 0.0)
            
            context_parts.append(f"Source {i+1} ({codex_name}): {content}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the LLM"""
        prompt = f"""Berilgan kontekst asosida savolga javob bering. Agar kontekstda savolga javob berish uchun yetarli ma'lumot bo'lmasa, buni aytib bering.

Kontekst:
{context}

Savol: {question}

Javob:"""
        return prompt
    
    def _extract_answer(self, response: str, question: str) -> str:
        """Extract answer from LLM response"""
        # Find the answer section (try both English and Uzbek)
        answer_start = response.find("Answer:")
        if answer_start == -1:
            answer_start = response.find("Javob:")
        
        if answer_start != -1:
            if "Answer:" in response:
                answer = response[answer_start + len("Answer:"):].strip()
            else:
                answer = response[answer_start + len("Javob:"):].strip()
        else:
            # Fallback: return everything after the question
            question_start = response.find(question)
            if question_start != -1:
                answer = response[question_start + len(question):].strip()
            else:
                answer = response.strip()
        
        # Clean up the answer
        answer = answer.replace("Question:", "").replace("Savol:", "").strip()
        
        # If answer is too short or seems incomplete, provide a fallback
        if len(answer) < 10:
            answer = "Kechirasiz, mavjud kontekst asosida to'liq javob yarata olmadim. Iltimos, savolingizni qayta tuzing yoki batafsilroq ma'lumot bering."
        
        return answer

# Global LLM service instance
llm_service = LLMService()
