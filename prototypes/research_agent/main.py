"""
Gemini LLM Wrapper for Research Assistant
"""

import google.generativeai as genai
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Wrapper for Google's Gemini API"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 2048
    ):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google AI API key
            model_name: Model to use (gemini-pro)
            temperature: Generation temperature (0.0-1.0)
            max_output_tokens: Max tokens to generate
        """
        genai.configure(api_key=api_key)
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Generation config
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        
        # Stats
        self.call_count = 0
        self.total_tokens = 0
        
        logger.info(f"Gemini LLM initialized: {model_name}")
    
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: The prompt to generate from
            system_instruction: Optional system instruction
        
        Returns:
            Generated text
        """
        self.call_count += 1
        
        try:
            # Create full prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            logger.debug(f"Generating response (call #{self.call_count})")
            
            # Generate
            response = self.model.generate_content(full_prompt)
            
            # Extract text
            result = response.text
            
            logger.debug(f"Generated {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def chat(self, messages: list) -> str:
        """
        Chat-style generation
        
        Args:
            messages: List of {"role": "user"/"model", "content": str}
        
        Returns:
            Generated response
        """
        self.call_count += 1
        
        try:
            # Start chat
            chat = self.model.start_chat(history=[])
            
            # Add history
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
            
            # Send final message
            response = chat.send_message(messages[-1]["content"])
            
            return response.text
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "model": self.model_name,
            "total_calls": self.call_count,
            "temperature": self.temperature
        }


def create_llm(api_key: str, **kwargs) -> GeminiLLM:
    """Factory function to create LLM instance"""
    return GeminiLLM(api_key=api_key, **kwargs)