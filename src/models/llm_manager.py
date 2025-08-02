"""
LLM Manager for abstracting different language model providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using OpenAI."""
        prompt = f"""Classify the following text into these categories: {', '.join(categories)}

Text: {text}

Respond ONLY with a JSON object where each key is a category and each value is a confidence score between 0.0 and 1.0. Do not include any other text.

Example format:
{{
  "category1": 0.8,
  "category2": 0.2,
  "category3": 0.1
}}"""
        
        try:
            response = await self.generate(prompt)
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                scores = json.loads(response)
            
            # Ensure all categories are present
            for category in categories:
                if category not in scores:
                    scores[category] = 0.0
            
            return scores
        except Exception as e:
            logger.warning(f"Failed to parse classification response: {e}")
            # Fallback to equal distribution
            return {cat: 1.0 / len(categories) for cat in categories}
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using Anthropic."""
        prompt = f"""Classify the following text into these categories: {', '.join(categories)}

Text: {text}

Respond ONLY with a JSON object where each key is a category and each value is a confidence score between 0.0 and 1.0. Do not include any other text.

Example format:
{{
  "category1": 0.8,
  "category2": 0.2,
  "category3": 0.1
}}"""
        
        try:
            response = await self.generate(prompt)
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                scores = json.loads(response)
            
            # Ensure all categories are present
            for category in categories:
                if category not in scores:
                    scores[category] = 0.0
            
            return scores
        except Exception as e:
            logger.warning(f"Failed to parse classification response: {e}")
            # Fallback to equal distribution
            return {cat: 1.0 / len(categories) for cat in categories}
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Anthropic."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Anthropic embedding error: {e}")
            raise


class LocalProvider(LLMProvider):
    """Local LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        # Initialize local model (placeholder)
        logger.warning("Local LLM provider not fully implemented")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local model."""
        # Placeholder implementation
        return f"Local model response to: {prompt[:50]}..."
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using local model."""
        return {cat: 0.5 for cat in categories}
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using local model."""
        # Placeholder - return random embeddings
        import random
        return [random.random() for _ in range(384)]


class LLMManager:
    """Manager for different LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = config.get("default_provider", "openai")
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        providers_config = self.config.get("providers", {})
        
        for provider_name, provider_config in providers_config.items():
            try:
                config = LLMConfig(
                    provider=provider_name,
                    model=provider_config.get("model", "gpt-4"),
                    temperature=provider_config.get("temperature", 0.1),
                    max_tokens=provider_config.get("max_tokens", 2000),
                    api_key=provider_config.get("api_key")
                )
                
                if provider_name == "openai":
                    self.providers[provider_name] = OpenAIProvider(config)
                elif provider_name == "anthropic":
                    self.providers[provider_name] = AnthropicProvider(config)
                elif provider_name == "local":
                    self.providers[provider_name] = LocalProvider(config)
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
    
    async def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """Generate text using specified or default provider."""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].generate(prompt, **kwargs)
    
    async def classify(self, text: str, categories: List[str], provider: Optional[str] = None) -> Dict[str, float]:
        """Classify text using specified or default provider."""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].classify(text, categories)
    
    async def embed(self, text: str, provider: Optional[str] = None) -> List[float]:
        """Generate embeddings using specified or default provider."""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].embed(text)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys()) 