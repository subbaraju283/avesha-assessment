"""
LLM Manager for handling different language model providers.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def resolve_env_vars(value: str) -> str:
    """Resolve environment variables in string values like ${VAR_NAME}."""
    if isinstance(value, str) and "${" in value:
        # Replace ${VAR_NAME} with environment variable value
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
    return value


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Resolve environment variables after initialization."""
        if self.api_key:
            self.api_key = resolve_env_vars(self.api_key)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    async def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using the LLM."""
        pass
    
    def get_langchain_llm(self):
        """Get LangChain-compatible LLM instance."""
        raise NotImplementedError("LangChain integration not implemented for this provider")


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
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
    
    def get_langchain_llm(self):
        """Get LangChain-compatible OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.api_key
            )
        except ImportError:
            logger.warning("langchain-openai not available")
            return None


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
    
    def get_langchain_llm(self):
        """Get LangChain-compatible Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                anthropic_api_key=self.api_key
            )
        except ImportError:
            logger.warning("langchain-anthropic not available")
            return None


class LLMManager:
    """Manager for handling different LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        llm_config = self.config.get("llm", {})

        # Initialize OpenAI provider
        if "openai" in llm_config:
            openai_config = LLMConfig(
                provider="openai",
                model=llm_config["openai"].get("model", "gpt-4"),
                temperature=llm_config["openai"].get("temperature", 0.1),
                max_tokens=llm_config["openai"].get("max_tokens", 2000),
                api_key=llm_config["openai"].get("api_key")
            )
            try:
                self.providers["openai"] = OpenAIProvider(openai_config)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        # Initialize Anthropic provider
        elif "anthropic" in llm_config:
            anthropic_config = LLMConfig(
                provider="anthropic",
                model=llm_config["anthropic"].get("model", "claude-3-sonnet-20240229"),
                temperature=llm_config["anthropic"].get("temperature", 0.1),
                max_tokens=llm_config["anthropic"].get("max_tokens", 2000),
                api_key=llm_config["anthropic"].get("api_key")
            )
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_config)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")
        else:
            raise ValueError("No LLM providers could be initialized")
    
    async def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """Generate text using specified or default provider."""
        provider_name = provider or self.config.get("default_provider", list(self.providers.keys())[0])
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].generate(prompt, **kwargs)
    
    async def classify(self, text: str, categories: List[str], provider: Optional[str] = None) -> Dict[str, float]:
        """Classify text using specified or default provider."""
        provider_name = provider or self.config.get("default_provider", list(self.providers.keys())[0])
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].classify(text, categories)
    
    async def embed(self, text: str, provider: Optional[str] = None) -> List[float]:
        """Generate embeddings using specified or default provider."""
        provider_name = provider or self.config.get("default_provider", list(self.providers.keys())[0])
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return await self.providers[provider_name].embed(text)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_langchain_llm(self, provider: Optional[str] = None):
        """Get LangChain-compatible LLM instance."""
        provider_name = provider or self.config.get("default_provider", list(self.providers.keys())[0])
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        return self.providers[provider_name].get_langchain_llm() 