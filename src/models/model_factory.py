"""
🌙 Moon Dev's Model Factory
Built with love by Moon Dev 🚀

This module manages all available AI models and provides a unified interface.
"""

# Standard library imports
import os
import random
import sys

# Fix Windows console encoding for emojis (must be done early)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Python < 3.7 or already configured
        pass

# Standard library from imports
from pathlib import Path
from typing import Dict, Optional, Type

# Third-party from imports
from dotenv import load_dotenv
from termcolor import cprint

# Local from imports
try:
    from .base_model import BaseModel
    from .claude_model import ClaudeModel
    from .deepseek_model import DeepSeekModel
    from .gemini_model import GeminiModel
    from .groq_model import GroqModel
    from .ollama_model import OllamaModel
    from .openai_model import OpenAIModel
    from .xai_model import XAIModel
except ImportError:
    from base_model import BaseModel
    from claude_model import ClaudeModel
    from deepseek_model import DeepSeekModel
    from gemini_model import GeminiModel
    from groq_model import GroqModel
    from ollama_model import OllamaModel
    from openai_model import OpenAIModel
    from xai_model import XAIModel

class ModelFactory:
    """Factory for creating and managing AI models"""
    
    # Map model types to their implementations
    MODEL_IMPLEMENTATIONS = {
        "claude": ClaudeModel,
        "groq": GroqModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,  # Re-enabled with Gemini 2.5 models
        "deepseek": DeepSeekModel,
        "ollama": OllamaModel,  # Add Ollama implementation
        "xai": XAIModel  # xAI Grok models
    }
    
    # Default models for each type (Updated October 2025)
    DEFAULT_MODELS = {
        "claude": "claude-haiku-4-5-20251001",  # Claude Haiku 4.5 (Oct 2025)
        "groq": "mixtral-8x7b-32768",           # Fast Mixtral model
        "openai": "gpt-5-2025-08-07",           # GPT-5 (Aug 2025)
        "gemini": "gemini-2.5-flash",           # Gemini 2.5 Flash (Oct 2025)
        "deepseek": "deepseek-reasoner",        # Enhanced reasoning model
        "ollama": "llama3.2",                   # Meta's Llama 3.2
        "xai": "grok-2-latest"                  # xAI's latest Grok
    }
    
    def __init__(self):
        cprint("\n[*] Creating new ModelFactory instance...", "cyan")

        # Load environment variables first
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        cprint(f"\n[*] Loading environment from: {env_path}", "cyan")
        load_dotenv(dotenv_path=env_path)
        cprint("[+] Environment loaded", "green")
        
        self._models: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        initialized = False

        cprint("\n[*] Moon Dev's Model Factory Initialization", "cyan")
        cprint("=" * 50, "cyan")

        # Debug current environment without exposing values
        cprint("\n[*] Environment Check:", "cyan")
        for key in ["GROQ_API_KEY", "OPENAI_KEY", "ANTHROPIC_KEY", "DEEPSEEK_KEY", "GROK_API_KEY", "GEMINI_KEY"]:
            value = os.getenv(key)
            if value and len(value.strip()) > 0:
                cprint(f"  |-- {key}: Found ({len(value)} chars)", "green")
            else:
                cprint(f"  |-- {key}: Not found or empty", "red")
        
        # Try to initialize each model type
        for model_type, key_name in self._get_api_key_mapping().items():
            cprint(f"\n[*] Initializing {model_type} model...", "cyan")
            cprint(f"  |-- Looking for {key_name}...", "cyan")
            
            if api_key := os.getenv(key_name):
                try:
                    cprint(f"  |-- Found {key_name} ({len(api_key)} chars)", "green")
                    cprint(f"  |-- Getting model class for {model_type}...", "cyan")

                    if model_type not in self.MODEL_IMPLEMENTATIONS:
                        cprint(f"  |-- [!] Model type not found in implementations!", "red")
                        cprint(f"  `-- Available implementations: {list(self.MODEL_IMPLEMENTATIONS.keys())}", "yellow")
                        continue

                    model_class = self.MODEL_IMPLEMENTATIONS[model_type]
                    cprint(f"  |-- Using model class: {model_class.__name__}", "cyan")

                    # Create instance with more detailed error handling
                    try:
                        cprint(f"  |-- Creating model instance...", "cyan")
                        cprint(f"  |-- Default model name: {self.DEFAULT_MODELS[model_type]}", "cyan")
                        model_instance = model_class(api_key)
                        cprint(f"  |-- Model instance created", "green")

                        # Test if instance is properly initialized
                        cprint(f"  |-- Testing model availability...", "cyan")
                        if model_instance.is_available():
                            self._models[model_type] = model_instance
                            initialized = True
                            cprint(f"  `-- [+] Successfully initialized {model_type}", "green")
                        else:
                            cprint(f"  `-- [!] Model instance created but not available", "yellow")
                    except Exception as instance_error:
                        cprint(f"  |-- [!] Error creating model instance", "yellow")
                        cprint(f"  |-- Error type: {type(instance_error).__name__}", "yellow")
                        cprint(f"  |-- Error message: {str(instance_error)}", "yellow")
                        if hasattr(instance_error, '__traceback__'):
                            import traceback
                            cprint(f"  `-- Traceback:\n{traceback.format_exc()}", "yellow")

                except Exception as e:
                    cprint(f"  |-- [!] Failed to initialize {model_type} model", "yellow")
                    cprint(f"  |-- Error type: {type(e).__name__}", "yellow")
                    cprint(f"  |-- Error message: {str(e)}", "yellow")
                    if hasattr(e, '__traceback__'):
                        import traceback
                        cprint(f"  `-- Traceback:\n{traceback.format_exc()}", "yellow")
            else:
                cprint(f"  `-- [i] {key_name} not found", "blue")
        
        # Initialize Ollama separately since it doesn't need an API key
        try:
            cprint("\n[*] Initializing Ollama model...", "cyan")
            model_class = self.MODEL_IMPLEMENTATIONS["ollama"]
            model_instance = model_class(model_name=self.DEFAULT_MODELS["ollama"])

            if model_instance.is_available():
                self._models["ollama"] = model_instance
                initialized = True
                cprint("[+] Successfully initialized Ollama", "green")
            else:
                cprint("[!] Ollama server not available - make sure 'ollama serve' is running", "yellow")
        except Exception as e:
            cprint(f"[!] Failed to initialize Ollama: {str(e)}", "red")
        
        cprint("\n" + "=" * 50, "cyan")
        cprint(f"[*] Initialization Summary:", "cyan")
        cprint(f"  |-- Models attempted: {len(self._get_api_key_mapping()) + 1}", "cyan")  # +1 for Ollama
        cprint(f"  |-- Models initialized: {len(self._models)}", "cyan")
        cprint(f"  `-- Available models: {list(self._models.keys())}", "cyan")

        if not initialized:
            cprint("\n[!] No AI models available - check API keys and Ollama server", "yellow")
            cprint("Required environment variables:", "yellow")
            for model_type, key_name in self._get_api_key_mapping().items():
                cprint(f"  |-- {key_name} (for {model_type})", "yellow")
            cprint("  `-- Add these to your .env file", "yellow")
            cprint("\nFor Ollama:", "yellow")
            cprint("  `-- Make sure 'ollama serve' is running", "yellow")
        else:
            # Print available models
            cprint("\n[*] Available AI Models:", "cyan")
            for model_type, model in self._models.items():
                cprint(f"  |-- {model_type}: {model.model_name}", "green")
            cprint("  `-- Moon Dev's Model Factory Ready!", "green")
    
    def get_model(self, model_type: str, model_name: Optional[str] = None) -> Optional[BaseModel]:
        """Get a specific model instance"""
        cprint(f"\n[*] Requesting model: {model_type} ({model_name or 'default'})", "cyan")

        if model_type not in self.MODEL_IMPLEMENTATIONS:
            cprint(f"[!] Invalid model type: '{model_type}'", "red")
            cprint("Available types:", "yellow")
            for available_type in self.MODEL_IMPLEMENTATIONS.keys():
                cprint(f"  |-- {available_type}", "yellow")
            return None

        if model_type not in self._models:
            key_name = self._get_api_key_mapping().get(model_type)
            if key_name:
                cprint(f"[!] Model type '{model_type}' not available - check {key_name} in .env", "red")
            else:
                cprint(f"[!] Model type '{model_type}' not available", "red")
            return None

        model = self._models[model_type]
        if model_name and model.model_name != model_name:
            cprint(f"[*] Reinitializing {model_type} with model {model_name}...", "cyan")
            try:
                # Special handling for Ollama models
                if model_type == "ollama":
                    model = self.MODEL_IMPLEMENTATIONS[model_type](model_name=model_name)
                else:
                    # For API-based models that need a key
                    if api_key := os.getenv(self._get_api_key_mapping()[model_type]):
                        model = self.MODEL_IMPLEMENTATIONS[model_type](api_key, model_name=model_name)
                    else:
                        cprint(f"[!] API key not found for {model_type}", "red")
                        return None

                self._models[model_type] = model
                cprint(f"[+] Successfully reinitialized with new model", "green")
            except Exception as e:
                cprint(f"[!] Failed to initialize {model_type} with model {model_name}", "red")
                cprint(f"[!] Error type: {type(e).__name__}", "red")
                cprint(f"[!] Error: {str(e)}", "red")
                return None

        return model
    
    def _get_api_key_mapping(self) -> Dict[str, str]:
        """Get mapping of model types to their API key environment variable names"""
        return {
            "claude": "ANTHROPIC_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_KEY",
            "gemini": "GEMINI_KEY",  # Re-enabled with Gemini 2.5 models
            "deepseek": "DEEPSEEK_KEY",
            "xai": "GROK_API_KEY",  # Grok/xAI uses GROK_API_KEY
            # Ollama doesn't need an API key as it runs locally
        }
    
    @property
    def available_models(self) -> Dict[str, list]:
        """Get all available models and their configurations"""
        return {
            model_type: model.AVAILABLE_MODELS
            for model_type, model in self._models.items()
        }
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a specific model type is available"""
        return model_type in self._models and self._models[model_type].is_available()

    def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None):
        """Generate a response from the model with no caching"""
        try:
            # Add random nonce to prevent caching
            nonce = f"_{random.randint(1, 1000000)}"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_content}{nonce}"}  # Add nonce to force new response
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else self.max_tokens
            )
            
            return response.choices[0].message
            
        except Exception as e:
            if "503" in str(e):
                raise e  # Let the retry logic handle 503s
            cprint(f"[!] Model error: {str(e)}", "red")
            return None

# Create a singleton instance
model_factory = ModelFactory() 