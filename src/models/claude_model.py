"""
🌙 Moon Dev's Claude Model Implementation
Built with love by Moon Dev 🚀
"""

# Third-party from imports
from anthropic import Anthropic
from termcolor import cprint

# Local from imports
try:
    from .base_model import BaseModel, ModelResponse
except ImportError:
    from base_model import BaseModel, ModelResponse

class ClaudeModel(BaseModel):
    """Implementation for Anthropic's Claude models"""
    
    AVAILABLE_MODELS = {
        # Claude 4.5 Series (Latest - 2025)
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 - Smartest model for complex agents and coding ($3/$15 per MTok) - 200K/1M context",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5 - Fastest with near-frontier intelligence ($1/$5 per MTok) - 200K context",

        # Claude 4.1 Series (Latest - 2025)
        "claude-opus-4-1-20250805": "Claude Opus 4.1 - Exceptional for specialized reasoning ($15/$75 per MTok) - 200K context",

        # Claude 3.5 Series (Legacy - Still Available)
        "claude-3-5-sonnet-20250122": "Claude 3.5 Sonnet (Jan 2025)",
        "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet - Auto-updates",
        "claude-3-5-haiku-20250107": "Claude 3.5 Haiku (Jan 2025)",
        "claude-3-5-haiku-latest": "Claude 3.5 Haiku - Auto-updates",

        # Claude 3 Series (Legacy)
        "claude-3-opus-20240229": "Claude 3 Opus (Feb 2024)",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet (Feb 2024)",
        "claude-3-haiku-20240307": "Claude 3 Haiku (Mar 2024)"
    }

    def __init__(self, api_key: str, model_name: str = "claude-haiku-4-5-20251001", **kwargs):
        self.model_name = model_name
        super().__init__(api_key, **kwargs)
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            self.client = Anthropic(api_key=self.api_key)
            cprint(f"[+] Initialized Claude model: {self.model_name}", "green")
        except Exception as e:
            cprint(f"[!] Failed to initialize Claude model: {str(e)}", "red")
            self.client = None
    
    def generate_response(self, 
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using Claude"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )
            
            return ModelResponse(
                content=response.content[0].text.strip(),
                raw_response=response,
                model_name=self.model_name,
                usage={"completion_tokens": response.usage.output_tokens}
            )
            
        except Exception as e:
            cprint(f"[!] Claude generation error: {str(e)}", "red")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude is available"""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "claude" 