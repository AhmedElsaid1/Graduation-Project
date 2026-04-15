from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, GeminiProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config
        
    def create(self, provider: str):
        
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                base_url = self.config.OPENAI_BASE_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
                generation_model_id=self.config.GENERATION_MODEL_ID
            )
        
        if provider == LLMEnums.GEMINI.value:
            return GeminiProvider(
                api_keys=self.config.GOOGLE_API_KEYS,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
                generation_model_id=self.config.GENERATION_MODEL_ID
            )
        
        return None