from ..LLMInterface import LLMInterface
from ..LLMEnums import LLMMessageRole
from openai import OpenAI
from typing import Type, TypeVar
from pydantic import BaseModel
import logging

T = TypeVar("T", bound=BaseModel)

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, generation_model_id: str,
                       base_url: str=None,
                       default_input_max_characters: int=10000,
                       default_generation_max_output_tokens: int=2048,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.base_url = base_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = generation_model_id

        self.client = OpenAI(
            api_key = self.api_key,
            base_url = self.base_url
        )

        self.logger = logging.getLogger(__name__)
        

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=LLMMessageRole.USER.value)
        )

        response = self.client.chat.completions.create(
            model = self.generation_model_id,
            messages = chat_history,
            max_tokens = max_output_tokens,
            temperature = temperature
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message["content"]


    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

    def generate_structured_output(self, prompt: str, output_schema: Type[T],
                                   chat_history: list = [], max_tokens: int = None,
                                   temperature: float = None) -> T:
        """Not yet implemented for OpenAI provider."""
        self.logger.warning("generate_structured_output is not implemented for OpenAIProvider. Returning None.")
        return None

    def generate_with_tools(self, prompt: str, tools: list,
                            max_tokens: int = None,
                            temperature: float = None) -> str:
        """Not yet implemented for OpenAI provider."""
        self.logger.warning("generate_with_tools is not implemented for OpenAIProvider. Returning None.")
        return None
        return None