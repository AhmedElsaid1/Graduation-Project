from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMInterface(ABC):

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_tokens: int = 2048,
                          temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str) -> str:
        pass

    @abstractmethod
    def generate_structured_output(self, prompt: str, output_schema: Type[T],
                                   chat_history: list = [], max_tokens: int = None,
                                   temperature: float = None) -> T:
        """
        Generate a response that is guaranteed to conform to the given Pydantic schema.
        Returns a validated Pydantic model instance, or None on failure.
        """
        pass

    @abstractmethod
    def generate_with_tools(self, prompt: str, tools: list,
                            max_tokens: int = None,
                            temperature: float = None) -> str:
        """
        Run an agentic tool-calling loop: the LLM may invoke any tool in
        `tools` zero or more times before returning its final text answer.
        Returns the final text response, or None on failure.
        """
        pass