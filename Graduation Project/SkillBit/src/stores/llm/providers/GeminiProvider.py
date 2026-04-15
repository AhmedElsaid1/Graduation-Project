from ..LLMInterface import LLMInterface
from ..LLMEnums import LLMMessageRole
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Type, TypeVar
from pydantic import BaseModel
import logging

T = TypeVar("T", bound=BaseModel)

class GeminiProvider(LLMInterface):

    def __init__(self, api_keys: list, generation_model_id: str,
                         default_input_max_characters: int=10000,
                         default_generation_max_output_tokens: int=2048,
                         default_generation_temperature: float=0.7):

        self.api_keys = api_keys

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = generation_model_id

        self.api_dispatcher_index = 0

        self.client = ChatGoogleGenerativeAI(
            google_api_key=self.api_keys[self.api_dispatcher_index],
            model=self.generation_model_id,
            max_tokens=self.default_generation_max_output_tokens,
            temperature=self.default_generation_temperature
        )

        self.logger = logging.getLogger(__name__)


    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    

    def switch_api_key(self):
        self.api_dispatcher_index = (self.api_dispatcher_index + 1) % len(self.api_keys)
        new_api_key = self.api_keys[self.api_dispatcher_index]

        self.client.google_api_key = new_api_key

        self.logger.info(f"Switched to new Gemini API key at index {self.api_dispatcher_index}")
    

    def generate_text(self, prompt: str, chat_history: list=[], max_tokens: int=None,
                             temperature: float = None):
        
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        final_chat_history = []
        
        max_tokens = max_tokens if max_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        for message in chat_history:
            final_chat_history.append(
                self.construct_prompt(prompt=message["content"], role=message["role"])
            )
        
        final_chat_history.append(
            self.construct_prompt(prompt=prompt, role=LLMMessageRole.USER.value)
        )

        response = self.client.invoke(final_chat_history)

        self.switch_api_key()

        if not response.content:
            self.logger.error("Error while generating text with Gemini")
            return None

        return response.content


    def construct_prompt(self, prompt: str, role: str):
        return (
            str(role),
            str(self.process_text(prompt))
        )

    def generate_structured_output(self, prompt: str, output_schema: Type[T],
                                   chat_history: list = [], max_tokens: int = None,
                                   temperature: float = None) -> T:
        """
        Uses LangChain's with_structured_output to bind the Pydantic schema
        to the Gemini model, guaranteeing the response matches the schema exactly.
        A temporary client is created with the caller-supplied token budget so the
        global default (which may be too small) does not truncate the response.
        Returns a validated Pydantic model instance, or None on failure.
        """
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        try:
            effective_max_tokens = max_tokens or self.default_generation_max_output_tokens
            effective_temperature = temperature or self.default_generation_temperature

            # Build a temporary client with the exact token budget for this call.
            # with_structured_output respects the max_tokens baked into the client.
            scoped_client = ChatGoogleGenerativeAI(
                google_api_key=self.api_keys[self.api_dispatcher_index],
                model=self.generation_model_id,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature
            )

            structured_client = scoped_client.with_structured_output(output_schema)

            final_chat_history = []
            for message in chat_history:
                final_chat_history.append(
                    self.construct_prompt(prompt=message["content"], role=message["role"])
                )

            final_chat_history.append(
                self.construct_prompt(prompt=prompt, role=LLMMessageRole.USER.value)
            )

            result = structured_client.invoke(final_chat_history)

            self.switch_api_key()

            if not result:
                self.logger.error("Gemini structured output returned empty result")
                return None

            return result

        except Exception as e:
            self.logger.error(f"Error during Gemini structured output generation: {e}")
            return None

    def generate_with_tools(self, prompt: str, tools: list,
                            max_tokens: int = None,
                            temperature: float = None) -> str:
        """
        Agentic tool-calling loop using bind_tools.
        The LLM can call any tool in `tools` repeatedly until it produces
        a final text answer with no pending tool calls.
        Returns the final text content, or None on failure.
        """
        if not self.client or not self.generation_model_id:
            self.logger.error("Gemini client or model not set")
            return None

        try:
            effective_max_tokens = max_tokens or self.default_generation_max_output_tokens
            effective_temperature = temperature or self.default_generation_temperature

            scoped_client = ChatGoogleGenerativeAI(
                google_api_key=self.api_keys[self.api_dispatcher_index],
                model=self.generation_model_id,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )
            llm_with_tools = scoped_client.bind_tools(tools)

            # Build a tool name→callable map for execution
            tool_map = {t.name: t for t in tools}

            messages = [HumanMessage(content=prompt)]

            # Agentic loop — max 10 rounds to prevent infinite loops
            for _ in range(10):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                # No tool calls → final answer
                if not response.tool_calls:
                    break

                # Execute every tool call the model requested
                for tc in response.tool_calls:
                    tool_fn = tool_map.get(tc["name"])
                    if tool_fn is None:
                        self.logger.warning(f"Unknown tool requested: {tc['name']}")
                        continue
                    self.logger.info(f"Tool call: {tc['name']}({tc['args']})")
                    tool_result = tool_fn.invoke(tc["args"])
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tc["id"],
                        )
                    )

            self.switch_api_key()

            final = messages[-1]
            return final.content if hasattr(final, "content") else str(final)

        except Exception as e:
            self.logger.error(f"Error during Gemini tool-calling generation: {e}")
            return None