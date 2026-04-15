from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    GEMINI = "GEMINI"


class LLMMessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"