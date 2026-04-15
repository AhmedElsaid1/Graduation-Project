from .BaseController import BaseController
from models.QuizModels import (
    QuizQuestion, QuizResponse, QuizRequest,
    MCQOption, DifficultyLevel, QuizTopic, QuizQuestionsOutput
)
from fastapi import Request
import json
import logging
import re

# Token budget reserved specifically for quiz generation.
# 25 questions × ~350 tokens each (question + options + explanation) ≈ 8 750 tokens.
# Keep well above that ceiling so no response is truncated.
QUIZ_MAX_TOKENS = 16000


class QuizController(BaseController):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _build_quiz_prompt(self, request: QuizRequest) -> str:
        topic_label = request.topic.value.replace("_", " ").title()
        total = request.easy_count + request.medium_count + request.hard_count

        prompt = f"""You are an expert Computer Science professor. Generate a quiz for CS students to assess their knowledge level.

Generate EXACTLY {total} multiple-choice questions (MCQ) about {topic_label} Computer Science concepts and code.

You MUST follow this EXACT distribution — no more, no less:
- EXACTLY {request.easy_count} questions with difficulty = "easy"
- EXACTLY {request.medium_count} questions with difficulty = "medium"
- EXACTLY {request.hard_count} questions with difficulty = "hard"

Difficulty definitions:
- easy: basic concepts, definitions, simple logic (e.g. what is a variable, what does a loop do)
- medium: intermediate concepts, algorithms, problem solving (e.g. Big-O, recursion, SQL joins)
- hard: advanced topics, complex algorithms, system design (e.g. concurrency, NP-completeness, distributed systems)

Requirements:
- Each question must have exactly 4 options labeled A, B, C, D
- Questions should cover a variety of CS sub-topics unless a specific topic is requested
- Include code snippets in questions where appropriate
- The explanation must be thorough and educational, explaining WHY the correct answer is right and why the others are wrong
- Order: first all easy questions, then all medium questions, then all hard questions
- Make sure questions are diverse, well-written, and truly test CS knowledge"""

        return prompt

    # ------------------------------------------------------------------ #
    #  Fallback helpers (used only when structured output is unavailable) #
    # ------------------------------------------------------------------ #

    def _build_json_prompt(self, request: QuizRequest) -> str:
        """Extends the base prompt with explicit JSON formatting instructions for the text fallback."""
        base = self._build_quiz_prompt(request)
        json_instruction = """

Return ONLY a valid JSON array with this exact structure (no markdown, no extra text):
[
  {
    "question_number": 1,
    "difficulty": "easy",
    "topic": "topic name",
    "question": "question text here",
    "options": [
      {"label": "A", "text": "option A text"},
      {"label": "B", "text": "option B text"},
      {"label": "C", "text": "option C text"},
      {"label": "D", "text": "option D text"}
    ],
    "correct_answer": "A",
    "explanation": "detailed explanation of why A is correct and others are wrong"
  }
]"""
        return base + json_instruction

    # ------------------------------------------------------------------ #
    #  Distribution enforcement                                           #
    # ------------------------------------------------------------------ #

    def _enforce_distribution(
        self,
        questions: list[QuizQuestion],
        easy_count: int,
        medium_count: int,
        hard_count: int,
    ) -> list[QuizQuestion] | None:
        """
        Slice the question list to match the requested distribution exactly,
        regardless of what difficulty labels the LLM assigned.

        Strategy:
        - Bucket all questions by the difficulty the LLM assigned.
        - Trim each bucket to the requested count.
        - Re-number questions sequentially (easy → medium → hard).
        - Warn if a bucket is short; return what we have.
        """
        easy   = [q for q in questions if q.difficulty == DifficultyLevel.EASY][:easy_count]
        medium = [q for q in questions if q.difficulty == DifficultyLevel.MEDIUM][:medium_count]
        hard   = [q for q in questions if q.difficulty == DifficultyLevel.HARD][:hard_count]

        shortfalls = []
        if len(easy)   < easy_count:   shortfalls.append(f"easy: got {len(easy)}, wanted {easy_count}")
        if len(medium) < medium_count: shortfalls.append(f"medium: got {len(medium)}, wanted {medium_count}")
        if len(hard)   < hard_count:   shortfalls.append(f"hard: got {len(hard)}, wanted {hard_count}")

        if shortfalls:
            self.logger.warning(
                f"LLM did not return enough questions for some buckets — {', '.join(shortfalls)}"
            )

        final = easy + medium + hard
        if not final:
            return None

        # Re-number sequentially
        for idx, q in enumerate(final):
            q.question_number = idx + 1

        return final

    def _parse_llm_response(self, raw_response: str) -> list:
        """
        Parse plain-text LLM JSON response, stripping markdown fences if present.
        If the response is truncated mid-string (token limit hit), attempts to
        recover all complete question objects that were already serialised.
        """
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # ── Happy path ───────────────────────────────────────────────────
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"Full JSON parse failed (likely truncated): {e}. "
                "Attempting partial recovery…"
            )

        # ── Partial recovery ─────────────────────────────────────────────
        # Find the last complete question object by locating the last "},"
        # or "}" that closes a top-level array element, then close the array.
        last_brace = cleaned.rfind("},")
        if last_brace == -1:
            last_brace = cleaned.rfind("}")

        if last_brace != -1:
            recovered = cleaned[: last_brace + 1].strip()
            # Make sure it starts with "["
            if not recovered.startswith("["):
                recovered = "[" + recovered
            recovered += "]"

            try:
                questions = json.loads(recovered)
                self.logger.info(
                    f"Partial recovery succeeded: recovered {len(questions)} question(s)."
                )
                return questions
            except json.JSONDecodeError as e2:
                self.logger.error(f"Partial recovery also failed: {e2}")

        self.logger.error("Could not recover any questions from the truncated response.")
        return None

    def _validate_and_build_questions(self, raw_questions: list) -> list[QuizQuestion]:
        """Validate each raw dict from the text fallback path and map to QuizQuestion."""
        questions = []
        valid_labels = {"A", "B", "C", "D"}

        for idx, q in enumerate(raw_questions):
            try:
                options = [
                    MCQOption(label=opt["label"], text=opt["text"])
                    for opt in q.get("options", [])
                ]

                difficulty_raw = q.get("difficulty", "easy").lower()
                try:
                    difficulty = DifficultyLevel(difficulty_raw)
                except ValueError:
                    difficulty = DifficultyLevel.EASY

                correct = q.get("correct_answer", "A").strip().upper()
                if correct not in valid_labels:
                    correct = "A"

                questions.append(QuizQuestion(
                    question_number=q.get("question_number", idx + 1),
                    difficulty=difficulty,
                    topic=q.get("topic", "General CS"),
                    question=q.get("question", ""),
                    options=options,
                    correct_answer=correct,
                    explanation=q.get("explanation", "")
                ))
            except Exception as e:
                self.logger.warning(f"Skipping malformed question at index {idx}: {e}")

        return questions

    # ------------------------------------------------------------------ #
    #  Primary: structured output                                        #
    # ------------------------------------------------------------------ #

    def _generate_via_structured_output(
        self, generation_client, quiz_request: QuizRequest
    ) -> list[QuizQuestion] | None:
        """
        Ask the LLM to return output that is guaranteed to conform to
        QuizQuestionsOutput (a Pydantic model wrapping List[QuizQuestion]).
        Passes QUIZ_MAX_TOKENS so the provider creates a scoped client with
        enough budget to fit all 25 questions + explanations.
        Returns the validated list of questions, or None on failure.
        """
        prompt = self._build_quiz_prompt(quiz_request)

        result: QuizQuestionsOutput = generation_client.generate_structured_output(
            prompt=prompt,
            output_schema=QuizQuestionsOutput,
            chat_history=[],
            max_tokens=QUIZ_MAX_TOKENS,
            temperature=0.7
        )

        if not result or not result.questions:
            self.logger.error("Structured output returned no questions")
            return None

        self.logger.info(f"Structured output succeeded: {len(result.questions)} question(s) returned")
        return result.questions

    # ------------------------------------------------------------------ #
    #  Fallback: plain text → JSON parse                                   #
    # ------------------------------------------------------------------ #

    def _generate_via_text(
        self, generation_client, quiz_request: QuizRequest
    ) -> list[QuizQuestion] | None:
        """
        Ask the LLM to return a raw JSON string and parse it manually.
        Used when the provider does not support structured output.
        Passes QUIZ_MAX_TOKENS to avoid mid-response truncation.
        """
        prompt = self._build_json_prompt(quiz_request)

        raw_response = generation_client.generate_text(
            prompt=prompt,
            chat_history=[],
            max_tokens=QUIZ_MAX_TOKENS,
            temperature=0.7
        )

        if not raw_response:
            self.logger.error("LLM returned empty text response")
            return None

        raw_questions = self._parse_llm_response(raw_response)
        if not raw_questions:
            return None

        return self._validate_and_build_questions(raw_questions)

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    async def generate_quiz(self, request: Request, quiz_request: QuizRequest) -> QuizResponse:
        generation_client = request.app.generation_client

        if not generation_client:
            self.logger.error("LLM generation client is not available")
            return None

        self.logger.info(
            f"Generating quiz: topic={quiz_request.topic.value}, "
            f"easy={quiz_request.easy_count}, medium={quiz_request.medium_count}, "
            f"hard={quiz_request.hard_count}"
        )

        # Try structured output first (stable, schema-validated)
        questions = self._generate_via_structured_output(generation_client, quiz_request)

        if not questions:
            self.logger.warning(
                "Structured output failed or not supported — falling back to text + JSON parse"
            )
            questions = self._generate_via_text(generation_client, quiz_request)

        if not questions:
            self.logger.error("Both generation paths failed")
            return None

        # ── 3. Enforce exact distribution regardless of what the LLM returned ──
        questions = self._enforce_distribution(
            questions=questions,
            easy_count=quiz_request.easy_count,
            medium_count=quiz_request.medium_count,
            hard_count=quiz_request.hard_count,
        )

        if not questions:
            self.logger.error("No questions remain after distribution enforcement")
            return None

        topic_label = quiz_request.topic.value.replace("_", " ").title()

        return QuizResponse(
            total_questions=len(questions),
            easy_count=sum(1 for q in questions if q.difficulty == DifficultyLevel.EASY),
            medium_count=sum(1 for q in questions if q.difficulty == DifficultyLevel.MEDIUM),
            hard_count=sum(1 for q in questions if q.difficulty == DifficultyLevel.HARD),
            topic=topic_label,
            questions=questions
        )
