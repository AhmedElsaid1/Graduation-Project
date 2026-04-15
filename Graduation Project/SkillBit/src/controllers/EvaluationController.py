from .BaseController import BaseController
from models.QuizModels import (
    EvaluationRequest, EvaluationResponse, EvaluationLLMOutput,
    QuestionResult, DifficultyLevel, StudyRecommendation, Resource
)
from utils.WebSearchTool import search_youtube_for_topic
from fastapi import Request
import logging

# Enough tokens for rich feedback + recommendations across 25 questions
EVALUATION_MAX_TOKENS = 8000


class EvaluationController(BaseController):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    #  Step 1 — Grade answers locally (no LLM needed)                    #
    # ------------------------------------------------------------------ #

    def _grade_answers(self, eval_request: EvaluationRequest) -> tuple[list[QuestionResult], dict]:
        """
        Compare each student answer against the correct answer from the
        original question list. Returns:
        - A list of QuestionResult (one per question)
        - A stats dict with per-difficulty correct/total counts
        """
        answer_map = {
            ans.question_number: ans.selected_answer.strip().upper()
            for ans in eval_request.student_answers
        }

        results: list[QuestionResult] = []
        stats = {
            "easy":   {"correct": 0, "total": 0},
            "medium": {"correct": 0, "total": 0},
            "hard":   {"correct": 0, "total": 0},
        }

        for q in eval_request.questions:
            student_ans = answer_map.get(q.question_number, "—")
            is_correct = student_ans == q.correct_answer.strip().upper()

            difficulty_key = q.difficulty.value
            stats[difficulty_key]["total"] += 1
            if is_correct:
                stats[difficulty_key]["correct"] += 1

            results.append(QuestionResult(
                question_number=q.question_number,
                difficulty=q.difficulty,
                topic=q.topic,
                question=q.question,
                student_answer=student_ans,
                correct_answer=q.correct_answer,
                is_correct=is_correct,
                explanation=q.explanation,
            ))

        return results, stats

    # ------------------------------------------------------------------ #
    #  Step 2 — Build a rich prompt for the LLM                           #
    # ------------------------------------------------------------------ #

    def _build_evaluation_prompt(
        self,
        eval_request: EvaluationRequest,
        question_results: list[QuestionResult],
        stats: dict,
    ) -> str:
        total = len(question_results)
        total_correct = sum(1 for r in question_results if r.is_correct)
        score_pct = round((total_correct / total) * 100, 1) if total else 0

        # Build a compact per-question summary for the LLM
        q_lines = []
        for r in question_results:
            status = "✓ Correct" if r.is_correct else f"✗ Wrong (chose {r.student_answer}, correct: {r.correct_answer})"
            q_lines.append(
                f"  Q{r.question_number} [{r.difficulty.value.upper()}] [{r.topic}]: {status}"
            )
        questions_summary = "\n".join(q_lines)

        # Identify weak topics (answered wrong)
        wrong_topics = list({r.topic for r in question_results if not r.is_correct})
        correct_topics = list({r.topic for r in question_results if r.is_correct})

        prompt = f"""You are an expert Computer Science mentor evaluating a CS student's quiz performance.

== QUIZ RESULTS ==
Total questions : {total}
Score           : {total_correct}/{total} ({score_pct}%)
Easy  score     : {stats['easy']['correct']}/{stats['easy']['total']}
Medium score    : {stats['medium']['correct']}/{stats['medium']['total']}
Hard score      : {stats['hard']['correct']}/{stats['hard']['total']}

Per-question breakdown:
{questions_summary}

Topics the student got WRONG: {', '.join(wrong_topics) if wrong_topics else 'None'}
Topics the student got RIGHT : {', '.join(correct_topics) if correct_topics else 'None'}

== YOUR TASK ==
Based on this performance data, provide a thorough personalised evaluation including:

1. overall_feedback      — An encouraging, honest paragraph summarising their performance and mindset advice.
2. level_assessment      — One of: Beginner | Elementary | Intermediate | Advanced | Expert
3. strengths_and_weaknesses — Topics and difficulty levels they excelled at vs struggled with.
4. study_recommendations — A prioritised list (most urgent first) of topics to study. For each topic:
   - topic name
   - reason why it's recommended
   - priority: "high", "medium", or "low"
5. next_topic            — The single most impactful topic they should start studying immediately.
6. next_topic_reason     — A clear, specific explanation of why that topic is their best next step.

Be specific, actionable, and encouraging. Base everything strictly on the quiz data above."""

        return prompt

    # ------------------------------------------------------------------ #
    #  Step 3 — Call the LLM with structured output                       #
    # ------------------------------------------------------------------ #

    def _get_llm_evaluation(
        self,
        generation_client,
        prompt: str,
    ) -> EvaluationLLMOutput | None:
        """Use with_structured_output for a schema-guaranteed LLM response."""
        result = generation_client.generate_structured_output(
            prompt=prompt,
            output_schema=EvaluationLLMOutput,
            chat_history=[],
            max_tokens=EVALUATION_MAX_TOKENS,
            temperature=0.4,
        )

        if not result:
            self.logger.error("LLM returned empty structured evaluation")
        return result

    # ------------------------------------------------------------------ #
    #  Step 4 — Search YouTube directly (no LLM — prevents hallucination) #
    # ------------------------------------------------------------------ #

    def _enrich_with_resources(
        self,
        generation_client,
        llm_output: EvaluationLLMOutput,
    ) -> list[StudyRecommendation]:
        """
        For each study recommendation, call DuckDuckGo directly in Python
        and use the returned URLs verbatim — the LLM never touches them,
        so it cannot hallucinate video IDs.
        """
        enriched: list[StudyRecommendation] = []

        for rec in llm_output.study_recommendations:
            self.logger.info(f"Searching YouTube for: '{rec.topic}'")
            raw_results = search_youtube_for_topic(rec.topic)

            resources = [
                Resource(title=r["title"], url=r["url"], source="YouTube")
                for r in raw_results
                if r.get("url") and r.get("title")
            ]

            enriched.append(StudyRecommendation(
                topic=rec.topic,
                reason=rec.reason,
                priority=rec.priority,
                suggested_resources=resources,
            ))

        return enriched

    # ------------------------------------------------------------------ #
    #  Public entry point                                                #
    # ------------------------------------------------------------------ #

    async def evaluate(
        self,
        request: Request,
        eval_request: EvaluationRequest,
    ) -> EvaluationResponse:
        generation_client = request.app.generation_client

        if not generation_client:
            self.logger.error("LLM generation client is not available")
            return None

        # 1. Grade locally
        question_results, stats = self._grade_answers(eval_request)

        total = len(question_results)
        total_correct = sum(1 for r in question_results if r.is_correct)
        score_pct = round((total_correct / total) * 100, 1) if total else 0.0

        self.logger.info(
            f"Grading complete: {total_correct}/{total} ({score_pct}%) — "
            f"easy {stats['easy']['correct']}/{stats['easy']['total']}, "
            f"medium {stats['medium']['correct']}/{stats['medium']['total']}, "
            f"hard {stats['hard']['correct']}/{stats['hard']['total']}"
        )

        # 2. Build LLM prompt
        prompt = self._build_evaluation_prompt(eval_request, question_results, stats)

        # 3. Get structured LLM evaluation
        llm_output: EvaluationLLMOutput = self._get_llm_evaluation(generation_client, prompt)

        if not llm_output:
            return None

        # 4. Let the LLM search YouTube for each recommendation
        self.logger.info("Fetching YouTube resources via tool-calling loop…")
        enriched_recommendations = self._enrich_with_resources(generation_client, llm_output)

        # 5. Assemble the final response
        return EvaluationResponse(
            total_questions=total,
            total_correct=total_correct,
            score_percentage=score_pct,

            easy_correct=stats["easy"]["correct"],
            easy_total=stats["easy"]["total"],
            medium_correct=stats["medium"]["correct"],
            medium_total=stats["medium"]["total"],
            hard_correct=stats["hard"]["correct"],
            hard_total=stats["hard"]["total"],

            question_results=question_results,

            level_assessment=llm_output.level_assessment,
            overall_feedback=llm_output.overall_feedback,
            strengths_and_weaknesses=llm_output.strengths_and_weaknesses,
            study_recommendations=enriched_recommendations,
            next_topic=llm_output.next_topic,
            next_topic_reason=llm_output.next_topic_reason,
        )
