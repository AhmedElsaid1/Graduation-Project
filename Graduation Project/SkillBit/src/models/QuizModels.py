from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuizTopic(str, Enum):
    DATA_STRUCTURES = "data_structures"
    ALGORITHMS = "algorithms"
    OOP = "oop"
    DATABASES = "databases"
    NETWORKING = "networking"
    OPERATING_SYSTEMS = "operating_systems"
    PROGRAMMING_FUNDAMENTALS = "programming_fundamentals"
    SOFTWARE_ENGINEERING = "software_engineering"
    MIXED = "mixed"


class MCQOption(BaseModel):
    label: str = Field(..., description="Option label: A, B, C, or D")
    text: str = Field(..., description="The option text")


class QuizQuestion(BaseModel):
    question_number: int = Field(..., description="The sequential number of the question")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the question")
    topic: str = Field(..., description="CS topic this question belongs to")
    question: str = Field(..., description="The question text")
    options: List[MCQOption] = Field(..., description="List of 4 MCQ options")
    correct_answer: str = Field(..., description="The label of the correct option (A, B, C, or D)")
    explanation: str = Field(..., description="Detailed explanation of why the correct answer is right")


class QuizResponse(BaseModel):
    total_questions: int = Field(..., description="Total number of questions in the quiz")
    easy_count: int = Field(..., description="Number of easy questions")
    medium_count: int = Field(..., description="Number of medium questions")
    hard_count: int = Field(..., description="Number of hard questions")
    topic: str = Field(..., description="Quiz topic")
    questions: List[QuizQuestion] = Field(..., description="List of quiz questions")


class QuizQuestionsOutput(BaseModel):
    """Structured output schema passed to with_structured_output."""
    questions: List[QuizQuestion] = Field(..., description="List of generated MCQ questions")


class QuizRequest(BaseModel):
    topic: QuizTopic = Field(
        default=QuizTopic.MIXED,
        description="CS topic for the quiz questions"
    )
    easy_count: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of easy questions (default: 10)"
    )
    medium_count: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of medium questions (default: 10)"
    )
    hard_count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of hard questions (default: 5)"
    )


# ======================================================================
#  Evaluation models
# ======================================================================

class StudentAnswer(BaseModel):
    question_number: int = Field(..., description="The question number being answered")
    selected_answer: str = Field(..., description="The label the student chose: A, B, C, or D")


class EvaluationRequest(BaseModel):
    questions: List[QuizQuestion] = Field(
        ...,
        description="The original quiz questions exactly as returned by /quiz/generate"
    )
    student_answers: List[StudentAnswer] = Field(
        ...,
        description="The student's answers, one per question"
    )


class QuestionResult(BaseModel):
    question_number: int = Field(..., description="Question number")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of this question")
    topic: str = Field(..., description="CS topic of this question")
    question: str = Field(..., description="The question text")
    student_answer: str = Field(..., description="The label the student chose")
    correct_answer: str = Field(..., description="The correct label")
    is_correct: bool = Field(..., description="Whether the student answered correctly")
    explanation: str = Field(..., description="Explanation of the correct answer")


class Resource(BaseModel):
    title: str = Field(..., description="Title of the resource")
    url: str = Field(..., description="Direct URL to the resource")
    source: str = Field(..., description="Source platform e.g. YouTube, freeCodeCamp, GeeksForGeeks")


class StudyRecommendation(BaseModel):
    topic: str = Field(..., description="Recommended topic to study next")
    reason: str = Field(..., description="Why this topic is recommended based on the student's performance")
    priority: str = Field(..., description="Priority level: high, medium, or low")
    suggested_resources: List[Resource] = Field(
        default_factory=list,
        description="Real web-searched resources (YouTube videos and free articles) for this topic"
    )


class StudyRecommendationLLM(BaseModel):
    """Used only inside EvaluationLLMOutput — plain strings before web search enrichment."""
    topic: str = Field(..., description="Recommended topic to study next")
    reason: str = Field(..., description="Why this topic is recommended based on the student's performance")
    priority: str = Field(..., description="Priority level: high, medium, or low")


class StrengthWeakness(BaseModel):
    strong_topics: List[str] = Field(..., description="Topics the student performed well in")
    weak_topics: List[str] = Field(..., description="Topics the student struggled with")
    strong_difficulty_levels: List[str] = Field(
        ...,
        description="Difficulty levels the student handled well"
    )
    weak_difficulty_levels: List[str] = Field(
        ...,
        description="Difficulty levels the student struggled with"
    )


class EvaluationLLMOutput(BaseModel):
    """Structured schema returned by the LLM for the evaluation."""
    overall_feedback: str = Field(
        ...,
        description="A personalised, encouraging paragraph summarising the student's overall performance"
    )
    level_assessment: str = Field(
        ...,
        description=(
            "The student's assessed CS knowledge level: "
            "'Beginner', 'Elementary', 'Intermediate', 'Advanced', or 'Expert'"
        )
    )
    strengths_and_weaknesses: StrengthWeakness = Field(
        ...,
        description="Breakdown of strong and weak areas"
    )
    study_recommendations: List[StudyRecommendationLLM] = Field(
        ...,
        description="Ordered list of study recommendations (most important first) — resources will be added via web search"
    )
    next_topic: str = Field(
        ...,
        description="The single most important topic the student should study next"
    )
    next_topic_reason: str = Field(
        ...,
        description="Clear explanation of why this is the best next topic for this student"
    )


class EvaluationResponse(BaseModel):
    # ── Score summary ──────────────────────────────────────────────────
    total_questions: int = Field(..., description="Total number of questions")
    total_correct: int = Field(..., description="Number of correct answers")
    score_percentage: float = Field(..., description="Overall score as a percentage")

    easy_correct: int = Field(..., description="Correct answers in easy questions")
    easy_total: int = Field(..., description="Total easy questions")

    medium_correct: int = Field(..., description="Correct answers in medium questions")
    medium_total: int = Field(..., description="Total medium questions")

    hard_correct: int = Field(..., description="Correct answers in hard questions")
    hard_total: int = Field(..., description="Total hard questions")

    # ── Per-question results ───────────────────────────────────────────
    question_results: List[QuestionResult] = Field(
        ...,
        description="Detailed result for every question"
    )

    # ── AI evaluation ─────────────────────────────────────────────────
    level_assessment: str = Field(..., description="Student's assessed CS knowledge level")
    overall_feedback: str = Field(..., description="Personalised overall feedback paragraph")
    strengths_and_weaknesses: StrengthWeakness = Field(..., description="Strong and weak areas")
    study_recommendations: List[StudyRecommendation] = Field(
        ...,
        description="Prioritised list of study recommendations"
    )
    next_topic: str = Field(..., description="The single most important topic to study next")
    next_topic_reason: str = Field(..., description="Why that topic is the best next step")
