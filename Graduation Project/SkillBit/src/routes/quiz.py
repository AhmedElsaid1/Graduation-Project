from fastapi import APIRouter, Request, HTTPException, status
from models.QuizModels import QuizRequest, QuizResponse, QuizTopic, EvaluationRequest, EvaluationResponse
from controllers.QuizController import QuizController
from controllers.EvaluationController import EvaluationController

quiz_router = APIRouter(
    prefix="/quiz",
    tags=["Quiz"]
)


@quiz_router.post(
    "/generate",
    response_model=QuizResponse,
    summary="Generate a CS Quiz",
    description=(
        "Generate a multiple-choice quiz for CS students. "
        "By default returns 25 questions: 10 easy, 10 medium, 5 hard. "
        "Each question includes 4 options (A–D), the correct answer label, "
        "and a detailed explanation of why the correct answer is right."
    )
)
async def generate_quiz(
    request: Request,
    quiz_request: QuizRequest = None
):
    if quiz_request is None:
        quiz_request = QuizRequest()

    controller = QuizController()
    result = await controller.generate_quiz(request=request, quiz_request=quiz_request)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate quiz. The AI service may be unavailable or returned an invalid response. Please try again."
        )

    return result


@quiz_router.get(
    "/generate",
    response_model=QuizResponse,
    summary="Generate a Default CS Quiz (GET)",
    description=(
        "Generate a default 25-question CS quiz using GET request. "
        "Returns 10 easy, 10 medium, and 5 hard questions on mixed CS topics. "
        "Each question includes 4 MCQ options, the correct answer, and a full explanation."
    )
)
async def generate_default_quiz(request: Request):
    quiz_request = QuizRequest()
    controller = QuizController()
    result = await controller.generate_quiz(request=request, quiz_request=quiz_request)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate quiz. The AI service may be unavailable or returned an invalid response. Please try again."
        )

    return result


@quiz_router.get(
    "/topics",
    summary="List Available Quiz Topics",
    description="Returns the list of all available CS topics you can request a quiz for."
)
async def get_quiz_topics():
    return {
        "topics": [
            {"value": t.value, "label": t.value.replace("_", " ").title()}
            for t in QuizTopic
        ]
    }


@quiz_router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate Student Quiz Answers",
    description=(
        "Submit the original quiz questions together with the student's selected answers. "
        "Returns: per-question grading, score breakdown by difficulty, an AI-generated "
        "level assessment (Beginner → Expert), personalised feedback, identified strengths "
        "and weaknesses, prioritised study recommendations, and the single best next topic "
        "for the student to focus on."
    )
)
async def evaluate_quiz(
    request: Request,
    eval_request: EvaluationRequest,
):
    controller = EvaluationController()
    result = await controller.evaluate(request=request, eval_request=eval_request)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to evaluate quiz. The AI service may be unavailable. Please try again."
        )

    return result
