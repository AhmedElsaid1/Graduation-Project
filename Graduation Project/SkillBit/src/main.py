from fastapi import FastAPI
from contextlib import asynccontextmanager
from utils.config import get_settings, Settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from routes import (
    base_router,
    quiz_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()

    llm_provider_factory = LLMProviderFactory(settings)

    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    
    yield
    
    # Shutdown
    pass


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(quiz_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)