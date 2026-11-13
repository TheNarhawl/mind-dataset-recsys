from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from mind_recsys.service.pipeline import TwoStageRecommender
from mind_recsys.service.schemas import RecommendRequest, RecommendResponse

_global_recommender: TwoStageRecommender | None = None


def set_recommender(recommender: TwoStageRecommender) -> None:
    global _global_recommender
    _global_recommender = recommender


def get_recommender() -> TwoStageRecommender:
    global _global_recommender
    if _global_recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender pipeline is not initialized.",
        )
    return _global_recommender


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup actions
    yield
    # Shutdown actions


def create_app(recommender: TwoStageRecommender | None = None) -> FastAPI:
    if recommender is not None:
        set_recommender(recommender)

    app = FastAPI(
        title="Two-Stage News Recommender System API",
        version="0.1.0",
        description="Candidate Retrieval (Qdrant) + Learning to Rank (CatBoostRanker) for Microsoft News (MIND)",
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "mind-recsys",
            "version": "0.1.0",
            "status": "active",
            "endpoints": "/api/v1/recommend, /health",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        is_ready = _global_recommender is not None
        return {
            "status": "ok" if is_ready else "initializing",
            "recommender_ready": str(is_ready),
        }

    @app.post("/api/v1/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest) -> RecommendResponse:
        rec = get_recommender()
        return rec.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            history_override=request.user_history_news_ids,
        )

    return app
