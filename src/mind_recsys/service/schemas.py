from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier, e.g. U13740")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    user_history_news_ids: list[str] | None = Field(
        default=None,
        description="Optional click history override for cold-start or ad-hoc simulation",
    )


class RecommendationItem(BaseModel):
    news_id: str
    score: float
    rank: int
    category: str = "General"
    subcategory: str = ""
    title: str = ""


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    feature_engineering_ms: float
    ranking_ms: float
    total_ms: float


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]
    latency: LatencyBreakdown
