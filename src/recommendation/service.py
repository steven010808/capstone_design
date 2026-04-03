from src.serving.schemas import (
    PipelineLatency,
    RecommendationItem,
    RecommendationResponse,
    SessionContext,
)


def run_recommendation(user_id: str, top_n: int, redis_client=None) -> RecommendationResponse:
    recent_clicks = []

    if redis_client is not None:
        redis_value = redis_client.get(f"user:{user_id}:recent_clicks")
        if redis_value:
            recent_clicks = redis_value.split(",")

    sample_recommendations = [
        RecommendationItem(
            product_id="P001",
            score=0.92,
            reason="recent_view",
            is_exploration=False,
        ),
        RecommendationItem(
            product_id="P002",
            score=0.88,
            reason="similar_users",
            is_exploration=False,
        ),
        RecommendationItem(
            product_id="P005",
            score=0.61,
            reason="mab_exploration",
            is_exploration=True,
        ),
    ]

    return RecommendationResponse(
        user_id=user_id,
        recommendations=sample_recommendations[:top_n],
        pipeline_latency=PipelineLatency(
            candidate_ms=35.0,
            ranking_ms=42.0,
            reranking_ms=8.0,
            total_ms=85.0,
        ),
        session_context=SessionContext(
            recent_clicks=recent_clicks,
            session_interest="캐주얼" if recent_clicks else None,
        ),
    )