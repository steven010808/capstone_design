from typing import List, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    top_k: int = 10


class SearchResultItem(BaseModel):
    product_id: str
    name: str
    score: float
    price: int


class SearchResponse(BaseModel):
    search_type: str
    results: List[SearchResultItem]
    latency_ms: float
    total_count: int


class PipelineLatency(BaseModel):
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


class SessionContext(BaseModel):
    recent_clicks: List[str]
    session_interest: Optional[str] = None


class RecommendationItem(BaseModel):
    product_id: str
    score: float
    reason: str
    is_exploration: bool


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    pipeline_latency: PipelineLatency
    session_context: SessionContext