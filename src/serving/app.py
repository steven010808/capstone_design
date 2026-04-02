from typing import List, Optional #data가 무슨 타입인지 힌트

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Multimodal Search & Recommendation API",
    version="0.1.0"
)


class SearchRequest(BaseModel):  ## 사용자가 보낼 query 구조
    query_text: Optional[str] = None
    top_k: int = 10


class SearchResultItem(BaseModel):
    product_id: str
    name: str
    score: float   #나중에 실재 점수
    price: int


class SearchResponse(BaseModel):
    search_type: str     #검색 방식 image/ text/ image+text
    results: List[SearchResultItem]
    latency_ms: float
    total_count: int

# recommand pipeline 단계별 시간
class PipelineLatency(BaseModel):
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


#recommandation을 위한 최근 정보 (redis 로다가)
class SessionContext(BaseModel):
    recent_clicks: List[str]
    session_interest: Optional[str] = None



#개추받은 상품의 구조
class RecommendationItem(BaseModel):
    product_id: str
    score: float
    reason: str
    is_exploration: bool

#개추 API 최종 출력 형식
class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    pipeline_latency: PipelineLatency
    session_context: SessionContext


@app.get("/")
def root():
    return {"message": "hello"}


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return SearchResponse(
        search_type="text",
        results=[],
        latency_ms=0.0,
        total_count=0
    )


@app.get("/api/recommend", response_model=RecommendationResponse)
def recommend(user_id: str, top_n: int = 10):
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[],
        pipeline_latency=PipelineLatency(
            candidate_ms=0.0,
            ranking_ms=0.0,
            reranking_ms=0.0,
            total_ms=0.0
        ),
        session_context=SessionContext(
            recent_clicks=[],
            session_interest=None
        )
    )