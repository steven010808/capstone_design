from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.common.config import load_config
from src.common.redis_client import get_redis_client
from src.recommendation.service import run_recommendation
from src.search.service import run_search
from src.simulator.service import get_simulator_status
from src.serving.schemas import (
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.config = cfg
    app.state.redis = None
    app.state.redis_ok = False

    try:
        redis_client = get_redis_client()
        redis_client.ping()
        app.state.redis = redis_client
        app.state.redis_ok = True
        print("Redis connection: OK")
    except Exception as e:
        print(f"Redis connection failed: {e}")

    yield


app = FastAPI(
    title="Multimodal Search & Recommendation API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "hello",
        "app_name": app.state.config["app"]["name"],
        "redis_connected": app.state.redis_ok,
    }


@app.get("/api/system/status")
def system_status():
    cfg = app.state.config
    return {
        "app_name": cfg["app"]["name"],
        "api_port": cfg["server"]["api_port"],
        "redis": {
            "host": cfg["redis"]["host"],
            "port": cfg["redis"]["port"],
            "connected": app.state.redis_ok,
        },
    }


@app.get("/api/simulator/status")
def simulator_status():
    return get_simulator_status()


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return run_search(request)


@app.get("/api/recommend", response_model=RecommendationResponse)
def recommend(user_id: str, top_n: int = 10):
    redis_client = app.state.redis if app.state.redis_ok else None
    return run_recommendation(user_id=user_id, top_n=top_n, redis_client=redis_client)