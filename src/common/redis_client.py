import redis
from src.common.config import load_config


def get_redis_client():
    cfg = load_config()
    redis_cfg = cfg["redis"]

    client = redis.Redis(
        host=redis_cfg["host"],
        port=redis_cfg["port"],
        db=redis_cfg.get("db", 0),
        decode_responses=True,
    )
    return client