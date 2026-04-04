from __future__ import annotations

from typing import Any


def get_demo_metrics() -> dict[str, Any]:
    return {
        "source": "demo",
        "search": {
            "mrr": 0.58,
            "ndcg": 0.53,
            "latency_ms_p95": 42.0,
            "search_type_mix": {
                "text": 62,
                "image": 21,
                "hybrid": 17,
            },
        },
        "recommendation": {
            "hitrate_at_50": 0.22,
            "ndcg_at_50": 0.09,
            "coverage": 0.24,
            "recall_at_300": 0.32,
            "pipeline_latency_ms": {
                "candidate": 45.0,
                "ranking": 62.0,
                "reranking": 12.0,
                "total": 127.0,
            },
        },
        "ab_test": {
            "control_cvr": 0.032,
            "treatment_cvr": 0.041,
            "lift_pct": 28.0,
            "p_value": 0.003,
            "confidence_interval": "[0.4%, 1.4%]",
            "control_users": 5000,
            "treatment_users": 5000,
            "exploration_rate_control": 0.00,
            "exploration_rate_treatment": 0.15,
        },
        "monitor": {
            "ctr": 0.071,
            "hitrate": 0.21,
            "threshold_hitrate": 0.20,
            "new_logs_since_last_train": 12345,
            "retrain_threshold": 10000,
            "model_version": "v2.1-demo",
            "alert": "정상 (demo)",
        },
    }