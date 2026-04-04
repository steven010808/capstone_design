from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.common.config import load_config
from src.dashboard.demo_metrics import get_demo_metrics


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_dir = Path(config["paths"]["data_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])

    return {
        "products": data_dir / "raw" / "products.csv",
        "users": data_dir / "raw" / "users.csv",
        "events": data_dir / "raw" / "events.csv",
        "train": data_dir / "processed" / "train_events.csv",
        "valid": data_dir / "processed" / "valid_events.csv",
        "test": data_dir / "processed" / "test_events.csv",
        "search_metrics": artifacts_dir / "dashboard" / "search_metrics.json",
        "recommendation_metrics": artifacts_dir / "dashboard" / "recommendation_metrics.json",
        "ab_metrics": artifacts_dir / "dashboard" / "ab_test_metrics.json",
        "monitor_metrics": artifacts_dir / "dashboard" / "monitor_metrics.json",
    }


def load_dataset_summary() -> dict[str, Any]:
    config = load_config()
    paths = _resolve_paths(config)

    products_df = _safe_read_csv(paths["products"])
    users_df = _safe_read_csv(paths["users"])
    events_df = _safe_read_csv(paths["events"])
    train_df = _safe_read_csv(paths["train"])
    valid_df = _safe_read_csv(paths["valid"])
    test_df = _safe_read_csv(paths["test"])

    summary: dict[str, Any] = {
        "products_count": len(products_df),
        "users_count": len(users_df),
        "events_count": len(events_df),
        "train_count": len(train_df),
        "valid_count": len(valid_df),
        "test_count": len(test_df),
        "latest_event_ts": None,
        "event_type_counts": pd.Series(dtype="int64"),
        "top_category_counts": pd.Series(dtype="int64"),
        "persona_event_pivot": pd.DataFrame(),
        "split_counts_df": pd.DataFrame(
            {
                "split": ["train", "valid", "test"],
                "count": [len(train_df), len(valid_df), len(test_df)],
            }
        ),
        "products_preview": pd.DataFrame(),
    }

    if not events_df.empty and "timestamp" in events_df.columns:
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce")
        summary["latest_event_ts"] = str(events_df["timestamp"].max())

    if not events_df.empty and "event_type" in events_df.columns:
        summary["event_type_counts"] = events_df["event_type"].value_counts()

    if not products_df.empty and "top_category" in products_df.columns:
        summary["top_category_counts"] = products_df["top_category"].value_counts()

    if (
        not events_df.empty
        and "persona" in events_df.columns
        and "event_type" in events_df.columns
    ):
        summary["persona_event_pivot"] = (
            events_df.groupby(["persona", "event_type"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

    preview_cols = [
        col
        for col in ["product_id", "name", "top_category", "mid_category", "price", "source"]
        if col in products_df.columns
    ]
    if preview_cols:
        summary["products_preview"] = products_df[preview_cols].head(10)

    return summary


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metric_bundle() -> dict[str, Any]:
    config = load_config()
    paths = _resolve_paths(config)

    search_metrics = _load_json(paths["search_metrics"])
    recommendation_metrics = _load_json(paths["recommendation_metrics"])
    ab_metrics = _load_json(paths["ab_metrics"])
    monitor_metrics = _load_json(paths["monitor_metrics"])

    if all([search_metrics, recommendation_metrics, ab_metrics, monitor_metrics]):
        return {
            "source": "real",
            "search": search_metrics,
            "recommendation": recommendation_metrics,
            "ab_test": ab_metrics,
            "monitor": monitor_metrics,
        }

    return get_demo_metrics()


def fetch_system_status() -> dict[str, Any]:
    config = load_config()
    api_port = int(config["server"]["api_port"])

    candidate_urls = [
        f"http://api:{api_port}/api/system/status",
        f"http://localhost:{api_port}/api/system/status",
    ]

    for url in candidate_urls:
        try:
            with urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
                payload["_url"] = url
                payload["_reachable"] = True
                return payload
        except URLError:
            continue
        except Exception:
            continue

    return {
        "_reachable": False,
        "_url": None,
        "message": "API status endpoint not reachable.",
    }