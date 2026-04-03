import os

import pandas as pd
import requests
import streamlit as st

from src.common.config import load_config
st.set_page_config(
    page_title="Multimodal Search & Recommendation Dashboard",
    page_icon="📊",
    layout="wide"
)

cfg = load_config()
api_base_url = os.getenv("API_BASE_URL", f"http://localhost:{cfg['server']['api_port']}")

st.title("멀티모달 검색 및 추천 대시보드")
st.write("현재는 Day 4용 최소 화면입니다. 이후 검색/추천/A-B 테스트 지표를 연결할 예정입니다.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="MRR", value="0.00")

with col2:
    st.metric(label="HitRate", value="0.00")

with col3:
    st.metric(label="Coverage", value="0.00")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Search Metrics", "Recommendation Metrics", "A/B Test", "System Status"]
)

sample_recommendations = pd.DataFrame(
    [
        {"product_id": "P001", "name": "블랙 오버핏 후드티", "score": 0.91, "reason": "recent_view"},
        {"product_id": "P002", "name": "그레이 와이드 팬츠", "score": 0.87, "reason": "similar_users"},
        {"product_id": "P003", "name": "화이트 스니커즈", "score": 0.84, "reason": "popular_item"},
        {"product_id": "P004", "name": "네이비 맨투맨", "score": 0.80, "reason": "category_match"},
        {"product_id": "P005", "name": "카키 바람막이", "score": 0.78, "reason": "exploration"},
    ]
)

with tab1:
    st.subheader("Search Metrics")
    st.write("검색 API 응답과 기본 지표를 표시할 영역입니다.")

    try:
        res = requests.post(
            f"{api_base_url}/api/search",
            json={"query_text": "black hoodie", "top_k": 10},
            timeout=5,
        )
        data = res.json()

        st.success("Search API 호출 성공")
        st.json(data)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(label="Latency (ms)", value=data.get("latency_ms", 0.0))
        metric_col2.metric(label="Total Count", value=data.get("total_count", 0))

        results = data.get("results", [])
        if results:
            results_df = pd.DataFrame(results)
            st.write("Search Results")
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("현재 search 결과는 더미 상태라 비어 있습니다.")

    except Exception as e:
        st.error(f"Search API 호출 실패: {e}")

with tab2:
    st.subheader("Recommendation Metrics")
    st.write("추천 API 응답과 세션 정보를 표시할 영역입니다.")

    try:
        res = requests.get(
            f"{api_base_url}/api/recommend",
            params={"user_id": "U001", "top_n": 10},
            timeout=5,
        )
        data = res.json()

        st.success("Recommend API 호출 성공")
        st.json(data)

        recent_clicks = data.get("session_context", {}).get("recent_clicks", [])

        summary_df = pd.DataFrame(
            [
                {"field": "user_id", "value": data.get("user_id")},
                {"field": "recent_clicks_count", "value": len(recent_clicks)},
                {"field": "session_interest", "value": data.get("session_context", {}).get("session_interest")},
            ]
        )
        st.dataframe(summary_df, use_container_width=True)

        if recent_clicks:
            clicks_df = pd.DataFrame({"recent_clicks": recent_clicks})
            st.write("Recent Clicks")
            st.dataframe(clicks_df, use_container_width=True)
        else:
            st.info("recent_clicks가 비어 있습니다.")

        recommendations = data.get("recommendations", [])
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.write("API Recommendation Results")
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("현재 recommendation 결과는 더미 상태라 비어 있습니다.")
            st.write("샘플 테이블")
            st.dataframe(sample_recommendations, use_container_width=True)

    except Exception as e:
        st.error(f"Recommend API 호출 실패: {e}")


with tab3:
    st.subheader("A/B Test")
    st.write("A/B 테스트 결과를 표시할 영역입니다.")
    st.metric(label="p-value", value="-")
    st.metric(label="95% CI", value="-")

with tab4:
    st.subheader("System Status")
    st.write("현재 시스템 상태를 표시할 영역입니다.")

    st.write("App name:", cfg["app"]["name"])
    st.write("API base URL:", api_base_url)
    st.write("Redis host (config):", cfg["redis"]["host"])
    st.write("Redis port (config):", cfg["redis"]["port"])

    try:
        res = requests.get(f"{api_base_url}/api/system/status", timeout=5)
        st.success("System Status API 호출 성공")
        st.json(res.json())
    except Exception as e:
        st.error(f"System Status API 호출 실패: {e}")