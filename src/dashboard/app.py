import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Multimodal Search & Recommendation Dashboard",
    page_icon="📊",
    layout="wide"
)

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
    st.write("검색 품질 지표를 표시할 영역입니다.")
    st.metric(label="NDCG@10", value="0.00")
    st.metric(label="Latency (ms)", value="0")

with tab2:
    st.subheader("Recommendation Metrics")
    st.write("추천 결과 샘플과 추천 성능 지표를 표시할 영역입니다.")
    st.dataframe(sample_recommendations, use_container_width=True)

with tab3:
    st.subheader("A/B Test")
    st.write("A/B 테스트 결과를 표시할 영역입니다.")
    st.metric(label="p-value", value="-")
    st.metric(label="95% CI", value="-")

with tab4:
    st.subheader("System Status")
    st.write("현재 시스템 상태를 표시할 영역입니다.")
    st.success("Dashboard skeleton is running.")