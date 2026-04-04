from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import (
    fetch_system_status,
    load_dataset_summary,
    load_metric_bundle,
)


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="근스톤 | Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

# ---------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------
COLORS = {
    "bg": "#f4f7fb",
    "panel": "#ffffff",
    "text": "#0f172a",
    "muted": "#64748b",
    "line": "#e2e8f0",
    "blue": "#2563eb",
    "blue_soft": "#eff6ff",
    "violet": "#7c3aed",
    "violet_soft": "#f5f3ff",
    "green": "#16a34a",
    "green_soft": "#ecfdf3",
    "amber": "#d97706",
    "amber_soft": "#fff7ed",
    "red": "#dc2626",
    "red_soft": "#fef2f2",
    "slate": "#94a3b8",
    "navy": "#0b1220",
    "hero_1": "#0b1220",
    "hero_2": "#16233b",
    "hero_3": "#1d4ed8",
}

# ---------------------------------------------------------
# Global styles
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: Inter, Pretendard, Apple SD Gothic Neo, Noto Sans KR, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(37,99,235,0.06), transparent 22%),
            radial-gradient(circle at top right, rgba(124,58,237,0.05), transparent 20%),
            {COLORS["bg"]};
    }}

    .block-container {{
        max-width: 1520px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}

    [data-testid="stSidebar"] {{
        background: #fbfcfe;
        border-right: 1px solid {COLORS["line"]};
    }}

    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 1rem;
    }}

    .brand-wrap {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }}

    .brand-title {{
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        color: {COLORS["text"]};
    }}

    .brand-subtitle {{
        color: {COLORS["muted"]};
        font-size: 0.92rem;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        background: white;
        border: 1px solid {COLORS["line"]};
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
        font-size: 0.84rem;
        font-weight: 700;
        color: {COLORS["text"]};
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at 10% 20%, rgba(255,255,255,0.16), transparent 18%),
            radial-gradient(circle at 80% 10%, rgba(255,255,255,0.10), transparent 18%),
            linear-gradient(135deg, {COLORS["hero_1"]} 0%, {COLORS["hero_2"]} 42%, {COLORS["hero_3"]} 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 30px;
        padding: 30px 34px 28px 34px;
        color: white;
        box-shadow: 0 26px 60px rgba(15,23,42,0.22);
        margin-bottom: 1.1rem;
    }}

    .hero-title {{
        font-size: 2.35rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.55rem;
    }}

    .hero-subtitle {{
        font-size: 1rem;
        color: rgba(255,255,255,0.86);
        line-height: 1.55;
        max-width: 900px;
        margin-bottom: 1rem;
    }}

    .hero-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        color: rgba(255,255,255,0.95);
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        backdrop-filter: blur(8px);
    }}

    .nav-strip {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 1rem;
    }}

    .nav-card {{
        background: white;
        border: 1px solid {COLORS["line"]};
        border-radius: 22px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 10px 28px rgba(15,23,42,0.05);
    }}

    .nav-step {{
        font-size: 0.77rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .nav-step.offline {{ color: {COLORS["blue"]}; }}
    .nav-step.online {{ color: {COLORS["violet"]}; }}
    .nav-step.monitor {{ color: {COLORS["green"]}; }}

    .nav-title {{
        font-size: 1.02rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.3rem;
    }}

    .nav-desc {{
        font-size: 0.9rem;
        color: {COLORS["muted"]};
        line-height: 1.45;
    }}

    .top-kpi {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid {COLORS["line"]};
        border-radius: 24px;
        padding: 16px 18px 14px 18px;
        box-shadow: 0 10px 24px rgba(15,23,42,0.04);
    }}

    .zone {{
        border-radius: 28px;
        padding: 18px;
        margin-bottom: 18px;
        border: 1px solid {COLORS["line"]};
        box-shadow: 0 12px 28px rgba(15,23,42,0.05);
    }}

    .zone.offline {{
        background: linear-gradient(180deg, {COLORS["blue_soft"]} 0%, #ffffff 28%);
        border-left: 8px solid {COLORS["blue"]};
    }}

    .zone.online {{
        background: linear-gradient(180deg, {COLORS["violet_soft"]} 0%, #ffffff 28%);
        border-left: 8px solid {COLORS["violet"]};
    }}

    .zone.monitor {{
        background: linear-gradient(180deg, {COLORS["green_soft"]} 0%, #ffffff 28%);
        border-left: 8px solid {COLORS["green"]};
    }}

    .zone-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 0.8rem;
    }}

    .zone-title {{
        font-size: 1.25rem;
        font-weight: 850;
        color: {COLORS["text"]};
        letter-spacing: -0.02em;
    }}

    .zone-desc {{
        color: {COLORS["muted"]};
        font-size: 0.93rem;
        line-height: 1.55;
        max-width: 760px;
        margin-top: 0.18rem;
    }}

    .zone-badge {{
        font-size: 0.76rem;
        font-weight: 800;
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        white-space: nowrap;
        border: 1px solid transparent;
    }}

    .zone-badge.offline {{
        color: {COLORS["blue"]};
        background: #dbeafe;
        border-color: #bfdbfe;
    }}

    .zone-badge.online {{
        color: {COLORS["violet"]};
        background: #ede9fe;
        border-color: #ddd6fe;
    }}

    .zone-badge.monitor {{
        color: {COLORS["green"]};
        background: #dcfce7;
        border-color: #bbf7d0;
    }}

    .panel {{
        background: {COLORS["panel"]};
        border: 1px solid {COLORS["line"]};
        border-radius: 22px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
        height: 100%;
    }}

    .panel-title {{
        font-size: 0.95rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.2rem;
    }}

    .panel-desc {{
        color: {COLORS["muted"]};
        font-size: 0.84rem;
        margin-bottom: 0.8rem;
        line-height: 1.45;
    }}

    .sub-kpi {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid {COLORS["line"]};
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.03);
    }}

    .meta-rail {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid {COLORS["line"]};
        border-radius: 20px;
        padding: 14px 16px;
    }}

    .meta-title {{
        font-size: 0.85rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.45rem;
    }}

    .meta-text {{
        color: {COLORS["muted"]};
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    .status-chip-ok {{
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: rgba(22,163,74,0.10);
        color: {COLORS["green"]};
        border: 1px solid rgba(22,163,74,0.18);
        font-size: 0.78rem;
        font-weight: 800;
    }}

    .status-chip-bad {{
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: rgba(220,38,38,0.10);
        color: {COLORS["red"]};
        border: 1px solid rgba(220,38,38,0.18);
        font-size: 0.78rem;
        font-weight: 800;
    }}

    .small-note {{
        color: {COLORS["muted"]};
        font-size: 0.84rem;
    }}

    .footer-note {{
        color: {COLORS["muted"]};
        font-size: 0.84rem;
        margin-top: 0.5rem;
    }}

    div[data-testid="stMetric"] {{
        background: transparent !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-weight: 850;
        color: {COLORS["text"]};
    }}

    div[data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]};
        font-weight: 700;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: white;
        border: 1px solid {COLORS["line"]};
        border-radius: 14px;
        padding: 10px 14px;
    }}

    .stTabs [aria-selected="true"] {{
        background: #eef4ff !important;
        border-color: #c7d7fe !important;
        color: {COLORS["blue"]} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
dataset = load_dataset_summary()
metrics = load_metric_bundle()
system_status = fetch_system_status()

metric_source = metrics.get("source", "demo")
source_badge = "Demo metrics" if metric_source == "demo" else "Real metrics"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## Dashboard Control")
    st.caption("운영 / 발표 공용 모드")

    st.markdown(
        f"""
        <div class="meta-rail">
            <div class="meta-title">Metric Source</div>
            <div class="meta-text">
                {source_badge}<br><br>
                <b>Latest Event</b><br>
                {dataset.get("latest_event_ts") or "N/A"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 표시 옵션")
    show_preview = st.toggle("상품 미리보기", value=True)
    show_system_json = st.toggle("System JSON", value=False)

    st.markdown("### 현재 데이터 규모")
    st.write(f"Products: {dataset['products_count']:,}")
    st.write(f"Users: {dataset['users_count']:,}")
    st.write(f"Events: {dataset['events_count']:,}")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def metric_box(label: str, value: str, delta: str | None = None) -> None:
    st.markdown('<div class="sub-kpi">', unsafe_allow_html=True)
    st.metric(label, value, delta=delta)
    st.markdown("</div>", unsafe_allow_html=True)


def apply_common_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=12, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    return fig


def make_funnel_chart(event_counts: pd.Series) -> go.Figure:
    order = ["search", "view", "cart", "purchase"]
    values = [int(event_counts.get(k, 0)) for k in order]
    fig = go.Figure(
        go.Funnel(
            y=["Search", "View", "Cart", "Purchase"],
            x=values,
            textinfo="value+percent initial",
            marker=dict(color=["#94a3b8", "#2563eb", "#7c3aed", "#16a34a"]),
        )
    )
    return apply_common_layout(fig, 350)


def make_persona_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return apply_common_layout(go.Figure(), 350)

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            colorscale=[
                [0.0, "#eff6ff"],
                [0.25, "#dbeafe"],
                [0.5, "#93c5fd"],
                [0.75, "#3b82f6"],
                [1.0, "#1d4ed8"],
            ],
            text=df.values,
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    return apply_common_layout(fig, 350)


def make_ab_chart(m: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Control",
            x=["CVR", "Exploration"],
            y=[m["control_cvr"], m["exploration_rate_control"]],
            marker_color="#94a3b8",
            text=[f"{m['control_cvr']:.3f}", f"{m['exploration_rate_control']:.3f}"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Treatment",
            x=["CVR", "Exploration"],
            y=[m["treatment_cvr"], m["exploration_rate_treatment"]],
            marker_color="#2563eb",
            text=[f"{m['treatment_cvr']:.3f}", f"{m['exploration_rate_treatment']:.3f}"],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group", yaxis_title="Rate")
    return apply_common_layout(fig, 320)


def make_latency_chart(latencies: dict) -> go.Figure:
    stages = ["candidate", "ranking", "reranking", "total"]
    values = [latencies[k] for k in stages]
    fig = px.bar(
        x=stages,
        y=values,
        text=values,
        labels={"x": "", "y": "Latency (ms)"},
    )
    fig.update_traces(marker_color="#2563eb")
    return apply_common_layout(fig, 320)


def make_search_mix_chart(search_mix: dict) -> go.Figure:
    fig = px.pie(
        names=list(search_mix.keys()),
        values=list(search_mix.values()),
        hole=0.62,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(colors=["#2563eb", "#7c3aed", "#0891b2"]),
    )
    return apply_common_layout(fig, 320)


def make_category_chart(top_category_counts: pd.Series) -> go.Figure:
    df = top_category_counts.reset_index()
    df.columns = ["top_category", "count"]
    fig = px.bar(df, x="top_category", y="count", text="count")
    fig.update_traces(marker_color="#2563eb")
    return apply_common_layout(fig, 340)


def make_split_chart(split_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(split_df, x="split", y="count", text="count")
    fig.update_traces(marker_color="#0f766e")
    return apply_common_layout(fig, 320)


def make_event_type_chart(event_counts: pd.Series) -> go.Figure:
    df = event_counts.reset_index()
    df.columns = ["event_type", "count"]
    fig = px.bar(df, x="event_type", y="count", text="count")
    fig.update_traces(marker_color="#7c3aed")
    return apply_common_layout(fig, 320)

# ---------------------------------------------------------
# Topbar
# ---------------------------------------------------------
top_left, top_right = st.columns([1.2, 0.8])
with top_left:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-title">근스톤 Evaluation Dashboard</div>
                <div class="brand-subtitle">
                    검색 · 추천 · 실험 · 운영 상태를 분리해서 보여주는 배포형 분석 대시보드
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_right:
    reachable = system_status.get("_reachable", False)
    chip = (
        "<span class='status-chip-ok'>API Reachable</span>"
        if reachable
        else "<span class='status-chip-bad'>API Unreachable</span>"
    )
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; gap:10px; flex-wrap:wrap; margin-top:10px;">
            <div class="status-pill">{source_badge}</div>
            <div class="status-pill">{chip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Evaluation, Experimentation & Monitoring</div>
        <div class="hero-subtitle">
            오프라인 모델 검증, 온라인 A/B 테스트, 실시간 운영 상태를 분리된 영역에서 직관적으로 확인할 수 있도록 설계된 대시보드
        </div>
        <span class="hero-pill">Spec-aligned</span>
        <span class="hero-pill">Production-style UI</span>
        <span class="hero-pill">Replaceable with Real Metrics</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if metric_source == "demo":
    st.info(
        "현재 Search / Recommendation / A/B 지표는 예시 데이터다. "
        "이후 artifacts/dashboard/*.json 파일을 생성하면 동일 UI에서 실제 결과로 자동 치환된다."
    )

# ---------------------------------------------------------
# Stage navigation
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="nav-strip">
        <div class="nav-card">
            <div class="nav-step offline">Stage 1</div>
            <div class="nav-title">Offline Evaluation</div>
            <div class="nav-desc">검색 품질(MRR, NDCG)과 추천 성능(HitRate, Recall, Coverage) 검증</div>
        </div>
        <div class="nav-card">
            <div class="nav-step online">Stage 2</div>
            <div class="nav-title">Online Evaluation</div>
            <div class="nav-desc">Control / Treatment 비교, Lift, p-value, 신뢰구간 기반 A/B 테스트 결과 검토</div>
        </div>
        <div class="nav-card">
            <div class="nav-step monitor">Stage 3</div>
            <div class="nav-title">Live Monitoring</div>
            <div class="nav-desc">시스템 연결 상태, 데이터 파이프라인 규모, 이벤트 분포 및 최신 상태 점검</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="top-kpi">', unsafe_allow_html=True)
    st.metric("Products", f"{dataset['products_count']:,}")
    st.caption("raw products.csv")
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    st.markdown('<div class="top-kpi">', unsafe_allow_html=True)
    st.metric("Users", f"{dataset['users_count']:,}")
    st.caption("raw users.csv")
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
    st.markdown('<div class="top-kpi">', unsafe_allow_html=True)
    st.metric("Events", f"{dataset['events_count']:,}")
    st.caption("raw events.csv")
    st.markdown("</div>", unsafe_allow_html=True)

with k4:
    st.markdown('<div class="top-kpi">', unsafe_allow_html=True)
    st.metric("Latest Event", dataset["latest_event_ts"] or "N/A")
    st.caption("latest timestamp")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Offline Evaluation Zone
# ---------------------------------------------------------
st.markdown('<div class="zone offline">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="zone-header">
        <div>
            <div class="zone-title">Offline Evaluation Zone</div>
            <div class="zone-desc">
                모델 자체 품질을 검증하는 구간. 검색 품질(MRR, NDCG), 추천 성능(HitRate, Recall, Coverage),
                파이프라인 latency를 기반으로 오프라인 성능을 평가한다.
            </div>
        </div>
        <div class="zone-badge offline">OFFLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

off_left, off_right = st.columns([1.12, 1.0])

with off_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Search Quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">텍스트/이미지/하이브리드 검색에 대한 핵심 오프라인 지표와 search mix를 함께 표시</div>',
        unsafe_allow_html=True,
    )
    row = st.columns(3)
    with row[0]:
        metric_box("MRR", f"{metrics['search']['mrr']:.3f}")
    with row[1]:
        metric_box("NDCG", f"{metrics['search']['ndcg']:.3f}")
    with row[2]:
        metric_box("Search p95", f"{metrics['search']['latency_ms_p95']:.1f} ms")
    st.plotly_chart(
        make_search_mix_chart(metrics["search"]["search_type_mix"]),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with off_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Recommendation Quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">HitRate, Recall, Coverage와 추천 파이프라인 latency로 offline recommendation 성능을 표시</div>',
        unsafe_allow_html=True,
    )
    row = st.columns(4)
    with row[0]:
        metric_box("HitRate@50", f"{metrics['recommendation']['hitrate_at_50']:.3f}")
    with row[1]:
        metric_box("NDCG@50", f"{metrics['recommendation']['ndcg_at_50']:.3f}")
    with row[2]:
        metric_box("Coverage", f"{metrics['recommendation']['coverage']:.3f}")
    with row[3]:
        metric_box("Recall@300", f"{metrics['recommendation']['recall_at_300']:.3f}")
    st.plotly_chart(
        make_latency_chart(metrics["recommendation"]["pipeline_latency_ms"]),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Online Evaluation Zone
# ---------------------------------------------------------
st.markdown('<div class="zone online">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="zone-header">
        <div>
            <div class="zone-title">Online Evaluation Zone</div>
            <div class="zone-desc">
                실제 사용자 노출 실험에 해당하는 구간. Control / Treatment 성과 비교, 전환율 Lift,
                p-value, 신뢰구간을 통해 온라인 전략의 효과를 검증한다.
            </div>
        </div>
        <div class="zone-badge online">ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

on_left, on_right = st.columns([1.0, 1.0])

with on_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Experiment KPI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">A/B 테스트의 핵심 성과 지표를 별도 KPI 카드로 분리해 직관적으로 확인</div>',
        unsafe_allow_html=True,
    )
    row1 = st.columns(3)
    with row1[0]:
        metric_box("Control CVR", f"{metrics['ab_test']['control_cvr'] * 100:.2f}%")
    with row1[1]:
        metric_box("Treatment CVR", f"{metrics['ab_test']['treatment_cvr'] * 100:.2f}%")
    with row1[2]:
        metric_box("Lift", f"+{metrics['ab_test']['lift_pct']:.1f}%")

    row2 = st.columns(2)
    with row2[0]:
        metric_box("p-value", f"{metrics['ab_test']['p_value']:.3f}")
    with row2[1]:
        metric_box("95% CI", metrics["ab_test"]["confidence_interval"])

    st.markdown("</div>", unsafe_allow_html=True)

with on_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Control vs Treatment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">Control과 Treatment의 CVR 및 탐험률 차이를 차트로 직접 비교</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_ab_chart(metrics["ab_test"]),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Live Monitoring Zone
# ---------------------------------------------------------
st.markdown('<div class="zone monitor">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="zone-header">
        <div>
            <div class="zone-title">Live Monitoring Zone</div>
            <div class="zone-desc">
                실제 생성된 raw / processed 데이터를 기반으로 이벤트 퍼널, persona 분포, split 분포,
                시스템 상태를 점검하는 운영 모니터링 구간.
            </div>
        </div>
        <div class="zone-badge monitor">MONITORING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mon_left, mon_mid, mon_right = st.columns([1.05, 1.05, 0.9])

with mon_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Observed Event Funnel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">실제 raw events.csv에서 읽은 Search → View → Cart → Purchase 퍼널</div>',
        unsafe_allow_html=True,
    )
    if not dataset["event_type_counts"].empty:
        st.plotly_chart(
            make_funnel_chart(dataset["event_type_counts"]),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    else:
        st.warning("events.csv를 찾지 못했다.")
    st.markdown("</div>", unsafe_allow_html=True)

with mon_mid:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Persona × Event Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">persona별 이벤트 생성 분포를 heatmap으로 표시</div>',
        unsafe_allow_html=True,
    )
    if not dataset["persona_event_pivot"].empty:
        st.plotly_chart(
            make_persona_heatmap(dataset["persona_event_pivot"]),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    else:
        st.warning("persona-event matrix를 만들 수 없다.")
    st.markdown("</div>", unsafe_allow_html=True)

with mon_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Operational Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-desc">실시간 운영 상태와 모니터링 수치를 요약</div>',
        unsafe_allow_html=True,
    )

    row = st.columns(2)
    with row[0]:
        metric_box("CTR", f"{metrics['monitor']['ctr']:.3f}")
    with row[1]:
        metric_box("HitRate", f"{metrics['monitor']['hitrate']:.3f}")

    row2 = st.columns(2)
    with row2[0]:
        metric_box("Threshold", f"{metrics['monitor']['threshold_hitrate']:.2f}")
    with row2[1]:
        metric_box("New Logs", f"{metrics['monitor']['new_logs_since_last_train']:,}")

    st.markdown(
        f"""
        <div class="meta-rail" style="margin-top:8px;">
            <div class="meta-title">System Meta</div>
            <div class="meta-text">
                <b>Model Version</b>: {metrics['monitor']['model_version']}<br>
                <b>Alert</b>: {metrics['monitor']['alert']}<br>
                <b>Latest Event</b>: {dataset['latest_event_ts'] or "N/A"}
            </div>
            <div style="margin-top:8px;">
                {"<span class='status-chip-ok'>API Reachable</span>" if system_status.get("_reachable") else "<span class='status-chip-bad'>API Unreachable</span>"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Detail tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Search Detail", "Recommendation Detail", "Experiment Detail", "System Detail"]
)

with tab1:
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.markdown("### Search KPI")
        st.metric("MRR", f"{metrics['search']['mrr']:.3f}")
        st.metric("NDCG", f"{metrics['search']['ndcg']:.3f}")
        st.metric("Latency p95", f"{metrics['search']['latency_ms_p95']:.1f} ms")
    with c2:
        st.markdown("### Product Category Distribution")
        if not dataset["top_category_counts"].empty:
            st.plotly_chart(
                make_category_chart(dataset["top_category_counts"]),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.warning("products.csv를 찾지 못했다.")

with tab2:
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.markdown("### Recommendation KPI")
        st.metric("HitRate@50", f"{metrics['recommendation']['hitrate_at_50']:.3f}")
        st.metric("Coverage", f"{metrics['recommendation']['coverage']:.3f}")
        st.metric("Recall@300", f"{metrics['recommendation']['recall_at_300']:.3f}")
    with c2:
        st.markdown("### Split Distribution")
        st.plotly_chart(
            make_split_chart(dataset["split_counts_df"]),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

with tab3:
    st.markdown("### A/B Test Summary")
    ab_df = pd.DataFrame(
        {
            "metric": ["CVR", "Exploration Rate"],
            "control": [
                metrics["ab_test"]["control_cvr"],
                metrics["ab_test"]["exploration_rate_control"],
            ],
            "treatment": [
                metrics["ab_test"]["treatment_cvr"],
                metrics["ab_test"]["exploration_rate_treatment"],
            ],
        }
    )
    st.dataframe(ab_df, use_container_width=True, height=180)
    st.success(
        f"Lift +{metrics['ab_test']['lift_pct']:.1f}% | "
        f"p-value = {metrics['ab_test']['p_value']:.3f} | "
        f"95% CI = {metrics['ab_test']['confidence_interval']}"
    )

with tab4:
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.markdown("### System Endpoint")
        if system_status.get("_reachable"):
            st.success("API system status reachable")
        else:
            st.warning("API system status endpoint not reachable")

        st.markdown("### Latest Event Timestamp")
        st.write(dataset["latest_event_ts"] or "N/A")

        if show_system_json:
            st.markdown("### Raw System JSON")
            st.json(system_status)

    with c2:
        st.markdown("### Event Type Distribution")
        if not dataset["event_type_counts"].empty:
            st.plotly_chart(
                make_event_type_chart(dataset["event_type_counts"]),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.warning("events.csv를 찾지 못했다.")

if show_preview:
    st.markdown("### Product Preview")
    if not dataset["products_preview"].empty:
        st.dataframe(dataset["products_preview"], use_container_width=True, height=320)
    else:
        st.warning("products preview를 만들 수 없다.")

st.markdown("---")
st.markdown(
    """
    <div class="footer-note">
    현재는 명세서 기반 예시 지표와 실제 raw / processed 데이터 요약을 함께 표시한다.
    이후 <code>artifacts/dashboard/*.json</code> 파일을 생성하면 동일 UI에서 실제 실험 결과로 자동 치환된다.
    </div>
    """,
    unsafe_allow_html=True,
)