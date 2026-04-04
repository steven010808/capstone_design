from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_mode(series: pd.Series) -> Optional[int]:
    """Return mode if available, else None."""
    if series.empty:
        return None
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return None
    try:
        return int(mode_values.iloc[0])
    except Exception:
        return None


def _normalize_article_id(value: object) -> str:
    """
    H&M article_id는 앞자리에 0이 포함될 수 있으므로 반드시 문자열 10자리로 유지한다.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # 숫자로 읽혔더라도 10자리로 복구 시도
    if text.isdigit():
        return text.zfill(10)
    return text


def _build_image_path(article_id: str, images_root: Optional[Path]) -> tuple[str, int]:
    """
    H&M image path 규칙:
    images/{article_id[:3]}/{article_id}.jpg
    실제 파일 존재 여부를 같이 반환.
    """
    if not article_id:
        return "", 0

    relative_path = Path(article_id[:3]) / f"{article_id}.jpg"

    if images_root is None:
        # images 폴더가 아직 없으면 relative path만 기록
        return str(relative_path).replace("\\", "/"), 0

    full_path = images_root / relative_path
    has_image = int(full_path.exists())
    return str(relative_path).replace("\\", "/"), has_image


def load_articles(articles_path: Path) -> pd.DataFrame:
    df = pd.read_csv(articles_path, dtype={"article_id": str, "product_code": str})
    df["article_id"] = df["article_id"].map(_normalize_article_id)
    df["product_code"] = df["product_code"].astype(str).str.strip()

    # detail_desc 결측 방어
    if "detail_desc" in df.columns:
        df["detail_desc"] = df["detail_desc"].fillna("").astype(str)
    else:
        df["detail_desc"] = ""

    return df


def load_transactions(transactions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        transactions_path,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )
    df["article_id"] = df["article_id"].map(_normalize_article_id)

    # price는 numeric 강제
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def build_transaction_price_stats(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    article_id별 가격 및 구매 통계를 만든다.
    """
    grouped = transactions_df.groupby("article_id", as_index=False).agg(
        price=("price", "median"),
        price_mean=("price", "mean"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        purchase_count=("price", "size"),
        last_purchase_date=("t_dat", "max"),
    )

    sales_channel_mode = (
        transactions_df.groupby("article_id")["sales_channel_id"]
        .apply(_safe_mode)
        .reset_index(name="sales_channel_mode")
    )

    stats = grouped.merge(sales_channel_mode, on="article_id", how="left")
    stats["price_source"] = "transaction_median"
    return stats


def build_fallback_prices(articles_df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    """
    transaction이 없는 article에 대해 단계별 fallback 가격을 채운다.

    우선순위:
    1) 같은 product_type_name의 median price
    2) 같은 garment_group_name의 median price
    3) 같은 product_group_name의 median price
    4) 전체 global median
    """
    merged = articles_df.merge(
        product_df[["article_id", "price"]],
        on="article_id",
        how="left",
    )

    type_price = (
        merged.groupby("product_type_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_product_type")
    )
    garment_price = (
        merged.groupby("garment_group_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_garment_group")
    )
    product_group_price = (
        merged.groupby("product_group_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_product_group")
    )

    global_price = merged["price"].median()

    fallback_df = (
        articles_df.merge(type_price, on="product_type_name", how="left")
        .merge(garment_price, on="garment_group_name", how="left")
        .merge(product_group_price, on="product_group_name", how="left")
    )

    fallback_df["fallback_price"] = (
        fallback_df["fallback_price_product_type"]
        .fillna(fallback_df["fallback_price_garment_group"])
        .fillna(fallback_df["fallback_price_product_group"])
        .fillna(global_price)
    )

    fallback_df["fallback_source"] = fallback_df["fallback_price_product_type"].notna().map(
        {True: "product_type_median", False: None}
    )
    mask = fallback_df["fallback_source"].isna() & fallback_df["fallback_price_garment_group"].notna()
    fallback_df.loc[mask, "fallback_source"] = "garment_group_median"

    mask = fallback_df["fallback_source"].isna() & fallback_df["fallback_price_product_group"].notna()
    fallback_df.loc[mask, "fallback_source"] = "product_group_median"

    fallback_df["fallback_source"] = fallback_df["fallback_source"].fillna("global_median")

    return fallback_df[
        ["article_id", "fallback_price", "fallback_source"]
    ]


def preprocess_hm_products(
    articles_path: Path,
    transactions_path: Path,
    output_path: Path,
    images_root: Optional[Path] = None,
) -> pd.DataFrame:
    articles_df = load_articles(articles_path)
    transactions_df = load_transactions(transactions_path)

    tx_stats_df = build_transaction_price_stats(transactions_df)
    fallback_df = build_fallback_prices(articles_df, tx_stats_df)

    merged = articles_df.merge(tx_stats_df, on="article_id", how="left")
    merged = merged.merge(fallback_df, on="article_id", how="left")

    # price fallback 적용
    missing_price_mask = merged["price"].isna()
    merged.loc[missing_price_mask, "price"] = merged.loc[missing_price_mask, "fallback_price"]
    merged.loc[missing_price_mask, "price_source"] = merged.loc[missing_price_mask, "fallback_source"]

    # 거래가 없는 상품의 부가 통계 기본값
    merged["price_mean"] = merged["price_mean"].fillna(merged["price"])
    merged["price_min"] = merged["price_min"].fillna(merged["price"])
    merged["price_max"] = merged["price_max"].fillna(merged["price"])
    merged["purchase_count"] = merged["purchase_count"].fillna(0).astype(int)

    # category 매핑
    # H&M 원본 구조를 우리 simulator에서 쓰기 쉬운 형태로 단순화
    merged["top_category"] = merged["index_group_name"].fillna("").astype(str)
    merged["mid_category"] = merged["index_name"].fillna("").astype(str)
    merged["leaf_category"] = merged["product_type_name"].fillna("").astype(str)

    # 색상
    merged["color"] = merged["colour_group_name"].fillna("").astype(str)

    # name / description
    merged["name"] = merged["prod_name"].fillna("").astype(str)
    merged["description"] = merged["detail_desc"].fillna("").astype(str)

    # image_path / has_image
    image_results = merged["article_id"].map(
        lambda article_id: _build_image_path(article_id, images_root)
    )
    merged["image_path"] = image_results.map(lambda x: x[0])
    merged["has_image"] = image_results.map(lambda x: x[1]).astype(int)

    # source
    merged["source"] = "hm"

    # 최종 컬럼
    final_df = merged[
        [
            "article_id",
            "product_code",
            "name",
            "description",
            "top_category",
            "mid_category",
            "leaf_category",
            "product_group_name",
            "garment_group_name",
            "department_name",
            "color",
            "price",
            "price_source",
            "price_mean",
            "price_min",
            "price_max",
            "purchase_count",
            "last_purchase_date",
            "sales_channel_mode",
            "image_path",
            "has_image",
            "source",
        ]
    ].copy()

    final_df = final_df.rename(
        columns={
            "article_id": "product_id",
        }
    )

    # 정렬: 구매 많이 된 상품 우선, 그다음 product_id
    final_df = final_df.sort_values(
        by=["purchase_count", "product_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess H&M articles + transactions into one products master file."
    )
    parser.add_argument(
        "--articles",
        type=Path,
        required=True,
        help="Path to articles.csv",
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        required=True,
        help="Path to transactions_train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/hm/processed/hm_products_master.csv"),
        help="Output path for processed products master CSV",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional root path to H&M images directory",
    )

    args = parser.parse_args()

    final_df = preprocess_hm_products(
        articles_path=args.articles,
        transactions_path=args.transactions,
        output_path=args.output,
        images_root=args.images_root,
    )

    print("[H&M preprocess complete]")
    print(f"rows: {len(final_df):,}")
    print(f"output: {args.output}")
    print("\n[Price source]")
    print(final_df["price_source"].value_counts(dropna=False))
    print("\n[Has image]")
    print(final_df["has_image"].value_counts(dropna=False))
    print("\n[Top category sample]")
    print(final_df["top_category"].value_counts().head(10))


if __name__ == "__main__":
    main()