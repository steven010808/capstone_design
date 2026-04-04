from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect_hm_catalog(input_path: Path, output_dir: Path | None = None) -> None:
    df = pd.read_csv(input_path)

    required_cols = ["product_id", "top_category", "mid_category", "leaf_category"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["top_category", "mid_category", "leaf_category"]:
        df[col] = df[col].fillna("(missing)").astype(str).str.strip()
        df.loc[df[col] == "", col] = "(missing)"

    top_counts = (
        df.groupby("top_category")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "top_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    mid_counts = (
        df.groupby("mid_category")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "mid_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    leaf_counts = (
        df.groupby("leaf_category")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "leaf_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    top_mid_counts = (
        df.groupby(["top_category", "mid_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["top_category", "count", "mid_category"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    top_leaf_counts = (
        df.groupby(["top_category", "leaf_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["top_category", "count", "leaf_category"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    mid_leaf_counts = (
        df.groupby(["mid_category", "leaf_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["mid_category", "count", "leaf_category"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    print("\n[Catalog summary]")
    print(f"rows: {len(df):,}")
    print(f"unique top_category: {df['top_category'].nunique()}")
    print(f"unique mid_category: {df['mid_category'].nunique()}")
    print(f"unique leaf_category: {df['leaf_category'].nunique()}")

    print("\n[Top category counts]")
    print(top_counts.to_string(index=False))

    print("\n[Mid category counts]")
    print(mid_counts.to_string(index=False))

    print("\n[Leaf category counts]")
    print(leaf_counts.to_string(index=False))

    print("\n[Mid category counts by top category]")
    current_top = None
    for _, row in top_mid_counts.iterrows():
        top = row["top_category"]
        mid = row["mid_category"]
        count = row["count"]

        if top != current_top:
            current_top = top
            print(f"\n## {top}")
        print(f"  - {mid}: {count}")

    print("\n[Leaf category counts by top category]")
    current_top = None
    for _, row in top_leaf_counts.iterrows():
        top = row["top_category"]
        leaf = row["leaf_category"]
        count = row["count"]

        if top != current_top:
            current_top = top
            print(f"\n## {top}")
        print(f"  - {leaf}: {count}")

    print("\n[Leaf category counts by mid category]")
    current_mid = None
    for _, row in mid_leaf_counts.iterrows():
        mid = row["mid_category"]
        leaf = row["leaf_category"]
        count = row["count"]

        if mid != current_mid:
            current_mid = mid
            print(f"\n## {mid}")
        print(f"  - {leaf}: {count}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        top_counts.to_csv(output_dir / "hm_top_category_counts.csv", index=False, encoding="utf-8-sig")
        mid_counts.to_csv(output_dir / "hm_mid_category_counts.csv", index=False, encoding="utf-8-sig")
        leaf_counts.to_csv(output_dir / "hm_leaf_category_counts.csv", index=False, encoding="utf-8-sig")
        top_mid_counts.to_csv(output_dir / "hm_top_mid_category_counts.csv", index=False, encoding="utf-8-sig")
        top_leaf_counts.to_csv(output_dir / "hm_top_leaf_category_counts.csv", index=False, encoding="utf-8-sig")
        mid_leaf_counts.to_csv(output_dir / "hm_mid_leaf_category_counts.csv", index=False, encoding="utf-8-sig")

        print("\n[Saved files]")
        print(output_dir / "hm_top_category_counts.csv")
        print(output_dir / "hm_mid_category_counts.csv")
        print(output_dir / "hm_leaf_category_counts.csv")
        print(output_dir / "hm_top_mid_category_counts.csv")
        print(output_dir / "hm_top_leaf_category_counts.csv")
        print(output_dir / "hm_mid_leaf_category_counts.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect top/mid/leaf category distributions in hm_products_master.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/hm/processed/hm_products_master.csv"),
        help="Path to hm_products_master.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/hm/processed/category_stats"),
        help="Directory to save category summary CSVs",
    )

    args = parser.parse_args()
    inspect_hm_catalog(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()