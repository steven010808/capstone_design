from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any
import random


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("Distribution total must be positive.")
    return {key: value / total for key, value in distribution.items()}


@dataclass
class ProductRecord:
    product_id: str
    name: str
    brand: str
    brand_tier: str
    top_category: str
    mid_category: str
    leaf_category: str
    color: str
    style_tag: str
    base_price: int
    price: int
    discount_rate: float
    is_new: bool
    popularity_seed: float
    created_at: str
    description: str


def build_top_category_tree(
    structure: dict[str, dict[str, list[str]]],
) -> dict[str, list[tuple[str, str]]]:
    tree: dict[str, list[tuple[str, str]]] = {}
    for top_category, mid_map in structure.items():
        tree[top_category] = []
        for mid_category, leaf_categories in mid_map.items():
            for leaf_category in leaf_categories:
                tree[top_category].append((mid_category, leaf_category))
    return tree


def build_top_category_reference_prices(config: dict[str, Any]) -> dict[str, float]:
    catalog_cfg = config["simulator"]["catalog"]
    structure = catalog_cfg["structure"]
    leaf_price_map = catalog_cfg["leaf_category_base_price"]

    references: dict[str, float] = {}

    for top_category, mid_map in structure.items():
        leaf_reference_prices: list[float] = []
        for leaf_categories in mid_map.values():
            for leaf_category in leaf_categories:
                price_cfg = leaf_price_map[leaf_category]
                ref_price = 0.4 * float(price_cfg["min"]) + 0.6 * float(price_cfg["max"])
                leaf_reference_prices.append(ref_price)

        references[top_category] = sum(leaf_reference_prices) / len(leaf_reference_prices)

    return references


def get_brand_tier(config: dict[str, Any], brand: str) -> str:
    brand_meta = config["simulator"]["catalog"]["brand_catalog"][brand]
    return str(brand_meta["tier"])


def sample_brand(
    brand_catalog: dict[str, Any],
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    brand_names = list(brand_catalog.keys())
    weights = [float(brand_catalog[name]["sampling_weight"]) for name in brand_names]
    brand_name = rng.choices(brand_names, weights=weights, k=1)[0]
    return brand_name, brand_catalog[brand_name]


def sample_top_category_for_brand(
    brand: str,
    brand_category_affinity: dict[str, dict[str, float]],
    rng: random.Random,
) -> str:
    affinity = normalize_distribution(
        {k: float(v) for k, v in brand_category_affinity[brand].items()}
    )
    categories = list(affinity.keys())
    weights = [affinity[category] for category in categories]
    return rng.choices(categories, weights=weights, k=1)[0]


def sample_leaf_under_top(
    top_category: str,
    top_category_tree: dict[str, list[tuple[str, str]]],
    rng: random.Random,
) -> tuple[str, str]:
    return rng.choice(top_category_tree[top_category])


def sample_base_price(
    leaf_category: str,
    base_price_map: dict[str, dict[str, int]],
    rng: random.Random,
) -> int:
    price_cfg = base_price_map[leaf_category]
    return rng.randrange(int(price_cfg["min"]), int(price_cfg["max"]) + 1, 1000)


def generate_products(config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    simulator_cfg = config["simulator"]
    catalog_cfg = simulator_cfg["catalog"]
    product_count = int(simulator_cfg["scale"]["products"])

    structure = catalog_cfg["structure"]
    top_category_tree = build_top_category_tree(structure)
    brand_catalog = catalog_cfg["brand_catalog"]
    base_price_map = catalog_cfg["leaf_category_base_price"]
    brand_category_affinity = catalog_cfg["brand_category_affinity"]

    colors = list(catalog_cfg["colors"])
    style_tags = list(catalog_cfg["style_tags"])

    now = datetime.utcnow()
    records: list[dict[str, Any]] = []

    for idx in range(product_count):
        brand, brand_meta = sample_brand(brand_catalog, rng)
        brand_tier = str(brand_meta["tier"])

        top_category = sample_top_category_for_brand(
            brand=brand,
            brand_category_affinity=brand_category_affinity,
            rng=rng,
        )
        mid_category, leaf_category = sample_leaf_under_top(
            top_category=top_category,
            top_category_tree=top_category_tree,
            rng=rng,
        )

        color = rng.choice(colors)
        style_tag = rng.choice(style_tags)

        base_price = sample_base_price(
            leaf_category=leaf_category,
            base_price_map=base_price_map,
            rng=rng,
        )

        multiplier = rng.uniform(
            float(brand_meta["multiplier_min"]),
            float(brand_meta["multiplier_max"]),
        )

        days_ago = rng.randint(0, 180)
        created_dt = now - timedelta(days=days_ago)
        is_new = days_ago <= 7

        discount_rate = round(rng.uniform(0.0, float(brand_meta["discount_max"])), 2)
        popularity_seed = round(rng.uniform(0.0, 1.0), 4)

        raw_price = base_price * multiplier
        discounted_price = int(round(raw_price * (1 - discount_rate), -3))
        price = max(10000, discounted_price)

        name = f"{brand} {color} {leaf_category}"
        description = (
            f"{top_category}/{mid_category}/{leaf_category} 카테고리의 "
            f"{style_tag} 스타일 상품"
        )

        record = ProductRecord(
            product_id=f"P{idx + 1:06d}",
            name=name,
            brand=brand,
            brand_tier=brand_tier,
            top_category=top_category,
            mid_category=mid_category,
            leaf_category=leaf_category,
            color=color,
            style_tag=style_tag,
            base_price=base_price,
            price=price,
            discount_rate=discount_rate,
            is_new=is_new,
            popularity_seed=popularity_seed,
            created_at=created_dt.isoformat(),
            description=description,
        )
        records.append(asdict(record))

    return records