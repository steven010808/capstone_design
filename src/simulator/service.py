from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import pandas as pd

from src.common.config import load_config
from src.simulator.personas import load_persona_profiles, sample_persona_name
from src.simulator.catalog import (
    generate_products,
    build_top_category_reference_prices,
    get_brand_tier,
)


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("Distribution total must be positive.")
    return {key: value / total for key, value in distribution.items()}


def build_price_based_distribution(
    top_category_reference_prices: dict[str, float],
    affordability_alpha: float,
) -> dict[str, float]:
    raw_scores = {
        category: 1.0 / (reference_price ** affordability_alpha)
        for category, reference_price in top_category_reference_prices.items()
    }
    return normalize_distribution(raw_scores)


def build_prior_distribution(
    category_prior: dict[str, float],
    top_categories: list[str],
) -> dict[str, float]:
    if not category_prior:
        uniform = 1.0 / len(top_categories)
        return {category: uniform for category in top_categories}

    filled = {category: float(category_prior.get(category, 0.0)) for category in top_categories}
    return normalize_distribution(filled)


def build_brand_anchor_distribution(
    favorite_brand: str,
    brand_category_affinity: dict[str, dict[str, float]],
    top_categories: list[str],
) -> dict[str, float]:
    affinity = brand_category_affinity[favorite_brand]
    filled = {category: float(affinity.get(category, 0.0)) for category in top_categories}
    return normalize_distribution(filled)


def apply_noise_to_distribution(
    distribution: dict[str, float],
    rng: random.Random,
    noise_cfg: dict[str, Any],
) -> dict[str, float]:
    mode = str(noise_cfg.get("mode", "blend_uniform"))
    strength = float(noise_cfg.get("strength", 0.0))

    if strength <= 0:
        return distribution

    categories = list(distribution.keys())

    if mode == "blend_uniform":
        random_vector = {category: rng.random() + 1e-6 for category in categories}
        random_distribution = normalize_distribution(random_vector)

        blended = {
            category: (1.0 - strength) * distribution[category]
            + strength * random_distribution[category]
            for category in categories
        }
        return normalize_distribution(blended)

    raise ValueError(f"Unsupported noise mode: {mode}")


def get_top_categories(
    distribution: dict[str, float],
    top_k: int,
) -> list[str]:
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    return [category for category, _ in ranked[:top_k]]


def adjust_budget_for_brand_tier(
    budget_min: int,
    budget_max: int,
    brand_tier: str,
    budget_adjustment_cfg: dict[str, Any],
) -> tuple[int, int]:
    cfg = budget_adjustment_cfg[brand_tier]

    adjusted_min = int(max(
        budget_min * float(cfg["min_multiplier"]),
        int(cfg["min_floor"]),
    ))
    adjusted_max = int(max(
        budget_max * float(cfg["max_multiplier"]),
        int(cfg["max_floor"]),
    ))

    if adjusted_min > adjusted_max:
        adjusted_min, adjusted_max = adjusted_max, adjusted_min

    return adjusted_min, adjusted_max


def sample_favorite_brand(
    favorite_brand_pool: list[str],
    rng: random.Random,
    sampling_cfg: dict[str, Any],
) -> str:
    mode = str(sampling_cfg.get("mode", "uniform"))

    if mode == "uniform":
        return rng.choice(favorite_brand_pool)

    raise ValueError(f"Unsupported favorite brand sampling mode: {mode}")


def get_simulator_status() -> dict[str, Any]:
    config = load_config()
    sim_cfg = config["simulator"]

    return {
        "status": "day2_ready",
        "message": "Simulator Day 2 structure is ready.",
        "output_format": sim_cfg["output_format"],
        "scale": sim_cfg["scale"],
        "target_scale": sim_cfg["target_scale"],
    }


def generate_users(config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    simulator_cfg = config["simulator"]
    profiles = load_persona_profiles(config)
    user_count = int(simulator_cfg["scale"]["users"])

    catalog_cfg = simulator_cfg["catalog"]
    top_category_reference_prices = build_top_category_reference_prices(config)
    top_categories = list(top_category_reference_prices.keys())
    brand_category_affinity = catalog_cfg["brand_category_affinity"]
    noise_cfg = simulator_cfg["preference_noise"]
    budget_adjustment_cfg = simulator_cfg["budget_adjustment_by_tier"]
    top_k_categories = int(simulator_cfg["user_preference"]["top_k_categories"])
    favorite_brand_sampling_cfg = simulator_cfg["favorite_brand_sampling"]

    users: list[dict[str, Any]] = []

    for idx in range(user_count):
        persona_name = sample_persona_name(profiles, rng)
        profile = profiles[persona_name]

        favorite_brand = ""
        favorite_brand_tier = ""
        budget_min = profile.budget_min
        budget_max = profile.budget_max

        if profile.distribution_mode == "price_based":
            base_distribution = build_price_based_distribution(
                top_category_reference_prices=top_category_reference_prices,
                affordability_alpha=float(profile.affordability_alpha or 1.0),
            )
        elif profile.distribution_mode == "prior_based":
            base_distribution = build_prior_distribution(
                category_prior=profile.category_prior,
                top_categories=top_categories,
            )
        elif profile.distribution_mode == "brand_anchor":
            if not profile.favorite_brand_pool:
                raise ValueError(
                    f"Persona '{profile.name}' requires favorite_brand_pool for brand_anchor mode."
                )

            favorite_brand = sample_favorite_brand(
                favorite_brand_pool=profile.favorite_brand_pool,
                rng=rng,
                sampling_cfg=favorite_brand_sampling_cfg,
            )
            favorite_brand_tier = get_brand_tier(config, favorite_brand)

            budget_min, budget_max = adjust_budget_for_brand_tier(
                budget_min=budget_min,
                budget_max=budget_max,
                brand_tier=favorite_brand_tier,
                budget_adjustment_cfg=budget_adjustment_cfg,
            )

            base_distribution = build_brand_anchor_distribution(
                favorite_brand=favorite_brand,
                brand_category_affinity=brand_category_affinity,
                top_categories=top_categories,
            )
        else:
            raise ValueError(
                f"Unsupported distribution_mode for user generation: {profile.distribution_mode}"
            )

        final_distribution = apply_noise_to_distribution(
            distribution=base_distribution,
            rng=rng,
            noise_cfg=noise_cfg,
        )

        top_preferred_categories = get_top_categories(
            distribution=final_distribution,
            top_k=top_k_categories,
        )

        record: dict[str, Any] = {
            "user_id": f"U{idx + 1:06d}",
            "persona": persona_name,
            "price_sensitivity": profile.price_sensitivity,
            "base_conversion_rate": profile.base_conversion_rate,
            "preferred_top_categories": "|".join(top_preferred_categories),
            "preferred_brand_tiers": "|".join(profile.preferred_brand_tiers),
            "favorite_brand": favorite_brand,
            "favorite_brand_tier": favorite_brand_tier,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "signup_days_ago": rng.randint(0, 365),
        }

        for category, value in final_distribution.items():
            record[f"category_pref_{category}"] = round(value, 4)

        users.append(record)

    return users


def save_dataframe(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        df.to_csv(output_path.with_suffix(".csv"), index=False)
    else:
        raise ValueError(f"Unsupported output format for Day 2: {output_format}")


def print_day2_summary(products_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    print("\n[Products by brand_tier]")
    print(products_df["brand_tier"].value_counts())

    print("\n[Average price by brand_tier]")
    print(products_df.groupby("brand_tier")["price"].mean().round(0))

    print("\n[Products by top_category]")
    print(products_df["top_category"].value_counts())

    print("\n[Users by persona]")
    print(users_df["persona"].value_counts())

    print("\n[Average user category preference by persona]")
    preference_columns = [column for column in users_df.columns if column.startswith("category_pref_")]
    grouped = users_df.groupby("persona")[preference_columns].mean().round(3)
    print(grouped)

    print("\n[Brand loyalist favorite brands]")
    loyalists = users_df[users_df["persona"] == "brand_loyalist"]
    if not loyalists.empty:
        print(loyalists["favorite_brand"].value_counts())


def run_day2_sample_generation() -> dict[str, Any]:
    config = load_config()
    rng = random.Random(int(config["app"]["random_seed"]))

    data_dir = Path(config["paths"]["data_dir"])
    output_format = config["simulator"]["output_format"]

    products = generate_products(config, rng)
    users = generate_users(config, rng)

    products_df = pd.DataFrame(products)
    users_df = pd.DataFrame(users)

    save_dataframe(products_df, data_dir / "raw" / "products", output_format)
    save_dataframe(users_df, data_dir / "raw" / "users", output_format)

    print_day2_summary(products_df, users_df)

    return {
        "products": len(products_df),
        "users": len(users_df),
        "output_format": output_format,
        "data_dir": str(data_dir / "raw"),
    }


if __name__ == "__main__":
    result = run_day2_sample_generation()
    print("[Simulator Day2]", result)