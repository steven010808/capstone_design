from pathlib import Path
import yaml

_CONFIG_CACHE = None


def load_config():
    global _CONFIG_CACHE

    if _CONFIG_CACHE is None:
        base_dir = Path(__file__).resolve().parents[2]
        config_path = base_dir / "configs" / "config.yaml"

        with config_path.open("r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE