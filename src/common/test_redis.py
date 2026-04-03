from src.common.redis_client import get_redis_client


def main():
    client = get_redis_client()

    key = "user:U001:recent_clicks"
    value = "item_101,item_202,item_303"

    client.set(key, value)
    loaded = client.get(key)

    print("saved key:", key)
    print("loaded value:", loaded)


if __name__ == "__main__":
    main()