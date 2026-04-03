from src.serving.schemas import SearchRequest, SearchResponse, SearchResultItem


def run_search(request: SearchRequest) -> SearchResponse:
    sample_items = [
        SearchResultItem(
            product_id="P001",
            name="블랙 오버핏 후드티",
            score=0.91,
            price=39900,
        ),
        SearchResultItem(
            product_id="P002",
            name="그레이 와이드 팬츠",
            score=0.87,
            price=45900,
        ),
        SearchResultItem(
            product_id="P003",
            name="화이트 스니커즈",
            score=0.84,
            price=69900,
        ),
        SearchResultItem(
            product_id="P004",
            name="네이비 맨투맨",
            score=0.80,
            price=32900,
        ),
        SearchResultItem(
            product_id="P005",
            name="카키 바람막이",
            score=0.78,
            price=55900,
        ),
    ]

    query = (request.query_text or "").lower()

    if query:
        filtered_items = [item for item in sample_items if query in item.name.lower()]
    else:
        filtered_items = sample_items

    limited_items = filtered_items[: request.top_k]

    return SearchResponse(
        search_type="text",
        results=limited_items,
        latency_ms=12.5,
        total_count=len(filtered_items),
    )