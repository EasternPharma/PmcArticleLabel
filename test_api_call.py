from ApiCall import ApiCall

API_BASE_URL = "http://localhost:1368"
BATCH_SIZE = 5


def test_health_check(client: ApiCall):
    print("--- health_check ---")
    result = client.health_check()
    print(f"  Result: {'OK' if result else 'FAILED'}")
    return result


def test_get_unlabeled_articles(client: ApiCall):
    print(f"\n--- get_unlabeled_articles (batch_size={BATCH_SIZE}) ---")
    articles = client.get_unlabeled_articles(batch_size=BATCH_SIZE)
    print(f"  Returned {len(articles)} article(s).")
    for i, a in enumerate(articles, 1):
        title_preview = (a.Title or "N/A")[:80]
        print(f"  [{i}] PmcId={a.PmcId}  Title={title_preview!r}")
    return articles


if __name__ == "__main__":
    client = ApiCall(base_url=API_BASE_URL)
    test_health_check(client)
    test_get_unlabeled_articles(client)
