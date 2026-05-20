import requests
from typing import Optional
from DTO.SimpleArticleLabelDTO import SimpleArticleLabelDTO
from DTO.ArticleLlmResponse import ArticleLlmResponse


class ApiCall:
    """Handles all HTTP communication with the article management server."""

    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize the API client with the server base URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_unlabeled_articles(self, batch_size: int = 100) -> list[SimpleArticleLabelDTO]:
        """Fetch a batch of unlabeled articles from the server. Returns an empty list on failure."""
        url = f"{self.base_url}/api/v1/articles/unlabeled"
        try:
            response = self.session.get(url, params={"batch_size": batch_size}, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ApiCall] Failed to fetch unlabeled articles: {e}")
            return []

        try:
            return [SimpleArticleLabelDTO(**item) for item in response.json()]
        except Exception as e:
            print(f"[ApiCall] Failed to parse articles response: {e}")
            return []

    def update_article_labels(self, results: list[ArticleLlmResponse]) -> bool:
        """Send labeling results back to the server. Returns True on success, False on failure."""
        url = f"{self.base_url}/api/v1/articles/labels"
        payload = [result.model_dump() for result in results]
        try:
            response = self.session.put(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"[ApiCall] Failed to update article labels: {e}")
            return False
