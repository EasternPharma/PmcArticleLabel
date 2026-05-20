import json
import re
from openai import OpenAI
from tqdm import tqdm
from DTO.SimpleArticleLabelDTO import SimpleArticleLabelDTO
from DTO.ArticleLlmResponse import ArticleLlmResponse

_SYSTEM_PROMPT = """You are a specialized AI trained to classify biomedical articles based on their relevance to Human Complementary and Alternative Medicine (CAM) and Human Nutrition.

Objective:
Analyze the provided article and assign it one of the following categories:

- "WHITE" – Clearly within the scope of human CAM or human nutrition.
- "BLACK" – Clearly outside the scope.
- "GRAY" – Borderline, unclear, or ambiguous relevance.

INCLUSION CRITERIA ("WHITE"):
An article qualifies as "WHITE" if it involves **human applications** of any of the following:
- Herbal medicine or botanical therapies (e.g., ginseng, turmeric)
- Natural or dietary supplements (vitamins, probiotics, amino acids, minerals)
- Functional foods or nutraceuticals with bioactive compounds (e.g., curcumin)
- Traditional medicine systems (e.g., Ayurveda, Chinese medicine)
- Essential oils, medicinal mushrooms, sports nutrition products
- Any naturally derived health intervention **not classified as a pharmaceutical**

OUTPUT FORMAT:
Respond only with valid JSON in the following format, no extra text:
{
    "label": "WHITE" | "BLACK" | "GRAY",
    "reason": "1 sentence explanation",
    "confidence_score": float between 0.0 and 1.0
}"""


class ArticleLabelHelper:
    """Handles prompt construction, vLLM inference, and response parsing for article labeling."""

    def __init__(self, vllm_base_url: str, model_name: str):
        """Initialize the vLLM client with the server URL and model name."""
        self.model_name = model_name
        self.client = OpenAI(base_url=f"{vllm_base_url}/v1", api_key="not-required")

    def build_prompt(self, article: SimpleArticleLabelDTO) -> str:
        """Build the user-facing prompt text from an article's title and abstract."""
        return (
            f"Title: {article.Title or 'N/A'}\n\n"
            f"Abstract:\n{article.AbstractText or 'N/A'}"
        )

    def _extract_json(self, text: str) -> str:
        """
        Extract a JSON object from text that may contain surrounding content.
        When vLLM reasoning-parser is active, message.content is already clean JSON.
        This fallback handles cases where extra text appears around the JSON block.
        """
        text = text.strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _parse_response(self, pmc_id: int, raw_content: str) -> ArticleLlmResponse | None:
        """Extract and parse the JSON answer into an ArticleLlmResponse. Returns None if parsing fails."""
        try:
            clean = self._extract_json(raw_content)
            data = json.loads(clean)

            raw_label = data.get("label", "").upper().strip()
            label_map = {"WHITE": 1, "BLACK": 2, "GRAY": 3}
            label = label_map.get(raw_label, 0)

            reason = data.get("reason") or data.get("reasoning")

            raw_confidence = data.get("confidence_score")
            confidence = 0.0
            if raw_confidence is not None:
                try:
                    confidence = float(raw_confidence)
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.0

            return ArticleLlmResponse(
                PmcId=pmc_id,
                Label=label,
                Confidence=confidence,
                Reasoning=reason,
            )
        except Exception as e:
            print(f"[ArticleLabelHelper] Failed to parse response for PmcId={pmc_id}: {e}")
            return None

    def label_article(self, article: SimpleArticleLabelDTO) -> ArticleLlmResponse | None:
        """Send a single article to the vLLM model and return the parsed label result. Returns None on error."""
        prompt = self.build_prompt(article)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=512,
            )
            raw_content = response.choices[0].message.content
            return self._parse_response(article.PmcId, raw_content)
        except Exception as e:
            print(f"[ArticleLabelHelper] vLLM call failed for PmcId={article.PmcId}: {e}")
            return None

    def label_batch(self, articles: list[SimpleArticleLabelDTO]) -> list[ArticleLlmResponse]:
        """Label a list of articles sequentially, showing a progress bar. Skips articles that fail."""
        results: list[ArticleLlmResponse] = []
        for article in tqdm(articles, desc="Labeling articles", unit="article", leave=False):
            result = self.label_article(article)
            if result is not None:
                results.append(result)
        return results
