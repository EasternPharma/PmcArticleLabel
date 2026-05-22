import json
import re
from openai import OpenAI
from tqdm import tqdm
from DTO.SimpleArticleLabelDTO import SimpleArticleLabelDTO
from DTO.ArticleLlmResponse import ArticleLlmResponse

_SYSTEM_PROMPT = """#Role#
You are a specialized biomedical literature classifier. You determine whether scientific articles fall within the scope of Human Complementary and Alternative Medicine (CAM) or Human Nutrition.

#Definitions#
===
WHITE – Clearly IN scope: the article studies a human application of CAM or nutrition.
BLACK – Clearly OUT of scope: no meaningful connection to human CAM or nutrition.
GRAY  – Borderline: mixed signals, animal-only studies with human implications, or genuinely ambiguous relevance.
===

#Inclusion criteria for WHITE#
The article must involve HUMAN subjects (or direct human application intent) AND at least one of:
- Herbal / botanical therapies (ginseng, turmeric, echinacea, etc.)
- Dietary or natural supplements (vitamins, minerals, probiotics, amino acids, omega-3s)
- Functional foods or nutraceuticals (curcumin, polyphenols, bioactive food compounds)
- Traditional medicine systems (Ayurveda, TCM, Unani, Naturopathy)
- Essential oils, medicinal mushrooms, sports nutrition
- Any naturally-derived health intervention NOT classified as a pharmaceutical drug

#Automatic BLACK signals#
- Pure animal or in-vitro study with no stated human translation intent
- Industrial / agricultural / veterinary applications only
- Pharmaceutical drug trials (synthetic molecules, biologics)
- Basic biochemistry with no health intervention angle

#Task steps#
1. Identify whether the article involves HUMAN subjects or direct human applications.
2. Check whether the intervention or substance qualifies under the inclusion criteria above.
3. If both are true → WHITE. If neither → BLACK. If uncertain on either → GRAY.
4. Assign a confidence_score: how certain are you of your label (0.0 = total uncertainty, 1.0 = certain).

#Output format#
Respond ONLY with valid JSON. No preamble, no markdown fences, no extra text.
{
    "label": "WHITE" | "BLACK" | "GRAY",
    "reason": "<one sentence, cite the key deciding factor>",
    "confidence_score": <float 0.0–1.0>
}
"""


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

    def _parse_response(self, pmc_id: int, raw_content: str) -> ArticleLlmResponse:
        """Extract and parse the JSON answer into an ArticleLlmResponse. Returns Label=0 on parse failure."""
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
                LlmModel=self.model_name,
                Reasoning=reason,
            )
        except Exception as e:
            print(f"[ArticleLabelHelper] Failed to parse response for PmcId={pmc_id}: {e}")
            return ArticleLlmResponse(
                PmcId=pmc_id,
                Label=0,
                Confidence=0.0,
                LlmModel=self.model_name,
                Reasoning=f"Parse error: {e}",
            )

    def label_article(self, article: SimpleArticleLabelDTO) -> ArticleLlmResponse | None:
        """Send a single article to the vLLM model and return the parsed label result. Returns None on network/API error."""
        prompt = self.build_prompt(article)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            message = response.choices[0].message
            raw_content = (message.content or "") or (getattr(message, "reasoning_content", None) or "")
            return self._parse_response(article.PmcId, raw_content)
        except Exception as e:
            print(f"[ArticleLabelHelper] vLLM call failed for PmcId={article.PmcId}: {e}")
            return None

    def label_batch(self, articles: list[SimpleArticleLabelDTO]) -> list[ArticleLlmResponse]:
        """Label a list of articles sequentially, showing a progress bar. Skips only on network/API failure."""
        results: list[ArticleLlmResponse] = []
        for article in tqdm(articles, desc="Labeling articles", unit="article", leave=False):
            result = self.label_article(article)
            if result is not None:  # None means vLLM call itself failed — skip those
                results.append(result)
        return results
