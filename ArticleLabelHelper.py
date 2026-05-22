import asyncio
import json
import re
import aiohttp
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
        self.vllm_base_url = vllm_base_url
        self.llm_cal_url = f"{vllm_base_url}/v1/chat/completions"
        print(f"[ArticleLabelHelper] Initialized with vLLM base URL: {vllm_base_url}")
        print(f"[ArticleLabelHelper] LLM call URL: {self.llm_cal_url}")

    def build_prompt(self, article: SimpleArticleLabelDTO) -> str:
        """Build the user-facing prompt text from an article's title and abstract."""
        return (
            _SYSTEM_PROMPT + "\n\n"
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
    
    async def _llm_call(
        self,
        session: aiohttp.ClientSession,
        pmc_id: int,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> ArticleLlmResponse:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with session.post(self.llm_cal_url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
        content = body["choices"][0]["message"].get("content") or ""
        return self._parse_response(pmc_id, content)

    async def run_batch(
        self,
        articles: list[SimpleArticleLabelDTO],
        batch_size: int,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> list[ArticleLlmResponse]:
        results: list[ArticleLlmResponse] = []
        connector = aiohttp.TCPConnector(limit=batch_size)
        timeout = aiohttp.ClientTimeout(total=600)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            pending: set[asyncio.Task[ArticleLlmResponse]] = set()

            for article in articles:
                prompt = self.build_prompt(article)
                task = asyncio.create_task(
                    self._llm_call(session, article.PmcId, prompt, max_tokens, temperature)
                )
                pending.add(task)

                if len(pending) >= batch_size:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        results.append(t.result())

            if pending:
                done, _ = await asyncio.wait(pending)
                for t in done:
                    results.append(t.result())

        return results

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

    def label_batch(self, articles: list[SimpleArticleLabelDTO]) -> list[ArticleLlmResponse]:
        """Label a list of articles in parallel via vLLM and return results."""
        return asyncio.run(self.run_batch(articles, batch_size=len(articles)))
