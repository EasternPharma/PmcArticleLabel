from typing import Optional
from pydantic import BaseModel


class ArticleLlmResponse(BaseModel):
    """Holds the structured labeling result for a single article produced by the LLM."""

    PmcId: int
    Label: int          # 1=WHITE, 2=BLACK, 3=GRAY, 0=unknown
    Confidence: float   # Normalized to [0.0, 1.0]
    Reasoning: Optional[str] = None
