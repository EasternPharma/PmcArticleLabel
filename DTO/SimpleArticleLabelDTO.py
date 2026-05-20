from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class SimpleArticleLabelDTO(BaseModel):
    """Represents a single unlabeled article fetched from the server, containing only the fields needed for labeling."""

    PmcId: int
    Title: Optional[str] = None
    AbstractText: Optional[str] = None