from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ContentCategory(str, Enum):
    SAFE = "SAFE"
    TOXICITY_HATE = "TOXICITY_HATE"
    FRAUD_SCAM = "FRAUD_SCAM"
    ADULT_EXPLICIT = "ADULT_EXPLICIT"
    VIOLENCE = "VIOLENCE"

class ModerationAction(str, Enum):
    APPROVE = "APPROVE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    REJECT = "REJECT"

class TextAnalysisResult(BaseModel):
    toxicity_score: float = Field(..., ge=0.0, le=1.0)
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    flagged_keywords: List[str] = Field(default_factory=list)
    sentiment: str = "NEUTRAL"

class ImageAnalysisResult(BaseModel):
    explicit_content_score: float = Field(..., ge=0.0, le=1.0)
    violence_score: float = Field(..., ge=0.0, le=1.0)
    feature_dimension: int = 128
    detected_labels: List[str] = Field(default_factory=list)

class ModerationRequest(BaseModel):
    ad_id: str
    headline: str
    body_text: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class ModerationResponse(BaseModel):
    ad_id: str
    overall_action: ModerationAction
    overall_risk_score: float = Field(..., ge=0.0, le=100.0)
    violation_categories: List[ContentCategory] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_analysis: TextAnalysisResult
    image_analysis: Optional[ImageAnalysisResult] = None
    decision_reason: str
