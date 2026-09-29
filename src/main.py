from fastapi import FastAPI, HTTPException, status
from src.models import ModerationRequest, ModerationResponse
from src.decision_engine import DecisionEngine

app = FastAPI(
    title="Multimodal AI Ad Safety & Content Moderation API",
    description="Microservice for NLP text safety, computer vision raster extraction, and aggregated policy enforcement.",
    version="1.0.0"
)

engine = DecisionEngine()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "UP",
        "service": "multimodal-ad-moderator",
        "models": ["TextSafetyClassifier-v1", "ImageSafetyExtractor-v1"]
    }

@app.post("/api/v1/moderate", response_model=ModerationResponse, status_code=status.HTTP_200_OK)
def moderate_ad_content(payload: ModerationRequest):
    if not payload.headline and not payload.body_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least headline or body_text must be provided."
        )
    return engine.evaluate(payload)
