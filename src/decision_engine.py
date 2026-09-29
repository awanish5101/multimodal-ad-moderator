from typing import List, Tuple
from src.models import (
    ModerationRequest,
    ModerationResponse,
    ModerationAction,
    ContentCategory,
    TextAnalysisResult,
    ImageAnalysisResult
)
from src.text_classifier import TextSafetyClassifier
from src.image_extractor import ImageSafetyExtractor

class DecisionEngine:
    """
    Multimodal Policy Decision Engine.
    Combines text & image model scores to yield aggregated risk decisions.
    """

    def __init__(self):
        self.text_model = TextSafetyClassifier()
        self.image_model = ImageSafetyExtractor()

    def evaluate(self, request: ModerationRequest) -> ModerationResponse:
        full_text = f"{request.headline} {request.body_text}"
        text_res: TextAnalysisResult = self.text_model.analyze(full_text)

        # Image inference
        image_res = None
        if request.image_base64:
            image_res = self.image_model.extract_from_base64(request.image_base64)
        elif request.image_url:
            image_res = self.image_model.extract_from_mock("normal")

        # Aggregate Risk Assessment
        violations: List[ContentCategory] = []
        risk_components = [
            text_res.toxicity_score * 50.0,
            text_res.fraud_score * 55.0
        ]

        if text_res.toxicity_score >= 0.5:
            violations.append(ContentCategory.TOXICITY_HATE)
        if text_res.fraud_score >= 0.5:
            violations.append(ContentCategory.FRAUD_SCAM)

        if image_res:
            risk_components.append(image_res.explicit_content_score * 40.0)
            risk_components.append(image_res.violence_score * 35.0)

            if image_res.explicit_content_score >= 0.6:
                violations.append(ContentCategory.ADULT_EXPLICIT)
            if image_res.violence_score >= 0.6:
                violations.append(ContentCategory.VIOLENCE)

        overall_risk = min(100.0, sum(risk_components))

        # Decision Thresholds
        if overall_risk >= 65.0 or len(violations) >= 2:
            action = ModerationAction.REJECT
            reason = f"Automated rejection: Policy violations detected in {', '.join([v.value for v in violations])}"
        elif overall_risk >= 35.0 or len(violations) == 1:
            action = ModerationAction.FLAG_FOR_REVIEW
            reason = f"Flagged for human trust & safety review: Elevated risk score ({overall_risk:.1f}/100)"
        else:
            action = ModerationAction.APPROVE
            reason = "Content passed all automated multimodal safety checks."
            violations.append(ContentCategory.SAFE)

        confidence = 0.94 if (overall_risk > 80 or overall_risk < 20) else 0.82

        return ModerationResponse(
            ad_id=request.ad_id,
            overall_action=action,
            overall_risk_score=round(overall_risk, 1),
            violation_categories=violations,
            confidence=confidence,
            text_analysis=text_res,
            image_analysis=image_res,
            decision_reason=reason
        )
