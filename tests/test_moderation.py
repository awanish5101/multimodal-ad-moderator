import pytest
from src.text_classifier import TextSafetyClassifier
from src.image_extractor import ImageSafetyExtractor
from src.decision_engine import DecisionEngine
from src.models import ModerationRequest, ModerationAction, ContentCategory

@pytest.fixture
def classifier():
    return TextSafetyClassifier()

@pytest.fixture
def extractor():
    return ImageSafetyExtractor()

@pytest.fixture
def engine():
    return DecisionEngine()

def test_clean_text_analysis(classifier):
    res = classifier.analyze("Discover the new summer collection with 20% discount on clothing")
    assert res.toxicity_score < 0.2
    assert res.fraud_score < 0.2
    assert res.sentiment == "POSITIVE"
    assert len(res.flagged_keywords) == 0

def test_fraud_text_detection(classifier):
    res = classifier.analyze("Guaranteed return on wire transfer! Double your money with no risk investment today!")
    assert res.fraud_score >= 0.5
    assert "guaranteed return" in res.flagged_keywords
    assert "wire transfer" in res.flagged_keywords
    assert res.sentiment == "NEGATIVE"

def test_image_safety_mock(extractor):
    res_clean = extractor.extract_from_mock("normal")
    assert res_clean.explicit_content_score < 0.2
    assert res_clean.violence_score < 0.2

    res_nsfw = extractor.extract_from_mock("explicit")
    assert res_nsfw.explicit_content_score > 0.8
    assert "nsfw_flagged" in res_nsfw.detected_labels

def test_end_to_end_approval(engine):
    req = ModerationRequest(
        ad_id="AD-001",
        headline="Eco-friendly insulated water bottles",
        body_text="BPA free stainless steel bottles available in multiple colors."
    )
    decision = engine.evaluate(req)
    assert decision.overall_action == ModerationAction.APPROVE
    assert decision.overall_risk_score < 30.0
    assert ContentCategory.SAFE in decision.violation_categories

def test_end_to_end_rejection(engine):
    req = ModerationRequest(
        ad_id="AD-999",
        headline="Urgent crypto payout and free money guaranteed return",
        body_text="Send wire transfer now for instant double payout guaranteed return"
    )
    decision = engine.evaluate(req)
    assert decision.overall_action in [ModerationAction.REJECT, ModerationAction.FLAG_FOR_REVIEW]
    assert decision.overall_risk_score >= 50.0
    assert ContentCategory.FRAUD_SCAM in decision.violation_categories
