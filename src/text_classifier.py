import re
import numpy as np
from typing import List, Tuple
from src.models import TextAnalysisResult

class TextSafetyClassifier:
    """
    NLP & Heuristic Text Safety Analysis Model.
    Computes toxicity, scam/fraud propensity, and keyword violations.
    """

    FRAUD_KEYWORDS = {
        "guaranteed return", "free money", "wire transfer", "crypto payout", 
        "double your money", "no risk investment", "unclaimed prize", "urgent transfer"
    }

    TOXICITY_KEYWORDS = {
        "hate", "scam", "idiot", "attack", "kill", "threat", "violence", "fraudulent"
    }

    def __init__(self):
        # Simulated vocabulary-based TF-IDF weight vector for lightweight scoring
        self.risk_weights = {
            "money": 0.25,
            "cash": 0.3,
            "winner": 0.4,
            "urgent": 0.35,
            "click": 0.2,
            "free": 0.3
        }

    def analyze(self, text: str) -> TextAnalysisResult:
        normalized = text.lower()
        flagged_words = []

        # 1. Regex & Pattern Matching for Fraud
        fraud_hits = 0
        for pattern in self.FRAUD_KEYWORDS:
            if pattern in normalized:
                fraud_hits += 1
                flagged_words.append(pattern)

        # 2. Toxicity analysis
        tokens = re.findall(r"\b\w+\b", normalized)
        toxic_hits = 0
        weight_score = 0.0

        for token in tokens:
            if token in self.TOXICITY_KEYWORDS:
                toxic_hits += 1
                flagged_words.append(token)
            if token in self.risk_weights:
                weight_score += self.risk_weights[token]

        # Calculate normalized scores (0.0 to 1.0)
        toxicity_score = min(1.0, (toxic_hits * 0.4) + (0.1 if toxic_hits > 0 else 0.0))
        fraud_score = min(1.0, (fraud_hits * 0.45) + min(0.4, weight_score * 0.15))

        sentiment = "NEGATIVE" if (toxicity_score > 0.4 or fraud_score > 0.4) else "NEUTRAL"
        if toxicity_score < 0.2 and fraud_score < 0.2:
            sentiment = "POSITIVE"

        return TextAnalysisResult(
            toxicity_score=round(toxicity_score, 3),
            fraud_score=round(fraud_score, 3),
            flagged_keywords=list(set(flagged_words)),
            sentiment=sentiment
        )
