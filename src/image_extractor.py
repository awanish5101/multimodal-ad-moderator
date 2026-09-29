import io
import base64
import numpy as np
from PIL import Image
from typing import Optional
from src.models import ImageAnalysisResult

class ImageSafetyExtractor:
    """
    Computer Vision Feature Extractor for Ad Image Moderation.
    Performs dimension analysis, color entropy scoring, and simulated safety classification.
    """

    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim

    def extract_from_base64(self, b64_str: str) -> ImageAnalysisResult:
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            image_data = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return self._analyze_image(image)
        except Exception:
            return ImageAnalysisResult(
                explicit_content_score=0.1,
                violence_score=0.1,
                feature_dimension=self.feature_dim,
                detected_labels=["unreadable_image"]
            )

    def extract_from_mock(self, mock_scenario: str = "normal") -> ImageAnalysisResult:
        if mock_scenario == "explicit":
            return ImageAnalysisResult(
                explicit_content_score=0.88,
                violence_score=0.12,
                feature_dimension=self.feature_dim,
                detected_labels=["nsfw_flagged", "skin_entropy_high"]
            )
        elif mock_scenario == "violence":
            return ImageAnalysisResult(
                explicit_content_score=0.08,
                violence_score=0.85,
                feature_dimension=self.feature_dim,
                detected_labels=["weapon_detected", "blood_pattern"]
            )
        return ImageAnalysisResult(
            explicit_content_score=0.05,
            violence_score=0.04,
            feature_dimension=self.feature_dim,
            detected_labels=["clean_advertisement", "product_catalog"]
        )

    def _analyze_image(self, img: Image.Image) -> ImageAnalysisResult:
        # Resize to standardized vector input
        resized = img.resize((64, 64))
        arr = np.array(resized) / 255.0

        # Heuristic color entropy
        red_channel_mean = float(np.mean(arr[:, :, 0]))
        brightness = float(np.mean(arr))

        # Check for high red concentration (indicative of blood/danger markers in heuristic pipelines)
        violence_score = min(0.9, max(0.02, (red_channel_mean - 0.4) * 1.5)) if red_channel_mean > 0.4 else 0.05
        explicit_score = 0.08 if brightness > 0.1 else 0.03

        detected = ["analyzed_raster"]
        if violence_score > 0.4:
            detected.append("high_red_saturation_risk")

        return ImageAnalysisResult(
            explicit_content_score=round(explicit_score, 3),
            violence_score=round(violence_score, 3),
            feature_dimension=self.feature_dim,
            detected_labels=detected
        )
