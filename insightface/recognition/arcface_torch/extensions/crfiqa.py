"""
CR-FIQA: Certainty Ratio Face Image Quality Assessment.

Extension stub for quality assessment / quality filtering
of face embeddings.

Concept:
- CR-FIQA is a POST-EMBEDDING quality assessment method (NOT a backbone)
- Estimates quality score from recognition model embeddings
- Can be used to filter low-quality images or weight embeddings
  during template aggregation / matching

Integration point:
- After embedding extraction in eval_degraded.py or inference
- Quality scores can be used for:
  - Filtering: reject samples below quality threshold
  - Weighting: weight embeddings by quality in template aggregation
  - Analysis: correlate quality with recognition accuracy

Status: STUB — not yet implemented
"""


class CRFIQAQualityEstimator:
    """CR-FIQA quality estimation from face embeddings.

    Usage:
        estimator = CRFIQAQualityEstimator(backbone)
        quality_scores = estimator.estimate(images)
        # Use scores for filtering or weighting
    """

    def __init__(self, backbone=None):
        raise NotImplementedError(
            "CR-FIQA extension is not yet implemented. "
            "See docstring for integration guidelines.")

    def estimate(self, images):
        """Estimate quality scores for a batch of face images.

        Args:
            images: torch.Tensor (B, 3, H, W)
        Returns:
            quality_scores: torch.Tensor (B,) in [0, 1]
        """
        raise NotImplementedError

    def filter_by_quality(self, embeddings, quality_scores, threshold=0.3):
        """Filter embeddings by quality threshold.

        Args:
            embeddings: np.ndarray (N, D)
            quality_scores: np.ndarray (N,)
            threshold: minimum quality score
        Returns:
            filtered_embeddings, filtered_indices
        """
        raise NotImplementedError
