"""
ARoFace: Alignment Robustness for Face Recognition.

Extension stub for improving recognition robustness against
alignment errors on low-quality face images.

Concept:
- ARoFace is a training/augmentation strategy (NOT a backbone)
- Adds alignment-aware augmentations during training
- Helps the recognition model be robust to imperfect face alignment
- Particularly useful for low-quality / in-the-wild scenarios

Integration point:
- Hook into training loop in train_lightweight.py
- When cfg.use_aroface = True, apply ARoFace augmentation strategy
- Can be combined with any backbone and any loss function

Status: STUB — not yet implemented
"""


class ARoFaceAugmentation:
    """ARoFace alignment robustness augmentation.

    Apply random alignment perturbations during TRAINING to make
    the recognition model robust to alignment errors at test time.

    This is different from degradation/alignment_perturb which is
    only used during EVALUATION.

    Usage in train_lightweight.py:
        if cfg.use_aroface:
            aroface = ARoFaceAugmentation(cfg)
            img = aroface.augment(img)  # before backbone forward
    """

    def __init__(self, config=None):
        raise NotImplementedError(
            "ARoFace extension is not yet implemented. "
            "See docstring for integration guidelines.")

    def augment(self, images):
        """Apply ARoFace augmentation to a batch of images.

        Args:
            images: torch.Tensor (B, 3, H, W)
        Returns:
            augmented images: torch.Tensor (B, 3, H, W)
        """
        raise NotImplementedError
