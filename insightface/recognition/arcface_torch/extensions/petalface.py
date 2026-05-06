"""
PETALface: Low-Resolution Adaptation for Face Recognition.

Extension stub for handling resolution mismatch between
gallery (high-res) and probe (low-res) face images.

Concept:
- PETALface is an ADAPTATION method for recognition (NOT a backbone)
- Addresses gallery-probe resolution mismatch
- Learns to map low-resolution probe embeddings to match
  high-resolution gallery embedding space
- Particularly useful for surveillance/CCTV face recognition

Integration point:
- Can be integrated as a post-processing step after embedding extraction
- Or as an auxiliary training objective in train_lightweight.py
- When cfg.use_petalface = True, enable adaptation module

Status: STUB — not yet implemented
"""


class PETALfaceAdapter:
    """PETALface low-resolution adaptation module.

    Usage during training:
        adapter = PETALfaceAdapter(embedding_size=512)
        # In training loop:
        hr_embeddings = backbone(hr_images)
        lr_embeddings = backbone(lr_images)
        adaptation_loss = adapter.compute_loss(hr_embeddings, lr_embeddings)
        total_loss = recognition_loss + alpha * adaptation_loss

    Usage during inference:
        # Adapt probe (low-res) embeddings before matching
        probe_emb = backbone(probe_images)
        adapted_emb = adapter.adapt(probe_emb)
        # Match adapted_emb against gallery
    """

    def __init__(self, embedding_size=512):
        raise NotImplementedError(
            "PETALface extension is not yet implemented. "
            "See docstring for integration guidelines.")

    def compute_loss(self, hr_embeddings, lr_embeddings):
        """Compute adaptation loss between HR and LR embeddings."""
        raise NotImplementedError

    def adapt(self, lr_embeddings):
        """Adapt low-resolution embeddings to HR embedding space."""
        raise NotImplementedError
