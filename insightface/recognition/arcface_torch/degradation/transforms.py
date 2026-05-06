"""
Image degradation transforms for evaluation under low-quality conditions.

All transforms:
- Accept numpy array (H, W, 3) uint8
- Return numpy array (H, W, 3) uint8
- Are deterministic given the same seed
- Support severity levels 1-5
"""

import cv2
import io
import numpy as np

SUPPORTED_DEGRADATIONS = [
    "gaussian_blur",
    "motion_blur",
    "low_resolution",
    "jpeg_compression",
    "low_illumination",
    "alignment_perturb",
]

# Severity parameter tables
_GAUSSIAN_BLUR_SIGMA = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}
_MOTION_BLUR_KERNEL = {1: 3, 2: 5, 3: 9, 4: 13, 5: 15}
_LOW_RES_SIZE = {1: 56, 2: 42, 3: 28, 4: 20, 5: 14}
_JPEG_QUALITY = {1: 75, 2: 50, 3: 30, 4: 15, 5: 10}
_LOW_ILLUM_GAMMA = {1: 1.3, 2: 1.6, 3: 2.0, 4: 2.7, 5: 3.5}
_ALIGN_PERTURB_PX = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def apply_gaussian_blur(image, severity=1):
    """Apply Gaussian blur with severity-dependent sigma."""
    sigma = _GAUSSIAN_BLUR_SIGMA.get(severity, 2.0)
    # Kernel size must be odd, large enough for sigma
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(image, severity=1, rng=None):
    """Apply directional motion blur."""
    kernel_size = _MOTION_BLUR_KERNEL.get(severity, 9)
    if rng is None:
        rng = np.random.default_rng()
    # Random angle for motion direction
    angle = rng.uniform(0, 360)
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    # Draw line through center at given angle
    cos_val = np.cos(np.radians(angle))
    sin_val = np.sin(np.radians(angle))
    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * cos_val))
        y = int(round(center + offset * sin_val))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_low_resolution(image, severity=1):
    """Downsample then upsample back to original size."""
    h, w = image.shape[:2]
    low_size = _LOW_RES_SIZE.get(severity, 28)
    small = cv2.resize(image, (low_size, low_size),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(image, severity=1):
    """Apply JPEG compression artifacts."""
    quality = _JPEG_QUALITY.get(severity, 30)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_low_illumination(image, severity=1):
    """Reduce brightness via gamma correction (gamma > 1 darkens)."""
    gamma = _LOW_ILLUM_GAMMA.get(severity, 2.0)
    # Build lookup table for efficiency
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_alignment_perturb(image, severity=1, rng=None):
    """Apply small random affine perturbation (synthetic alignment error).

    This simulates alignment errors by applying small translation/rotation
    on the already-aligned 112x112 face crop. Does NOT change the aligner.
    """
    if rng is None:
        rng = np.random.default_rng()
    max_px = _ALIGN_PERTURB_PX.get(severity, 3)
    h, w = image.shape[:2]
    # Random translation
    tx = rng.uniform(-max_px, max_px)
    ty = rng.uniform(-max_px, max_px)
    # Small random rotation (proportional to severity)
    angle = rng.uniform(-severity, severity)
    # Affine matrix
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# Degradation function registry
_DEGRADATION_FN = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "low_resolution": apply_low_resolution,
    "jpeg_compression": apply_jpeg_compression,
    "low_illumination": apply_low_illumination,
    "alignment_perturb": apply_alignment_perturb,
}


class DegradationTransform:
    """Configurable, reproducible image degradation for evaluation."""

    def __init__(self, degradation_type, severity=1, seed=42):
        if degradation_type not in SUPPORTED_DEGRADATIONS:
            raise ValueError(
                f"Unknown degradation: {degradation_type}. "
                f"Supported: {SUPPORTED_DEGRADATIONS}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be 1-5, got {severity}")

        self.degradation_type = degradation_type
        self.severity = severity
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fn = _DEGRADATION_FN[degradation_type]

    def apply(self, image):
        """Apply degradation to image (H, W, 3) uint8 numpy array."""
        # Functions that need rng
        if self.degradation_type in ("motion_blur", "alignment_perturb"):
            return self._fn(image, severity=self.severity, rng=self.rng)
        return self._fn(image, severity=self.severity)

    def __repr__(self):
        return (f"DegradationTransform(type={self.degradation_type}, "
                f"severity={self.severity}, seed={self.seed})")
