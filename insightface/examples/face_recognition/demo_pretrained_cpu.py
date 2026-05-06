import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_INSIGHTFACE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON_PACKAGE = REPO_INSIGHTFACE_ROOT / "python-package"
if str(LOCAL_PYTHON_PACKAGE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGE))

from insightface.app import FaceAnalysis


def load_first_face_embedding(app: FaceAnalysis, image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"No face detected in: {image_path}")
    if len(faces) > 1:
        print(f"[WARN] Multiple faces in {image_path}. Using the first one.")

    return faces[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo face verification using InsightFace pretrained model on CPU"
    )
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--model", default="buffalo_l", help="Model pack name")
    parser.add_argument(
        "--model-root",
        type=str,
        default=str((Path.cwd() / ".insightface").resolve()),
        help="Model cache root directory (default: ./.insightface in current workspace)",
    )
    parser.add_argument("--threshold", type=float, default=0.65, help="Similarity threshold")
    parser.add_argument("--det-size", type=int, default=640, help="Detection input size")
    args = parser.parse_args()

    if not Path(args.img1).exists():
        raise FileNotFoundError(f"Missing image: {args.img1}")
    if not Path(args.img2).exists():
        raise FileNotFoundError(f"Missing image: {args.img2}")

    app = FaceAnalysis(
        name=args.model,
        root=args.model_root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    emb1 = load_first_face_embedding(app, args.img1)
    emb2 = load_first_face_embedding(app, args.img2)

    sim = cosine_similarity(emb1, emb2)
    same = sim >= args.threshold

    print(f"Model: {args.model}")
    print(f"Similarity: {sim:.4f}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Prediction: {'SAME PERSON' if same else 'DIFFERENT PERSON'}")


if __name__ == "__main__":
    main()
