import argparse
import sys
from pathlib import Path

import numpy as np

REPO_INSIGHTFACE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON_PACKAGE = REPO_INSIGHTFACE_ROOT / "python-package"
if str(LOCAL_PYTHON_PACKAGE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGE))

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick smoke test for InsightFace pretrained model on CPU"
    )
    parser.add_argument("--model", type=str, default="buffalo_s",
                        help="Model pack name: buffalo_s, buffalo_l, antelopev2")
    parser.add_argument(
        "--model-root",
        type=str,
        default=str((Path.cwd() / ".insightface").resolve()),
        help="Model cache root directory (default: ./.insightface in current workspace)",
    )
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size")
    args = parser.parse_args()

    print(f"[INFO] Loading model pack: {args.model}")
    app = FaceAnalysis(
        name=args.model,
        root=args.model_root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    print("[INFO] Loading built-in sample image: t1")
    img = ins_get_image("t1")
    faces = app.get(img)

    if len(faces) == 0:
        print("[FAIL] No face detected on sample image.")
        return 1

    print(f"[INFO] Detected faces: {len(faces)}")

    emb = faces[0].embedding.astype(np.float32)
    sim_self = cosine_similarity(emb, emb)
    print(f"[INFO] Self similarity of first face: {sim_self:.6f}")

    if len(faces) >= 2:
        emb2 = faces[1].embedding.astype(np.float32)
        sim_12 = cosine_similarity(emb, emb2)
        print(f"[INFO] Similarity between face#1 and face#2: {sim_12:.6f}")

    if sim_self < 0.999:
        print("[FAIL] Embedding quality check failed (self similarity too low).")
        return 2

    print("[PASS] Pretrained CPU smoke test is OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
