import argparse
from pathlib import Path
import sys

import cv2

REPO_INSIGHTFACE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON_PACKAGE = REPO_INSIGHTFACE_ROOT / "python-package"
if str(LOCAL_PYTHON_PACKAGE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGE))

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual demo for InsightFace pretrained model on CPU"
    )
    parser.add_argument("--model", type=str, default="buffalo_s")
    parser.add_argument(
        "--model-root",
        type=str,
        default=str((Path.cwd() / ".insightface").resolve()),
        help="Model cache root directory (default: ./.insightface in current workspace)",
    )
    parser.add_argument("--input", type=str, default="t1",
                        help="Image path. Use 't1' to load built-in sample.")
    parser.add_argument("--output", type=str, default="insightface/examples/face_recognition/demo_output.jpg")
    parser.add_argument("--det-size", type=int, default=640)
    args = parser.parse_args()

    app = FaceAnalysis(
        name=args.model,
        root=args.model_root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    if args.input.lower() == "t1":
        img = ins_get_image("t1")
    else:
        img = cv2.imread(args.input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")

    faces = app.get(img)
    out = app.draw_on(img, faces)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)

    print(f"Detected faces: {len(faces)}")
    print(f"Saved visualized output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
