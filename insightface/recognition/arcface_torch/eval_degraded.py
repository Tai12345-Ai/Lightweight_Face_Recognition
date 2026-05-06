"""
Degradation-aware evaluation for face recognition.

Evaluates recognition accuracy under various image degradations
using the same verification protocol as clean evaluation.

Usage:
    # Clean evaluation only
    python eval_degraded.py --network mbf --weight model.pt --rec /path/to/data

    # Degraded evaluation
    python eval_degraded.py \\
        --network mbf --weight model.pt --rec /path/to/data \\
        --targets lfw,cfp_fp,agedb_30 \\
        --degradations gaussian_blur,low_resolution,jpeg_compression \\
        --severities 1,3,5 --seed 42
"""

import argparse
import datetime
import os
import pickle
import sys

import numpy as np
import sklearn
import torch

try:
    import mxnet as mx
    from mxnet import ndarray as nd
except ImportError:
    mx = None

from backbones import get_model
from eval.verification import evaluate
from degradation.transforms import DegradationTransform, SUPPORTED_DEGRADATIONS


@torch.no_grad()
def load_bin_as_numpy(path, image_size=(112, 112)):
    """Load .bin verification set and return raw image arrays + labels.

    Returns:
        images: np.ndarray (N, H, W, 3) uint8
        issame_list: list of bool
    """
    assert mx is not None, "mxnet is required to load .bin verification files"
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')

    num_images = len(issame_list) * 2
    images = np.empty((num_images, image_size[0], image_size[1], 3),
                      dtype=np.uint8)

    for idx in range(num_images):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()  # (H, W, 3) RGB uint8
        if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
            img = mx.image.resize_short(
                mx.nd.array(img), image_size[0]).asnumpy()
        images[idx] = img
        if idx % 1000 == 0:
            print(f'  loading bin: {idx}/{num_images}')

    return images, issame_list


@torch.no_grad()
def extract_embeddings(images, backbone, batch_size=64, device='cuda'):
    """Extract embeddings from numpy images (N, H, W, 3) uint8.

    Performs flip augmentation and averages embeddings (same as verification.test).
    """
    num_images = images.shape[0]
    embeddings_list = []

    for flip in [False, True]:
        embeddings = np.zeros((num_images, 512), dtype=np.float32)
        ba = 0
        while ba < num_images:
            bb = min(ba + batch_size, num_images)
            batch_imgs = images[ba:bb].copy()

            if flip:
                batch_imgs = batch_imgs[:, :, ::-1, :].copy()

            # (B, H, W, 3) -> (B, 3, H, W), normalize
            batch_tensor = torch.from_numpy(
                batch_imgs.transpose(0, 3, 1, 2).astype(np.float32))
            batch_tensor = ((batch_tensor / 255.0) - 0.5) / 0.5
            batch_tensor = batch_tensor.to(device)

            output = backbone(batch_tensor).cpu().numpy()
            embeddings[ba:bb] = output
            ba = bb

        embeddings_list.append(embeddings)

    # Average flip embeddings then normalize
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    return embeddings


def eval_with_degradation(images, issame_list, backbone, degradation=None,
                          batch_size=64, device='cuda'):
    """Evaluate with optional degradation applied to images."""
    eval_images = images.copy()

    if degradation is not None:
        print(f'  applying {degradation}...')
        for i in range(len(eval_images)):
            eval_images[i] = degradation.apply(eval_images[i])

    embeddings = extract_embeddings(eval_images, backbone, batch_size, device)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list,
                                                  nrof_folds=10)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return acc_mean, acc_std


def main():
    parser = argparse.ArgumentParser(
        description='Degradation-aware Face Recognition Evaluation')
    parser.add_argument('--network', type=str, default='mbf',
                        help='Backbone name')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to model weights (.pt)')
    parser.add_argument('--rec', type=str, required=True,
                        help='Path to dataset dir containing .bin eval files')
    parser.add_argument('--targets', type=str, default='lfw,cfp_fp,agedb_30',
                        help='Comma-separated verification targets')
    parser.add_argument('--degradations', type=str, default='',
                        help='Comma-separated degradation types (empty=clean only)')
    parser.add_argument('--severities', type=str, default='1,3,5',
                        help='Comma-separated severity levels (1-5)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding_size', type=int, default=512)
    args = parser.parse_args()

    # Parse arguments
    targets = [t.strip() for t in args.targets.split(',')]
    severities = [int(s.strip()) for s in args.severities.split(',')]
    degradations = []
    if args.degradations and args.degradations.lower() != 'none':
        degradations = [d.strip() for d in args.degradations.split(',')]

    # Validate degradation names
    for d in degradations:
        if d not in SUPPORTED_DEGRADATIONS:
            print(f"ERROR: Unknown degradation '{d}'. "
                  f"Supported: {SUPPORTED_DEGRADATIONS}")
            sys.exit(1)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading model: {args.network} from {args.weight}')
    backbone = get_model(args.network, dropout=0, fp16=False,
                         num_features=args.embedding_size)
    backbone.load_state_dict(torch.load(args.weight, map_location=device))
    backbone = backbone.to(device)
    backbone.eval()

    # Results storage
    results = {}  # {target: {condition: (mean, std)}}

    for target_name in targets:
        bin_path = os.path.join(args.rec, target_name + '.bin')
        if not os.path.exists(bin_path):
            print(f'WARNING: {bin_path} not found, skipping {target_name}')
            continue

        print(f'\n=== {target_name.upper()} ===')
        images, issame_list = load_bin_as_numpy(bin_path)
        results[target_name] = {}

        # Clean evaluation
        print('  Evaluating clean...')
        clean_acc, clean_std = eval_with_degradation(
            images, issame_list, backbone, None, args.batch_size, device)
        results[target_name]['clean'] = (clean_acc, clean_std)
        print(f'  Clean: {clean_acc*100:.2f}% (±{clean_std*100:.2f})')

        # Degraded evaluation
        for deg_name in degradations:
            for sev in severities:
                condition = f'{deg_name}_s{sev}'
                deg = DegradationTransform(deg_name, severity=sev,
                                           seed=args.seed)
                print(f'  Evaluating {condition}...')
                deg_acc, deg_std = eval_with_degradation(
                    images, issame_list, backbone, deg,
                    args.batch_size, device)
                results[target_name][condition] = (deg_acc, deg_std)
                drop = (clean_acc - deg_acc) * 100
                print(f'  {condition}: {deg_acc*100:.2f}% '
                      f'(drop: {drop:+.2f}%)')

    # Print summary table
    print('\n' + '=' * 80)
    print('DEGRADATION EVALUATION SUMMARY')
    print(f'Model: {args.network} | Weight: {args.weight}')
    print('=' * 80)

    for target_name, target_results in results.items():
        print(f'\n--- {target_name.upper()} ---')
        clean_acc = target_results['clean'][0]
        print(f'  {"Condition":<30} {"Accuracy":>10} {"Drop":>10}')
        print(f'  {"-"*30} {"-"*10} {"-"*10}')
        print(f'  {"Clean":<30} {clean_acc*100:>9.2f}% {"---":>10}')

        for condition, (acc, std) in target_results.items():
            if condition == 'clean':
                continue
            drop = (clean_acc - acc) * 100
            print(f'  {condition:<30} {acc*100:>9.2f}% {drop:>+9.2f}%')

    print('\n' + '=' * 80)


if __name__ == '__main__':
    main()
