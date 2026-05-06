"""
Model efficiency benchmark for face recognition backbones.

Reports parameter count, model size, FLOPs, and CPU inference time.

Usage:
    # Single backbone
    python benchmark_model.py --network mbf

    # Compare multiple backbones
    python benchmark_model.py --networks mbf,shufflefacenet,vargfacenet
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

from backbones import get_model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    """Get model file size in MB (FP32)."""
    # Save to temporary file to measure actual size
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_tmp_model_size.pt')
    try:
        torch.save(model.state_dict(), tmp)
        size_mb = os.path.getsize(tmp) / (1024 * 1024)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return size_mb


def measure_cpu_inference(model, input_size=(1, 3, 112, 112),
                          num_warmup=10, num_runs=100):
    """Measure CPU inference time (ms)."""
    model = model.cpu()
    model.eval()
    dummy = torch.randn(*input_size)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def compute_flops(model, input_size=(1, 3, 112, 112)):
    """Compute FLOPs if ptflops is available."""
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, 112, 112), as_strings=False,
            print_per_layer_stat=False, verbose=False)
        return macs / (1000**3)  # GFLOPs
    except ImportError:
        return None


def benchmark_single(network_name, embedding_size=512):
    """Benchmark a single backbone."""
    print(f'\n{"="*60}')
    print(f'  Benchmark: {network_name}')
    print(f'{"="*60}')

    model = get_model(network_name, dropout=0, fp16=False,
                      num_features=embedding_size)
    model.eval()

    # Parameters
    total_params, trainable_params = count_parameters(model)
    print(f'  Parameters:     {total_params/1e6:.2f} M '
          f'(trainable: {trainable_params/1e6:.2f} M)')

    # Model size
    size_mb = get_model_size_mb(model)
    print(f'  Model size:     {size_mb:.1f} MB (FP32)')

    # FLOPs
    gflops = compute_flops(model)
    if gflops is not None:
        print(f'  FLOPs:          {gflops:.3f} GFLOPs')
        if hasattr(model, 'extra_gflops'):
            print(f'  Extra GFLOPs:   {model.extra_gflops:.3f}')
            print(f'  Total GFLOPs:   {gflops + model.extra_gflops:.3f}')
    else:
        print(f'  FLOPs:          N/A (install ptflops: pip install ptflops)')

    # CPU inference time (batch=1)
    mean_ms, std_ms = measure_cpu_inference(model, (1, 3, 112, 112))
    print(f'  CPU (batch=1):  {mean_ms:.1f} ms ± {std_ms:.1f} ms')

    # CPU inference time (batch=16)
    mean_ms_16, std_ms_16 = measure_cpu_inference(model, (16, 3, 112, 112),
                                                   num_warmup=3, num_runs=20)
    print(f'  CPU (batch=16): {mean_ms_16:.1f} ms ± {std_ms_16:.1f} ms')

    # Verify output shape
    with torch.no_grad():
        dummy = torch.randn(1, 3, 112, 112)
        out = model(dummy)
        print(f'  Output shape:   {list(out.shape)} '
              f'(expected [1, {embedding_size}])')

    return {
        'network': network_name,
        'params_m': total_params / 1e6,
        'size_mb': size_mb,
        'gflops': gflops,
        'cpu_ms_b1': mean_ms,
        'cpu_ms_b16': mean_ms_16,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark face recognition backbones')
    parser.add_argument('--network', type=str, default=None,
                        help='Single backbone to benchmark')
    parser.add_argument('--networks', type=str, default=None,
                        help='Comma-separated list of backbones')
    parser.add_argument('--embedding_size', type=int, default=512)
    args = parser.parse_args()

    if args.network is None and args.networks is None:
        args.networks = "mbf,shufflefacenet,vargfacenet"

    if args.network:
        networks = [args.network]
    else:
        networks = [n.strip() for n in args.networks.split(',')]

    all_results = []
    for net in networks:
        try:
            result = benchmark_single(net, args.embedding_size)
            all_results.append(result)
        except Exception as e:
            print(f'  ERROR benchmarking {net}: {e}')

    # Comparison table
    if len(all_results) > 1:
        print(f'\n{"="*80}')
        print('  COMPARISON TABLE')
        print(f'{"="*80}')
        header = (f'  {"Network":<20} {"Params(M)":>10} {"Size(MB)":>10} '
                  f'{"GFLOPs":>10} {"CPU b1(ms)":>12} {"CPU b16(ms)":>12}')
        print(header)
        print(f'  {"-"*20} {"-"*10} {"-"*10} {"-"*10} {"-"*12} {"-"*12}')
        for r in all_results:
            gflops_str = f"{r['gflops']:.3f}" if r['gflops'] else "N/A"
            print(f'  {r["network"]:<20} {r["params_m"]:>10.2f} '
                  f'{r["size_mb"]:>10.1f} {gflops_str:>10} '
                  f'{r["cpu_ms_b1"]:>12.1f} {r["cpu_ms_b16"]:>12.1f}')


if __name__ == '__main__':
    main()
