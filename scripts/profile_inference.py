"""Inference profiler for measuring latency and memory usage."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.model_factory_torch import get_model


def profile_inference(
    model_path: Path,
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
    input_size: int = 224,
) -> None:
    """Profile model inference latency and memory usage."""
    
    print(f"📊 Profiling model: {model_path}")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Input size: {input_size}")
    print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
    print("")
    
    # Load model
    device_obj = torch.device(device)
    
    # Infer model architecture from path or use default
    model_name = "efficientnet_b0"  # Default
    if "mobilenet" in str(model_path).lower():
        model_name = "mobilenetv3_small"
    elif "resnet" in str(model_path).lower():
        model_name = "resnet18"
    
    print(f"Loading model ({model_name})...")
    model = get_model(model_name, num_classes=10, pretrained=False)  # Placeholder num_classes
    
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device_obj)
        model.load_state_dict(state_dict)
        print("✓ Loaded checkpoint")
    else:
        print("⚠️  Checkpoint not found, using random weights")
    
    model.to(device_obj)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device_obj)
    
    # Memory tracking
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({num_runs} runs)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    # Results
    latencies_arr = np.array(latencies)
    
    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean latency:   {latencies_arr.mean():.2f} ms")
    print(f"Median latency: {np.median(latencies_arr):.2f} ms")
    print(f"Min latency:    {latencies_arr.min():.2f} ms")
    print(f"Max latency:    {latencies_arr.max():.2f} ms")
    print(f"Std dev:        {latencies_arr.std():.2f} ms")
    print(f"P50:            {np.percentile(latencies_arr, 50):.2f} ms")
    print(f"P95:            {np.percentile(latencies_arr, 95):.2f} ms")
    print(f"P99:            {np.percentile(latencies_arr, 99):.2f} ms")
    
    if device == "cuda" and torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"\nPeak GPU memory: {peak_mem:.2f} MB")
    
    # Throughput
    throughput = (batch_size * num_runs) / (sum(latencies) / 1000)
    print(f"\nThroughput:     {throughput:.2f} images/sec")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Profile inference latency and memory usage for trained models"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint (e.g., models/run_001/best.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmarking",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs before benchmarking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (square)",
    )
    
    args = parser.parse_args()
    
    profile_inference(
        model_path=args.model,
        device=args.device,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        batch_size=args.batch_size,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
