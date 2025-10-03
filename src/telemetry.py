"""Telemetry and logging utilities for inference tracking."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


INFERENCE_LOG_PATH = Path("logs") / "inference_log.csv"
INFERENCE_LOG_COLUMNS = [
    "timestamp",
    "run_id",
    "model_path",
    "image_name",
    "top1_label",
    "top1_confidence",
    "latency_ms",
]


def log_inference(
    run_id: Optional[str],
    model_path: str,
    image_name: str,
    top1_label: str,
    top1_confidence: float,
    latency_ms: float,
) -> None:
    """Log a single inference event to inference_log.csv atomically."""
    INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    write_header = not INFERENCE_LOG_PATH.exists()
    temp_path = INFERENCE_LOG_PATH.with_suffix(".tmp")
    
    # Read existing content
    existing_content = ""
    if INFERENCE_LOG_PATH.exists():
        existing_content = INFERENCE_LOG_PATH.read_text(encoding="utf-8")
    
    # Write atomically
    with temp_path.open("w", newline="", encoding="utf-8") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=INFERENCE_LOG_COLUMNS)
        
        if existing_content:
            tmp.write(existing_content)
            if not existing_content.endswith("\n"):
                tmp.write("\n")
        else:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id or "unknown",
            "model_path": model_path,
            "image_name": image_name,
            "top1_label": top1_label,
            "top1_confidence": f"{top1_confidence:.4f}",
            "latency_ms": f"{latency_ms:.2f}",
        })
    
    temp_path.replace(INFERENCE_LOG_PATH)
