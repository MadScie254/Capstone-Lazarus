from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTENSIONS:
        yield from directory.rglob(f"*{ext}")


@dataclass
class ClassSummary:
    name: str
    image_count: int
    sample_paths: List[Path]


class DatasetManager:
    """Convenience helpers for exploring the on-disk dataset."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not self.class_dirs:
            raise RuntimeError(f"No class folders discovered in {self.data_dir}")

        self.class_names: List[str] = sorted(d.name for d in self.class_dirs)
        self._class_cache: Dict[str, ClassSummary] = {}

    def _build_summary(self, class_name: str) -> ClassSummary:
        if class_name not in self._class_cache:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                raise KeyError(f"Unknown class: {class_name}")

            paths = [path for path in _iter_image_paths(class_dir)]
            self._class_cache[class_name] = ClassSummary(
                name=class_name,
                image_count=len(paths),
                sample_paths=paths[: min(24, len(paths))],
            )
        return self._class_cache[class_name]

    def get_class_statistics(self) -> Dict[str, object]:
        """Return aggregate dataset statistics for dashboard consumption."""

        summaries = [self._build_summary(cls) for cls in self.class_names]
        class_distribution = {summary.name: summary.image_count for summary in summaries}
        total_images = sum(class_distribution.values())
        valid_images = total_images
        corrupted_images = 0
        imbalance_ratio = 0.0
        counts = class_distribution.values()
        if counts:
            minimum = min(counts)
            maximum = max(counts)
            if minimum > 0:
                imbalance_ratio = maximum / minimum

        dataframe = pd.DataFrame(
            [
                {
                    "class": summary.name,
                    "images": summary.image_count,
                }
                for summary in summaries
            ]
        )

        return {
            "total_images": total_images,
            "valid_images": valid_images,
            "corrupted_images": corrupted_images,
            "num_classes": len(self.class_names),
            "class_distribution": class_distribution,
            "class_names": self.class_names,
            "imbalance_ratio": imbalance_ratio,
            "dataframe": dataframe,
        }

    def list_samples(self, class_name: str, limit: int = 12) -> List[Path]:
        summary = self._build_summary(class_name)
        return summary.sample_paths[:limit]

    def sample_preview(self, max_images: int = 12) -> List[Path]:
        samples = list(
            itertools.islice(
                (path for cls in self.class_names for path in self.list_samples(cls, limit=max_images)),
                max_images,
            )
        )
        return samples

    def verify_image(self, path: Path) -> bool:
        try:
            with Image.open(path) as handle:
                handle.verify()
            return True
        except Exception:
            return False

    def refresh(self) -> None:
        self._class_cache.clear()
        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.class_names = sorted(d.name for d in self.class_dirs)
