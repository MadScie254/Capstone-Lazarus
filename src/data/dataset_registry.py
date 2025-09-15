"""
Dataset Registry and Management for CAPSTONE-LAZARUS
===================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import shutil

logger = logging.getLogger(__name__)

@dataclass
class DatasetMetadata:
    """Metadata for dataset registration"""
    name: str
    version: str
    description: str
    path: str
    task_type: str  # classification, regression, etc.
    num_samples: int
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    created_at: str = ""
    updated_at: str = ""
    checksum: str = ""
    tags: List[str] = None
    preprocessing_steps: List[str] = None
    validation_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []
        if self.validation_metrics is None:
            self.validation_metrics = {}

class DatasetRegistry:
    """Registry for managing datasets and their metadata"""
    
    def __init__(self, registry_path: str = "data/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.datasets = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load dataset registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save dataset registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.datasets, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for dataset integrity"""
        hash_md5 = hashlib.md5()
        
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        elif path.is_dir():
            # Calculate checksum for directory contents
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def register_dataset(self, metadata: DatasetMetadata, overwrite: bool = False) -> bool:
        """
        Register a new dataset
        
        Args:
            metadata: Dataset metadata
            overwrite: Whether to overwrite existing dataset
            
        Returns:
            Success status
        """
        dataset_key = f"{metadata.name}:{metadata.version}"
        
        if dataset_key in self.datasets and not overwrite:
            logger.warning(f"Dataset {dataset_key} already exists. Use overwrite=True to update.")
            return False
        
        # Calculate checksum
        dataset_path = Path(metadata.path)
        if dataset_path.exists():
            metadata.checksum = self._calculate_checksum(dataset_path)
        
        # Update timestamp
        metadata.updated_at = datetime.now().isoformat()
        
        # Store in registry
        self.datasets[dataset_key] = asdict(metadata)
        self._save_registry()
        
        logger.info(f"Dataset {dataset_key} registered successfully")
        return True
    
    def get_dataset(self, name: str, version: str = "latest") -> Optional[DatasetMetadata]:
        """
        Get dataset metadata
        
        Args:
            name: Dataset name
            version: Dataset version or "latest"
            
        Returns:
            Dataset metadata or None
        """
        if version == "latest":
            # Find latest version
            matching_datasets = [key for key in self.datasets.keys() if key.startswith(f"{name}:")]
            if not matching_datasets:
                return None
            dataset_key = max(matching_datasets)  # Assumes lexicographic ordering works for versions
        else:
            dataset_key = f"{name}:{version}"
        
        if dataset_key in self.datasets:
            return DatasetMetadata(**self.datasets[dataset_key])
        
        return None
    
    def list_datasets(self, task_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[DatasetMetadata]:
        """
        List registered datasets with optional filtering
        
        Args:
            task_type: Filter by task type
            tags: Filter by tags (must contain all specified tags)
            
        Returns:
            List of dataset metadata
        """
        datasets = []
        
        for dataset_data in self.datasets.values():
            metadata = DatasetMetadata(**dataset_data)
            
            # Apply filters
            if task_type and metadata.task_type != task_type:
                continue
            
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            
            datasets.append(metadata)
        
        return sorted(datasets, key=lambda x: x.updated_at, reverse=True)
    
    def delete_dataset(self, name: str, version: str, remove_files: bool = False) -> bool:
        """
        Delete dataset from registry
        
        Args:
            name: Dataset name
            version: Dataset version
            remove_files: Whether to also remove dataset files
            
        Returns:
            Success status
        """
        dataset_key = f"{name}:{version}"
        
        if dataset_key not in self.datasets:
            logger.warning(f"Dataset {dataset_key} not found in registry")
            return False
        
        dataset_data = self.datasets[dataset_key]
        
        if remove_files:
            dataset_path = Path(dataset_data['path'])
            if dataset_path.exists():
                if dataset_path.is_dir():
                    shutil.rmtree(dataset_path)
                else:
                    dataset_path.unlink()
                logger.info(f"Removed dataset files at {dataset_path}")
        
        del self.datasets[dataset_key]
        self._save_registry()
        
        logger.info(f"Dataset {dataset_key} removed from registry")
        return True
    
    def validate_dataset(self, name: str, version: str) -> Dict[str, Any]:
        """
        Validate dataset integrity and consistency
        
        Args:
            name: Dataset name
            version: Dataset version
            
        Returns:
            Validation results
        """
        metadata = self.get_dataset(name, version)
        if not metadata:
            return {"valid": False, "error": "Dataset not found"}
        
        results = {"valid": True, "checks": []}
        
        # Check if dataset path exists
        dataset_path = Path(metadata.path)
        if not dataset_path.exists():
            results["valid"] = False
            results["checks"].append({"check": "path_exists", "passed": False, "message": "Dataset path not found"})
        else:
            results["checks"].append({"check": "path_exists", "passed": True, "message": "Dataset path exists"})
        
        # Check checksum if available
        if metadata.checksum and dataset_path.exists():
            current_checksum = self._calculate_checksum(dataset_path)
            checksum_valid = current_checksum == metadata.checksum
            results["checks"].append({
                "check": "checksum",
                "passed": checksum_valid,
                "message": "Checksum matches" if checksum_valid else "Checksum mismatch - dataset may be corrupted"
            })
            if not checksum_valid:
                results["valid"] = False
        
        return results
    
    def add_validation_metrics(self, name: str, version: str, metrics: Dict[str, float]):
        """Add validation metrics to dataset metadata"""
        dataset_key = f"{name}:{version}"
        
        if dataset_key in self.datasets:
            self.datasets[dataset_key]["validation_metrics"].update(metrics)
            self.datasets[dataset_key]["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Added validation metrics to {dataset_key}")
        else:
            logger.warning(f"Dataset {dataset_key} not found")
    
    def tag_dataset(self, name: str, version: str, tags: List[str]):
        """Add tags to dataset"""
        dataset_key = f"{name}:{version}"
        
        if dataset_key in self.datasets:
            existing_tags = set(self.datasets[dataset_key]["tags"])
            existing_tags.update(tags)
            self.datasets[dataset_key]["tags"] = list(existing_tags)
            self.datasets[dataset_key]["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Added tags {tags} to {dataset_key}")
        else:
            logger.warning(f"Dataset {dataset_key} not found")
    
    def search_datasets(self, query: str) -> List[DatasetMetadata]:
        """Search datasets by name, description, or tags"""
        query = query.lower()
        results = []
        
        for dataset_data in self.datasets.values():
            metadata = DatasetMetadata(**dataset_data)
            
            # Search in name, description, and tags
            searchable_text = f"{metadata.name} {metadata.description} {' '.join(metadata.tags)}".lower()
            
            if query in searchable_text:
                results.append(metadata)
        
        return sorted(results, key=lambda x: x.updated_at, reverse=True)
    
    def export_metadata(self, output_path: str):
        """Export registry metadata to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.datasets, f, indent=2, default=str)
        
        logger.info(f"Registry metadata exported to {output_path}")
    
    def import_metadata(self, input_path: str, merge: bool = True):
        """Import registry metadata from file"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Import file {input_path} not found")
            return
        
        try:
            with open(input_path, 'r') as f:
                imported_data = json.load(f)
            
            if merge:
                self.datasets.update(imported_data)
            else:
                self.datasets = imported_data
            
            self._save_registry()
            logger.info(f"Registry metadata imported from {input_path}")
        
        except Exception as e:
            logger.error(f"Failed to import metadata: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_datasets = len(self.datasets)
        
        if total_datasets == 0:
            return {"total_datasets": 0}
        
        task_types = {}
        total_samples = 0
        
        for dataset_data in self.datasets.values():
            metadata = DatasetMetadata(**dataset_data)
            
            # Count by task type
            task_types[metadata.task_type] = task_types.get(metadata.task_type, 0) + 1
            
            # Sum samples
            total_samples += metadata.num_samples
        
        return {
            "total_datasets": total_datasets,
            "total_samples": total_samples,
            "task_type_distribution": task_types,
            "average_samples_per_dataset": total_samples / total_datasets
        }