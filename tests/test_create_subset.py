"""
Unit tests for create_subset.py
Tests subset creation, balanced sampling, and file operations
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import os
import sys
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

# Import the functions we want to test
from create_subset import (
    collect_classes, 
    create_subset,
    safe_symlink
)

class TestCreateSubset:
    """Test suite for subset creation functionality"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset structure for testing"""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample classes with files
        classes_data = {
            'class_a': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'class_b': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'class_c': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg']
        }
        
        for class_name, filenames in classes_data.items():
            class_dir = Path(temp_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for filename in filenames:
                # Create dummy image files
                file_path = class_dir / filename
                file_path.write_text(f"dummy content for {filename}")
        
        yield temp_dir, classes_data
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def subset_output_dir(self):
        """Create temporary directory for subset output"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_collect_classes(self, sample_dataset):
        """Test class and sample collection"""
        data_dir, expected_classes = sample_dataset
        
        classes_dict = collect_classes(Path(data_dir))
        
        # Check that all classes are found
        assert set(classes_dict.keys()) == set(expected_classes.keys())
        
        # Check sample counts
        for class_name, expected_files in expected_classes.items():
            assert len(classes_dict[class_name]) == len(expected_files)
            
        # Check that files have full paths
        for class_name, file_paths in classes_dict.items():
            for file_path in file_paths:
                assert file_path.exists()
                assert class_name in str(file_path)
    
    def test_create_subset_full_workflow(self, sample_dataset, subset_output_dir):
        """Test the complete subset creation workflow"""
        data_dir, expected_classes = sample_dataset
        subset_dir = subset_output_dir
        
        # Create subset
        result = create_subset(
            data_dir=Path(data_dir),
            out_dir=Path(subset_dir),
            samples_per_class=2,
            val_ratio=0.5,
            seed=42
        )
        
        # Check directory structure
        subset_path = Path(subset_dir)
        train_dir = subset_path / 'train'
        val_dir = subset_path / 'val'
        
        assert train_dir.exists()
        assert val_dir.exists()
        
        # Check that all classes exist in both train and val
        for class_name in expected_classes.keys():
            assert (train_dir / class_name).exists()
            assert (val_dir / class_name).exists()
            
            # Check that there are files in each class directory
            train_files = list((train_dir / class_name).glob('*'))
            val_files = list((val_dir / class_name).glob('*'))
            
            assert len(train_files) > 0
            assert len(val_files) > 0
            
            # Total should not exceed requested samples per class
            assert len(train_files) + len(val_files) <= 2
    
    def test_deterministic_behavior(self, sample_dataset):
        """Test that subset creation is deterministic with same random seed"""
        data_dir, _ = sample_dataset
        
        # Create two temporary directories for comparison
        subset_dir1 = tempfile.mkdtemp()
        subset_dir2 = tempfile.mkdtemp()
        
        try:
            # Create subsets with same parameters
            create_subset(Path(data_dir), Path(subset_dir1), samples_per_class=2, 
                         val_ratio=0.5, seed=42)
            create_subset(Path(data_dir), Path(subset_dir2), samples_per_class=2, 
                         val_ratio=0.5, seed=42)
            
            # Compare directory structures (both should have same structure)
            for split in ['train', 'val']:
                dir1_classes = set(d.name for d in (Path(subset_dir1) / split).iterdir() if d.is_dir())
                dir2_classes = set(d.name for d in (Path(subset_dir2) / split).iterdir() if d.is_dir())
                assert dir1_classes == dir2_classes
            
        finally:
            shutil.rmtree(subset_dir1, ignore_errors=True)
            shutil.rmtree(subset_dir2, ignore_errors=True)
    
    def test_empty_directory_handling(self):
        """Test handling of empty or non-existent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / 'empty'
            empty_dir.mkdir()
            
            # Should raise an appropriate error
            with pytest.raises(Exception):
                collect_classes(empty_dir)
    
    def test_invalid_samples_per_class(self, sample_dataset):
        """Test handling of invalid samples_per_class values"""
        data_dir, _ = sample_dataset
        
        with tempfile.TemporaryDirectory() as subset_dir:
            # Test negative value
            with pytest.raises(Exception):  # Could be ValueError or other
                create_subset(Path(data_dir), Path(subset_dir), samples_per_class=-1)
            
            # Test zero value  
            with pytest.raises(Exception):  # Could be ValueError or other
                create_subset(Path(data_dir), Path(subset_dir), samples_per_class=0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])