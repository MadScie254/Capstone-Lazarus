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
    collect_classes_and_samples, 
    create_balanced_subset,
    safe_symlink,
    create_subset
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
    
    def test_collect_classes_and_samples(self, sample_dataset):
        """Test class and sample collection"""
        data_dir, expected_classes = sample_dataset
        
        classes_dict = collect_classes_and_samples(data_dir)
        
        # Check that all classes are found
        assert set(classes_dict.keys()) == set(expected_classes.keys())
        
        # Check sample counts
        for class_name, expected_files in expected_classes.items():
            assert len(classes_dict[class_name]) == len(expected_files)
            
        # Check that files have full paths
        for class_name, file_paths in classes_dict.items():
            for file_path in file_paths:
                assert os.path.exists(file_path)
                assert class_name in file_path
    
    def test_create_balanced_subset_sufficient_samples(self, sample_dataset):
        """Test balanced subset creation when all classes have enough samples"""
        data_dir, _ = sample_dataset
        classes_dict = collect_classes_and_samples(data_dir)
        
        # Request 2 samples per class (all classes have at least 3)
        samples_per_class = 2
        train_data, val_data = create_balanced_subset(
            classes_dict, samples_per_class, val_split=0.5, random_state=42
        )
        
        # Check that we get the right number of classes
        assert len(train_data) == len(classes_dict)
        assert len(val_data) == len(classes_dict)
        
        # Check sample counts per class
        for class_name in classes_dict.keys():
            expected_train_samples = samples_per_class // 2  # 50% split = 1 sample
            expected_val_samples = samples_per_class - expected_train_samples  # = 1 sample
            
            assert len(train_data[class_name]) == expected_train_samples
            assert len(val_data[class_name]) == expected_val_samples
    
    def test_create_balanced_subset_insufficient_samples(self, sample_dataset):
        """Test balanced subset creation when some classes don't have enough samples"""
        data_dir, _ = sample_dataset
        classes_dict = collect_classes_and_samples(data_dir)
        
        # Request 10 samples per class (class_b only has 3)
        samples_per_class = 10
        train_data, val_data = create_balanced_subset(
            classes_dict, samples_per_class, val_split=0.5, random_state=42
        )
        
        # Should be limited by the class with fewest samples (class_b with 3)
        max_possible = min(len(files) for files in classes_dict.values())
        
        for class_name, files in classes_dict.items():
            total_selected = len(train_data[class_name]) + len(val_data[class_name])
            assert total_selected <= max_possible
            assert total_selected <= len(files)
    
    def test_safe_symlink_creates_symlink(self, sample_dataset, subset_output_dir):
        """Test that safe_symlink creates symlinks when possible"""
        data_dir, _ = sample_dataset
        source_file = Path(data_dir) / 'class_a' / 'img1.jpg'
        target_file = Path(subset_output_dir) / 'test_link.jpg'
        
        # Ensure parent directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = safe_symlink(source_file, target_file)
        
        # Check result
        assert target_file.exists()
        
        # On systems that support symlinks, result should be True
        # On systems that don't (some Windows), result might be False (copy fallback)
        assert isinstance(result, bool)
        
        # File content should match
        assert target_file.read_text() == source_file.read_text()
    
    def test_safe_symlink_fallback_to_copy(self, sample_dataset, subset_output_dir):
        """Test that safe_symlink falls back to copy when symlink fails"""
        data_dir, _ = sample_dataset
        source_file = Path(data_dir) / 'class_a' / 'img1.jpg'
        target_file = Path(subset_output_dir) / 'test_copy.jpg'
        
        # Ensure parent directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Force copy by making a mock that always raises for symlink
        import shutil as shutil_orig
        
        def mock_symlink(*args, **kwargs):
            raise OSError("Symlink not supported")
        
        # Patch os.symlink temporarily
        original_symlink = getattr(os, 'symlink', None)
        os.symlink = mock_symlink
        
        try:
            result = safe_symlink(source_file, target_file)
            
            # Should have fallen back to copy (result = False)
            assert result is False
            assert target_file.exists()
            assert target_file.read_text() == source_file.read_text()
        finally:
            # Restore original symlink function
            if original_symlink:
                os.symlink = original_symlink
            elif hasattr(os, 'symlink'):
                delattr(os, 'symlink')
    
    def test_create_subset_full_workflow(self, sample_dataset, subset_output_dir):
        """Test the complete subset creation workflow"""
        data_dir, expected_classes = sample_dataset
        subset_dir = subset_output_dir
        
        # Create subset
        result = create_subset(
            data_dir=data_dir,
            subset_dir=subset_dir,
            samples_per_class=2,
            val_split=0.5,
            random_state=42
        )
        
        # Check return value
        assert result is True
        
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
    
    def test_create_subset_with_metadata(self, sample_dataset, subset_output_dir):
        """Test subset creation saves metadata correctly"""
        data_dir, _ = sample_dataset
        subset_dir = subset_output_dir
        
        # Create subset
        create_subset(
            data_dir=data_dir,
            subset_dir=subset_dir,
            samples_per_class=2,
            val_split=0.3,
            random_state=123
        )
        
        # Check metadata file
        metadata_file = Path(subset_dir) / 'subset_metadata.json'
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata content
        assert metadata['samples_per_class'] == 2
        assert metadata['val_split'] == 0.3
        assert metadata['random_state'] == 123
        assert 'total_samples' in metadata
        assert 'classes' in metadata
        assert len(metadata['classes']) == 3  # We have 3 classes
    
    def test_deterministic_behavior(self, sample_dataset):
        """Test that subset creation is deterministic with same random seed"""
        data_dir, _ = sample_dataset
        
        # Create two temporary directories for comparison
        subset_dir1 = tempfile.mkdtemp()
        subset_dir2 = tempfile.mkdtemp()
        
        try:
            # Create subsets with same parameters
            create_subset(data_dir, subset_dir1, samples_per_class=2, 
                         val_split=0.5, random_state=42)
            create_subset(data_dir, subset_dir2, samples_per_class=2, 
                         val_split=0.5, random_state=42)
            
            # Compare metadata
            with open(Path(subset_dir1) / 'subset_metadata.json', 'r') as f:
                meta1 = json.load(f)
            with open(Path(subset_dir2) / 'subset_metadata.json', 'r') as f:
                meta2 = json.load(f)
            
            # Should be identical (excluding timestamps)
            meta1.pop('created_at', None)
            meta2.pop('created_at', None)
            assert meta1 == meta2
            
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
                collect_classes_and_samples(str(empty_dir))
    
    def test_invalid_samples_per_class(self, sample_dataset):
        """Test handling of invalid samples_per_class values"""
        data_dir, _ = sample_dataset
        
        with tempfile.TemporaryDirectory() as subset_dir:
            # Test negative value
            with pytest.raises(ValueError):
                create_subset(data_dir, subset_dir, samples_per_class=-1)
            
            # Test zero value  
            with pytest.raises(ValueError):
                create_subset(data_dir, subset_dir, samples_per_class=0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])