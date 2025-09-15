"""
Data Validation and Quality Checks for CAPSTONE-LAZARUS
=======================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results of data validation"""
    passed: bool
    score: float  # 0.0 to 1.0
    checks: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class DataValidator:
    """Comprehensive data validation and quality assessment"""
    
    def __init__(self, config=None):
        self.config = config
        
    def validate_dataset(self, 
                        dataset_path: str, 
                        dataset_type: str = "auto",
                        sample_size: Optional[int] = None) -> ValidationResult:
        """
        Comprehensive dataset validation
        
        Args:
            dataset_path: Path to dataset
            dataset_type: Type of dataset (image, tabular, auto)
            sample_size: Number of samples to validate (None = all)
            
        Returns:
            ValidationResult object
        """
        logger.info(f"Validating dataset at {dataset_path}")
        
        path = Path(dataset_path)
        if not path.exists():
            return ValidationResult(
                passed=False,
                score=0.0,
                checks={},
                warnings=[],
                errors=[f"Dataset path {dataset_path} does not exist"]
            )
        
        # Auto-detect dataset type
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(path)
        
        if dataset_type == "image":
            return self._validate_image_dataset(path, sample_size)
        elif dataset_type == "tabular":
            return self._validate_tabular_dataset(path, sample_size)
        else:
            return ValidationResult(
                passed=False,
                score=0.0,
                checks={},
                warnings=[],
                errors=[f"Unsupported dataset type: {dataset_type}"]
            )
    
    def _detect_dataset_type(self, path: Path) -> str:
        """Auto-detect dataset type"""
        if path.is_file() and path.suffix.lower() == '.csv':
            return "tabular"
        
        if path.is_dir():
            # Check for image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            has_images = any(
                f.suffix.lower() in image_extensions 
                for f in path.rglob("*") 
                if f.is_file()
            )
            if has_images:
                return "image"
            
            # Check for CSV files
            if list(path.glob("*.csv")):
                return "tabular"
        
        return "unknown"
    
    def _validate_image_dataset(self, path: Path, sample_size: Optional[int] = None) -> ValidationResult:
        """Validate image dataset"""
        checks = {}
        warnings = []
        errors = []
        
        try:
            # Check directory structure
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            checks["has_subdirectories"] = len(subdirs) > 0
            
            if len(subdirs) == 0:
                errors.append("No class directories found")
            else:
                checks["num_classes"] = len(subdirs)
                checks["class_names"] = [d.name for d in subdirs]
            
            # Check image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            total_images = 0
            class_distributions = {}
            corrupted_images = []
            image_sizes = []
            
            for class_dir in subdirs:
                if not class_dir.is_dir():
                    continue
                    
                class_images = [
                    f for f in class_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
                
                class_distributions[class_dir.name] = len(class_images)
                total_images += len(class_images)
                
                # Sample images for detailed validation
                sample_images = class_images
                if sample_size:
                    sample_images = class_images[:min(sample_size, len(class_images))]
                
                for img_path in sample_images:
                    try:
                        # Try to load image
                        image = tf.keras.utils.load_img(str(img_path))
                        image_sizes.append(image.size)
                    except Exception as e:
                        corrupted_images.append(str(img_path))
            
            checks["total_images"] = total_images
            checks["class_distribution"] = class_distributions
            checks["corrupted_images"] = len(corrupted_images)
            
            if corrupted_images:
                if len(corrupted_images) > 10:
                    warnings.append(f"Found {len(corrupted_images)} corrupted images")
                else:
                    warnings.extend([f"Corrupted image: {img}" for img in corrupted_images])
            
            # Check class balance
            if class_distributions:
                min_samples = min(class_distributions.values())
                max_samples = max(class_distributions.values())
                imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                checks["class_imbalance_ratio"] = imbalance_ratio
                
                if imbalance_ratio > 10:
                    warnings.append(f"High class imbalance detected (ratio: {imbalance_ratio:.2f})")
                elif imbalance_ratio > 3:
                    warnings.append(f"Moderate class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
            # Check image sizes
            if image_sizes:
                unique_sizes = list(set(image_sizes))
                checks["unique_image_sizes"] = len(unique_sizes)
                checks["image_size_variance"] = unique_sizes
                
                if len(unique_sizes) > 1:
                    warnings.append(f"Images have different sizes: {unique_sizes[:5]}...")
            
            # Overall assessment
            score = 1.0
            if errors:
                score = 0.0
            else:
                # Deduct for warnings and issues
                if corrupted_images:
                    score -= min(0.3, len(corrupted_images) / total_images)
                if len(unique_sizes) > 1:
                    score -= 0.1
                if "class_imbalance_ratio" in checks and checks["class_imbalance_ratio"] > 5:
                    score -= 0.2
            
            return ValidationResult(
                passed=len(errors) == 0,
                score=max(0.0, score),
                checks=checks,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                checks=checks,
                warnings=warnings,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def _validate_tabular_dataset(self, path: Path, sample_size: Optional[int] = None) -> ValidationResult:
        """Validate tabular dataset"""
        checks = {}
        warnings = []
        errors = []
        
        try:
            # Load data
            if path.is_file() and path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            elif path.is_dir():
                csv_files = list(path.glob("*.csv"))
                if not csv_files:
                    errors.append("No CSV files found in directory")
                    return ValidationResult(False, 0.0, checks, warnings, errors)
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                errors.append(f"Unsupported file format: {path.suffix}")
                return ValidationResult(False, 0.0, checks, warnings, errors)
            
            # Basic statistics
            checks["num_rows"] = len(df)
            checks["num_columns"] = len(df.columns)
            checks["column_names"] = list(df.columns)
            
            # Check for empty dataset
            if len(df) == 0:
                errors.append("Dataset is empty")
                return ValidationResult(False, 0.0, checks, warnings, errors)
            
            # Sample for detailed analysis
            if sample_size and len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=42)
            else:
                df_sample = df
            
            # Missing values analysis
            missing_values = df_sample.isnull().sum()
            checks["missing_values_per_column"] = missing_values.to_dict()
            checks["total_missing_values"] = missing_values.sum()
            checks["missing_value_percentage"] = (missing_values.sum() / (len(df_sample) * len(df_sample.columns))) * 100
            
            if missing_values.sum() > 0:
                high_missing_cols = missing_values[missing_values > len(df_sample) * 0.5].index.tolist()
                if high_missing_cols:
                    warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
            
            # Data type analysis
            dtypes_summary = df_sample.dtypes.value_counts().to_dict()
            checks["data_types"] = {str(k): v for k, v in dtypes_summary.items()}
            
            # Numerical columns analysis
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            checks["numeric_columns"] = len(numeric_cols)
            
            if numeric_cols:
                # Check for constant columns
                constant_cols = []
                for col in numeric_cols:
                    if df_sample[col].nunique() <= 1:
                        constant_cols.append(col)
                
                if constant_cols:
                    warnings.append(f"Constant columns detected: {constant_cols}")
                    checks["constant_columns"] = constant_cols
                
                # Check for outliers (simple IQR method)
                outlier_info = {}
                for col in numeric_cols[:10]:  # Check first 10 numeric columns
                    Q1 = df_sample[col].quantile(0.25)
                    Q3 = df_sample[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df_sample[(df_sample[col] < lower_bound) | (df_sample[col] > upper_bound)][col]
                    outlier_info[col] = len(outliers)
                
                checks["outliers_per_numeric_column"] = outlier_info
            
            # Categorical columns analysis
            categorical_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
            checks["categorical_columns"] = len(categorical_cols)
            
            if categorical_cols:
                high_cardinality_cols = []
                for col in categorical_cols:
                    unique_values = df_sample[col].nunique()
                    if unique_values > len(df_sample) * 0.8:  # High cardinality
                        high_cardinality_cols.append(col)
                
                if high_cardinality_cols:
                    warnings.append(f"High cardinality columns: {high_cardinality_cols}")
                    checks["high_cardinality_columns"] = high_cardinality_cols
            
            # Duplicates check
            num_duplicates = df_sample.duplicated().sum()
            checks["duplicate_rows"] = num_duplicates
            if num_duplicates > 0:
                warnings.append(f"Found {num_duplicates} duplicate rows")
            
            # Overall assessment
            score = 1.0
            
            # Deduct for issues
            if checks["missing_value_percentage"] > 50:
                score -= 0.4
            elif checks["missing_value_percentage"] > 20:
                score -= 0.2
            
            if num_duplicates > len(df_sample) * 0.1:
                score -= 0.2
            
            if len(warnings) > 5:
                score -= 0.1
            
            return ValidationResult(
                passed=len(errors) == 0,
                score=max(0.0, score),
                checks=checks,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                checks=checks,
                warnings=warnings,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def validate_data_drift(self, 
                           reference_data: np.ndarray, 
                           new_data: np.ndarray,
                           threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift between reference and new data using statistical tests
        
        Args:
            reference_data: Reference dataset
            new_data: New dataset to compare
            threshold: P-value threshold for significance
            
        Returns:
            Drift detection results
        """
        from scipy import stats
        
        results = {
            "drift_detected": False,
            "features_with_drift": [],
            "drift_scores": [],
            "p_values": [],
            "threshold": threshold
        }
        
        try:
            if reference_data.shape[1] != new_data.shape[1]:
                results["error"] = "Datasets have different number of features"
                return results
            
            for i in range(reference_data.shape[1]):
                ref_feature = reference_data[:, i]
                new_feature = new_data[:, i]
                
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
                
                results["drift_scores"].append(statistic)
                results["p_values"].append(p_value)
                
                if p_value < threshold:
                    results["features_with_drift"].append(i)
            
            results["drift_detected"] = len(results["features_with_drift"]) > 0
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def generate_data_profile(self, dataset_path: str) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        validation_result = self.validate_dataset(dataset_path)
        
        profile = {
            "dataset_path": dataset_path,
            "validation_result": asdict(validation_result),
            "profile_timestamp": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(validation_result)
        }
        
        return profile
    
    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        checks = validation_result.checks
        warnings = validation_result.warnings
        errors = validation_result.errors
        
        # Error-based recommendations
        if errors:
            recommendations.append("Fix critical errors before proceeding with model training")
        
        # Warning-based recommendations
        if "class_imbalance_ratio" in checks and checks["class_imbalance_ratio"] > 5:
            recommendations.append("Consider using class weighting, oversampling (SMOTE), or undersampling techniques")
        
        if "corrupted_images" in checks and checks["corrupted_images"] > 0:
            recommendations.append("Remove or repair corrupted images before training")
        
        if "missing_value_percentage" in checks and checks["missing_value_percentage"] > 10:
            recommendations.append("Address missing values using imputation or removal strategies")
        
        if "duplicate_rows" in checks and checks["duplicate_rows"] > 0:
            recommendations.append("Remove duplicate rows to avoid data leakage")
        
        if "unique_image_sizes" in checks and checks["unique_image_sizes"] > 1:
            recommendations.append("Resize all images to a consistent size before training")
        
        if "high_cardinality_columns" in checks:
            recommendations.append("Consider encoding or dimensionality reduction for high cardinality categorical features")
        
        if validation_result.score < 0.8:
            recommendations.append("Dataset quality is below recommended threshold. Review and clean data before training")
        
        return recommendations
    
    def save_validation_report(self, validation_result: ValidationResult, output_path: str):
        """Save validation report to file"""
        report = {
            "validation_result": asdict(validation_result),
            "recommendations": self._generate_recommendations(validation_result),
            "report_generated_at": datetime.now().isoformat()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")